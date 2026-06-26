"""推理入口 - 支持Task 1(GNN分类)和Task 2(序列推荐)

使用方法:
    # Task 1: GNN节点分类推理
    python infer.py --task task1 --data_path data/task1/graph.npz \
        --checkpoint output/task1/best_model.pt --output_path A1.csv

    # Task 2: 序列推荐推理
    python infer.py --task task2 --data_path data/task2/ \
        --checkpoint output/task2/best_model.pt --output_path A2.csv

输出格式:
    - A1.csv: node_id,predicted_category
    - A2.csv: user_id,top1,top2,...,top10

Agent可以修改推理逻辑,如集成学习、后处理等。
"""
import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import GNNClassifier, GRU4Rec, SASRec
from datasets import GraphDataset, RecDataset, load_rec_data
from rec_heuristics import (
    build_cooccur_stats,
    build_global_popularity,
    build_user_combo_profile_stats,
    build_user_profile_stats,
    choose_seq_col,
    get_user_combo_profile_counters,
    get_user_profile_counters,
    parse_item_counts,
    parse_seq,
    parse_user_profile_cols,
    rank_items,
)
from utils import (
    normalize_adj, normalize_adj_sparse,
    random_walk_normalize, random_walk_normalize_sparse, sparse_to_torch,
    preprocess_features,
    set_seed, get_device, setup_logger
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自主科研Agent推理脚本')

    # 任务相关
    parser.add_argument('--task', type=str, required=True, choices=['task1', 'task2'],
                        help='任务类型: task1=图分类, task2=序列推荐')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据路径: task1为.npz文件, task2为数据目录')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='模型检查点路径；task2启发式策略可不提供')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出CSV文件路径')

    # 推理相关
    parser.add_argument('--batch_size', type=int, default=256,
                        help='推理批次大小(Task 2)')
    parser.add_argument('--topk', type=int, default=10,
                        help='推荐Top-K数量(Task 2)')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 可选: 集成多个模型
    parser.add_argument('--ensemble_checkpoints', type=str, nargs='+', default=None,
                        help='多个检查点路径,用于集成推理')

    # A2推荐后处理/启发式融合
    parser.add_argument('--rec_strategy', type=str, default='model',
                        choices=['model', 'popular', 'last_item', 'history', 'hybrid'],
                        help='A2推荐策略: model=模型分数, popular=热门, history=历史共现, hybrid=融合')
    parser.add_argument('--history_filter', type=str, default='none',
                        choices=['none', 'soft', 'hard'],
                        help='A2是否处理历史item: none=保留, soft=降权, hard=删除')
    parser.add_argument('--seq_col', type=str, default='auto',
                        help='A2历史序列列名，默认自动选择 item_seq_dedup')
    parser.add_argument('--recent_n', type=int, default=20,
                        help='A2共现召回使用最近多少个历史item')
    parser.add_argument('--model_weight', type=float, default=1.0,
                        help='A2模型分数融合权重')
    parser.add_argument('--model_topn', type=int, default=300,
                        help='A2模型融合时每个用户保留多少个模型候选分数')
    parser.add_argument('--pop_weight', type=float, default=1.0,
                        help='A2热门度分数融合权重')
    parser.add_argument('--cooccur_weight', type=float, default=1.0,
                        help='A2历史共现分数融合权重')
    parser.add_argument('--user_weight', type=float, default=0.0,
                        help='A2用户画像分组热门度融合权重')
    parser.add_argument('--user_combo_weight', type=float, default=0.0,
                        help='A2用户画像前缀组合热门度融合权重')
    parser.add_argument('--user_combo_sizes', type=str, default='3,2,1',
                        help='A2用户画像前缀组合长度')
    parser.add_argument('--user_combo_min_count', type=int, default=5,
                        help='A2画像组合最少训练样本数')
    parser.add_argument('--history_count_weight', type=float, default=0.0,
                        help='A2当前用户历史item频次融合权重')
    parser.add_argument('--user_profile_cols', type=str, default='auto',
                        help='A2用户画像列，auto表示使用user.csv中除uid外全部列，空字符串表示关闭')
    parser.add_argument('--pop_penalty_weight', type=float, default=0.0,
                        help='A2热门item惩罚权重，用于提升推荐多样性')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 目标设备

    Returns:
        加载好的模型和参数字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = checkpoint.get('args', {})

    # 根据任务类型和模型类型创建模型
    task = args_dict.get('task', 'task1')
    model_type = args_dict.get('model_type', 'sage')

    if task == 'task1' or model_type in ['gcn', 'sage', 'gat']:
        # GNN分类器
        num_features = args_dict.get('num_features', 767)
        num_classes = args_dict.get('num_classes', 10)
        hidden_dim = args_dict.get('hidden_dim', 128)
        num_layers = args_dict.get('num_layers', 2)
        dropout = args_dict.get('dropout', 0.5)

        # 如果args_dict中没有,尝试从数据推断
        if num_features == 767 and num_classes == 10:
            logging.info("使用默认参数: num_features=767, num_classes=10")

        model = GNNClassifier(
            in_dim=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )
    else:
        # 序列推荐模型
        num_items = args_dict.get('num_items', 2156)
        embedding_dim = args_dict.get('embedding_dim', 64)
        hidden_dim = args_dict.get('hidden_dim', 128)
        num_layers = args_dict.get('num_layers', 1)
        dropout = args_dict.get('dropout', 0.2)
        max_len = args_dict.get('max_len', 50)
        num_heads = args_dict.get('num_heads', 2)

        if model_type == 'gru4rec':
            model = GRU4Rec(
                num_items=num_items,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                max_len=max_len
            )
        elif model_type == 'sasrec':
            model = SASRec(
                num_items=num_items,
                embedding_dim=embedding_dim,
                max_len=max_len,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info(f"模型加载成功: {model_type} (来自 {checkpoint_path})")
    logging.info(f"检查点信息: epoch={checkpoint.get('epoch', 'unknown')}")

    return model, args_dict


def _prepare_task1_tensors(data, model_args, device):
    """按单个检查点的训练配置准备A1推理张量

    不同检查点可能使用不同的邻接矩阵归一化或特征归一化方式。
    集成推理时必须逐个检查点复现对应的数据预处理，否则模型看到的
    输入分布会和训练时不一致，平均logits也会被错误输入污染。
    """
    normalize_type = model_args.get('normalize', 'symmetric')
    feature_norm = model_args.get('feature_norm', 'none')
    model_type = model_args.get('model_type', 'sage')
    adj_format = model_args.get('adj_format', 'dense')

    features_raw = preprocess_features(data['features'], method=feature_norm)
    if sp.issparse(features_raw):
        features = sparse_to_torch(features_raw, device=device)
    else:
        features = torch.FloatTensor(features_raw).to(device)

    use_sparse_adj = adj_format == 'sparse' and model_type == 'gcn'
    if normalize_type == 'symmetric':
        adj = normalize_adj_sparse(data['adj'], device=device) if use_sparse_adj else normalize_adj(data['adj']).to(device)
    elif normalize_type == 'random_walk':
        adj = random_walk_normalize_sparse(data['adj'], device=device) if use_sparse_adj else random_walk_normalize(data['adj']).to(device)
    else:
        if sp.issparse(data['adj']):
            adj = sparse_to_torch(data['adj'], device=device, return_sparse=use_sparse_adj)
        else:
            adj = torch.FloatTensor(data['adj']).to(device)

    return features, adj


def infer_task1(args):
    """Task 1推理 - GNN节点分类

    加载图数据和训练好的GNN模型,对测试节点进行分类预测,
    输出结果到CSV文件。
    """
    checkpoint_paths = args.ensemble_checkpoints or ([args.checkpoint] if args.checkpoint else [])
    if not checkpoint_paths:
        raise ValueError("Task 1推理必须提供 --checkpoint 或 --ensemble_checkpoints")

    logging.info("=" * 60)
    logging.info("开始 Task 1 推理 - GNN节点分类")
    logging.info("=" * 60)

    device = get_device(args.device)

    # 1. 加载数据
    logging.info(f"加载数据: {args.data_path}")
    data = GraphDataset.load(args.data_path)
    test_idx = data['test_idx']
    num_features = data['num_features']
    num_classes = data['num_classes']
    logging.info(f"测试节点数: {len(test_idx)}, 特征维度: {num_features}, 类别数: {num_classes}")

    test_idx_t = torch.LongTensor(test_idx).to(device)

    # 2. 逐个检查点推理并平均logits。
    # logits平均比标签投票更细，因为它保留每个类别的置信度信息。
    logging.info(f"开始推理，共 {len(checkpoint_paths)} 个检查点")
    logits_sum = None
    with torch.no_grad():
        for ckpt_id, checkpoint_path in enumerate(checkpoint_paths, start=1):
            logging.info(f"[{ckpt_id}/{len(checkpoint_paths)}] 加载检查点: {checkpoint_path}")
            model, model_args = load_model_from_checkpoint(checkpoint_path, device)
            features, adj = _prepare_task1_tensors(data, model_args, device)

            model.eval()
            logits = model(features, adj)[test_idx_t]
            if logits_sum is None:
                logits_sum = logits.detach().clone()
            else:
                if logits_sum.shape != logits.shape:
                    raise ValueError(
                        f"检查点输出shape不一致: {checkpoint_path}, "
                        f"{tuple(logits.shape)} != {tuple(logits_sum.shape)}"
                    )
                logits_sum += logits.detach()

            del model, features, adj, logits
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_logits = logits_sum / len(checkpoint_paths)
        predictions = torch.argmax(avg_logits, dim=1).cpu().numpy()

    logging.info(f"预测完成,预测类别分布:")
    unique, counts = np.unique(predictions, return_counts=True)
    for cls, cnt in zip(unique, counts):
        logging.info(f"  类别 {cls}: {cnt} 个")

    # 5. 保存结果
    # 输出格式: test_idx,label（与提交模板一致）
    result_df = pd.DataFrame({
        'test_idx': test_idx,
        'label': predictions
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    logging.info(f"结果已保存: {args.output_path} ({len(result_df)} 行)")

    return result_df


def infer_task2(args):
    """Task 2推理 - 序列推荐

    加载序列数据和训练好的推荐模型,为每个测试用户生成Top-10推荐,
    输出结果到CSV文件。
    """
    logging.info("=" * 60)
    logging.info("开始 Task 2 推理 - 序列推荐")
    logging.info("=" * 60)

    device = get_device(args.device)

    # 1. 加载数据
    logging.info(f"加载数据: {args.data_path}")
    rec_data = load_rec_data(args.data_path)

    # 读取测试数据 - 用于获取用户ID
    test_df = pd.read_csv(f'{args.data_path}/test.csv')
    test_seqs, test_targets, test_lens = rec_data['test']
    num_items = rec_data['num_items']
    logging.info(f"测试样本数: {len(test_seqs)}, 物品数: {num_items}")

    # 2. 加载模型
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    model_type = model_args.get('model_type', 'gru4rec')

    # 3. 创建DataLoader
    test_dataset = RecDataset(test_seqs, test_targets, test_lens)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 4. 推理
    logging.info(f"开始推理 (Top-{args.topk})...")
    all_predictions = []
    all_user_ids = []

    # 按用户分组的最后一个序列
    # 读取原始测试CSV来获取用户ID
    user_col = 'uid' if 'uid' in test_df.columns else 'user_id'
    user_ids = []
    seq_indices = []
    last_idx_per_user = {}

    for i, (uid, group) in enumerate(test_df.groupby(user_col)):
        items = group['item_id'].tolist()
        # 每个用户最后一个可预测的交互
        for j in range(1, len(items)):
            last_idx_per_user[uid] = len(user_ids)
            user_ids.append(uid)
            seq_indices.append(len(user_ids) - 1)

    # 如果测试数据是逐行格式,直接使用
    model.eval()
    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            item_seqs, targets, seq_lens = batch
            item_seqs = item_seqs.to(device)
            seq_lens = seq_lens.to(device)

            batch_size = item_seqs.size(0)

            # 获取序列表示
            if model_type == 'gru4rec':
                seq_repr = model(item_seqs, seq_lens)
            else:  # sasrec
                seq_repr = model(item_seqs)

            # 计算所有物品的分数
            all_item_emb = model.item_embedding.weight[1:]  # (num_items, embed_dim)
            scores = seq_repr @ all_item_emb.T  # (batch, num_items)

            # 排除历史交互过的物品
            for i in range(batch_size):
                hist_items = set(item_seqs[i].cpu().numpy())
                for item in hist_items:
                    if 1 <= item <= num_items:
                        scores[i, item - 1] = -1e10

            # 取Top-K
            top_scores, top_indices = torch.topk(scores, k=args.topk, dim=-1)
            top_indices = (top_indices + 1).cpu().numpy()  # 转回原始物品ID(1-based)

            all_predictions.extend(top_indices.tolist())

            batch_idx += 1
            if batch_idx % 100 == 0:
                logging.info(f"  已处理 {batch_idx * args.batch_size} 个样本...")

    # 5. 构建输出
    # 按用户聚合推荐结果
    # 每个用户取最后一次交互的推荐
    user_recommendations = {}

    # 重新按用户分组
    test_df = pd.read_csv(f'{args.data_path}/test.csv')
    user_col = 'uid' if 'uid' in test_df.columns else 'user_id'
    user_seq_idx = 0

    for user_id, group in test_df.groupby(user_col):
        items = group['item_id'].tolist()
        # 最后一个可预测位置
        n_seqs = len(items) - 1 if len(items) > 1 else 1
        # 取该用户最后一个序列的预测
        if user_seq_idx + n_seqs <= len(all_predictions):
            user_recommendations[user_id] = all_predictions[user_seq_idx + n_seqs - 1]
        else:
            # 回退:随机推荐
            user_recommendations[user_id] = list(range(1, args.topk + 1))
        user_seq_idx += n_seqs

    # 构建DataFrame
    result_rows = []
    for user_id in sorted(user_recommendations.keys()):
        recs = user_recommendations[user_id]
        row = {'user_id': user_id}
        for i in range(args.topk):
            row[f'top{i+1}'] = recs[i] if i < len(recs) else 0
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)

    # 确保列顺序: user_id, top1, top2, ..., top10
    col_order = ['user_id'] + [f'top{i}' for i in range(1, args.topk + 1)]
    result_df = result_df[col_order]

    # 6. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    logging.info(f"结果已保存: {args.output_path} ({len(result_df)} 用户)")

    # 输出一些统计信息
    logging.info("推荐结果统计:")
    for col in [f'top{i}' for i in range(1, min(4, args.topk + 1))]:
        logging.info(f"  {col}: {result_df[col].nunique()} 个不同物品")

    return result_df


def infer_task2_v2(args):
    """Task 2推理 - 支持模型/热门度/历史共现融合

    旧版推理只使用序列模型分数，并且默认硬排除历史item。离线验证发现，
    A2训练集中大量target会重复出现在历史序列中，硬排除会显著降低NDCG。
    因此这里把历史过滤改为可配置，并加入热门度、历史共现等轻量信号。
    """
    logging.info("=" * 60)
    logging.info("开始 Task 2 推理 (v2) - 序列推荐融合")
    logging.info("=" * 60)

    device = get_device(args.device)

    # 1. 读取训练/测试/候选item数据
    train_df = pd.read_csv(f'{args.data_path}/train.csv')
    test_df = pd.read_csv(f'{args.data_path}/test.csv')
    item_df = pd.read_csv(f'{args.data_path}/item.csv')
    user_col = 'uid' if 'uid' in test_df.columns else 'user_id'
    candidate_items = set(item_df['iid'].astype(str).tolist())
    train_seq_col = choose_seq_col(train_df, args.seq_col)
    test_seq_col = choose_seq_col(test_df, args.seq_col)

    global_pop = build_global_popularity(train_df)
    cooccur_stats = build_cooccur_stats(train_df, train_seq_col, recent_n=args.recent_n)

    user_cols = []
    user_lookup = None
    user_profile_stats = None
    user_combo_profile_stats = None
    user_path = f'{args.data_path}/user.csv'
    if args.user_weight > 0 or args.user_combo_weight > 0:
        if os.path.exists(user_path):
            user_df = pd.read_csv(user_path)
            user_cols = parse_user_profile_cols(user_df, user_col, args.user_profile_cols)
            if user_cols:
                if args.user_weight > 0:
                    user_profile_stats, user_lookup = build_user_profile_stats(
                        train_df=train_df,
                        user_df=user_df,
                        user_cols=user_cols,
                        user_col=user_col,
                    )
                    logging.info(f"启用用户画像单列融合: cols={user_cols}, weight={args.user_weight:.3f}")
                else:
                    user_lookup = user_df.set_index(user_col)
                if args.user_combo_weight > 0:
                    combo_sizes = [int(x.strip()) for x in args.user_combo_sizes.split(',') if x.strip()]
                    user_combo_profile_stats, user_lookup = build_user_combo_profile_stats(
                        train_df=train_df,
                        user_df=user_df,
                        user_cols=user_cols,
                        combo_sizes=combo_sizes,
                        min_count=args.user_combo_min_count,
                        user_col=user_col,
                    )
                    logging.info(
                        "启用用户画像组合融合: sizes=%s, min_count=%s, weight=%.3f",
                        combo_sizes, args.user_combo_min_count, args.user_combo_weight,
                    )
            else:
                logging.warning("启用用户画像融合，但未选择任何用户画像列，画像融合关闭")
        else:
            logging.warning("启用用户画像融合，但未找到user.csv，画像融合关闭")

    logging.info(
        "推荐配置: strategy=%s, history_filter=%s, seq_col=%s, recent_n=%s, "
        "weights(model/pop/cooccur/user/user_combo/hist_count)=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f, "
        "pop_penalty=%.3f",
        args.rec_strategy, args.history_filter, test_seq_col, args.recent_n,
        args.model_weight, args.pop_weight, args.cooccur_weight, args.user_weight,
        args.user_combo_weight, args.history_count_weight, args.pop_penalty_weight,
    )
    logging.info(f"候选item数: {len(candidate_items)}, target热门item数: {len(global_pop)}")

    # 2. 可选加载模型。如果只跑热门/共现策略，可以不提供checkpoint。
    model = None
    model_type = ''
    max_len = 50
    idx2iid = {}
    iid2idx = {}
    num_items = len(candidate_items)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        idx2iid = checkpoint.get('idx2iid', {})
        iid2idx = checkpoint.get('iid2idx', {})
        num_items = checkpoint.get('args', {}).get('num_items', len(idx2iid) or len(candidate_items))
        model, model_args = load_model_from_checkpoint(args.checkpoint, device)
        model_type = model_args.get('model_type', 'gru4rec')
        max_len = model_args.get('max_len', 50)
    elif args.rec_strategy == 'model':
        raise ValueError("rec_strategy=model 需要提供 --checkpoint；无checkpoint时请使用 history/popular/hybrid")
    elif args.rec_strategy == 'hybrid' and args.model_weight > 0:
        logging.warning("未提供checkpoint，hybrid策略将只融合热门度和历史共现")

    def precompute_model_scores():
        """批量计算测试用户模型Top-N分数，减少逐用户GPU调用开销"""
        if model is None or args.rec_strategy not in {'model', 'hybrid'}:
            return [None] * len(test_df)

        logging.info(
            f"批量计算模型Top-{args.model_topn}分数: batch_size={args.batch_size}, max_len={max_len}"
        )
        seqs = []
        lens = []
        for _, row in test_df.iterrows():
            items = parse_seq(row.get(test_seq_col))
            item_indices = [iid2idx.get(item, 0) for item in items]
            seq = item_indices[-max_len:]
            seq_len = len(seq)
            seq = [0] * (max_len - seq_len) + seq
            seqs.append(seq)
            lens.append(seq_len)

        model_scores_by_row = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(seqs), args.batch_size):
                end = min(start + args.batch_size, len(seqs))
                item_seq = torch.LongTensor(seqs[start:end]).to(device)
                seq_len_t = torch.LongTensor(lens[start:end]).to(device)

                if model_type == 'gru4rec':
                    seq_repr = model(item_seq, seq_len_t)
                else:
                    seq_repr = model(item_seq)
                all_item_emb = model.item_embedding.weight[1:]
                scores = seq_repr @ all_item_emb.T
                top_scores, top_indices = torch.topk(
                    scores, k=min(args.model_topn, scores.size(1)), dim=1
                )
                top_scores = top_scores.detach().cpu().numpy()
                top_indices = (top_indices.detach().cpu().numpy() + 1)

                for row_indices, row_scores in zip(top_indices, top_scores):
                    score_dict = {}
                    for model_idx, score in zip(row_indices, row_scores):
                        iid = idx2iid.get(int(model_idx))
                        if iid is not None:
                            score_dict[str(iid)] = float(score)
                    model_scores_by_row.append(score_dict)

                logging.info(f"  模型分数已处理 {end} / {len(seqs)}")
        return model_scores_by_row

    # 3. 为每个测试用户生成推荐。这里保持 test.csv 原始顺序，不再按uid排序。
    logging.info(f"为每个用户生成Top-{args.topk}推荐...")
    model_scores_by_row = precompute_model_scores()
    result_rows = []
    for row_idx, row in test_df.iterrows():
        user_id = str(row[user_col])
        items = parse_seq(row.get(test_seq_col))
        model_scores = model_scores_by_row[row_idx]
        user_counters = get_user_profile_counters(
            user_id=user_id,
            user_lookup=user_lookup,
            user_profile_stats=user_profile_stats,
            user_cols=user_cols,
        )
        user_combo_counters = get_user_combo_profile_counters(
            user_id=user_id,
            user_lookup=user_lookup,
            combo_profile_stats=user_combo_profile_stats,
            min_count=args.user_combo_min_count,
        )
        recs = rank_items(
            seq=items,
            candidate_items=candidate_items,
            global_pop=global_pop,
            cooccur_stats=cooccur_stats,
            topk=args.topk,
            strategy=args.rec_strategy,
            history_filter=args.history_filter,
            recent_n=args.recent_n,
            model_scores=model_scores,
            user_profile_counters=user_counters,
            user_combo_counters=user_combo_counters,
            history_counts=parse_item_counts(row.get('item_seq_counts')),
            model_weight=args.model_weight,
            pop_weight=args.pop_weight,
            cooccur_weight=args.cooccur_weight,
            user_weight=args.user_weight,
            user_combo_weight=args.user_combo_weight,
            history_count_weight=args.history_count_weight,
            pop_penalty_weight=args.pop_penalty_weight,
        )
        result_rows.append({
            'uid': user_id,
            'prediction': ','.join(recs)
        })

        if (row_idx + 1) % 1000 == 0:
            logging.info(f"  已处理 {row_idx + 1} / {len(test_df)} 个用户")

    # 4. 构建输出DataFrame
    # 提交格式: uid,prediction（逗号分隔的item id列表）
    result_df = pd.DataFrame(result_rows)

    # 5. 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    logging.info(f"结果已保存: {args.output_path} ({len(result_df)} 用户)")

    return result_df


def main():
    """主入口函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    setup_logger()
    logging.info(f"任务: {args.task}, 数据: {args.data_path}, 检查点: {args.checkpoint}")

    # 设备信息
    device = get_device(args.device)
    logging.info(f"使用设备: {device}")

    # 根据任务类型执行推理
    if args.task == 'task1':
        result = infer_task1(args)
    elif args.task == 'task2':
        # 使用v2版本(更简洁的按用户推理)
        result = infer_task2_v2(args)
    else:
        raise ValueError(f"未知的任务类型: {args.task}")

    logging.info("推理脚本执行完毕!")
    return result


if __name__ == '__main__':
    main()
