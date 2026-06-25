"""A2模型分数融合离线评估

用途：
1. 评估修复后的GRU4Rec/SASRec checkpoint 是否能给启发式推荐带来增益。
2. 批量计算验证集用户的模型Top-N分数，避免逐用户前向导致GPU利用率过低。
3. 搜索 model_weight / cooccur_weight / user_weight 等融合权重。
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from a2_offline_eval import ndcg_at_k, reciprocal_rank, split_train_val
from datasets import RecDataset, load_rec_data
from infer import load_model_from_checkpoint
from rec_heuristics import (
    build_cooccur_stats,
    build_global_popularity,
    build_user_profile_stats,
    get_user_profile_counters,
    parse_seq,
    parse_user_profile_cols,
    rank_items,
)
from utils import get_device


def parse_csv_float(value: str) -> List[float]:
    """解析逗号分隔浮点数"""
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_int(value: str) -> List[int]:
    """解析逗号分隔整数"""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A2模型+启发式融合验证")
    parser.add_argument("--data_path", type=str, default="data/rec_data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model_topn", type=int, default=200,
                        help="每个样本保留多少个模型候选分数")
    parser.add_argument("--seq_col", type=str, default="item_seq_raw")
    parser.add_argument("--recent_ns", type=str, default="6,8,10,12,15,20")
    parser.add_argument("--model_weights", type=str, default="0,0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--cooccur_weights", type=str, default="1.0")
    parser.add_argument("--user_weights", type=str, default="0,0.01,0.02,0.03,0.04")
    parser.add_argument("--pop_weights", type=str, default="1.0")
    parser.add_argument("--pop_penalty_weights", type=str, default="0")
    parser.add_argument("--history_filter", type=str, default="none",
                        choices=["none", "soft", "hard"])
    parser.add_argument("--user_profile_cols", type=str, default="auto")
    parser.add_argument("--top_results", type=int, default=30)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def compute_model_score_dicts(args, checkpoint_args, val_seqs, val_targets, val_lens, idx2iid):
    """批量计算验证集模型Top-N分数"""
    device = get_device(args.device)
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    model_type = model_args.get("model_type", checkpoint_args.get("model_type", "gru4rec"))
    model.eval()

    dataset = RecDataset(val_seqs, val_targets, val_lens)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    score_dicts: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader, start=1):
            item_seqs, _, seq_lens = batch
            item_seqs = item_seqs.to(device, non_blocking=device.type == "cuda")
            seq_lens = seq_lens.to(device, non_blocking=device.type == "cuda")

            if model_type == "gru4rec":
                seq_repr = model(item_seqs, seq_lens)
            else:
                seq_repr = model(item_seqs)

            scores = seq_repr @ model.item_embedding.weight[1:].T
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
                score_dicts.append(score_dict)

            if batch_id % 20 == 0:
                print(f"已计算模型分数: {len(score_dicts)} 条")

    return score_dicts


def main():
    """主入口"""
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    idx2iid = checkpoint.get("idx2iid", {})

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    candidate_items = set(item_df["iid"].astype(str).tolist())
    user_col = "uid" if "uid" in train_df.columns else "user_id"

    fit_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    rec_data = load_rec_data(
        args.data_path,
        max_seq_len=checkpoint_args.get("max_len", 50),
        seq_col=args.seq_col,
    )
    train_seqs, train_targets, train_lens = rec_data["train"]

    # 使用与 split_train_val 相同的随机排列，取出验证集对应的模型输入。
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_df))
    val_size = int(len(train_df) * args.val_ratio)
    val_indices = perm[:val_size]
    val_seqs = train_seqs[val_indices]
    val_targets = train_targets[val_indices]
    val_lens = train_lens[val_indices]

    print("=" * 100)
    print("A2模型+启发式融合验证")
    print("=" * 100)
    print(f"checkpoint={args.checkpoint}")
    print(f"验证样本={len(val_df)}, model_topn={args.model_topn}, batch_size={args.batch_size}")

    model_score_dicts = compute_model_score_dicts(
        args, checkpoint_args, val_seqs, val_targets, val_lens, idx2iid
    )
    global_pop = build_global_popularity(fit_df)

    user_cols = parse_user_profile_cols(user_df, user_col, args.user_profile_cols)
    user_profile_stats, user_lookup = build_user_profile_stats(
        train_df=fit_df,
        user_df=user_df,
        user_cols=user_cols,
        user_col=user_col,
    )

    recent_ns = parse_csv_int(args.recent_ns)
    model_weights = parse_csv_float(args.model_weights)
    cooccur_weights = parse_csv_float(args.cooccur_weights)
    user_weights = parse_csv_float(args.user_weights)
    pop_weights = parse_csv_float(args.pop_weights)
    pop_penalty_weights = parse_csv_float(args.pop_penalty_weights)

    results = []
    for recent_n in recent_ns:
        cooccur_stats = build_cooccur_stats(fit_df, args.seq_col, recent_n=recent_n)
        for model_weight in model_weights:
            for cooccur_weight in cooccur_weights:
                for user_weight in user_weights:
                    for pop_weight in pop_weights:
                        for pop_penalty_weight in pop_penalty_weights:
                            ndcgs, hits, mrrs = [], [], []
                            for row_pos, (_, row) in enumerate(val_df.iterrows()):
                                target = str(row["target_iid"])
                                user_id = str(row[user_col])
                                user_counters = get_user_profile_counters(
                                    user_id=user_id,
                                    user_lookup=user_lookup,
                                    user_profile_stats=user_profile_stats,
                                    user_cols=user_cols,
                                )
                                preds = rank_items(
                                    seq=parse_seq(row.get(args.seq_col)),
                                    candidate_items=candidate_items,
                                    global_pop=global_pop,
                                    cooccur_stats=cooccur_stats,
                                    topk=args.topk,
                                    strategy="hybrid",
                                    history_filter=args.history_filter,
                                    recent_n=recent_n,
                                    model_scores=model_score_dicts[row_pos],
                                    user_profile_counters=user_counters,
                                    model_weight=model_weight,
                                    pop_weight=pop_weight,
                                    cooccur_weight=cooccur_weight,
                                    user_weight=user_weight,
                                    pop_penalty_weight=pop_penalty_weight,
                                )
                                ndcgs.append(ndcg_at_k(preds, target, args.topk))
                                hits.append(1.0 if target in preds[:args.topk] else 0.0)
                                mrrs.append(reciprocal_rank(preds, target))

                            results.append({
                                "recent_n": recent_n,
                                "model_weight": model_weight,
                                "cooccur_weight": cooccur_weight,
                                "user_weight": user_weight,
                                "pop_weight": pop_weight,
                                "pop_penalty_weight": pop_penalty_weight,
                                "ndcg": float(np.mean(ndcgs)),
                                "hit": float(np.mean(hits)),
                                "mrr": float(np.mean(mrrs)),
                            })

    results = sorted(results, key=lambda item: item["ndcg"], reverse=True)
    print("\n排名 | recent_n | model_w | co_w | user_w | pop_w | pop_pen | NDCG@10 | Hit@10 | MRR")
    print("-" * 110)
    for rank, item in enumerate(results[:args.top_results], start=1):
        print(
            f"{rank:>4} | {item['recent_n']:>8} | {item['model_weight']:<7.3f} | "
            f"{item['cooccur_weight']:<4.2f} | {item['user_weight']:<6.3f} | "
            f"{item['pop_weight']:<5.2f} | {item['pop_penalty_weight']:<7.3f} | "
            f"{item['ndcg']:.6f} | {item['hit']:.6f} | {item['mrr']:.6f}"
        )

    best = results[0]
    print("-" * 110)
    print(
        "最佳参数: "
        f"recent_n={best['recent_n']}, model_weight={best['model_weight']}, "
        f"cooccur_weight={best['cooccur_weight']}, user_weight={best['user_weight']}, "
        f"pop_weight={best['pop_weight']}, pop_penalty_weight={best['pop_penalty_weight']}, "
        f"NDCG@{args.topk}={best['ndcg']:.6f}"
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"完整结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
