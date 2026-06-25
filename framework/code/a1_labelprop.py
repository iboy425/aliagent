"""A1标签传播后处理工具

这个脚本用于把GNN模型输出与图上的已知标签传播结果融合。

核心思想：
1. GNN输出每个节点属于10个类别的概率。
2. 训练集中已知标签可以沿图边向邻居传播，形成另一种类别置信度。
3. 两种置信度加权融合后再预测测试节点类别。

在验证模式下，脚本只使用训练划分内的标签做传播，用验证划分评估；
在推理模式下，脚本使用全部已知训练标签做传播，并生成A1.csv。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from datasets import GraphDataset
from infer import _prepare_task1_tensors, load_model_from_checkpoint
from utils import (
    compute_accuracy,
    get_device,
    normalize_adj,
    random_walk_normalize,
    sparse_to_torch,
    split_train_val,
    stratified_split,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 GNN + 标签传播融合工具")
    parser.add_argument("--data_path", type=str, required=True,
                        help="A1.npz数据路径")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="A1 checkpoint列表，可传入多个做logits平均")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备，如 cuda 或 cpu")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="验证集划分随机种子")
    parser.add_argument("--stratified_split", action="store_true",
                        help="使用分层验证集划分")
    parser.add_argument("--lp_normalize", type=str, default="random_walk",
                        choices=["random_walk", "symmetric", "none"],
                        help="标签传播使用的邻接矩阵归一化方式")

    # 网格搜索参数
    parser.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
                        help="验证模式搜索的传播系数列表")
    parser.add_argument("--num_iters", type=str, default="5,10,20,40",
                        help="验证模式搜索的传播轮数列表")
    parser.add_argument("--lp_weights", type=str, default="0,0.05,0.1,0.2,0.3,0.4,0.5",
                        help="验证模式搜索的标签传播融合权重列表")

    # 推理参数
    parser.add_argument("--alpha", type=float, default=None,
                        help="推理时使用的传播系数")
    parser.add_argument("--num_iter", type=int, default=None,
                        help="推理时使用的传播轮数")
    parser.add_argument("--lp_weight", type=float, default=None,
                        help="推理时使用的标签传播融合权重")
    parser.add_argument("--output_path", type=str, default="",
                        help="若提供，则使用指定参数生成A1.csv")
    parser.add_argument("--output_json", type=str, default="",
                        help="保存验证搜索结果JSON")
    return parser.parse_args()


def _parse_float_list(value):
    """解析逗号分隔浮点数列表"""
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value):
    """解析逗号分隔整数列表"""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _split_indices(data, args):
    """复现A1训练脚本中的验证集划分"""
    if args.stratified_split:
        return stratified_split(
            data["labels"], data["train_idx"], args.val_ratio, args.split_seed
        )
    return split_train_val(data["train_idx"], args.val_ratio, args.split_seed)


def _prepare_lp_adj(adj_raw, normalize_type, device):
    """准备标签传播使用的邻接矩阵"""
    if normalize_type == "random_walk":
        return random_walk_normalize(adj_raw).to(device)
    if normalize_type == "symmetric":
        return normalize_adj(adj_raw).to(device)
    if sp.issparse(adj_raw):
        return sparse_to_torch(adj_raw, device=device)
    return torch.FloatTensor(adj_raw).to(device)


def _build_label_seed(labels, label_idx, num_nodes, num_classes, device):
    """根据已知标签构造标签传播初始矩阵"""
    y0 = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)
    idx_t = torch.LongTensor(label_idx).to(device)
    label_t = torch.LongTensor(labels[label_idx]).to(device)
    y0[idx_t] = F.one_hot(label_t, num_classes=num_classes).float()
    return y0


def _propagate_labels(adj, y0, alpha, num_iter):
    """执行标签传播"""
    y = y0.clone()
    for _ in range(num_iter):
        y = alpha * (adj @ y) + (1.0 - alpha) * y0
    return y


def _average_model_logits(data, checkpoint_paths, device):
    """计算多个checkpoint的平均logits"""
    logits_sum = None
    with torch.no_grad():
        for ckpt_id, path in enumerate(checkpoint_paths, start=1):
            print(f"[模型推理] {ckpt_id}/{len(checkpoint_paths)} {path}")
            model, model_args = load_model_from_checkpoint(path, device)
            features, adj = _prepare_task1_tensors(data, model_args, device)
            logits = model(features, adj)
            logits_sum = logits.detach().clone() if logits_sum is None else logits_sum + logits.detach()

            del model, features, adj, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return logits_sum / len(checkpoint_paths)


def _score_accuracy(scores, labels, idx, device):
    """计算指定节点集合上的准确率"""
    idx_t = torch.LongTensor(idx).to(device)
    labels_t = torch.LongTensor(labels[idx]).to(device)
    return compute_accuracy(scores[idx_t], labels_t)


def run_search(data, model_probs, adj_lp, args, device):
    """在固定验证集上搜索标签传播参数"""
    train_fit_idx, val_idx = _split_indices(data, args)
    labels = data["labels"]
    num_nodes = data["num_nodes"]
    num_classes = data["num_classes"]

    y0 = _build_label_seed(labels, train_fit_idx, num_nodes, num_classes, device)
    baseline_acc = _score_accuracy(model_probs, labels, val_idx, device)

    rows = [{
        "alpha": None,
        "num_iter": None,
        "lp_weight": 0.0,
        "val_acc": baseline_acc,
        "kind": "model_only",
    }]

    alphas = _parse_float_list(args.alphas)
    num_iters = _parse_int_list(args.num_iters)
    lp_weights = _parse_float_list(args.lp_weights)

    print(f"\n模型原始验证准确率: {baseline_acc:.6f}")
    for alpha in alphas:
        for num_iter in num_iters:
            lp_scores = _propagate_labels(adj_lp, y0, alpha, num_iter)
            for lp_weight in lp_weights:
                scores = (1.0 - lp_weight) * model_probs + lp_weight * lp_scores
                acc = _score_accuracy(scores, labels, val_idx, device)
                rows.append({
                    "alpha": alpha,
                    "num_iter": num_iter,
                    "lp_weight": lp_weight,
                    "val_acc": acc,
                    "kind": "fusion",
                })

    rows = sorted(rows, key=lambda x: x["val_acc"], reverse=True)
    print("\n标签传播融合搜索 Top-20")
    for row in rows[:20]:
        print(
            f"val_acc={row['val_acc']:.6f}\t"
            f"alpha={row['alpha']}\tnum_iter={row['num_iter']}\t"
            f"lp_weight={row['lp_weight']}\t{row['kind']}"
        )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "baseline_acc": baseline_acc,
                "results": rows,
                "split_seed": args.split_seed,
                "val_ratio": args.val_ratio,
                "stratified_split": args.stratified_split,
                "lp_normalize": args.lp_normalize,
            }, f, ensure_ascii=False, indent=2)
        csv_path = os.path.splitext(args.output_json)[0] + ".csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n搜索结果已保存: {args.output_json}")
        print(f"搜索表格已保存: {csv_path}")

    return rows[0]


def run_infer(data, model_probs, adj_lp, args, device):
    """使用全部训练标签传播并生成A1.csv"""
    if args.alpha is None or args.num_iter is None or args.lp_weight is None:
        raise ValueError("生成A1.csv必须同时提供 --alpha、--num_iter、--lp_weight")

    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    num_nodes = data["num_nodes"]
    num_classes = data["num_classes"]

    y0 = _build_label_seed(labels, train_idx, num_nodes, num_classes, device)
    lp_scores = _propagate_labels(adj_lp, y0, args.alpha, args.num_iter)
    scores = (1.0 - args.lp_weight) * model_probs + args.lp_weight * lp_scores

    test_idx_t = torch.LongTensor(test_idx).to(device)
    predictions = torch.argmax(scores[test_idx_t], dim=1).cpu().numpy()

    result_df = pd.DataFrame({
        "test_idx": test_idx,
        "label": predictions,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nA1推理完成，类别分布:")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt}")
    print(f"结果已保存: {args.output_path}")


def main():
    """主入口"""
    args = parse_args()
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)

    logits = _average_model_logits(data, args.checkpoints, device)
    model_probs = F.softmax(logits, dim=1)
    adj_lp = _prepare_lp_adj(data["adj"], args.lp_normalize, device)

    best = run_search(data, model_probs, adj_lp, args, device)
    print(
        "\n当前搜索最佳参数: "
        f"alpha={best['alpha']}, num_iter={best['num_iter']}, "
        f"lp_weight={best['lp_weight']}, val_acc={best['val_acc']:.6f}"
    )

    if args.output_path:
        run_infer(data, model_probs, adj_lp, args, device)


if __name__ == "__main__":
    main()
