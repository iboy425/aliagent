"""A1 Correct and Smooth 后处理工具

Correct and Smooth 是一种图半监督节点分类后处理方法：

1. Correct：在训练节点上比较 `真实标签 one-hot` 和 `模型预测概率`，
   得到预测残差，并把残差沿图传播到未标注节点，用于纠正模型错误。
2. Smooth：把训练节点的真实标签作为锚点，把修正后的预测概率沿图平滑，
   利用图同质性让相邻节点预测更一致。

本脚本不重新训练模型，只读取已有 A1 checkpoint 的 logits，因此适合在
提交前快速搜索后处理参数。
"""
import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

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
    normalize_adj_sparse,
    random_walk_normalize,
    random_walk_normalize_sparse,
    sparse_to_torch,
    split_train_val,
    stratified_split,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 Correct and Smooth 后处理")
    parser.add_argument("--data_path", type=str, required=True,
                        help="A1.npz 数据路径")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="A1 checkpoint 列表，可传入多个做 logits 平均")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备，如 cuda 或 cpu")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="验证集划分随机种子")
    parser.add_argument("--stratified_split", action="store_true",
                        help="使用分层验证集划分")
    parser.add_argument("--cs_normalize", type=str, default="random_walk",
                        choices=["random_walk", "symmetric", "none"],
                        help="C&S 传播使用的邻接矩阵归一化方式")

    # 搜索参数。默认范围偏保守，先验证方法是否有效，再扩大搜索。
    parser.add_argument("--correct_alphas", type=str, default="0.3,0.5,0.7,0.85,0.95",
                        help="Correct 阶段传播系数列表")
    parser.add_argument("--correct_iters", type=str, default="5,10,20,40",
                        help="Correct 阶段传播轮数列表")
    parser.add_argument("--correct_weights", type=str, default="0,0.25,0.5,0.75,1.0,1.5",
                        help="Correct 残差融合权重列表")
    parser.add_argument("--smooth_alphas", type=str, default="0.3,0.5,0.7,0.85,0.95",
                        help="Smooth 阶段传播系数列表")
    parser.add_argument("--smooth_iters", type=str, default="5,10,20,40",
                        help="Smooth 阶段传播轮数列表")
    parser.add_argument("--smooth_weights", type=str, default="0,0.25,0.5,0.75,1.0",
                        help="Smooth 预测融合权重列表")
    parser.add_argument("--pseudo_thresholds", type=str, default="",
                        help="伪标签置信度阈值列表；空字符串表示搜索时关闭伪标签")
    parser.add_argument("--pseudo_weights", type=str, default="0.5,1.0",
                        help="伪标签软锚点权重列表")

    # 推理参数。若 output_path 存在但未显式指定，则使用搜索得到的最佳参数。
    parser.add_argument("--correct_alpha", type=float, default=None,
                        help="推理使用的 Correct 传播系数")
    parser.add_argument("--correct_iter", type=int, default=None,
                        help="推理使用的 Correct 传播轮数")
    parser.add_argument("--correct_weight", type=float, default=None,
                        help="推理使用的 Correct 残差融合权重")
    parser.add_argument("--smooth_alpha", type=float, default=None,
                        help="推理使用的 Smooth 传播系数")
    parser.add_argument("--smooth_iter", type=int, default=None,
                        help="推理使用的 Smooth 传播轮数")
    parser.add_argument("--smooth_weight", type=float, default=None,
                        help="推理使用的 Smooth 预测融合权重")
    parser.add_argument("--pseudo_threshold", type=float, default=None,
                        help="推理使用的伪标签置信度阈值；不提供则按搜索最佳或关闭")
    parser.add_argument("--pseudo_weight", type=float, default=None,
                        help="推理使用的伪标签软锚点权重")
    parser.add_argument("--output_path", type=str, default="",
                        help="若提供，则生成 A1.csv")
    parser.add_argument("--output_json", type=str, default="",
                        help="保存搜索结果 JSON")
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点数"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_score_rows(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """把每一行非负分数归一化为概率分布

    如果某一行全为 0，则用均匀分布修复，避免后续 `argmax` 被异常值影响。
    """
    scores = scores.clamp(min=0)
    row_sum = scores.sum(dim=1, keepdim=True)
    zero_mask = row_sum.squeeze(1) <= eps
    normalized = scores / row_sum.clamp(min=eps)
    if zero_mask.any():
        normalized[zero_mask] = 1.0 / scores.size(1)
    return normalized


def propagate_with_restart(
    adj: torch.Tensor,
    seed: torch.Tensor,
    alpha: float,
    num_iter: int,
) -> torch.Tensor:
    """带重启项的图传播

    公式：`Y_{t+1} = alpha * A * Y_t + (1 - alpha) * Y_0`。
    """
    propagated = seed.clone()
    for _ in range(num_iter):
        propagated = alpha * (adj @ propagated) + (1.0 - alpha) * seed
    return propagated


def _build_onehot(
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    num_nodes: int,
    num_classes: int,
) -> torch.Tensor:
    """根据训练节点标签构造 one-hot 标签矩阵"""
    onehot = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=labels.device)
    onehot[train_idx] = F.one_hot(labels[train_idx], num_classes=num_classes).float()
    return onehot


def correct_predictions(
    model_probs: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    adj: torch.Tensor,
    alpha: float,
    num_iter: int,
    correction_weight: float,
) -> torch.Tensor:
    """Correct 阶段：传播训练节点预测残差并修正模型概率"""
    num_nodes, num_classes = model_probs.shape
    label_onehot = _build_onehot(labels, train_idx, num_nodes, num_classes)

    residual_seed = torch.zeros_like(model_probs)
    residual_seed[train_idx] = label_onehot[train_idx] - model_probs[train_idx]
    residual = propagate_with_restart(adj, residual_seed, alpha=alpha, num_iter=num_iter)

    corrected = model_probs + correction_weight * residual
    return normalize_score_rows(corrected)


def smooth_predictions(
    corrected_probs: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    adj: torch.Tensor,
    alpha: float,
    num_iter: int,
    smooth_weight: float,
    pseudo_idx: Optional[torch.Tensor] = None,
    pseudo_labels: Optional[torch.Tensor] = None,
    pseudo_weight: float = 0.0,
) -> torch.Tensor:
    """Smooth 阶段：以训练标签为锚点平滑预测概率"""
    num_nodes, num_classes = corrected_probs.shape
    label_onehot = _build_onehot(labels, train_idx, num_nodes, num_classes)

    smooth_seed = corrected_probs.clone()
    smooth_seed[train_idx] = label_onehot[train_idx]
    if (
        pseudo_idx is not None
        and pseudo_labels is not None
        and pseudo_weight > 0
        and len(pseudo_idx) > 0
    ):
        pseudo_onehot = F.one_hot(pseudo_labels, num_classes=num_classes).float()
        old_seed = smooth_seed[pseudo_idx]
        smooth_seed[pseudo_idx] = (
            (1.0 - pseudo_weight) * old_seed
            + pseudo_weight * pseudo_onehot
        )
    propagated = propagate_with_restart(adj, smooth_seed, alpha=alpha, num_iter=num_iter)

    smoothed = (1.0 - smooth_weight) * corrected_probs + smooth_weight * propagated
    return normalize_score_rows(smoothed)


def correct_and_smooth(
    model_probs: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    adj: torch.Tensor,
    params: Dict,
    pseudo_idx: Optional[torch.Tensor] = None,
    pseudo_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """执行完整 Correct and Smooth 流程"""
    corrected = correct_predictions(
        model_probs=model_probs,
        labels=labels,
        train_idx=train_idx,
        adj=adj,
        alpha=params["correct_alpha"],
        num_iter=params["correct_iter"],
        correction_weight=params["correct_weight"],
    )
    return smooth_predictions(
        corrected_probs=corrected,
        labels=labels,
        train_idx=train_idx,
        adj=adj,
        alpha=params["smooth_alpha"],
        num_iter=params["smooth_iter"],
        smooth_weight=params["smooth_weight"],
        pseudo_idx=pseudo_idx,
        pseudo_labels=pseudo_labels,
        pseudo_weight=params.get("pseudo_weight", 0.0) or 0.0,
    )


def build_pseudo_labels(
    model_probs: torch.Tensor,
    candidate_idx: np.ndarray,
    threshold: Optional[float],
    device: torch.device,
) -> Sequence[torch.Tensor]:
    """根据模型置信度选择伪标签节点"""
    if threshold is None:
        empty_idx = torch.LongTensor([]).to(device)
        empty_labels = torch.LongTensor([]).to(device)
        return empty_idx, empty_labels

    candidate_t = torch.LongTensor(candidate_idx).to(device)
    if len(candidate_t) == 0:
        empty_labels = torch.LongTensor([]).to(device)
        return candidate_t, empty_labels

    conf, pred = model_probs[candidate_t].max(dim=1)
    keep = conf >= threshold
    return candidate_t[keep], pred[keep]


def _split_indices(data: Dict, args) -> Sequence[np.ndarray]:
    """复现 A1 训练脚本中的验证集划分"""
    if args.stratified_split:
        return stratified_split(data["labels"], data["train_idx"], args.val_ratio, args.split_seed)
    return split_train_val(data["train_idx"], args.val_ratio, args.split_seed)


def _prepare_cs_adj(adj_raw, normalize_type: str, device: torch.device) -> torch.Tensor:
    """准备 C&S 使用的归一化邻接矩阵"""
    if normalize_type == "random_walk":
        if sp.issparse(adj_raw):
            return random_walk_normalize_sparse(adj_raw, device=device)
        return random_walk_normalize(adj_raw).to(device)
    if normalize_type == "symmetric":
        if sp.issparse(adj_raw):
            return normalize_adj_sparse(adj_raw, device=device)
        return normalize_adj(adj_raw).to(device)
    if sp.issparse(adj_raw):
        return sparse_to_torch(adj_raw, device=device, return_sparse=True).coalesce()
    return torch.FloatTensor(adj_raw).to(device)


def _average_model_probs(data: Dict, checkpoints: Iterable[str], device: torch.device) -> torch.Tensor:
    """计算多个 checkpoint 的平均预测概率"""
    logits_sum: Optional[torch.Tensor] = None
    checkpoint_list = list(checkpoints)
    with torch.no_grad():
        for ckpt_id, path in enumerate(checkpoint_list, start=1):
            print(f"[模型推理] {ckpt_id}/{len(checkpoint_list)} {path}")
            model, model_args = load_model_from_checkpoint(path, device)
            features, adj = _prepare_task1_tensors(data, model_args, device)
            logits = model(features, adj)
            logits_sum = logits.detach().clone() if logits_sum is None else logits_sum + logits.detach()

            del model, features, adj, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if logits_sum is None:
        raise ValueError("至少需要一个 checkpoint")
    return F.softmax(logits_sum / len(checkpoint_list), dim=1)


def _score_accuracy(scores: torch.Tensor, labels: torch.Tensor, idx: np.ndarray, device: torch.device) -> float:
    """计算指定节点集合上的准确率"""
    idx_t = torch.LongTensor(idx).to(device)
    return compute_accuracy(scores[idx_t], labels[idx_t])


def _iter_param_grid(args):
    """遍历 C&S 参数网格"""
    pseudo_thresholds = [None]
    if args.pseudo_thresholds:
        pseudo_thresholds = [None] + _parse_float_list(args.pseudo_thresholds)
    pseudo_weights = [0.0] if pseudo_thresholds == [None] else [0.0] + _parse_float_list(args.pseudo_weights)
    for correct_alpha in _parse_float_list(args.correct_alphas):
        for correct_iter in _parse_int_list(args.correct_iters):
            for correct_weight in _parse_float_list(args.correct_weights):
                for smooth_alpha in _parse_float_list(args.smooth_alphas):
                    for smooth_iter in _parse_int_list(args.smooth_iters):
                        for smooth_weight in _parse_float_list(args.smooth_weights):
                            for pseudo_threshold in pseudo_thresholds:
                                for pseudo_weight in pseudo_weights:
                                    if pseudo_threshold is None and pseudo_weight > 0:
                                        continue
                                    if pseudo_threshold is not None and pseudo_weight <= 0:
                                        continue
                                    yield {
                                        "correct_alpha": correct_alpha,
                                        "correct_iter": correct_iter,
                                        "correct_weight": correct_weight,
                                        "smooth_alpha": smooth_alpha,
                                        "smooth_iter": smooth_iter,
                                        "smooth_weight": smooth_weight,
                                        "pseudo_threshold": pseudo_threshold,
                                        "pseudo_weight": pseudo_weight if pseudo_threshold is not None else 0.0,
                                    }


def run_search(
    data: Dict,
    model_probs: torch.Tensor,
    labels: torch.Tensor,
    adj: torch.Tensor,
    args,
    device: torch.device,
) -> Dict:
    """在固定验证集上搜索 C&S 参数"""
    fit_idx, val_idx = _split_indices(data, args)
    fit_idx_t = torch.LongTensor(fit_idx).to(device)
    pseudo_candidate_idx = np.setdiff1d(
        np.arange(data["labels"].shape[0]),
        fit_idx,
        assume_unique=False,
    )

    baseline_acc = _score_accuracy(model_probs, labels, val_idx, device)
    rows = [{
        "kind": "model_only",
        "val_acc": baseline_acc,
        "correct_alpha": None,
        "correct_iter": None,
        "correct_weight": 0.0,
        "smooth_alpha": None,
        "smooth_iter": None,
        "smooth_weight": 0.0,
        "pseudo_threshold": None,
        "pseudo_weight": 0.0,
        "pseudo_count": 0,
    }]
    print(f"\n模型原始验证准确率: {baseline_acc:.6f}")

    for params in _iter_param_grid(args):
        pseudo_idx, pseudo_labels = build_pseudo_labels(
            model_probs=model_probs,
            candidate_idx=pseudo_candidate_idx,
            threshold=params.get("pseudo_threshold"),
            device=device,
        )
        scores = correct_and_smooth(
            model_probs=model_probs,
            labels=labels,
            train_idx=fit_idx_t,
            adj=adj,
            params=params,
            pseudo_idx=pseudo_idx,
            pseudo_labels=pseudo_labels,
        )
        val_acc = _score_accuracy(scores, labels, val_idx, device)
        row = {"kind": "correct_smooth", "val_acc": val_acc}
        row.update(params)
        row["pseudo_count"] = int(len(pseudo_idx))
        rows.append(row)

    rows = sorted(rows, key=lambda item: item["val_acc"], reverse=True)
    print("\nCorrect and Smooth 搜索 Top-20")
    for row in rows[:20]:
        print(
            f"val_acc={row['val_acc']:.6f}\t"
            f"correct=({row['correct_alpha']},{row['correct_iter']},{row['correct_weight']})\t"
            f"smooth=({row['smooth_alpha']},{row['smooth_iter']},{row['smooth_weight']})\t"
            f"pseudo=({row.get('pseudo_threshold')},{row.get('pseudo_weight')},{row.get('pseudo_count', 0)})\t"
            f"{row['kind']}"
        )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        payload = {
            "baseline_acc": baseline_acc,
            "results": rows,
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "stratified_split": args.stratified_split,
            "cs_normalize": args.cs_normalize,
            "checkpoints": list(args.checkpoints),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        csv_path = os.path.splitext(args.output_json)[0] + ".csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n搜索结果已保存: {args.output_json}")
        print(f"搜索表格已保存: {csv_path}")

    return rows[0]


def _infer_params_from_args(args, best: Dict) -> Dict:
    """决定推理使用的参数；未显式提供时回退到搜索最佳参数"""
    keys = [
        "correct_alpha", "correct_iter", "correct_weight",
        "smooth_alpha", "smooth_iter", "smooth_weight",
        "pseudo_threshold", "pseudo_weight",
    ]
    params = {}
    for key in keys:
        value = getattr(args, key)
        params[key] = best[key] if value is None else value
    return params


def run_infer(
    data: Dict,
    model_probs: torch.Tensor,
    labels: torch.Tensor,
    adj: torch.Tensor,
    params: Dict,
    args,
    device: torch.device,
):
    """使用全部训练标签执行 C&S 并生成 A1.csv"""
    train_idx_t = torch.LongTensor(data["train_idx"]).to(device)
    pseudo_idx, pseudo_labels = build_pseudo_labels(
        model_probs=model_probs,
        candidate_idx=data["test_idx"],
        threshold=params.get("pseudo_threshold"),
        device=device,
    )
    scores = correct_and_smooth(
        model_probs=model_probs,
        labels=labels,
        train_idx=train_idx_t,
        adj=adj,
        params=params,
        pseudo_idx=pseudo_idx,
        pseudo_labels=pseudo_labels,
    )

    test_idx_t = torch.LongTensor(data["test_idx"]).to(device)
    predictions = torch.argmax(scores[test_idx_t], dim=1).cpu().numpy()
    result_df = pd.DataFrame({
        "test_idx": data["test_idx"],
        "label": predictions,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)

    unique, counts = np.unique(predictions, return_counts=True)
    print("\nA1 C&S 推理完成，类别分布:")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt}")
    print(f"推理参数: {params}")
    print(f"伪标签数量: {len(pseudo_idx)}")
    print(f"结果已保存: {args.output_path}")


def main():
    """主入口"""
    args = parse_args()
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)

    model_probs = _average_model_probs(data, args.checkpoints, device)
    labels = torch.LongTensor(data["labels"]).to(device)
    adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)

    best = run_search(data, model_probs, labels, adj, args, device)
    print(
        "\n当前搜索最佳参数: "
        f"correct=({best['correct_alpha']},{best['correct_iter']},{best['correct_weight']}), "
        f"smooth=({best['smooth_alpha']},{best['smooth_iter']},{best['smooth_weight']}), "
        f"val_acc={best['val_acc']:.6f}"
    )

    if args.output_path:
        infer_params = _infer_params_from_args(args, best)
        run_infer(data, model_probs, labels, adj, infer_params, args, device)


if __name__ == "__main__":
    main()
