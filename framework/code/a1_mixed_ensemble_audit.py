"""A1 异构模型集成无泄漏审计

本脚本用于评估不同 A1 模型族的概率融合效果。每个 split 只使用同一
split 下训练出的 checkpoint，在该 split 的验证集上评估，避免训练泄漏。

当前支持：

- SIGN/MLP checkpoint
- GNN checkpoint（GCN/GAT/SAGE）
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from datasets import GraphDataset
from utils import get_device, split_train_val, stratified_split
from a1_correct_smooth import (
    _average_model_probs,
    _prepare_cs_adj,
    _score_accuracy,
    correct_and_smooth,
    normalize_score_rows,
)
from a1_sign_infer import load_sign_probs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1异构模型集成审计")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split_seeds", type=str, default="42,777,2024,2026,3407")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--stratified_split", action="store_true")
    parser.add_argument("--cs_normalize", type=str, default="random_walk",
                        choices=["random_walk", "symmetric", "none"])
    parser.add_argument("--smooth_weights", type=str, default="0.5,0.75,1.0")
    parser.add_argument("--base_dir", type=str, default="output",
                        help="framework输出目录根路径")
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数列表"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点列表"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def split_indices(data: Dict, split_seed: int, val_ratio: float, stratified: bool) -> Tuple[np.ndarray, np.ndarray]:
    """划分训练/验证节点"""
    if stratified:
        return stratified_split(data["labels"], data["train_idx"], val_ratio, split_seed)
    return split_train_val(data["train_idx"], val_ratio, split_seed)


def checkpoint_paths(base_dir: str, split_seed: int) -> Dict[str, Tuple[str, str]]:
    """返回当前 split 可用的模型 checkpoint 路径

    value格式：(kind, path)，kind用于决定加载器。
    """
    split = f"split{split_seed}"
    return {
        "sign_rw": (
            "sign",
            os.path.join(
                base_dir,
                "exp050_a1_sign_audit",
                "sign_rw_undir_k5_h512_l2_do04_block",
                split,
                "best_model.pt",
            ),
        ),
        "sign_sym": (
            "sign",
            os.path.join(
                base_dir,
                "exp050_a1_sign_audit",
                "sign_sym_undir_k5_h512_l2_do04_block",
                split,
                "best_model.pt",
            ),
        ),
        "sign_row": (
            "sign",
            os.path.join(
                base_dir,
                "exp050_a1_sign_audit",
                "sign_sym_undir_k5_h512_l2_do04_rowblock",
                split,
                "best_model.pt",
            ),
        ),
        "gat": (
            "gnn",
            os.path.join(
                base_dir,
                "exp049_a1_fixed_epoch_no_leak_audit",
                "gat_h256_heads4_seed2026_e270",
                split,
                "best_model.pt",
            ),
        ),
        "gcn": (
            "gnn",
            os.path.join(
                base_dir,
                "exp049_a1_fixed_epoch_no_leak_audit",
                "gcn_h256_seed777_e120",
                split,
                "best_model.pt",
            ),
        ),
    }


def candidate_weight_sets() -> List[Dict[str, float]]:
    """预定义异构融合权重集合

    先用少量有解释性的组合做大方向判断，避免在同一验证集上过度搜索。
    """
    return [
        {"sign_rw": 1.0},
        {"sign_rw": 0.8, "sign_row": 0.2},
        {"sign_rw": 0.8, "sign_sym": 0.2},
        {"sign_rw": 0.7, "sign_row": 0.15, "sign_sym": 0.15},
        {"sign_rw": 0.85, "gat": 0.15},
        {"sign_rw": 0.85, "gcn": 0.15},
        {"sign_rw": 0.75, "sign_row": 0.15, "gat": 0.10},
        {"sign_rw": 0.70, "sign_row": 0.15, "sign_sym": 0.10, "gat": 0.05},
        {"sign_rw": 0.65, "sign_row": 0.15, "sign_sym": 0.10, "gat": 0.05, "gcn": 0.05},
        {"sign_rw": 0.5, "sign_row": 0.25, "sign_sym": 0.25},
        {"sign_row": 1.0},
        {"sign_sym": 1.0},
    ]


def load_all_probs(data: Dict, paths: Dict[str, Tuple[str, str]], device: torch.device) -> Dict[str, torch.Tensor]:
    """加载当前 split 的全部模型概率"""
    probs = {}
    for name, (kind, path) in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少checkpoint: {path}")
        print(f"  [加载] {name}: {path}")
        if kind == "sign":
            probs[name] = load_sign_probs(data, path, device)
        elif kind == "gnn":
            probs[name] = _average_model_probs(data, [path], device)
        else:
            raise ValueError(f"未知模型类型: {kind}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return probs


def mix_probs(probs: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """按权重融合概率"""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("融合权重总和必须大于0")
    mixed = None
    for name, weight in weights.items():
        part = probs[name] * (weight / total)
        mixed = part.clone() if mixed is None else mixed + part
    return normalize_score_rows(mixed)


def evaluate_split(data: Dict, split_seed: int, args, device: torch.device) -> List[Dict]:
    """评估单个 split 的所有融合候选"""
    fit_idx, val_idx = split_indices(data, split_seed, args.val_ratio, args.stratified_split)
    labels = torch.LongTensor(data["labels"]).to(device)
    adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)
    fit_idx_t = torch.LongTensor(fit_idx).to(device)
    model_probs = load_all_probs(data, checkpoint_paths(args.base_dir, split_seed), device)

    rows = []
    for weights in candidate_weight_sets():
        mixed = mix_probs(model_probs, weights)
        model_acc = _score_accuracy(mixed, labels, val_idx, device)
        rows.append({
            "split_seed": split_seed,
            "candidate": "+".join(f"{k}:{v:g}" for k, v in weights.items()),
            "weights": weights,
            "kind": "model_only",
            "val_acc": model_acc,
            "smooth_weight": 0.0,
        })
        for smooth_weight in parse_float_list(args.smooth_weights):
            params = {
                "correct_alpha": 0.3,
                "correct_iter": 5,
                "correct_weight": 0.0,
                "smooth_alpha": 0.7,
                "smooth_iter": 5,
                "smooth_weight": smooth_weight,
            }
            scores = correct_and_smooth(
                model_probs=mixed,
                labels=labels,
                train_idx=fit_idx_t,
                adj=adj,
                params=params,
            )
            rows.append({
                "split_seed": split_seed,
                "candidate": "+".join(f"{k}:{v:g}" for k, v in weights.items()),
                "weights": weights,
                "kind": "correct_smooth",
                "val_acc": _score_accuracy(scores, labels, val_idx, device),
                **params,
            })
    rows.sort(key=lambda item: item["val_acc"], reverse=True)
    return rows


def summarize(all_rows: List[Dict]) -> List[Dict]:
    """按候选+后处理参数汇总跨split表现"""
    groups: Dict[str, List[Dict]] = {}
    for row in all_rows:
        key = json.dumps({
            "candidate": row["candidate"],
            "kind": row["kind"],
            "smooth_weight": row.get("smooth_weight", 0.0),
        }, sort_keys=True)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, rows in groups.items():
        values = [float(row["val_acc"]) for row in rows]
        if not values:
            continue
        meta = json.loads(key)
        summary.append({
            **meta,
            "weights": rows[0]["weights"],
            "n": len(values),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "details": rows,
        })
    summary.sort(key=lambda item: (item["mean"], item["min"]), reverse=True)
    return summary


def main():
    """主入口"""
    args = parse_args()
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for split_seed in parse_int_list(args.split_seeds):
        print("\n" + "=" * 100)
        print(f"[Split] {split_seed}")
        print("=" * 100)
        rows = evaluate_split(data, split_seed, args, device)
        all_rows.extend(rows)
        split_path = os.path.join(args.output_dir, f"split{split_seed}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({"results": rows}, f, ensure_ascii=False, indent=2)
        print("Top-8")
        for row in rows[:8]:
            print(
                f"val_acc={row['val_acc']:.6f}\t{row['candidate']}\t"
                f"{row['kind']}\tsmooth={row.get('smooth_weight', 0.0)}"
            )

    summary = summarize(all_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[汇总] A1异构融合审计 Top-20")
    print("=" * 100)
    for row in summary[:20]:
        print(
            f"mean={row['mean']:.6f}\tmin={row['min']:.6f}\tstd={row['std']:.6f}\t"
            f"n={row['n']}\t{row['candidate']}\t{row['kind']}\tsmooth={row['smooth_weight']}"
        )


if __name__ == "__main__":
    main()
