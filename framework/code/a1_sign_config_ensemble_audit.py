"""A1 SIGN不同配置checkpoint融合审计

本脚本用于比较多个 SIGN 配置在同一 split 下的概率融合效果。

使用场景：
- Exp-059 发现 `undirected+reverse` 均值更高，但部分 split 不如纯
  `undirected`；
- 这说明两类配置可能互补，适合做概率层融合，而不是只押单一配置。

审计方式：
- 每个 split 只加载该 split 下训练出来的 checkpoint；
- 验证集划分与训练脚本一致；
- 不使用验证标签构造任何特征，避免泄漏。
"""
import argparse
import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from datasets import GraphDataset
from utils import get_device, split_train_val, stratified_split
from a1_correct_smooth import (
    _prepare_cs_adj,
    _score_accuracy,
    correct_and_smooth,
    normalize_score_rows,
)
from a1_sign_infer import load_sign_probs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 SIGN配置融合审计")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split_seeds", type=str, default="42,777,2024,2026,3407")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--stratified_split", action="store_true")
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        required=True,
        help="格式：name=目录。目录下必须有 split{seed}/best_model.pt",
    )
    parser.add_argument("--grid_step", type=float, default=0.1,
                        help="权重网格步长，默认0.1")
    parser.add_argument("--cs_normalize", type=str, default="random_walk",
                        choices=["random_walk", "symmetric", "none"])
    parser.add_argument("--smooth_weights", type=str, default="0",
                        help="C&S smooth_weight列表；默认只评估model_only等价配置")
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数列表"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点列表"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_sources(values: List[str]) -> Dict[str, str]:
    """解析 name=path 形式的数据源"""
    sources = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"source必须是name=path格式: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"source必须是name=path格式: {item}")
        sources[name] = path
    if len(sources) < 2:
        raise ValueError("至少需要两个source才有融合意义")
    return sources


def split_indices(data: Dict, split_seed: int, val_ratio: float, stratified: bool) -> Tuple[np.ndarray, np.ndarray]:
    """划分训练/验证节点"""
    if stratified:
        return stratified_split(data["labels"], data["train_idx"], val_ratio, split_seed)
    return split_train_val(data["train_idx"], val_ratio, split_seed)


def build_weight_grid(names: List[str], step: float) -> List[Dict[str, float]]:
    """构造权重网格

    为了避免过度搜索，这里只做 0.1 粒度的 simplex 网格。
    3个source时一共66组，计算量很小。
    """
    if step <= 0 or step > 1:
        raise ValueError("grid_step必须在(0,1]之间")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("grid_step需要能整除1，例如0.1或0.05")

    rows = []
    for counts in itertools.product(range(units + 1), repeat=len(names)):
        if sum(counts) != units:
            continue
        weights = {name: count / units for name, count in zip(names, counts)}
        if sum(weights.values()) <= 0:
            continue
        rows.append(weights)
    rows.sort(key=lambda item: tuple(item[name] for name in names))
    return rows


def weight_key(weights: Dict[str, float]) -> str:
    """权重组合转为稳定字符串"""
    return "+".join(f"{name}:{weight:g}" for name, weight in weights.items() if weight > 0)


def load_split_probs(
    data: Dict,
    sources: Dict[str, str],
    split_seed: int,
    label_idx: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """加载某个split的全部source概率

    `label_idx` 是当前 split 的训练折。SIGN 的标签传播特征只能使用
    这些标签；如果使用全部 train_idx，会把验证标签泄漏进特征。
    """
    probs = {}
    for name, base_dir in sources.items():
        path = os.path.join(base_dir, f"split{split_seed}", "best_model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少checkpoint: {path}")
        print(f"  [加载] {name}: {path}")
        probs[name] = load_sign_probs(data, path, device, label_idx=label_idx)
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
        if weight <= 0:
            continue
        part = probs[name] * (weight / total)
        mixed = part.clone() if mixed is None else mixed + part
    return normalize_score_rows(mixed)


def evaluate_split(
    data: Dict,
    sources: Dict[str, str],
    weight_grid: List[Dict[str, float]],
    split_seed: int,
    args,
    device: torch.device,
) -> List[Dict]:
    """评估单个split"""
    fit_idx, val_idx = split_indices(data, split_seed, args.val_ratio, args.stratified_split)
    labels = torch.LongTensor(data["labels"]).to(device)
    fit_idx_t = torch.LongTensor(fit_idx).to(device)
    adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)
    probs = load_split_probs(data, sources, split_seed, fit_idx, device)

    rows = []
    for weights in weight_grid:
        mixed = mix_probs(probs, weights)
        base_acc = _score_accuracy(mixed, labels, val_idx, device)
        rows.append({
            "split_seed": split_seed,
            "candidate": weight_key(weights),
            "weights": weights,
            "kind": "model_only",
            "val_acc": base_acc,
            "smooth_weight": 0.0,
        })
        for smooth_weight in parse_float_list(args.smooth_weights):
            if smooth_weight <= 0:
                continue
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
                "candidate": weight_key(weights),
                "weights": weights,
                "kind": "correct_smooth",
                "val_acc": _score_accuracy(scores, labels, val_idx, device),
                **params,
            })
    rows.sort(key=lambda item: item["val_acc"], reverse=True)
    return rows


def summarize(all_rows: List[Dict]) -> List[Dict]:
    """汇总跨split结果"""
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
    sources = parse_sources(args.sources)
    names = list(sources.keys())
    weight_grid = build_weight_grid(names, args.grid_step)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("A1 SIGN配置融合审计")
    print("=" * 100)
    print(f"sources={sources}")
    print(f"weight_grid={len(weight_grid)}组, grid_step={args.grid_step}")

    all_rows = []
    for split_seed in parse_int_list(args.split_seeds):
        print("\n" + "=" * 100)
        print(f"[Split] {split_seed}")
        print("=" * 100)
        rows = evaluate_split(data, sources, weight_grid, split_seed, args, device)
        all_rows.extend(rows)
        with open(os.path.join(args.output_dir, f"split{split_seed}.json"), "w", encoding="utf-8") as f:
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
    print("[汇总] Exp-060 A1 SIGN配置融合审计 Top-20")
    print("=" * 100)
    for row in summary[:20]:
        print(
            f"mean={row['mean']:.6f}\tmin={row['min']:.6f}\tstd={row['std']:.6f}\t"
            f"n={row['n']}\t{row['candidate']}\t{row['kind']}\tsmooth={row['smooth_weight']}"
        )


if __name__ == "__main__":
    main()
