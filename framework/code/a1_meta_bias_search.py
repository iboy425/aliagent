"""A1 元模型后的类别概率微调审计

当前线上最好 A1 来自 Exp075/078 的 OOF 元模型 stacking。它已经证明
有效，但只比 Exp066 高约 0.0007，说明可用空间很小。本脚本只做
低风险后处理：

1. 复用 Exp075 的 OOF 元模型训练方式；
2. 在每个 heldout split 上得到和正式推理一致的 final score；
3. 搜索单类别概率缩放，例如把 class 8 的概率乘以 1.05；
4. 只保留平均提升、最差 split、改动比例都可解释的候选。

正式 infer 阶段使用 OOF 元模型训练二层分类器，再应用 audit 选出的
类别缩放，生成 A1.csv。
"""
import argparse
import json
import os
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from datasets import GraphDataset
from utils import get_device
from a1_sign_meta_stack import (
    average_fulltrain_sources,
    base_probs_from_sources,
    build_dataset_for_splits,
    graph_struct_features,
    load_split_sources,
    parse_float_list,
    parse_int_list,
    parse_infer_sources,
    parse_sources,
    source_meta_features,
    split_indices,
    train_logistic,
)


def parse_classes(value: str, num_classes: int) -> List[int]:
    """解析类别列表"""
    if not value:
        return list(range(num_classes))
    classes = [int(item.strip()) for item in value.split(",") if item.strip()]
    for cls in classes:
        if cls < 0 or cls >= num_classes:
            raise ValueError(f"类别越界: {cls}")
    return classes


def final_scores_from_meta(
    meta_proba: np.ndarray,
    base_probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """按 Exp075 置信度回退规则构造最终 score"""
    meta_pred = np.argmax(meta_proba, axis=1).astype(np.int64)
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    meta_conf = np.max(meta_proba, axis=1)
    use_meta = (meta_conf >= threshold) | (meta_pred == base_pred)
    scores = base_probs.copy()
    scores[use_meta] = meta_proba[use_meta]
    return scores


def apply_bias(scores: np.ndarray, bias: Mapping[str, float]) -> np.ndarray:
    """应用类别概率缩放并返回预测"""
    if not bias:
        return np.argmax(scores, axis=1).astype(np.int64)
    weights = np.ones(scores.shape[1], dtype=np.float32)
    for key, value in bias.items():
        weights[int(key)] = float(value)
    return np.argmax(scores * weights.reshape(1, -1), axis=1).astype(np.int64)


def make_bias_candidates(classes: List[int], factors: List[float]) -> List[Dict]:
    """生成单类别缩放候选"""
    candidates = [{"name": "identity", "bias": {}}]
    for cls in classes:
        for factor in factors:
            if abs(factor - 1.0) < 1e-12:
                continue
            candidates.append({
                "name": f"c{cls}_x{factor:g}",
                "bias": {str(cls): float(factor)},
            })
    return candidates


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    """准确率"""
    return float(np.mean(pred.astype(np.int64) == labels.astype(np.int64)))


def build_oof_final_scores(data: Dict, args, device) -> Tuple[List[Dict], Dict]:
    """构造每个 heldout split 的 final score"""
    audit = json.load(open(args.base_audit_json, encoding="utf-8"))
    base_best = audit["best"]
    threshold = float(base_best.get("threshold", 0.0))
    labels = data["labels"].astype(np.int64)

    # 先用现有工具加载每个 split 的 source 概率和元特征。
    split_data = build_dataset_for_splits(data, args, device)
    split_keys = sorted(split_data.keys(), key=int)

    rows = []
    for heldout in split_keys:
        train_keys = [key for key in split_keys if key != heldout]
        train_x = np.concatenate([split_data[key]["x"] for key in train_keys], axis=0)
        train_y = np.concatenate([split_data[key]["y"] for key in train_keys], axis=0)
        model = train_logistic(train_x, train_y, float(base_best["C"]), base_best["class_weight"])

        val_idx = split_data[heldout]["val_idx"].astype(np.int64)
        meta_proba = model.predict_proba(split_data[heldout]["x"])
        base_probs = base_probs_from_sources(split_data[heldout]["source_probs"])[val_idx]
        scores = final_scores_from_meta(meta_proba, base_probs, threshold)
        base_pred = np.argmax(scores, axis=1).astype(np.int64)
        rows.append({
            "heldout": int(heldout),
            "val_idx": val_idx,
            "labels": labels[val_idx],
            "scores": scores.astype(np.float32),
            "base_pred": base_pred,
            "base_acc": accuracy(base_pred, labels[val_idx]),
        })
    return rows, base_best


def run_audit(args):
    """执行类别 bias 搜索"""
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)
    rows, base_best = build_oof_final_scores(data, args, device)
    num_classes = int(data["num_classes"])
    classes = parse_classes(args.classes, num_classes)
    factors = parse_float_list(args.factors)
    candidates = make_bias_candidates(classes, factors)

    results = []
    for candidate in candidates:
        split_rows = []
        for row in rows:
            pred = apply_bias(row["scores"], candidate["bias"])
            split_rows.append({
                "heldout": row["heldout"],
                "acc": accuracy(pred, row["labels"]),
                "base_acc": row["base_acc"],
                "gain": accuracy(pred, row["labels"]) - row["base_acc"],
                "changed": float(np.mean(pred != row["base_pred"])),
            })
        result = {
            "name": candidate["name"],
            "bias": candidate["bias"],
            "mean_acc": float(np.mean([item["acc"] for item in split_rows])),
            "mean_base": float(np.mean([item["base_acc"] for item in split_rows])),
            "mean_gain": float(np.mean([item["gain"] for item in split_rows])),
            "min_gain": float(np.min([item["gain"] for item in split_rows])),
            "mean_changed": float(np.mean([item["changed"] for item in split_rows])),
            "split_rows": split_rows,
        }
        results.append(result)

    results.sort(key=lambda item: (item["mean_gain"], item["min_gain"], -item["mean_changed"]), reverse=True)
    output = {
        "base_best": base_best,
        "best": results[0],
        "results": results,
        "classes": classes,
        "factors": factors,
    }
    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A1 meta bias 审计 Top-20")
    print("=" * 100)
    for item in results[:20]:
        print(
            f"{item['name']:<12}\tmean={item['mean_acc']:.6f}\t"
            f"gain={item['mean_gain']:+.6f}\tmin={item['min_gain']:+.6f}\t"
            f"changed={item['mean_changed']:.4%}\tbias={item['bias']}"
        )
    print(f"\n结果已保存: {out_path}")


def build_full_scores(data: Dict, args, device, base_best: Mapping) -> np.ndarray:
    """构造正式推理用 full final score"""
    labels = data["labels"].astype(np.int64)
    source_paths = parse_infer_sources(args.sources)
    source_probs = average_fulltrain_sources(data, source_paths, device)
    struct = graph_struct_features(data)
    meta = np.concatenate([source_meta_features(source_probs), struct], axis=1)

    if args.oof_sources:
        oof_sources = parse_sources(args.oof_sources)
        oof_rows = []
        oof_labels = []
        for split_seed in parse_int_list(args.split_seeds):
            fit_idx, val_idx = split_indices(data, split_seed, args.val_ratio, args.stratified_split)
            split_probs = load_split_sources(data, oof_sources, split_seed, fit_idx, device)
            split_meta = np.concatenate([source_meta_features(split_probs), struct], axis=1)
            oof_rows.append(split_meta[val_idx])
            oof_labels.append(labels[val_idx])
        train_x = np.concatenate(oof_rows, axis=0)
        train_y = np.concatenate(oof_labels, axis=0)
    else:
        train_idx = data["train_idx"].astype(np.int64)
        train_x = meta[train_idx]
        train_y = labels[train_idx]

    model = train_logistic(train_x, train_y, float(base_best["C"]), base_best["class_weight"])
    test_idx = data["test_idx"].astype(np.int64)
    meta_proba = model.predict_proba(meta[test_idx])
    base_probs = base_probs_from_sources(source_probs)[test_idx]
    return final_scores_from_meta(meta_proba, base_probs, float(base_best.get("threshold", 0.0)))


def run_infer(args):
    """根据 audit 最优 bias 生成 A1"""
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    audit = json.load(open(args.audit_json, encoding="utf-8"))
    best = audit["best"]
    base_best = audit["base_best"]
    scores = build_full_scores(data, args, device, base_best)
    pred = apply_bias(scores, best["bias"])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    test_idx = data["test_idx"].astype(np.int64)
    pd.DataFrame({"test_idx": test_idx, "label": pred}).to_csv(args.output_path, index=False)

    base_pred = np.argmax(scores, axis=1).astype(np.int64)
    unique, counts = np.unique(pred, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    result = {
        "best": best,
        "base_best": base_best,
        "changed_vs_unbiased": float(np.mean(pred != base_pred)),
        "class_distribution": distribution,
        "output_path": args.output_path,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A1 meta bias 推理完成")
    print("=" * 100)
    print(f"best={best}")
    print(f"changed_vs_unbiased={result['changed_vs_unbiased']:.4%}")
    print(f"class_distribution={distribution}")
    print(f"output={args.output_path}")


def add_common_args(parser):
    """添加共用参数"""
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base_audit_json", default="output/exp075_a1_meta_stack_threshold_audit/summary.json")
    parser.add_argument("--split_seeds", default="42,777,2024,2026,3407")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--stratified_split", action="store_true")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="A1元模型类别bias搜索")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    add_common_args(audit)
    audit.add_argument("--output_dir", required=True)
    audit.add_argument("--sources", nargs="+", required=True)
    audit.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    audit.add_argument("--factors", default="0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.3")

    infer = sub.add_parser("infer")
    infer.add_argument("--audit_json", required=True)
    infer.add_argument("--sources", nargs="+", required=True)
    infer.add_argument("--oof_sources", nargs="+", default=[])
    infer.add_argument("--output_path", required=True)
    infer.add_argument("--output_json", default="")
    add_common_args(infer)
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    if args.command == "audit":
        run_audit(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
