"""A1 SIGN 专家的正则化 OOF 元模型 stacking

手工按类别选择专家只能表达很粗的规则。这个脚本把多个 SIGN 专家的
概率、置信度、margin 和简单结构特征拼成元特征，再用强正则的
LogisticRegression 学习二层分类器。

审计方式仍然是 leave-one-split-out：
- 每次用 4 个 split 的验证节点训练元模型；
- 在剩下 1 个 split 的验证节点上评估；
- 这样可以估计元模型是否真的泛化，而不是只记住验证集。
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from datasets import GraphDataset
from utils import get_device, split_train_val, stratified_split
from a1_sign_classwise_stack import parse_sources, parse_infer_sources
from a1_sign_infer import load_sign_probs


def parse_int_list(value: str) -> List[int]:
    """解析整数列表"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """解析浮点列表"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def split_indices(data: Dict, split_seed: int, val_ratio: float, stratified: bool) -> Tuple[np.ndarray, np.ndarray]:
    """划分训练/验证节点"""
    if stratified:
        return stratified_split(data["labels"], data["train_idx"], val_ratio, split_seed)
    return split_train_val(data["train_idx"], val_ratio, split_seed)


def graph_struct_features(data: Dict) -> np.ndarray:
    """构造简单结构元特征"""
    adj = data["adj"].tocsr().astype(np.float32)
    undirected = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    out_degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    in_degree = np.asarray(adj.sum(axis=0)).reshape(-1)
    degree = np.asarray(undirected.sum(axis=1)).reshape(-1)
    features = np.column_stack([
        np.log1p(degree),
        np.log1p(in_degree),
        np.log1p(out_degree),
        (degree <= 1).astype(np.float32),
        (degree <= 3).astype(np.float32),
    ]).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def _row_normalize_features(features: np.ndarray) -> np.ndarray:
    """对元特征做列标准化"""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def label_neighbor_meta_features(data: Dict, label_idx: np.ndarray, hops: int = 2) -> np.ndarray:
    """构造无泄漏邻居标签统计元特征

    这里的 `label_idx` 必须是当前 split 的拟合集节点。验证节点标签不参与
    统计，因此可以用于 OOF 审计；正式测试时则使用全部训练标签。
    """
    labels = data["labels"].astype(np.int64)
    num_nodes = int(data.get("num_nodes", data["adj"].shape[0]))
    num_classes = int(labels[labels >= 0].max()) + 1
    label_idx = np.asarray(label_idx, dtype=np.int64)

    seed = np.zeros((num_nodes, num_classes), dtype=np.float32)
    valid_label_idx = label_idx[labels[label_idx] >= 0]
    seed[valid_label_idx, labels[valid_label_idx]] = 1.0

    adj = data["adj"].tocsr().astype(np.float32)
    modes = {
        "directed": (adj > 0).astype(np.float32).tocsr(),
        "reverse": (adj.T > 0).astype(np.float32).tocsr(),
        "undirected": ((adj + adj.T) > 0).astype(np.float32).tocsr(),
    }

    blocks = []
    for mat in modes.values():
        cur = seed
        for _ in range(max(int(hops), 0)):
            cur = mat @ cur
            total = cur.sum(axis=1, keepdims=True).astype(np.float32)
            prob = cur / np.maximum(total, 1e-6)
            top = prob.max(axis=1, keepdims=True)
            sorted_prob = np.sort(prob, axis=1)
            margin = sorted_prob[:, -1:] - sorted_prob[:, -2:-1]
            entropy = -np.sum(
                np.clip(prob, 1e-9, 1.0) * np.log(np.clip(prob, 1e-9, 1.0)),
                axis=1,
                keepdims=True,
            )
            blocks.extend([
                prob.astype(np.float32),
                np.log1p(total).astype(np.float32),
                top.astype(np.float32),
                margin.astype(np.float32),
                entropy.astype(np.float32),
                (total <= 0).astype(np.float32),
            ])
    return _row_normalize_features(np.concatenate(blocks, axis=1))


def source_meta_features(source_probs: Mapping[str, np.ndarray]) -> np.ndarray:
    """把多个 source 概率转成元特征"""
    names = sorted(source_probs.keys())
    blocks = []
    for name in names:
        probs = source_probs[name].astype(np.float32)
        sorted_probs = np.sort(probs, axis=1)
        top1 = sorted_probs[:, -1:]
        top2 = sorted_probs[:, -2:-1]
        margin = top1 - top2
        entropy = -np.sum(np.clip(probs, 1e-9, 1.0) * np.log(np.clip(probs, 1e-9, 1.0)), axis=1, keepdims=True)
        blocks.extend([probs, top1, margin, entropy])

    # 额外加入专家之间的均值和分歧度。
    stack = np.stack([source_probs[name].astype(np.float32) for name in names], axis=0)
    mean_probs = stack.mean(axis=0)
    std_probs = stack.std(axis=0)
    blocks.extend([mean_probs, std_probs])
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_meta_matrix(
    source_probs: Mapping[str, np.ndarray],
    struct: np.ndarray,
    label_meta: np.ndarray = None,
) -> np.ndarray:
    """合并专家概率、结构特征和可选邻居标签统计特征"""
    blocks = [source_meta_features(source_probs), struct]
    if label_meta is not None:
        blocks.append(label_meta)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def load_split_sources(
    data: Dict,
    sources: Mapping[str, str],
    split_seed: int,
    label_idx: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """加载 split source 概率"""
    out = {}
    for name, base_dir in sources.items():
        path = os.path.join(base_dir, f"split{split_seed}", "best_model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少checkpoint: {path}")
        print(f"  [加载] split={split_seed} source={name} {path}")
        out[name] = load_sign_probs(data, path, device, label_idx=label_idx).cpu().numpy().astype(np.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def build_dataset_for_splits(data: Dict, args, device: torch.device):
    """构造每个 split 的元特征和标签"""
    sources = parse_sources(args.sources)
    struct = graph_struct_features(data)
    rows = {}
    labels = data["labels"].astype(np.int64)

    for split_seed in parse_int_list(args.split_seeds):
        print("\n" + "=" * 100)
        print(f"[Split] {split_seed}")
        print("=" * 100)
        fit_idx, val_idx = split_indices(data, split_seed, args.val_ratio, args.stratified_split)
        source_probs = load_split_sources(data, sources, split_seed, fit_idx, device)
        label_meta = (
            label_neighbor_meta_features(data, fit_idx, args.label_neighbor_hops)
            if args.use_label_neighbor_meta
            else None
        )
        meta = build_meta_matrix(source_probs, struct, label_meta)
        rows[str(split_seed)] = {
            "fit_idx": fit_idx,
            "val_idx": val_idx,
            "x": meta[val_idx],
            "y": labels[val_idx],
            "source_probs": source_probs,
        }
    return rows


def base_probs_from_sources(source_probs: Mapping[str, np.ndarray]) -> np.ndarray:
    """当前线上主线对应的 base 概率"""
    return 0.3 * source_probs["undir"] + 0.7 * source_probs["undir_reverse"]


def acc(pred: np.ndarray, y: np.ndarray) -> float:
    """准确率"""
    return float(np.mean(pred.astype(np.int64) == y.astype(np.int64)))


def train_logistic(x: np.ndarray, y: np.ndarray, c_value: float, class_weight: str):
    """训练一个正则化 LogisticRegression"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("A1元模型stacking需要scikit-learn") from exc

    cw = "balanced" if class_weight == "balanced" else None
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=cw,
            max_iter=500,
            multi_class="ovr",
            solver="liblinear",
            random_state=42,
        ),
    ).fit(x, y)


def run_audit(args):
    """执行 LOO 审计"""
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)
    split_data = build_dataset_for_splits(data, args, device)
    split_keys = sorted(split_data.keys(), key=int)

    results = []
    thresholds = parse_float_list(args.thresholds)
    for c_value in parse_float_list(args.c_values):
        for class_weight in args.class_weights.split(","):
            class_weight = class_weight.strip()
            if not class_weight:
                continue
            threshold_rows = {threshold: [] for threshold in thresholds}
            for heldout in split_keys:
                train_keys = [key for key in split_keys if key != heldout]
                train_x = np.concatenate([split_data[key]["x"] for key in train_keys], axis=0)
                train_y = np.concatenate([split_data[key]["y"] for key in train_keys], axis=0)
                model = train_logistic(train_x, train_y, c_value, class_weight)
                meta_proba = model.predict_proba(split_data[heldout]["x"])
                meta_pred = np.argmax(meta_proba, axis=1).astype(np.int64)
                meta_conf = np.max(meta_proba, axis=1)

                val_idx = split_data[heldout]["val_idx"]
                base_probs = base_probs_from_sources(split_data[heldout]["source_probs"])
                base_pred = np.argmax(base_probs[val_idx], axis=1)
                base_acc = acc(base_pred, split_data[heldout]["y"])

                for threshold in thresholds:
                    # threshold=0 表示纯元模型；阈值越高，越多低置信改动回退到base。
                    use_meta = (meta_conf >= threshold) | (meta_pred == base_pred)
                    pred = np.where(use_meta, meta_pred, base_pred)
                    meta_acc = acc(pred, split_data[heldout]["y"])
                    threshold_rows[threshold].append({
                        "heldout": int(heldout),
                        "meta_acc": meta_acc,
                        "base_acc": base_acc,
                        "gain": meta_acc - base_acc,
                        "changed": float(np.mean(pred != base_pred)),
                    })

            for threshold, loo_rows in threshold_rows.items():
                mean_meta = float(np.mean([row["meta_acc"] for row in loo_rows]))
                mean_base = float(np.mean([row["base_acc"] for row in loo_rows]))
                result = {
                    "C": c_value,
                    "class_weight": class_weight,
                    "threshold": threshold,
                    "mean_meta": mean_meta,
                    "mean_base": mean_base,
                    "mean_gain": mean_meta - mean_base,
                    "min_gain": float(np.min([row["gain"] for row in loo_rows])),
                    "mean_changed": float(np.mean([row["changed"] for row in loo_rows])),
                    "loo_rows": loo_rows,
                }
                results.append(result)
                print(
                    f"C={c_value:g}\tclass_weight={class_weight}\tthreshold={threshold:g}\t"
                    f"mean_meta={mean_meta:.6f}\tmean_base={mean_base:.6f}\t"
                    f"gain={mean_meta - mean_base:+.6f}\tmin_gain={result['min_gain']:+.6f}\t"
                    f"changed={result['mean_changed']:.4%}"
                )

    results.sort(key=lambda item: (item["mean_gain"], item["min_gain"]), reverse=True)
    output = {
        "results": results,
        "best": results[0],
        "split_seeds": split_keys,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("A1 元模型 stacking 审计 Top-10")
    print("=" * 100)
    for item in results[:10]:
        print(
            f"C={item['C']:g}\tclass_weight={item['class_weight']}\t"
            f"threshold={item['threshold']:g}\tmean_meta={item['mean_meta']:.6f}\tgain={item['mean_gain']:+.6f}\t"
            f"min_gain={item['min_gain']:+.6f}"
        )
    print(f"\n结果已保存: {os.path.join(args.output_dir, 'summary.json')}")


def average_fulltrain_sources(data: Dict, source_paths: Mapping[str, List[str]], device: torch.device) -> Dict[str, np.ndarray]:
    """加载并平均全标签 source 概率"""
    out = {}
    for name, paths in source_paths.items():
        avg = None
        for idx, path in enumerate(paths, start=1):
            print(f"  [推理] source={name} {idx}/{len(paths)} {path}")
            probs = load_sign_probs(data, path, device, label_idx=data["train_idx"]).cpu().numpy().astype(np.float32)
            avg = probs if avg is None else avg + probs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        avg = avg / len(paths)
        out[name] = avg / np.maximum(avg.sum(axis=1, keepdims=True), 1e-12)
    return out


def run_infer(args):
    """训练全量元模型并推理测试集"""
    audit = json.load(open(args.audit_json, encoding="utf-8"))
    best = audit["best"]
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    labels = data["labels"].astype(np.int64)
    source_paths = parse_infer_sources(args.sources)
    source_probs = average_fulltrain_sources(data, source_paths, device)
    struct = graph_struct_features(data)
    label_meta = (
        label_neighbor_meta_features(data, data["train_idx"], args.label_neighbor_hops)
        if args.use_label_neighbor_meta
        else None
    )
    meta = build_meta_matrix(source_probs, struct, label_meta)

    if args.oof_sources:
        print("=" * 100)
        print("构造 OOF 元模型训练集")
        print("=" * 100)
        oof_sources = parse_sources(args.oof_sources)
        oof_rows = []
        oof_labels = []
        for split_seed in parse_int_list(args.split_seeds):
            fit_idx, val_idx = split_indices(data, split_seed, args.val_ratio, args.stratified_split)
            split_probs = load_split_sources(data, oof_sources, split_seed, fit_idx, device)
            split_label_meta = (
                label_neighbor_meta_features(data, fit_idx, args.label_neighbor_hops)
                if args.use_label_neighbor_meta
                else None
            )
            split_meta = build_meta_matrix(split_probs, struct, split_label_meta)
            oof_rows.append(split_meta[val_idx])
            oof_labels.append(labels[val_idx])
        train_x = np.concatenate(oof_rows, axis=0)
        train_y = np.concatenate(oof_labels, axis=0)
        print(f"OOF训练样本数: {len(train_y)}")
    else:
        # 后备模式：使用所有 train_idx 的全标签 source 概率训练元模型。
        # 这个模式可能偏乐观，正式候选应优先提供 --oof_sources。
        train_idx = data["train_idx"].astype(np.int64)
        train_x = meta[train_idx]
        train_y = labels[train_idx]

    model = train_logistic(
        train_x,
        train_y,
        float(best["C"]),
        best["class_weight"],
    )
    test_idx = data["test_idx"].astype(np.int64)
    meta_proba = model.predict_proba(meta[test_idx])
    meta_pred = np.argmax(meta_proba, axis=1).astype(np.int64)
    meta_conf = np.max(meta_proba, axis=1)
    base_probs = base_probs_from_sources(source_probs)
    base_pred = np.argmax(base_probs[test_idx], axis=1).astype(np.int64)
    threshold = float(best.get("threshold", 0.0))
    use_meta = (meta_conf >= threshold) | (meta_pred == base_pred)
    pred = np.where(use_meta, meta_pred, base_pred).astype(np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pd.DataFrame({"test_idx": test_idx, "label": pred}).to_csv(args.output_path, index=False)
    unique, counts = np.unique(pred, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    result = {
        "best": best,
        "class_distribution": distribution,
        "changed_vs_base": float(np.mean(pred != base_pred)),
        "output_path": args.output_path,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A1 元模型 stacking 推理完成")
    print("=" * 100)
    print(f"best={best}")
    print(f"changed_vs_base={result['changed_vs_base']:.4%}")
    print(f"class_distribution={distribution}")
    print(f"output={args.output_path}")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="A1 SIGN正则化元模型stacking")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--data_path", required=True)
    audit.add_argument("--output_dir", required=True)
    audit.add_argument("--device", default=None)
    audit.add_argument("--split_seeds", default="42,777,2024,2026,3407")
    audit.add_argument("--val_ratio", type=float, default=0.1)
    audit.add_argument("--stratified_split", action="store_true")
    audit.add_argument("--sources", nargs="+", required=True)
    audit.add_argument("--c_values", default="0.02,0.05,0.1,0.2,0.5,1.0")
    audit.add_argument("--class_weights", default="none,balanced")
    audit.add_argument("--thresholds", default="0,0.4,0.5,0.6,0.7,0.8",
                       help="元模型覆盖base所需置信度；0表示不回退")
    audit.add_argument("--use_label_neighbor_meta", action="store_true",
                       help="为二层元模型加入无泄漏邻居标签统计特征")
    audit.add_argument("--label_neighbor_hops", type=int, default=2,
                       help="邻居标签统计传播跳数")

    infer = sub.add_parser("infer")
    infer.add_argument("--data_path", required=True)
    infer.add_argument("--audit_json", required=True)
    infer.add_argument("--sources", nargs="+", required=True)
    infer.add_argument("--oof_sources", nargs="+", default=[],
                       help="OOF二层训练source，格式 name=split目录；正式推理建议提供")
    infer.add_argument("--split_seeds", default="42,777,2024,2026,3407")
    infer.add_argument("--val_ratio", type=float, default=0.1)
    infer.add_argument("--stratified_split", action="store_true")
    infer.add_argument("--device", default=None)
    infer.add_argument("--use_label_neighbor_meta", action="store_true",
                       help="为二层元模型加入无泄漏邻居标签统计特征")
    infer.add_argument("--label_neighbor_hops", type=int, default=2,
                       help="邻居标签统计传播跳数")
    infer.add_argument("--output_path", required=True)
    infer.add_argument("--output_json", default="")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    if args.command == "audit":
        run_audit(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
