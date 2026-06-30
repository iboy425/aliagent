"""A1 SIGN 专家的按类别轻量 stacking

全局概率平均假设每个专家在所有类别上都同样可靠，但 A1 类别分布很不均衡，
不同标签传播方向的错误模式也不同。本脚本做一个更细的融合：

1. 先用基准融合得到每个节点的预测类别；
2. 对“基准预测为某个类别”的节点，单独选择最可靠的专家融合权重；
3. 正式推理时也按基准预测类别选择对应权重。

为了避免把验证集调穿，audit 模式采用 leave-one-split-out：
- 用 4 个 split 的验证表现选择类别到候选权重的映射；
- 在剩下 1 个 split 上评估这个映射；
- 轮流做 5 次，得到更接近线上泛化的估计。
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from datasets import GraphDataset
from utils import get_device, split_train_val, stratified_split
from a1_correct_smooth import _prepare_cs_adj, correct_and_smooth, normalize_score_rows
from a1_sign_infer import load_sign_probs


DEFAULT_CANDIDATES = (
    "base=undir:0.3,undir_reverse:0.7|"
    "reverse=undir_reverse:1.0|"
    "undir=undir:1.0|"
    "u7r3=undir:0.7,undir_reverse:0.3|"
    "u2r8=undir:0.2,undir_reverse:0.8|"
    "u3r6a1=undir:0.3,undir_reverse:0.6,all_h2:0.1|"
    "u3r5a2=undir:0.3,undir_reverse:0.5,all_h2:0.2|"
    "u3r4a3=undir:0.3,undir_reverse:0.4,all_h2:0.3|"
    "u2r6a2=undir:0.2,undir_reverse:0.6,all_h2:0.2|"
    "a1=all_h2:1.0"
)


def parse_int_list(value: str) -> List[int]:
    """解析整数列表"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_sources(values: Sequence[str]) -> Dict[str, str]:
    """解析 name=path 形式的 split source"""
    sources = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source必须是 name=path: {value}")
        name, path = value.split("=", 1)
        sources[name.strip()] = path.strip()
    return sources


def parse_infer_sources(values: Sequence[str]) -> Dict[str, List[str]]:
    """解析推理 source，path 支持 glob"""
    sources = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"infer source必须是 name=glob: {value}")
        name, pattern = value.split("=", 1)
        paths = sorted(glob.glob(pattern.strip()))
        if not paths:
            raise FileNotFoundError(f"没有匹配到checkpoint: {value}")
        sources[name.strip()] = paths
    return sources


def parse_candidates(value: str) -> Dict[str, Dict[str, float]]:
    """解析候选融合权重"""
    candidates = {}
    for part in value.split("|"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"candidate必须是 name=w1:0.5,w2:0.5: {part}")
        name, body = part.split("=", 1)
        weights = {}
        for item in body.split(","):
            if not item.strip():
                continue
            src, weight = item.split(":", 1)
            weights[src.strip()] = float(weight)
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"candidate权重和必须大于0: {part}")
        candidates[name.strip()] = {k: v / total for k, v in weights.items()}
    if not candidates:
        raise ValueError("至少需要一个candidate")
    return candidates


def split_indices(data: Dict, split_seed: int, val_ratio: float, stratified: bool) -> Tuple[np.ndarray, np.ndarray]:
    """划分训练/验证节点"""
    if stratified:
        return stratified_split(data["labels"], data["train_idx"], val_ratio, split_seed)
    return split_train_val(data["train_idx"], val_ratio, split_seed)


def mix_probs(source_probs: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    """按权重融合 source 概率"""
    mixed = None
    for name, weight in weights.items():
        if name not in source_probs:
            raise KeyError(f"source_probs缺少 {name}")
        part = source_probs[name] * float(weight)
        mixed = part.clone() if mixed is None else mixed + part
    return normalize_score_rows(mixed)


def accuracy_for_nodes(pred: np.ndarray, labels: np.ndarray, nodes: Sequence[int]) -> float:
    """计算指定节点准确率"""
    if len(nodes) == 0:
        return 0.0
    nodes = np.asarray(nodes, dtype=np.int64)
    return float(np.mean(pred[nodes] == labels[nodes]))


def load_split_source_probs(
    data: Dict,
    sources: Mapping[str, str],
    split_seed: int,
    label_idx: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """加载某个 split 的各 source 概率"""
    out = {}
    for name, base_dir in sources.items():
        path = os.path.join(base_dir, f"split{split_seed}", "best_model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少checkpoint: {path}")
        print(f"  [加载] split={split_seed} source={name} {path}")
        out[name] = load_sign_probs(data, path, device, label_idx=label_idx)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def prepare_split_predictions(data: Dict, args, device: torch.device) -> Tuple[List[Dict], Dict[str, Dict]]:
    """预计算每个 split、每个候选的预测"""
    sources = parse_sources(args.sources)
    candidates = parse_candidates(args.candidates)
    labels = data["labels"].astype(np.int64)
    split_rows = []
    all_preds: Dict[str, Dict] = {}

    for split_seed in parse_int_list(args.split_seeds):
        print("\n" + "=" * 100)
        print(f"[Split] {split_seed}")
        print("=" * 100)
        fit_idx, val_idx = split_indices(data, split_seed, args.val_ratio, args.stratified_split)
        source_probs = load_split_source_probs(data, sources, split_seed, fit_idx, device)

        split_key = str(split_seed)
        all_preds[split_key] = {"val_idx": val_idx.tolist(), "candidates": {}}
        candidate_preds = {}
        candidate_probs = {}
        for cand_name, weights in candidates.items():
            probs = mix_probs(source_probs, weights)
            pred = torch.argmax(probs, dim=1).cpu().numpy().astype(np.int64)
            candidate_preds[cand_name] = pred
            candidate_probs[cand_name] = probs.cpu().numpy().astype(np.float32)
            acc = accuracy_for_nodes(pred, labels, val_idx)
            split_rows.append({
                "split_seed": split_seed,
                "candidate": cand_name,
                "val_acc": acc,
                "weights": weights,
            })
            all_preds[split_key]["candidates"][cand_name] = {"pred": pred}

        base_pred = candidate_preds[args.base_candidate]
        all_preds[split_key]["base_pred"] = base_pred
        all_preds[split_key]["candidate_probs"] = candidate_probs
        print("候选Top:")
        for row in sorted([r for r in split_rows if r["split_seed"] == split_seed], key=lambda x: x["val_acc"], reverse=True)[:8]:
            print(f"  {row['candidate']}\t{row['val_acc']:.6f}\t{row['weights']}")

    return split_rows, all_preds


def choose_mapping(
    train_split_keys: Sequence[str],
    all_preds: Mapping[str, Dict],
    labels: np.ndarray,
    candidates: Mapping[str, Dict[str, float]],
    base_candidate: str,
    min_bucket_count: int,
    min_gain: float,
) -> Dict[str, str]:
    """用若干 split 选择预测类别到候选的映射"""
    mapping = {}
    num_classes = int(labels[labels >= 0].max()) + 1
    for cls in range(num_classes):
        bucket_nodes = []
        correct_by_candidate = defaultdict(int)
        total = 0
        for split_key in train_split_keys:
            val_idx = np.asarray(all_preds[split_key]["val_idx"], dtype=np.int64)
            base_pred = np.asarray(all_preds[split_key]["base_pred"], dtype=np.int64)
            nodes = val_idx[base_pred[val_idx] == cls]
            if len(nodes) == 0:
                continue
            bucket_nodes.extend(nodes.tolist())
            total += len(nodes)
            for cand_name in candidates:
                cand_pred = np.asarray(all_preds[split_key]["candidates"][cand_name]["pred"], dtype=np.int64)
                correct_by_candidate[cand_name] += int(np.sum(cand_pred[nodes] == labels[nodes]))

        if total < min_bucket_count:
            mapping[str(cls)] = base_candidate
            continue
        base_acc = correct_by_candidate[base_candidate] / total
        best_name = max(candidates.keys(), key=lambda name: correct_by_candidate[name] / total)
        best_acc = correct_by_candidate[best_name] / total
        mapping[str(cls)] = best_name if best_acc >= base_acc + min_gain else base_candidate
    return mapping


def apply_mapping(
    split_key: str,
    mapping: Mapping[str, str],
    all_preds: Mapping[str, Dict],
    labels: np.ndarray,
    base_candidate: str,
) -> Dict:
    """在一个 split 上应用类别映射"""
    val_idx = np.asarray(all_preds[split_key]["val_idx"], dtype=np.int64)
    base_pred = np.asarray(all_preds[split_key]["base_pred"], dtype=np.int64)
    out_pred = np.asarray(all_preds[split_key]["candidates"][base_candidate]["pred"], dtype=np.int64).copy()

    usage = defaultdict(int)
    for node in val_idx:
        cls = str(int(base_pred[node]))
        cand_name = mapping.get(cls, base_candidate)
        usage[cand_name] += 1
        out_pred[node] = all_preds[split_key]["candidates"][cand_name]["pred"][node]

    return {
        "split_seed": int(split_key),
        "val_acc": accuracy_for_nodes(out_pred, labels, val_idx),
        "usage": dict(usage),
        "mapping": dict(mapping),
    }


def run_audit(args):
    """执行无泄漏类别 stacking 审计"""
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    labels = data["labels"].astype(np.int64)
    candidates = parse_candidates(args.candidates)
    os.makedirs(args.output_dir, exist_ok=True)

    split_rows, all_preds = prepare_split_predictions(data, args, device)
    split_keys = sorted(all_preds.keys(), key=int)

    loo_rows = []
    for heldout in split_keys:
        train_keys = [key for key in split_keys if key != heldout]
        mapping = choose_mapping(
            train_split_keys=train_keys,
            all_preds=all_preds,
            labels=labels,
            candidates=candidates,
            base_candidate=args.base_candidate,
            min_bucket_count=args.min_bucket_count,
            min_gain=args.min_gain,
        )
        result = apply_mapping(heldout, mapping, all_preds, labels, args.base_candidate)
        loo_rows.append(result)
        print(
            f"[LOO] heldout={heldout}\tval_acc={result['val_acc']:.6f}\t"
            f"usage={result['usage']}\tmapping={mapping}"
        )

    final_mapping = choose_mapping(
        train_split_keys=split_keys,
        all_preds=all_preds,
        labels=labels,
        candidates=candidates,
        base_candidate=args.base_candidate,
        min_bucket_count=args.min_bucket_count,
        min_gain=args.min_gain,
    )
    base_rows = [row for row in split_rows if row["candidate"] == args.base_candidate]
    output = {
        "base_candidate": args.base_candidate,
        "candidates": candidates,
        "base_mean": float(np.mean([row["val_acc"] for row in base_rows])),
        "loo_mean": float(np.mean([row["val_acc"] for row in loo_rows])),
        "loo_min": float(np.min([row["val_acc"] for row in loo_rows])),
        "loo_rows": loo_rows,
        "final_mapping": final_mapping,
        "split_candidate_rows": split_rows,
        "min_bucket_count": args.min_bucket_count,
        "min_gain": args.min_gain,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("A1 类别 stacking 审计汇总")
    print("=" * 100)
    print(f"base_mean={output['base_mean']:.6f}")
    print(f"loo_mean={output['loo_mean']:.6f}")
    print(f"loo_min={output['loo_min']:.6f}")
    print(f"final_mapping={final_mapping}")


def average_fulltrain_source_probs(
    data: Dict,
    source_paths: Mapping[str, List[str]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """加载全标签模型并按 source 求平均概率"""
    out = {}
    for name, paths in source_paths.items():
        avg = None
        for idx, path in enumerate(paths, start=1):
            print(f"  [推理] source={name} {idx}/{len(paths)} {path}")
            probs = load_sign_probs(data, path, device, label_idx=data["train_idx"])
            avg = probs if avg is None else avg + probs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        out[name] = normalize_score_rows(avg / len(paths))
    return out


def run_infer(args):
    """按审计得到的类别映射生成 A1.csv"""
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    source_paths = parse_infer_sources(args.sources)
    candidates = parse_candidates(args.candidates)
    audit = json.load(open(args.audit_json, encoding="utf-8"))
    mapping = audit["final_mapping"]
    base_candidate = audit["base_candidate"]
    source_probs = average_fulltrain_source_probs(data, source_paths, device)

    candidate_probs = {
        name: mix_probs(source_probs, weights)
        for name, weights in candidates.items()
    }
    base_pred = torch.argmax(candidate_probs[base_candidate], dim=1).cpu().numpy().astype(np.int64)
    final_scores = candidate_probs[base_candidate].clone()
    test_idx = data["test_idx"]
    usage = defaultdict(int)
    for node in test_idx:
        cls = str(int(base_pred[node]))
        cand_name = mapping.get(cls, base_candidate)
        usage[cand_name] += 1
        final_scores[node] = candidate_probs[cand_name][node]

    if args.smooth_weight > 0:
        labels = torch.LongTensor(data["labels"]).to(device)
        adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)
        final_scores = correct_and_smooth(
            model_probs=final_scores,
            labels=labels,
            train_idx=torch.LongTensor(data["train_idx"]).to(device),
            adj=adj,
            params={
                "correct_alpha": 0.3,
                "correct_iter": 5,
                "correct_weight": 0.0,
                "smooth_alpha": 0.7,
                "smooth_iter": 5,
                "smooth_weight": args.smooth_weight,
            },
        )

    preds = torch.argmax(final_scores[torch.LongTensor(test_idx).to(device)], dim=1).cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pd.DataFrame({"test_idx": test_idx, "label": preds}).to_csv(args.output_path, index=False)
    unique, counts = np.unique(preds, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}

    result = {
        "mapping": mapping,
        "usage": dict(usage),
        "class_distribution": distribution,
        "output_path": args.output_path,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A1 类别 stacking 推理完成")
    print("=" * 100)
    print(f"usage={dict(usage)}")
    print(f"class_distribution={distribution}")
    print(f"output={args.output_path}")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="A1 SIGN按类别 stacking")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--data_path", required=True)
    audit.add_argument("--output_dir", required=True)
    audit.add_argument("--device", default=None)
    audit.add_argument("--split_seeds", default="42,777,2024,2026,3407")
    audit.add_argument("--val_ratio", type=float, default=0.1)
    audit.add_argument("--stratified_split", action="store_true")
    audit.add_argument("--sources", nargs="+", required=True)
    audit.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    audit.add_argument("--base_candidate", default="base")
    audit.add_argument("--min_bucket_count", type=int, default=120)
    audit.add_argument("--min_gain", type=float, default=0.002)

    infer = sub.add_parser("infer")
    infer.add_argument("--data_path", required=True)
    infer.add_argument("--audit_json", required=True)
    infer.add_argument("--sources", nargs="+", required=True)
    infer.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    infer.add_argument("--device", default=None)
    infer.add_argument("--output_path", required=True)
    infer.add_argument("--output_json", default="")
    infer.add_argument("--cs_normalize", default="random_walk", choices=["random_walk", "symmetric", "none"])
    infer.add_argument("--smooth_weight", type=float, default=0.0)
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
