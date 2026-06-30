"""A1邻居多数类/LP专家门控融合

SIGN 标签传播特征已经给 A1 带来主要提升，但高纯度邻居节点仍可能被
全局 MLP 边界拉偏。本脚本做一个轻量后处理专家：

- 用已知训练标签统计每个节点的邻居多数类和纯度；
- 可选构造 LP 概率专家；
- 只在“邻居标签足够多、纯度足够高、模型置信度不够高”的节点上融合专家。

审计模式严格无泄漏：每个 split 只用 fit_idx 标签构造邻居/LP 专家，
验证节点标签只用于最终打分。
"""
import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from a1_correct_smooth import normalize_score_rows, propagate_with_restart
from a1_sign_infer import load_sign_probs
from datasets import GraphDataset
from utils import random_walk_normalize_sparse, stratified_split


SPLIT_SEEDS = [42, 777, 2024, 2026, 3407]


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点数"""
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数"""
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_bool_list(value: str) -> List[bool]:
    """解析布尔列表"""
    mapping = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}
    result = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"无法解析布尔值: {item}")
        result.append(mapping[item])
    return result


def parse_weights(value: str, count: int) -> List[float]:
    """解析并归一化checkpoint权重"""
    if not value:
        return [1.0 / count for _ in range(count)]
    weights = parse_float_list(value)
    if len(weights) != count:
        raise ValueError("checkpoint权重数量必须和checkpoint数量一致")
    total = sum(weights)
    if total <= 0:
        raise ValueError("checkpoint权重总和必须大于0")
    return [weight / total for weight in weights]


def make_undirected_adj(adj: sp.spmatrix) -> sp.csr_matrix:
    """构造无向二值邻接矩阵，不含自环"""
    out = (adj.tocsr() + adj.T.tocsr()).astype(np.float32)
    out.data[:] = 1.0
    out.setdiag(0)
    out.eliminate_zeros()
    return out.tocsr()


def build_neighbor_majority(
    adj: sp.spmatrix,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    num_classes: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """统计每个节点的已知邻居多数类专家"""
    undirected = make_undirected_adj(adj)
    known = np.zeros(labels.shape[0], dtype=bool)
    known[np.asarray(fit_idx, dtype=np.int64)] = True

    probs = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    labeled_counts = np.zeros(labels.shape[0], dtype=np.int32)
    purity = np.zeros(labels.shape[0], dtype=np.float32)
    majority_pred = np.zeros(labels.shape[0], dtype=np.int64)

    indptr = undirected.indptr
    indices = undirected.indices
    for node in range(undirected.shape[0]):
        neigh = indices[indptr[node]:indptr[node + 1]]
        if len(neigh) == 0:
            probs[node] = 1.0 / num_classes
            continue
        labeled_neigh = neigh[known[neigh]]
        if len(labeled_neigh) == 0:
            probs[node] = 1.0 / num_classes
            continue
        counts = np.bincount(labels[labeled_neigh].astype(np.int64), minlength=num_classes).astype(np.float32)
        total = float(counts.sum())
        pred = int(counts.argmax())
        probs[node] = counts / max(total, 1.0)
        labeled_counts[node] = int(total)
        purity[node] = float(counts[pred] / max(total, 1.0))
        majority_pred[node] = pred

    return (
        torch.from_numpy(probs),
        labeled_counts,
        purity,
        majority_pred,
    )


def build_lp_probs(
    adj: sp.spmatrix,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    num_classes: int,
    device: torch.device,
    alpha: float,
    num_iter: int,
) -> torch.Tensor:
    """构造简单 LP 概率专家"""
    undirected = make_undirected_adj(adj)
    adj_t = random_walk_normalize_sparse(undirected, device=device)
    seed = torch.zeros((labels.shape[0], num_classes), dtype=torch.float32, device=device)
    fit_t = torch.LongTensor(fit_idx).to(device)
    label_t = torch.LongTensor(labels[fit_idx]).to(device)
    seed[fit_t] = F.one_hot(label_t, num_classes=num_classes).float()
    lp = propagate_with_restart(adj_t, seed, alpha=alpha, num_iter=num_iter)
    return normalize_score_rows(lp)


def load_weighted_probs(
    data: Dict,
    checkpoints: Sequence[str],
    weights: Sequence[float],
    device: torch.device,
    label_idx: np.ndarray,
) -> torch.Tensor:
    """加载多个SIGN checkpoint并加权平均概率"""
    avg_probs = None
    for idx, (checkpoint, weight) in enumerate(zip(checkpoints, weights), start=1):
        print(f"[SIGN概率] {idx}/{len(checkpoints)} weight={weight:.6f} {checkpoint}")
        probs = load_sign_probs(data, checkpoint, device, label_idx=label_idx)
        avg_probs = probs.mul(weight) if avg_probs is None else avg_probs + probs.mul(weight)
    if avg_probs is None:
        raise ValueError("至少需要一个checkpoint")
    return normalize_score_rows(avg_probs)


def apply_gate(
    model_probs: torch.Tensor,
    expert_probs: torch.Tensor,
    labeled_counts: np.ndarray,
    purity: np.ndarray,
    majority_pred: np.ndarray,
    params: Dict,
    target_idx: np.ndarray,
) -> Tuple[torch.Tensor, int]:
    """按门控条件融合专家概率"""
    device = model_probs.device
    out = model_probs.clone()
    target_t = torch.LongTensor(target_idx).to(device)
    conf, pred = model_probs[target_t].max(dim=1)
    target_np = np.asarray(target_idx, dtype=np.int64)

    mask = (
        (labeled_counts[target_np] >= int(params["min_neighbors"]))
        & (purity[target_np] >= float(params["purity_threshold"]))
        & (conf.detach().cpu().numpy() <= float(params["max_model_conf"]))
    )
    if params.get("require_disagree", False):
        mask &= pred.detach().cpu().numpy() != majority_pred[target_np]

    selected_np = target_np[mask]
    if len(selected_np) > 0:
        selected_t = torch.LongTensor(selected_np).to(device)
        weight = float(params["expert_weight"])
        out[selected_t] = normalize_score_rows(
            (1.0 - weight) * out[selected_t] + weight * expert_probs[selected_t].to(device)
        )
    return out, int(len(selected_np))


def score_accuracy(scores: torch.Tensor, labels: np.ndarray, idx: np.ndarray, device: torch.device) -> float:
    """计算准确率"""
    idx_t = torch.LongTensor(idx).to(device)
    labels_t = torch.LongTensor(labels[idx]).to(device)
    pred = scores[idx_t].argmax(dim=1)
    return float((pred == labels_t).float().mean().item())


def split_paths(base_dir: str, split_seed: int) -> List[str]:
    """返回某个split对应的两个SIGN配置checkpoint"""
    return [
        os.path.join(base_dir, "standard_struct_label_undir_h3_rw", f"split{split_seed}", "best_model.pt"),
        os.path.join(base_dir, "standard_struct_label_undir_reverse_h3_rw", f"split{split_seed}", "best_model.pt"),
    ]


def iter_gate_params(args):
    """遍历门控参数"""
    for expert in [item.strip() for item in args.experts.split(",") if item.strip()]:
        for min_neighbors in parse_int_list(args.min_neighbors):
            for purity_threshold in parse_float_list(args.purity_thresholds):
                for max_model_conf in parse_float_list(args.max_model_confs):
                    for expert_weight in parse_float_list(args.expert_weights):
                        for require_disagree in parse_bool_list(args.require_disagrees):
                            yield {
                                "expert": expert,
                                "min_neighbors": min_neighbors,
                                "purity_threshold": purity_threshold,
                                "max_model_conf": max_model_conf,
                                "expert_weight": expert_weight,
                                "require_disagree": require_disagree,
                            }


def audit(args):
    """无泄漏多split审计"""
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data = GraphDataset.load(args.data_path)
    labels = data["labels"]
    all_params = list(iter_gate_params(args))
    aggregate = {idx: {"params": params, "accs": [], "selected": [], "baseline": []}
                 for idx, params in enumerate(all_params)}

    for split_seed in parse_int_list(args.split_seeds):
        print("\n" + "=" * 100)
        print(f"[A1 gate audit] split_seed={split_seed}")
        print("=" * 100)
        fit_idx, val_idx = stratified_split(labels, data["train_idx"], args.val_ratio, split_seed)
        checkpoints = split_paths(args.base_dir, split_seed)
        for checkpoint in checkpoints:
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(checkpoint)
        weights = parse_weights(args.checkpoint_weights, len(checkpoints))
        model_probs = load_weighted_probs(data, checkpoints, weights, device, label_idx=fit_idx)
        baseline = score_accuracy(model_probs, labels, val_idx, device)
        print(f"baseline={baseline:.6f}")

        maj_probs, labeled_counts, purity, majority_pred = build_neighbor_majority(
            data["adj"], labels, fit_idx, data["num_classes"]
        )
        maj_probs = maj_probs.to(device)
        lp_probs = None

        for idx, item in aggregate.items():
            params = item["params"]
            if params["expert"] == "majority":
                expert_probs = maj_probs
            elif params["expert"] == "lp":
                if lp_probs is None:
                    lp_probs = build_lp_probs(
                        data["adj"],
                        labels,
                        fit_idx,
                        data["num_classes"],
                        device,
                        alpha=args.lp_alpha,
                        num_iter=args.lp_iter,
                    )
                expert_probs = lp_probs
            else:
                raise ValueError(f"未知expert: {params['expert']}")

            scores, selected_count = apply_gate(
                model_probs=model_probs,
                expert_probs=expert_probs,
                labeled_counts=labeled_counts,
                purity=purity,
                majority_pred=majority_pred,
                params=params,
                target_idx=val_idx,
            )
            acc = score_accuracy(scores, labels, val_idx, device)
            item["accs"].append(acc)
            item["selected"].append(selected_count)
            item["baseline"].append(baseline)

    rows = []
    for item in aggregate.values():
        accs = np.array(item["accs"], dtype=np.float64)
        baseline = np.array(item["baseline"], dtype=np.float64)
        selected = np.array(item["selected"], dtype=np.float64)
        row = {
            **item["params"],
            "mean_acc": float(accs.mean()),
            "mean_gain": float((accs - baseline).mean()),
            "min_acc": float(accs.min()),
            "std_acc": float(accs.std()),
            "mean_selected": float(selected.mean()),
            "split_accs": [float(x) for x in accs],
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["mean_acc"], row["min_acc"], row["mean_gain"]), reverse=True)

    print("\nA1 gate审计 Top-20")
    for row in rows[:20]:
        print(
            f"mean={row['mean_acc']:.6f}\tgain={row['mean_gain']:+.6f}\t"
            f"min={row['min_acc']:.6f}\tstd={row['std_acc']:.6f}\t"
            f"selected={row['mean_selected']:.1f}\t"
            f"expert={row['expert']}\tmin_n={row['min_neighbors']}\t"
            f"purity={row['purity_threshold']}\tconf<={row['max_model_conf']}\t"
            f"w={row['expert_weight']}\tdisagree={row['require_disagree']}"
        )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"results": rows, "best": rows[0]}, f, ensure_ascii=False, indent=2)
        pd.DataFrame(rows).to_csv(os.path.splitext(args.output_json)[0] + ".csv", index=False)
        print(f"\n审计结果已保存: {args.output_json}")


def infer(args):
    """正式推理"""
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data = GraphDataset.load(args.data_path)
    labels = data["labels"]
    checkpoints = args.checkpoints
    weights = parse_weights(args.checkpoint_weights, len(checkpoints))
    model_probs = load_weighted_probs(data, checkpoints, weights, device, label_idx=data["train_idx"])
    maj_probs, labeled_counts, purity, majority_pred = build_neighbor_majority(
        data["adj"], labels, data["train_idx"], data["num_classes"]
    )
    if args.expert == "majority":
        expert_probs = maj_probs.to(device)
    elif args.expert == "lp":
        expert_probs = build_lp_probs(
            data["adj"], labels, data["train_idx"], data["num_classes"],
            device, alpha=args.lp_alpha, num_iter=args.lp_iter,
        )
    else:
        raise ValueError(f"未知expert: {args.expert}")

    params = {
        "expert": args.expert,
        "min_neighbors": args.min_neighbors_value,
        "purity_threshold": args.purity_threshold,
        "max_model_conf": args.max_model_conf,
        "expert_weight": args.expert_weight,
        "require_disagree": args.require_disagree,
    }
    scores, selected_count = apply_gate(
        model_probs=model_probs,
        expert_probs=expert_probs,
        labeled_counts=labeled_counts,
        purity=purity,
        majority_pred=majority_pred,
        params=params,
        target_idx=data["test_idx"],
    )
    test_idx_t = torch.LongTensor(data["test_idx"]).to(device)
    predictions = scores[test_idx_t].argmax(dim=1).detach().cpu().numpy()
    result_df = pd.DataFrame({"test_idx": data["test_idx"], "label": predictions})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)

    unique, counts = np.unique(predictions, return_counts=True)
    distribution = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
    print("\nA1 gate推理完成")
    print(f"selected_count={selected_count}")
    print(f"params={params}")
    print(f"类别分布={distribution}")
    print(f"结果已保存: {args.output_path}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "params": params,
                "selected_count": selected_count,
                "class_distribution": distribution,
                "checkpoints": checkpoints,
                "weights": weights,
            }, f, ensure_ascii=False, indent=2)


def build_parser():
    """构建命令行参数"""
    parser = argparse.ArgumentParser(description="A1邻居多数类/LP专家门控")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--data_path", default="data/cls_data/A1.npz")
    audit_parser.add_argument("--base_dir", default="output/exp059_a1_sign_directed_label_feature_audit")
    audit_parser.add_argument("--checkpoint_weights", default="0.3,0.7")
    audit_parser.add_argument("--device", default="cuda")
    audit_parser.add_argument("--val_ratio", type=float, default=0.1)
    audit_parser.add_argument("--split_seeds", default="42,777,2024,2026,3407")
    audit_parser.add_argument("--experts", default="majority,lp")
    audit_parser.add_argument("--min_neighbors", default="1,2,3")
    audit_parser.add_argument("--purity_thresholds", default="0.75,0.8,0.85,0.9,0.95,1.0")
    audit_parser.add_argument("--max_model_confs", default="0.55,0.65,0.75,0.85,0.95,1.01")
    audit_parser.add_argument("--expert_weights", default="0.2,0.35,0.5,0.65,0.8,1.0")
    audit_parser.add_argument("--require_disagrees", default="0,1")
    audit_parser.add_argument("--lp_alpha", type=float, default=0.5)
    audit_parser.add_argument("--lp_iter", type=int, default=20)
    audit_parser.add_argument("--output_json", default="")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--data_path", default="data/cls_data/A1.npz")
    infer_parser.add_argument("--checkpoints", nargs="+", required=True)
    infer_parser.add_argument("--checkpoint_weights", default="")
    infer_parser.add_argument("--device", default="cuda")
    infer_parser.add_argument("--expert", default="majority", choices=["majority", "lp"])
    infer_parser.add_argument("--min_neighbors_value", type=int, required=True)
    infer_parser.add_argument("--purity_threshold", type=float, required=True)
    infer_parser.add_argument("--max_model_conf", type=float, required=True)
    infer_parser.add_argument("--expert_weight", type=float, required=True)
    infer_parser.add_argument("--require_disagree", action="store_true")
    infer_parser.add_argument("--lp_alpha", type=float, default=0.5)
    infer_parser.add_argument("--lp_iter", type=int, default=20)
    infer_parser.add_argument("--output_path", required=True)
    infer_parser.add_argument("--output_json", default="")
    return parser


def main():
    """主入口"""
    args = build_parser().parse_args()
    if args.mode == "audit":
        audit(args)
    elif args.mode == "infer":
        infer(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
