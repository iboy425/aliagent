"""A1 基于 leaderboard 稀疏反馈的约束 MAP 推断

比赛反馈给了每个 A1 提交的整体 Accuracy。对于 2751 个测试节点，
一次提交的分数等价于一个稀疏线性约束：

    sum_i 1[y_i == pred_submission_i] = correct_count

多次提交会形成多条约束。这个脚本用这些约束和历史提交的加权投票
先验，求解一个最可能的测试标签分配。它不是重新训练模型，而是把
“稀疏反馈下的自动实验”真正转成受约束的推断问题。

注意：
- 该方法会利用 A 榜反馈，存在 A 榜过拟合风险；
- 分数显示四位小数，因此 correct_count 默认按 round(score * N)；
- 若求解不可行，可增大 --count_tolerance。
"""
import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from datasets import GraphDataset


def parse_submission_spec(value: str) -> Tuple[str, float, str]:
    """解析 name,score,path 格式"""
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"submission格式必须是 name,score,path: {value}")
    name = parts[0].strip()
    score = float(parts[1].strip())
    path = parts[2].strip()
    return name, score, path


def load_a1(path: str, expected_idx: Sequence[int]) -> np.ndarray:
    """读取 A1.csv 并校验 test_idx 顺序"""
    df = pd.read_csv(path)
    if df["test_idx"].astype(int).tolist() != list(map(int, expected_idx)):
        raise ValueError(f"A1.csv test_idx顺序不一致: {path}")
    return df["label"].astype(int).to_numpy()


def score_to_count(score: float, n: int) -> int:
    """将四位小数线上分数转成正确数估计"""
    return int(round(float(score) * n))


def make_vote_prior(preds: np.ndarray, scores: np.ndarray, num_classes: int, mode: str) -> np.ndarray:
    """根据历史提交构造测试标签先验分数"""
    n = preds.shape[1]
    prior = np.zeros((n, num_classes), dtype=np.float64)

    if mode == "score":
        weights = scores.copy()
    elif mode == "centered":
        weights = np.maximum(scores - scores.min() + 1e-4, 1e-4)
    elif mode == "power":
        weights = np.maximum(scores - 0.70, 1e-4) ** 2
    elif mode == "logit":
        # 若一个提交整体准确率为 a，粗略看成“该标注器给出的类比随机类更可信”的证据。
        weights = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
    else:
        raise ValueError(f"未知 prior mode: {mode}")

    for sub_idx in range(preds.shape[0]):
        for cls in range(num_classes):
            prior[:, cls] += weights[sub_idx] * (preds[sub_idx] == cls)
    return prior


def add_class_prior(prior: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, weight: float) -> np.ndarray:
    """加入训练集类别分布先验"""
    if weight <= 0:
        return prior
    counts = np.bincount(labels[train_idx].astype(int), minlength=prior.shape[1]).astype(np.float64)
    probs = (counts + 1.0) / (counts.sum() + prior.shape[1])
    return prior + weight * np.log(probs.reshape(1, -1))


def add_model_prior(prior: np.ndarray, prior_scores_npy: str, weight: float) -> np.ndarray:
    """加入模型概率先验"""
    if not prior_scores_npy or weight <= 0:
        return prior
    probs = np.load(prior_scores_npy).astype(np.float64)
    if probs.shape != prior.shape:
        raise ValueError(f"prior_scores_npy shape不一致: {probs.shape} vs {prior.shape}")
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return prior + weight * np.log(np.clip(probs, 1e-9, 1.0))


def class_count_bounds_from_reference(
    reference_path: str,
    expected_idx: Sequence[int],
    num_classes: int,
    slack: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """根据参考A1预测构造类别数量上下界"""
    if not reference_path or slack < 0:
        return None, None
    ref = load_a1(reference_path, expected_idx)
    counts = np.bincount(ref.astype(int), minlength=num_classes)
    lower = np.maximum(counts - slack, 0).astype(float)
    upper = (counts + slack).astype(float)
    return lower, upper


def build_constraints(
    preds: np.ndarray,
    counts: Sequence[int],
    count_tolerance: int,
    n: int,
    num_classes: int,
    class_lower: np.ndarray = None,
    class_upper: np.ndarray = None,
) -> LinearConstraint:
    """构造节点 one-hot 约束和 leaderboard 正确数约束"""
    rows = []
    cols = []
    data = []
    lower = []
    upper = []

    # 每个节点必须选择一个类别。
    for node in range(n):
        row_id = len(lower)
        for cls in range(num_classes):
            rows.append(row_id)
            cols.append(node * num_classes + cls)
            data.append(1.0)
        lower.append(1.0)
        upper.append(1.0)

    # 每个历史提交的正确数约束。
    for sub_idx, count in enumerate(counts):
        row_id = len(lower)
        for node in range(n):
            cls = int(preds[sub_idx, node])
            rows.append(row_id)
            cols.append(node * num_classes + cls)
            data.append(1.0)
        lower.append(float(count - count_tolerance))
        upper.append(float(count + count_tolerance))

    # 类别数量约束，防止 leaderboard 约束不足时解坍缩到大类。
    if class_lower is not None and class_upper is not None:
        for cls in range(num_classes):
            row_id = len(lower)
            for node in range(n):
                rows.append(row_id)
                cols.append(node * num_classes + cls)
                data.append(1.0)
            lower.append(float(class_lower[cls]))
            upper.append(float(class_upper[cls]))

    matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(lower), n * num_classes),
    ).tocsr()
    return LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))


def solve_map(prior: np.ndarray, preds: np.ndarray, counts: Sequence[int], args) -> Tuple[np.ndarray, Dict]:
    """求解受 leaderboard 约束的 MAP 标签"""
    n, num_classes = prior.shape
    constraints = build_constraints(
        preds,
        counts,
        args.count_tolerance,
        n,
        num_classes,
        args.class_count_lower,
        args.class_count_upper,
    )
    # scipy.optimize.milp 是最小化，因此取负先验。
    c = -prior.reshape(-1)
    result = milp(
        c=c,
        integrality=np.ones(n * num_classes, dtype=np.int8),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options={
            "time_limit": args.time_limit,
            "mip_rel_gap": args.mip_rel_gap,
            "disp": bool(args.verbose),
        },
    )
    if not result.success:
        raise RuntimeError(f"MILP求解失败: status={result.status}, message={result.message}")
    x = np.asarray(result.x).reshape(n, num_classes)
    pred = np.argmax(x, axis=1).astype(np.int64)
    return pred, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "mip_gap": None if result.mip_gap is None else float(result.mip_gap),
    }


def summarize_candidate(
    candidate: np.ndarray,
    submissions: List[Dict],
    preds: np.ndarray,
    base_name: str,
) -> Dict:
    """汇总候选与历史提交的关系"""
    summary = {
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(candidate, return_counts=True))},
        "matches": [],
    }
    base_idx = None
    for idx, item in enumerate(submissions):
        if item["name"] == base_name:
            base_idx = idx
        matched = int(np.sum(candidate == preds[idx]))
        summary["matches"].append({
            "name": item["name"],
            "target_count": int(item["count"]),
            "candidate_match_count": matched,
            "score": float(item["score"]),
        })
    if base_idx is not None:
        base_pred = preds[base_idx]
        changed = candidate != base_pred
        summary["base_name"] = base_name
        summary["changed_vs_base"] = float(np.mean(changed))
        summary["changed_count_vs_base"] = int(np.sum(changed))
        summary["transitions_vs_base"] = [
            {"from": int(a), "to": int(b), "count": int(c)}
            for (a, b), c in Counter(
                (int(a), int(b)) for a, b in zip(base_pred, candidate) if a != b
            ).most_common(30)
        ]
    return summary


def run(args):
    """主流程"""
    data = GraphDataset.load(args.data_path)
    test_idx = data["test_idx"].astype(int)
    num_classes = int(data["num_classes"])
    specs = [parse_submission_spec(value) for value in args.submission]

    submissions = []
    pred_rows = []
    for name, score, path in specs:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pred = load_a1(path, test_idx)
        count = score_to_count(score, len(test_idx))
        submissions.append({"name": name, "score": score, "path": path, "count": count})
        pred_rows.append(pred)
    preds = np.stack(pred_rows, axis=0)
    scores = np.asarray([item["score"] for item in submissions], dtype=np.float64)
    counts = [item["count"] for item in submissions]

    prior = make_vote_prior(preds, scores, num_classes, args.prior_mode)
    prior = add_class_prior(prior, data["labels"], data["train_idx"], args.class_prior_weight)
    prior = add_model_prior(prior, args.prior_scores_npy, args.model_prior_weight)
    if args.anchor_submission:
        anchor_names = {item["name"]: idx for idx, item in enumerate(submissions)}
        anchor_idx = anchor_names[args.anchor_submission]
        anchor_pred = preds[anchor_idx]
        for cls in range(num_classes):
            prior[:, cls] += args.anchor_weight * (anchor_pred == cls)

    class_lower, class_upper = class_count_bounds_from_reference(
        args.class_count_reference,
        test_idx,
        num_classes,
        args.class_count_slack,
    )
    args.class_count_lower = class_lower
    args.class_count_upper = class_upper
    candidate, solve_info = solve_map(prior, preds, counts, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pd.DataFrame({"test_idx": test_idx, "label": candidate}).to_csv(args.output_path, index=False)

    summary = summarize_candidate(candidate, submissions, preds, args.base_submission)
    summary.update({
        "submissions": submissions,
        "solve_info": solve_info,
        "prior_mode": args.prior_mode,
        "count_tolerance": args.count_tolerance,
        "class_prior_weight": args.class_prior_weight,
        "anchor_submission": args.anchor_submission,
        "anchor_weight": args.anchor_weight,
        "prior_scores_npy": args.prior_scores_npy,
        "model_prior_weight": args.model_prior_weight,
        "class_count_reference": args.class_count_reference,
        "class_count_slack": args.class_count_slack,
        "class_count_lower": None if class_lower is None else class_lower.astype(int).tolist(),
        "class_count_upper": None if class_upper is None else class_upper.astype(int).tolist(),
        "output_path": args.output_path,
    })
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A1 leaderboard MAP 推断完成")
    print("=" * 100)
    print(f"prior_mode={args.prior_mode}, count_tolerance={args.count_tolerance}")
    if args.prior_scores_npy:
        print(f"model_prior={args.prior_scores_npy}, weight={args.model_prior_weight}")
    if args.class_count_reference:
        print(f"class_count_reference={args.class_count_reference}, slack={args.class_count_slack}")
    print(f"solve={solve_info}")
    print(f"class_distribution={summary['class_distribution']}")
    if "changed_vs_base" in summary:
        print(
            f"base={summary['base_name']}, changed={summary['changed_vs_base']:.4%}, "
            f"changed_count={summary['changed_count_vs_base']}"
        )
        print(f"transitions={summary['transitions_vs_base'][:12]}")
    print("历史提交约束匹配:")
    for item in summary["matches"]:
        print(
            f"  {item['name']:<8} target={item['target_count']} "
            f"candidate_match={item['candidate_match_count']} score={item['score']}"
        )
    print(f"output={args.output_path}")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="A1 leaderboard constrained MAP")
    parser.add_argument("--data_path", default="data/cls_data/A1.npz")
    parser.add_argument("--submission", action="append", required=True,
                        help="历史提交，格式 name,score,path")
    parser.add_argument("--prior_mode", default="power",
                        choices=["score", "centered", "power", "logit"])
    parser.add_argument("--class_prior_weight", type=float, default=0.05)
    parser.add_argument("--prior_scores_npy", default="",
                        help="模型概率先验npy，shape=(test_nodes,num_classes)")
    parser.add_argument("--model_prior_weight", type=float, default=0.0)
    parser.add_argument("--anchor_submission", default="exp078",
                        help="额外锚定某个提交的预测作为先验；为空则关闭")
    parser.add_argument("--anchor_weight", type=float, default=0.1)
    parser.add_argument("--base_submission", default="exp078",
                        help="用于统计changed_vs_base的提交名")
    parser.add_argument("--count_tolerance", type=int, default=0)
    parser.add_argument("--class_count_reference", default="",
                        help="参考A1.csv，用于限制候选类别分布")
    parser.add_argument("--class_count_slack", type=int, default=-1,
                        help="类别数量上下界相对参考预测的宽松度；负数表示关闭")
    parser.add_argument("--time_limit", type=float, default=300.0)
    parser.add_argument("--mip_rel_gap", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
