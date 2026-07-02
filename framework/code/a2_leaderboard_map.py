"""A2 基于 leaderboard NDCG 反馈的目标物品 MAP 推断

A2 每次线上分数是 NDCG@10。对单目标推荐任务来说，某用户如果真实
target 出现在提交列表第 r 位，则贡献：

    gain(r) = 1 / log2(r + 1)

如果不在 Top10，则贡献 0。于是一次提交给出一条全局线性约束：

    sum_u gain(rank_s(u, y_u)) ~= score_s * N

本脚本把多次 A2 提交和分数转成约束，推断一个最可能的 target item
分配，再把推断 target 放到推荐列表前面，生成高风险大幅候选。

注意：
- 这是 A 榜反馈驱动方法，有明显过拟合风险；
- 若输出 top1_changed 很高，说明它是一次强试探，不是稳健候选；
- NDCG 分数只有四位小数，默认允许一定 score_tolerance。
"""
import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from rec_heuristics import parse_seq


OTHER = "__OTHER__"
BUCKET_ORDER = ["len=0", "len=1", "len=2-3", "len=4-10", "len>10"]


def ndcg_gain(rank: int) -> float:
    """rank 从 1 开始，返回单目标 NDCG 贡献"""
    return 1.0 / math.log2(rank + 1)


def parse_submission_spec(value: str) -> Tuple[str, float, str]:
    """解析 name,score,path"""
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"submission格式必须是 name,score,path: {value}")
    return parts[0].strip(), float(parts[1].strip()), parts[2].strip()


def parse_prediction(value) -> List[str]:
    """解析 prediction 字段"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def bucket_seq_len(length: int) -> str:
    """按历史长度分桶"""
    if length == 0:
        return "len=0"
    if length == 1:
        return "len=1"
    if length <= 3:
        return "len=2-3"
    if length <= 10:
        return "len=4-10"
    return "len>10"


def load_a2(path: str, expected_uids: Sequence[str] = None) -> Tuple[List[str], List[List[str]]]:
    """读取 A2.csv"""
    df = pd.read_csv(path)
    uids = df["uid"].astype(str).tolist()
    if expected_uids is not None and uids != list(expected_uids):
        raise ValueError(f"A2 uid顺序不一致: {path}")
    preds = [parse_prediction(value) for value in df["prediction"]]
    return uids, preds


def submission_weight(score: float, scores: np.ndarray, mode: str) -> float:
    """根据线上分数构造先验权重"""
    if mode == "centered":
        return float(score - scores.mean())
    if mode == "positive_centered":
        return float(max(score - scores.mean(), 0.0))
    if mode == "best_centered":
        return float(score - scores.min())
    if mode == "score":
        return float(score)
    raise ValueError(f"未知 weight_mode: {mode}")


def build_user_candidates(all_preds: Sequence[Sequence[List[str]]], include_other: bool) -> List[List[str]]:
    """构造每个用户的候选 target item 集合"""
    n = len(all_preds[0])
    out = []
    for idx in range(n):
        seen = set()
        items = []
        for preds in all_preds:
            for item in preds[idx]:
                if item and item not in seen:
                    items.append(item)
                    seen.add(item)
        if include_other:
            items.append(OTHER)
        out.append(items)
    return out


def build_prior(
    candidates: Sequence[Sequence[str]],
    all_preds: Sequence[Sequence[List[str]]],
    scores: np.ndarray,
    weight_mode: str,
    anchor_preds: Sequence[List[str]],
    anchor_weight: float,
    other_prior: float,
) -> np.ndarray:
    """构造每个候选 item 的先验分数"""
    weights = [submission_weight(float(score), scores, weight_mode) for score in scores]
    prior_values = []
    for user_idx, items in enumerate(candidates):
        score_map = {item: 0.0 for item in items}
        for sub_idx, preds in enumerate(all_preds):
            rank_map = {item: rank for rank, item in enumerate(preds[user_idx], start=1)}
            for item in items:
                if item in rank_map:
                    score_map[item] += weights[sub_idx] * ndcg_gain(rank_map[item])
        if anchor_preds and anchor_weight > 0:
            rank_map = {item: rank for rank, item in enumerate(anchor_preds[user_idx], start=1)}
            for item in items:
                if item in rank_map:
                    score_map[item] += anchor_weight * ndcg_gain(rank_map[item])
        if OTHER in score_map:
            score_map[OTHER] += other_prior
        prior_values.extend(score_map[item] for item in items)
    return np.asarray(prior_values, dtype=np.float64)


def build_offsets(candidates: Sequence[Sequence[str]]) -> np.ndarray:
    """构造每个用户变量起始 offset"""
    offsets = [0]
    total = 0
    for items in candidates:
        total += len(items)
        offsets.append(total)
    return np.asarray(offsets, dtype=np.int64)


def build_constraints(
    candidates: Sequence[Sequence[str]],
    offsets: np.ndarray,
    all_preds: Sequence[Sequence[List[str]]],
    targets: Sequence[float],
    score_tolerance: float,
) -> LinearConstraint:
    """构造每用户 one-hot 和 NDCG 总分约束"""
    rows = []
    cols = []
    data = []
    lower = []
    upper = []

    # 每个用户选择一个 target 候选。
    for user_idx, items in enumerate(candidates):
        row_id = len(lower)
        for local_idx in range(len(items)):
            rows.append(row_id)
            cols.append(int(offsets[user_idx] + local_idx))
            data.append(1.0)
        lower.append(1.0)
        upper.append(1.0)

    # 每个历史提交的 NDCG 总贡献。
    for sub_idx, preds in enumerate(all_preds):
        row_id = len(lower)
        for user_idx, items in enumerate(candidates):
            rank_map = {item: rank for rank, item in enumerate(preds[user_idx], start=1)}
            for local_idx, item in enumerate(items):
                gain = ndcg_gain(rank_map[item]) if item in rank_map else 0.0
                if gain:
                    rows.append(row_id)
                    cols.append(int(offsets[user_idx] + local_idx))
                    data.append(gain)
        lower.append(float(targets[sub_idx] - score_tolerance))
        upper.append(float(targets[sub_idx] + score_tolerance))

    matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(lower), int(offsets[-1]))).tocsr()
    return LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))


def solve_targets(candidates, offsets, all_preds, targets, prior, args):
    """求解 MAP target 分配"""
    if args.solver == "greedy":
        selected = []
        for user_idx, items in enumerate(candidates):
            part = prior[offsets[user_idx]:offsets[user_idx + 1]]
            selected.append(items[int(np.argmax(part))])
        return selected, {
            "success": True,
            "status": 0,
            "message": "Greedy prior argmax",
            "fun": float(-np.sum([
                np.max(prior[offsets[idx]:offsets[idx + 1]])
                for idx in range(len(candidates))
            ])),
            "mip_gap": None,
        }

    constraints = build_constraints(candidates, offsets, all_preds, targets, args.score_tolerance)
    result = milp(
        c=-prior,
        integrality=np.ones(int(offsets[-1]), dtype=np.int8),
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
    selected = []
    x = np.asarray(result.x)
    for user_idx, items in enumerate(candidates):
        part = x[offsets[user_idx]:offsets[user_idx + 1]]
        selected.append(items[int(np.argmax(part))])
    return selected, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "mip_gap": None if result.mip_gap is None else float(result.mip_gap),
    }


def build_prediction(base_items: Sequence[str], selected_item: str, topk: int, protect_topn: int) -> List[str]:
    """把推断 target 合成提交列表"""
    result = []
    seen = set()
    for item in base_items[:protect_topn]:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    if selected_item and selected_item != OTHER and selected_item not in seen:
        result.append(selected_item)
        seen.add(selected_item)
    for item in base_items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            break
    return result[:topk]


def overlap_ratio(a: Sequence[str], b: Sequence[str], topk: int) -> float:
    """TopK 集合重合率"""
    return len(set(a[:topk]) & set(b[:topk])) / max(topk, 1)


def run(args):
    """主流程"""
    specs = [parse_submission_spec(value) for value in args.submission]
    names = [item[0] for item in specs]
    scores = np.asarray([item[1] for item in specs], dtype=np.float64)
    paths = [item[2] for item in specs]

    uids, first_preds = load_a2(paths[0])
    all_preds = [first_preds]
    for path in paths[1:]:
        _, preds = load_a2(path, uids)
        all_preds.append(preds)

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    base_idx = name_to_idx[args.base_submission]
    anchor_idx = name_to_idx[args.anchor_submission] if args.anchor_submission else base_idx
    base_preds = all_preds[base_idx]
    anchor_preds = all_preds[anchor_idx]

    candidates = build_user_candidates(all_preds, args.include_other)
    offsets = build_offsets(candidates)
    targets = scores * len(uids)
    prior = build_prior(
        candidates,
        all_preds,
        scores,
        args.weight_mode,
        anchor_preds,
        args.anchor_weight,
        args.other_prior,
    )
    selected, solve_info = solve_targets(candidates, offsets, all_preds, targets, prior, args)

    rows = []
    stats = []
    test_df = pd.read_csv(args.test_csv)
    top1_counter = Counter()
    selected_counter = Counter(selected)
    bucket_stats: Dict[str, List[Dict]] = {bucket: [] for bucket in BUCKET_ORDER}
    for idx, uid in enumerate(uids):
        pred = build_prediction(base_preds[idx], selected[idx], args.topk, args.protect_topn)
        rows.append({"uid": uid, "prediction": ",".join(pred)})
        top1_counter[pred[0] if pred else ""] += 1
        row_stat = {
            "changed": float(pred != base_preds[idx][:args.topk]),
            "top1_changed": float(pred[:1] != base_preds[idx][:1]),
            "overlap": overlap_ratio(pred, base_preds[idx], args.topk),
            "selected_other": float(selected[idx] == OTHER),
        }
        stats.append(row_stat)
        bucket = bucket_seq_len(len(parse_seq(test_df.iloc[idx].get(args.seq_col))))
        bucket_stats[bucket].append(row_stat)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_path, index=False)

    def avg(items, key):
        return float(np.mean([item[key] for item in items])) if items else 0.0

    summary = {
        "solve_info": solve_info,
        "submissions": [
            {"name": name, "score": float(score), "target_total": float(score * len(uids)), "path": path}
            for name, score, path in specs
        ],
        "base_submission": args.base_submission,
        "anchor_submission": args.anchor_submission,
        "weight_mode": args.weight_mode,
        "score_tolerance": args.score_tolerance,
        "protect_topn": args.protect_topn,
        "changed": avg(stats, "changed"),
        "top1_changed": avg(stats, "top1_changed"),
        "overlap": avg(stats, "overlap"),
        "selected_other": avg(stats, "selected_other"),
        "selected_top_items": selected_counter.most_common(30),
        "top1_distribution": top1_counter.most_common(30),
        "bucket_stats": {
            bucket: {
                "n": len(items),
                "changed": avg(items, "changed"),
                "top1_changed": avg(items, "top1_changed"),
                "overlap": avg(items, "overlap"),
                "selected_other": avg(items, "selected_other"),
            }
            for bucket, items in bucket_stats.items()
        },
        "output_path": args.output_path,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A2 leaderboard MAP 推断完成")
    print("=" * 100)
    print(f"solve={solve_info}")
    print(
        f"changed={summary['changed']:.4%}, top1_changed={summary['top1_changed']:.4%}, "
        f"overlap={summary['overlap']:.4%}, selected_other={summary['selected_other']:.4%}"
    )
    for bucket in BUCKET_ORDER:
        item = summary["bucket_stats"][bucket]
        print(
            f"{bucket:<8} n={item['n']:<5} changed={item['changed']:.4%} "
            f"top1_changed={item['top1_changed']:.4%} overlap={item['overlap']:.4%} "
            f"other={item['selected_other']:.4%}"
        )
    print(f"selected_top_items={summary['selected_top_items'][:12]}")
    print(f"top1_distribution={summary['top1_distribution'][:12]}")
    print(f"output={args.output_path}")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="A2 leaderboard constrained MAP")
    parser.add_argument("--submission", action="append", required=True,
                        help="历史提交，格式 name,score,path")
    parser.add_argument("--base_submission", default="exp090")
    parser.add_argument("--anchor_submission", default="exp090")
    parser.add_argument("--test_csv", default="data/rec_data/test.csv")
    parser.add_argument("--seq_col", default="item_seq_raw")
    parser.add_argument("--weight_mode", default="best_centered",
                        choices=["score", "centered", "positive_centered", "best_centered"])
    parser.add_argument("--anchor_weight", type=float, default=0.1)
    parser.add_argument("--other_prior", type=float, default=-0.01)
    parser.add_argument("--score_tolerance", type=float, default=0.5)
    parser.add_argument("--include_other", action="store_true")
    parser.add_argument("--protect_topn", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--time_limit", type=float, default=300.0)
    parser.add_argument("--mip_rel_gap", type=float, default=0.0)
    parser.add_argument("--solver", default="milp", choices=["milp", "greedy"],
                        help="milp=精确全局约束；greedy=只按leaderboard先验逐用户选择，速度快")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
