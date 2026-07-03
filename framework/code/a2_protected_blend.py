"""A2保护TopN的候选融合

线上反馈说明 A2 对 Top1 漂移很敏感：
- Exp066 稳定；
- Exp069/078 只要 Top1 漂移偏大，线上 A2 就下降。

因此本工具尝试只保护 base 的前 N 个位置，其余位置用 alt 列表补强。
典型设置是 `keep_topn=1`：完全保留线上稳定 Top1，只让 Exp045 多
seed 模型参与第 2-10 位排序。
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from a2_bucket_blend import BUCKET_ORDER, bucket_seq_len, parse_prediction
from a2_feature_ranker import (
    build_rule_context,
    choose_seq_col,
    compute_bucket_weights,
    evaluate_prediction_lists,
    load_rankers,
    rank_with_fusion,
    score_dataframe_with_models,
    split_train_val,
    test_like_lengths,
    apply_fixed_test_like_truncation,
)
from rec_heuristics import parse_seq
from utils import set_seed


def parse_csv_list(value: str) -> List[str]:
    """解析逗号分隔字符串"""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    """解析整数列表"""
    return [int(item) for item in parse_csv_list(value)]


def protected_merge(
    base_items: Sequence[str],
    alt_items: Sequence[str],
    keep_topn: int,
    topk: int,
) -> List[str]:
    """保留 base 前 keep_topn 个位置，再用 alt/base 补齐"""
    result = []
    seen = set()
    for item in base_items[:keep_topn]:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    for item in alt_items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            return result[:topk]
    for item in base_items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            break
    return result[:topk]


def blend_lists(
    df: pd.DataFrame,
    base_preds: Sequence[Sequence[str]],
    alt_preds: Sequence[Sequence[str]],
    seq_col: str,
    buckets: Sequence[str],
    keep_topn: int,
    topk: int,
) -> List[List[str]]:
    """按历史桶做保护式融合"""
    selected = set(buckets)
    out = []
    for (_, row), base, alt in zip(df.iterrows(), base_preds, alt_preds):
        bucket = bucket_seq_len(len(parse_seq(row.get(seq_col))))
        if bucket in selected:
            out.append(protected_merge(base, alt, keep_topn, topk))
        else:
            out.append(list(base[:topk]))
    return out


def run_eval(args):
    """离线评估保护式融合"""
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_models, base_bundle = load_rankers(args.base_checkpoint, device)
    alt_models, alt_bundle = load_rankers(args.alt_checkpoint, device)
    if base_bundle.target_items != alt_bundle.target_items:
        raise ValueError("base/alt checkpoint target_items 不一致")

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    train_seq_col = choose_seq_col(train_df, args.seq_col)
    test_seq_col = choose_seq_col(test_df, args.seq_col)
    bucket_weights = compute_bucket_weights(test_df, test_seq_col)

    all_results = []
    split_summaries = []
    for split_seed in parse_int_list(args.split_seeds):
        fit_df, val_df = split_train_val(train_df, args.val_ratio, split_seed)
        if args.test_like_val:
            lengths = test_like_lengths(test_df, test_seq_col)
            val_df = apply_fixed_test_like_truncation(val_df, lengths, train_seq_col, split_seed)
        if args.max_val_samples > 0 and len(val_df) > args.max_val_samples:
            val_df = val_df.sample(n=args.max_val_samples, random_state=split_seed).reset_index(drop=True)

        rule_context = build_rule_context(args, fit_df, user_df, item_df, train_seq_col)
        user_lookup = user_df.set_index("uid")
        base_scores = score_dataframe_with_models(
            val_df, base_models, base_bundle, user_lookup, train_seq_col,
            args.max_len, args.batch_size, device,
        )
        alt_scores = score_dataframe_with_models(
            val_df, alt_models, alt_bundle, user_lookup, train_seq_col,
            args.max_len, args.batch_size, device,
        )
        base_preds = rank_with_fusion(val_df, base_scores, train_seq_col, rule_context, args, args.base_model_weight)
        alt_preds = rank_with_fusion(val_df, alt_scores, train_seq_col, rule_context, args, args.alt_model_weight)
        targets = val_df["target_iid"].astype(str).tolist()

        base_metrics = evaluate_prediction_lists(base_preds, targets, val_df, train_seq_col, args.topk, bucket_weights)
        alt_metrics = evaluate_prediction_lists(alt_preds, targets, val_df, train_seq_col, args.topk, bucket_weights)
        split_rows = []
        for keep_topn in parse_int_list(args.keep_topns):
            for bucket_expr in parse_csv_list(args.bucket_sets):
                buckets = [item for item in bucket_expr.split("+") if item]
                preds = blend_lists(val_df, base_preds, alt_preds, train_seq_col, buckets, keep_topn, args.topk)
                metrics = evaluate_prediction_lists(preds, targets, val_df, train_seq_col, args.topk, bucket_weights)
                row = {
                    "split_seed": split_seed,
                    "keep_topn": keep_topn,
                    "buckets": buckets,
                    "bucket_expr": bucket_expr,
                    "weighted_ndcg": metrics["weighted_ndcg"],
                    "ndcg": metrics["ndcg"],
                    "gain_vs_base": metrics["weighted_ndcg"] - base_metrics["weighted_ndcg"],
                    "gain_vs_alt": metrics["weighted_ndcg"] - alt_metrics["weighted_ndcg"],
                    "metrics": metrics,
                }
                split_rows.append(row)
                all_results.append(row)
        split_summaries.append({
            "split_seed": split_seed,
            "base_weighted_ndcg": base_metrics["weighted_ndcg"],
            "alt_weighted_ndcg": alt_metrics["weighted_ndcg"],
            "rows": split_rows,
        })
        print(
            f"[split={split_seed}] base={base_metrics['weighted_ndcg']:.6f} "
            f"alt={alt_metrics['weighted_ndcg']:.6f}"
        )

    grouped = defaultdict(list)
    for row in all_results:
        grouped[(row["keep_topn"], row["bucket_expr"])].append(row)
    summary = []
    for (keep_topn, bucket_expr), rows in grouped.items():
        gains = [row["gain_vs_base"] for row in rows]
        summary.append({
            "keep_topn": keep_topn,
            "bucket_expr": bucket_expr,
            "buckets": rows[0]["buckets"],
            "mean_gain": float(np.mean(gains)),
            "min_gain": float(np.min(gains)),
            "positive_splits": int(sum(1 for gain in gains if gain > 0)),
            "mean_weighted_ndcg": float(np.mean([row["weighted_ndcg"] for row in rows])),
            "rows": rows,
        })
    summary.sort(key=lambda item: (item["mean_gain"], item["min_gain"]), reverse=True)
    output = {"summary": summary, "split_summaries": split_summaries, "best": summary[0] if summary else {}}
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A2 保护TopN融合审计 Top-20")
    print("=" * 100)
    for item in summary[:20]:
        print(
            f"keep={item['keep_topn']}\tbuckets={item['bucket_expr']}\t"
            f"mean_gain={item['mean_gain']:+.6f}\tmin_gain={item['min_gain']:+.6f}\t"
            f"positive={item['positive_splits']}"
        )
    if args.output_json:
        print(f"\n结果已保存: {args.output_json}")


def overlap_ratio(items_a: Sequence[str], items_b: Sequence[str], topk: int) -> float:
    """计算 TopK 重合率"""
    return len(set(items_a[:topk]) & set(items_b[:topk])) / max(topk, 1)


def run_predict(args):
    """生成保护式融合 A2"""
    base_df = pd.read_csv(args.base_a2)
    alt_df = pd.read_csv(args.alt_a2)
    test_df = pd.read_csv(args.test_csv)
    if base_df["uid"].astype(str).tolist() != alt_df["uid"].astype(str).tolist():
        raise ValueError("base_a2 和 alt_a2 uid 顺序不一致")
    if base_df["uid"].astype(str).tolist() != test_df["uid"].astype(str).tolist():
        raise ValueError("A2 uid 顺序和 test.csv 不一致")

    if args.audit_json:
        audit = json.load(open(args.audit_json, encoding="utf-8"))
        best = audit["best"]
        keep_topn = int(best["keep_topn"])
        buckets = list(best["buckets"])
    else:
        keep_topn = int(args.keep_topn)
        buckets = parse_csv_list(args.buckets)

    rows = []
    stats = defaultdict(list)
    for idx, row in test_df.iterrows():
        bucket = bucket_seq_len(len(parse_seq(row.get(args.seq_col))))
        base_pred = parse_prediction(base_df.iloc[idx]["prediction"])
        alt_pred = parse_prediction(alt_df.iloc[idx]["prediction"])
        if bucket in set(buckets):
            pred = protected_merge(base_pred, alt_pred, keep_topn, args.topk)
        else:
            pred = base_pred[:args.topk]
        rows.append({"uid": str(row["uid"]), "prediction": ",".join(pred[:args.topk])})
        stats[bucket].append({
            "changed": float(base_pred[:args.topk] != pred[:args.topk]),
            "top1_changed": float(base_pred[:1] != pred[:1]),
            "overlap": overlap_ratio(base_pred, pred, args.topk),
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_path, index=False)

    def avg(items, key):
        return float(np.mean([item[key] for item in items])) if items else 0.0

    all_rows = [item for vals in stats.values() for item in vals]
    print("=" * 100)
    print("A2 保护TopN融合推理完成")
    print("=" * 100)
    print(f"keep_topn={keep_topn}, buckets={buckets}")
    print(f"output={args.output_path}")
    print(
        f"总体 changed={avg(all_rows, 'changed'):.4%}, "
        f"top1_changed={avg(all_rows, 'top1_changed'):.4%}, "
        f"overlap={avg(all_rows, 'overlap'):.4%}"
    )
    for bucket in BUCKET_ORDER:
        items = stats.get(bucket, [])
        print(
            f"{bucket:<8} n={len(items):<5} changed={avg(items, 'changed'):.4%} "
            f"top1_changed={avg(items, 'top1_changed'):.4%} "
            f"overlap={avg(items, 'overlap'):.4%}"
        )


def add_fusion_args(parser):
    """添加与 a2_feature_ranker 一致的融合参数"""
    parser.add_argument("--seq_col", default="item_seq_raw")
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--history_filter", default="none", choices=["none", "soft", "hard"])
    parser.add_argument("--recent_n", type=int, default=18)
    parser.add_argument("--cooccur_formula", default="jaccard",
                        choices=["log_count", "count", "confidence", "jaccard", "lift", "sqrt_lift", "pmi", "log_pmi"])
    parser.add_argument("--cooccur_weight", type=float, default=1.0)
    parser.add_argument("--cooccur_decay", type=float, default=1.0)
    parser.add_argument("--pop_weight", type=float, default=0.0)
    parser.add_argument("--pop_penalty_weight", type=float, default=0.0)
    parser.add_argument("--history_count_weight", type=float, default=0.0)
    parser.add_argument("--user_weight", type=float, default=0.01)
    parser.add_argument("--user_combo_weight", type=float, default=0.1)
    parser.add_argument("--user_combo_sizes", default="3,2,1")
    parser.add_argument("--user_combo_mode", default="prefix", choices=["prefix", "all"])
    parser.add_argument("--user_combo_min_count", type=int, default=5)
    parser.add_argument("--item_feature_cols", default="auto")
    parser.add_argument("--item_feature_weight", type=float, default=0.0)
    parser.add_argument("--item_feature_recent_n", type=int, default=10)
    parser.add_argument("--item_feature_min_count", type=int, default=20)


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="A2保护TopN融合")
    sub = parser.add_subparsers(dest="mode", required=True)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--data_path", default="data/rec_data")
    eval_p.add_argument("--base_checkpoint", required=True)
    eval_p.add_argument("--alt_checkpoint", required=True)
    eval_p.add_argument("--base_model_weight", type=float, default=1.0)
    eval_p.add_argument("--alt_model_weight", type=float, default=22.0)
    eval_p.add_argument("--split_seeds", default="42,777,2024")
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--val_ratio", type=float, default=0.1)
    eval_p.add_argument("--test_like_val", action="store_true")
    eval_p.add_argument("--max_val_samples", type=int, default=4000)
    eval_p.add_argument("--keep_topns", default="1,2")
    eval_p.add_argument(
        "--bucket_sets",
        default="len=2-3,len>10,len=2-3+len>10,len=2-3+len=4-10+len>10",
        help="逗号分隔候选桶集合，集合内部用+连接",
    )
    eval_p.add_argument("--output_json", default="")
    add_fusion_args(eval_p)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--base_a2", required=True)
    pred_p.add_argument("--alt_a2", required=True)
    pred_p.add_argument("--test_csv", default="data/rec_data/test.csv")
    pred_p.add_argument("--seq_col", default="item_seq_raw")
    pred_p.add_argument("--audit_json", default="")
    pred_p.add_argument("--keep_topn", type=int, default=1)
    pred_p.add_argument("--buckets", default="len=2-3,len>10")
    pred_p.add_argument("--topk", type=int, default=10)
    pred_p.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    if args.mode == "eval":
        run_eval(args)
    elif args.mode == "predict":
        run_predict(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
