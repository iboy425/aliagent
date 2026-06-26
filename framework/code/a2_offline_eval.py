"""A2推荐任务离线评估工具

线上A2的真实 `target_iid` 对我们不可见，所以不能在本地直接计算线上分数。
本脚本从 `train.csv` 中切出一部分样本作为验证集，用剩余样本构造推荐策略，
再用验证集的公开 `target_iid` 计算 NDCG@10 / Hit@10 / MRR。

它的作用不是替代线上评测，而是在每天提交次数有限时，先筛掉明显无效的想法。
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rec_heuristics import (
    build_cooccur_stats,
    build_global_popularity,
    build_user_combo_profile_stats,
    build_user_profile_stats,
    choose_seq_col,
    get_user_combo_profile_counters,
    get_user_profile_counters,
    parse_item_counts,
    parse_seq,
    parse_user_profile_cols,
    rank_items,
)



def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从train.csv随机切分拟合集和验证集"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    val_idx = perm[:val_size]
    fit_idx = perm[val_size:]
    return df.iloc[fit_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def ndcg_at_k(preds: Sequence[str], target: str, k: int) -> float:
    """计算单样本NDCG@K"""
    for rank, item in enumerate(preds[:k], start=1):
        if item == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def reciprocal_rank(preds: Sequence[str], target: str) -> float:
    """计算单样本倒数排名"""
    for rank, item in enumerate(preds, start=1):
        if item == target:
            return 1.0 / rank
    return 0.0


BUCKET_ORDER = ["len=0", "len=1", "len=2-3", "len=4-10", "len>10"]


def bucket_seq_len(length: int) -> str:
    """按历史长度分桶，方便观察冷启动和短历史效果"""
    if length == 0:
        return "len=0"
    if length == 1:
        return "len=1"
    if length <= 3:
        return "len=2-3"
    if length <= 10:
        return "len=4-10"
    return "len>10"


def compute_bucket_weights(df: pd.DataFrame, seq_col: str) -> Dict[str, float]:
    """根据指定数据集的历史长度分布计算桶权重"""
    counts = Counter()
    total = len(df)
    if total == 0:
        return {bucket: 0.0 for bucket in BUCKET_ORDER}

    for _, row in df.iterrows():
        counts[bucket_seq_len(len(parse_seq(row.get(seq_col))))] += 1
    return {bucket: counts.get(bucket, 0) / total for bucket in BUCKET_ORDER}


def _format_item_counts(items: Sequence[str]) -> str:
    """把可见历史序列重新编码为 item_seq_counts 字符串"""
    if not items:
        return ""
    counter = Counter(items)
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return ",".join(f"{item}:{counter[item]}" for item in seen)


def truncate_history_fields(
    df: pd.DataFrame,
    seq_col: str,
    keep_lengths: Sequence[int],
) -> pd.DataFrame:
    """按给定长度截断历史序列，并同步更新 item_seq_counts"""
    if len(df) != len(keep_lengths):
        raise ValueError("keep_lengths长度必须与df行数一致")

    out = df.copy()
    new_seqs = []
    new_counts = []
    for (_, row), keep_len in zip(df.iterrows(), keep_lengths):
        seq = parse_seq(row.get(seq_col))
        keep_len = max(int(keep_len), 0)
        visible = seq[-keep_len:] if keep_len > 0 else []
        new_seqs.append(",".join(visible))
        new_counts.append(_format_item_counts(visible))

    out[seq_col] = new_seqs
    if "item_seq_counts" in out.columns:
        out["item_seq_counts"] = new_counts
    return out


def apply_test_like_history_distribution(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_col: str,
    seed: int,
) -> pd.DataFrame:
    """把验证集历史截断为接近 test.csv 的历史长度分布

    线上测试集短历史和空历史用户很多。随机验证集若保留完整长历史，会
    高估历史共现策略。因此这里从 test.csv 的精确长度分布中抽样，
    对验证样本进行截断，模拟线上可见历史长度。
    """
    test_lengths = np.array([len(parse_seq(row.get(seq_col))) for _, row in test_df.iterrows()])
    if len(test_lengths) == 0:
        return val_df.copy()
    rng = np.random.default_rng(seed)
    sampled_lengths = rng.choice(test_lengths, size=len(val_df), replace=True)
    return truncate_history_fields(val_df, seq_col, sampled_lengths)


def add_weighted_metrics(metrics: Dict, bucket_weights: Dict[str, float]) -> Dict:
    """按目标桶权重聚合各桶指标"""
    for metric_name in ["ndcg", "hit", "mrr"]:
        weighted = 0.0
        for bucket, weight in bucket_weights.items():
            bucket_metrics = metrics.get("buckets", {}).get(bucket)
            if bucket_metrics is None:
                continue
            weighted += weight * bucket_metrics.get(metric_name, 0.0)
        metrics[f"weighted_{metric_name}"] = float(weighted)
    metrics["bucket_weights"] = dict(bucket_weights)
    return metrics


def evaluate_strategy(
    val_df: pd.DataFrame,
    seq_col: str,
    candidate_items: set,
    global_pop: Counter,
    cooccur_stats: Dict[str, Counter],
    topk: int,
    strategy: str,
    history_filter: str,
    recent_n: int,
    user_lookup: pd.DataFrame = None,
    user_profile_stats: Dict[str, Dict[str, Counter]] = None,
    user_combo_profile_stats: Dict = None,
    user_cols: Sequence[str] = (),
    user_weight: float = 0.0,
    user_combo_weight: float = 0.0,
    user_combo_min_count: int = 5,
    pop_penalty_weight: float = 0.0,
    history_count_weight: float = 0.0,
) -> Dict:
    """评估一种推荐策略"""
    rows = []
    bucket_rows = defaultdict(list)

    for _, row in val_df.iterrows():
        target = str(row["target_iid"])
        user_id = str(row["uid"]) if "uid" in row else str(row.get("user_id", ""))
        user_counters = get_user_profile_counters(
            user_id=user_id,
            user_lookup=user_lookup,
            user_profile_stats=user_profile_stats,
            user_cols=user_cols,
        )
        user_combo_counters = get_user_combo_profile_counters(
            user_id=user_id,
            user_lookup=user_lookup,
            combo_profile_stats=user_combo_profile_stats,
            min_count=user_combo_min_count,
        )
        preds = rank_items(
            seq=parse_seq(row.get(seq_col)),
            candidate_items=candidate_items,
            global_pop=global_pop,
            cooccur_stats=cooccur_stats,
            topk=topk,
            strategy=strategy,
            history_filter=history_filter,
            recent_n=recent_n,
            user_profile_counters=user_counters,
            user_combo_counters=user_combo_counters,
            history_counts=parse_item_counts(row.get("item_seq_counts")),
            user_weight=user_weight,
            user_combo_weight=user_combo_weight,
            history_count_weight=history_count_weight,
            pop_penalty_weight=pop_penalty_weight,
        )
        seq_len = len(parse_seq(row.get(seq_col)))
        score = {
            "ndcg": ndcg_at_k(preds, target, topk),
            "hit": 1.0 if target in preds[:topk] else 0.0,
            "mrr": reciprocal_rank(preds, target),
            "seq_len": seq_len,
        }
        rows.append(score)
        bucket_rows[bucket_seq_len(seq_len)].append(score)

    def avg(metric_rows: List[Dict], key: str) -> float:
        if not metric_rows:
            return 0.0
        return float(np.mean([item[key] for item in metric_rows]))

    metrics = {
        "strategy": strategy,
        "history_filter": history_filter,
        "topk": topk,
        "user_weight": user_weight,
        "user_combo_weight": user_combo_weight,
        "user_combo_min_count": user_combo_min_count,
        "pop_penalty_weight": pop_penalty_weight,
        "history_count_weight": history_count_weight,
        "samples": len(rows),
        "ndcg": avg(rows, "ndcg"),
        "hit": avg(rows, "hit"),
        "mrr": avg(rows, "mrr"),
        "buckets": {},
    }
    for bucket, metric_rows in sorted(bucket_rows.items()):
        metrics["buckets"][bucket] = {
            "samples": len(metric_rows),
            "ndcg": avg(metric_rows, "ndcg"),
            "hit": avg(metric_rows, "hit"),
            "mrr": avg(metric_rows, "mrr"),
        }
    return metrics


def print_metrics(metrics: Dict):
    """打印指标"""
    print(
        f"{metrics['strategy']:>9} | filter={metrics['history_filter']:<4} | "
        f"NDCG@{metrics['topk']}={metrics['ndcg']:.6f} | "
        f"Hit@{metrics['topk']}={metrics['hit']:.6f} | "
        f"MRR={metrics['mrr']:.6f} | samples={metrics['samples']}"
    )
    for bucket, vals in metrics["buckets"].items():
        print(
            f"  {bucket:<8} samples={vals['samples']:<5} "
            f"ndcg={vals['ndcg']:.6f} hit={vals['hit']:.6f} mrr={vals['mrr']:.6f}"
        )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A2推荐任务本地离线评估")
    parser.add_argument("--data_path", type=str, default="data/rec_data", help="推荐数据目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--topk", type=int, default=10, help="推荐列表长度")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "popular", "last_item", "history", "hybrid"],
        help="推荐策略",
    )
    parser.add_argument(
        "--history_filter",
        type=str,
        default="none",
        choices=["none", "soft", "hard"],
        help="是否过滤历史item",
    )
    parser.add_argument(
        "--seq_col",
        type=str,
        default="auto",
        help="历史序列列名，默认自动选择 item_seq_dedup",
    )
    parser.add_argument("--recent_n", type=int, default=20, help="共现统计使用最近多少个历史item")
    parser.add_argument("--user_weight", type=float, default=0.0, help="用户画像分组热门度融合权重")
    parser.add_argument("--user_combo_weight", type=float, default=0.0,
                        help="用户画像前缀组合热门度融合权重")
    parser.add_argument("--user_combo_sizes", type=str, default="3,2,1",
                        help="逗号分隔的用户画像前缀组合长度")
    parser.add_argument("--user_combo_min_count", type=int, default=5,
                        help="画像组合最少训练样本数")
    parser.add_argument(
        "--user_profile_cols",
        type=str,
        default="auto",
        help="用户画像列，auto表示使用user.csv中除uid外全部列，空字符串表示关闭",
    )
    parser.add_argument("--pop_penalty_weight", type=float, default=0.0, help="热门item惩罚权重")
    parser.add_argument("--history_count_weight", type=float, default=0.0, help="用户历史频次权重")
    parser.add_argument("--test_like_eval", action="store_true",
                        help="把验证集历史截断为接近test.csv的长度分布")
    parser.add_argument("--sort_metric", type=str, default="ndcg",
                        choices=["ndcg", "weighted_ndcg"],
                        help="选择最佳策略时使用的指标")
    parser.add_argument("--output_json", type=str, default="", help="指标JSON输出路径")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    train_path = os.path.join(args.data_path, "train.csv")
    item_path = os.path.join(args.data_path, "item.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    item_df = pd.read_csv(item_path)
    candidate_items = set(item_df["iid"].astype(str).tolist())
    seq_col = choose_seq_col(train_df, args.seq_col)
    user_col = "uid" if "uid" in train_df.columns else "user_id"

    fit_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    bucket_weights = compute_bucket_weights(test_df, seq_col)
    if args.test_like_eval:
        val_df = apply_test_like_history_distribution(val_df, test_df, seq_col, args.seed)
    global_pop = build_global_popularity(fit_df)
    cooccur_stats = build_cooccur_stats(fit_df, seq_col, recent_n=args.recent_n)

    user_cols = []
    user_lookup = None
    user_profile_stats = None
    user_combo_profile_stats = None
    if args.user_weight > 0 or args.user_combo_weight > 0:
        user_path = os.path.join(args.data_path, "user.csv")
        if os.path.exists(user_path):
            user_df = pd.read_csv(user_path)
            user_cols = parse_user_profile_cols(user_df, user_col, args.user_profile_cols)
            if user_cols:
                if args.user_weight > 0:
                    user_profile_stats, user_lookup = build_user_profile_stats(
                        train_df=fit_df,
                        user_df=user_df,
                        user_cols=user_cols,
                        user_col=user_col,
                    )
                else:
                    user_lookup = user_df.set_index(user_col)
                if args.user_combo_weight > 0:
                    combo_sizes = [int(x.strip()) for x in args.user_combo_sizes.split(",") if x.strip()]
                    user_combo_profile_stats, user_lookup = build_user_combo_profile_stats(
                        train_df=fit_df,
                        user_df=user_df,
                        user_cols=user_cols,
                        combo_sizes=combo_sizes,
                        min_count=args.user_combo_min_count,
                        user_col=user_col,
                    )
        else:
            print("警告: 启用用户画像融合，但未找到user.csv，画像融合关闭")

    print("=" * 80)
    print("A2离线评估")
    print("=" * 80)
    print(f"数据目录: {args.data_path}")
    print(f"序列列: {seq_col}")
    print(f"拟合集: {len(fit_df)} 行, 验证集: {len(val_df)} 行")
    print(f"候选item: {len(candidate_items)} 个, target去重: {train_df['target_iid'].nunique()} 个")
    print(f"history_filter: {args.history_filter}, recent_n: {args.recent_n}")
    print(
        f"user_weight: {args.user_weight}, user_combo_weight: {args.user_combo_weight}, "
        f"user_combo_sizes: {args.user_combo_sizes}, user_combo_min_count: {args.user_combo_min_count}, "
        f"user_cols: {user_cols}, pop_penalty_weight: {args.pop_penalty_weight}"
    )
    print(f"history_count_weight: {args.history_count_weight}")
    print(f"test_like_eval: {args.test_like_eval}, sort_metric: {args.sort_metric}")
    print(f"test bucket weights: {bucket_weights}")
    print("-" * 80)

    strategies = ["popular", "last_item", "history", "hybrid"] if args.strategy == "all" else [args.strategy]
    all_metrics = []
    for strategy in strategies:
        metrics = evaluate_strategy(
            val_df=val_df,
            seq_col=seq_col,
            candidate_items=candidate_items,
            global_pop=global_pop,
            cooccur_stats=cooccur_stats,
            topk=args.topk,
            strategy=strategy,
            history_filter=args.history_filter,
            recent_n=args.recent_n,
            user_lookup=user_lookup,
            user_profile_stats=user_profile_stats,
            user_combo_profile_stats=user_combo_profile_stats,
            user_cols=user_cols,
            user_weight=args.user_weight,
            user_combo_weight=args.user_combo_weight,
            user_combo_min_count=args.user_combo_min_count,
            pop_penalty_weight=args.pop_penalty_weight,
            history_count_weight=args.history_count_weight,
        )
        add_weighted_metrics(metrics, bucket_weights)
        all_metrics.append(metrics)
        print_metrics(metrics)
        print(
            f"  weighted_ndcg={metrics['weighted_ndcg']:.6f} "
            f"weighted_hit={metrics['weighted_hit']:.6f} "
            f"weighted_mrr={metrics['weighted_mrr']:.6f}"
        )
        print("-" * 80)

    best = max(all_metrics, key=lambda item: item[args.sort_metric])
    print(
        f"最佳策略: {best['strategy']}，NDCG@{args.topk}={best['ndcg']:.6f}，"
        f"weighted_NDCG@{args.topk}={best['weighted_ndcg']:.6f}，"
        f"Hit@{args.topk}={best['hit']:.6f}"
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"指标已保存: {args.output_json}")


if __name__ == "__main__":
    main()
