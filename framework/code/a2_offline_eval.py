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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_seq(value) -> List[str]:
    """解析逗号分隔的item序列"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def choose_seq_col(df: pd.DataFrame, seq_col: str) -> str:
    """确定使用哪个历史序列字段"""
    if seq_col != "auto":
        if seq_col not in df.columns:
            raise ValueError(f"指定的序列列不存在: {seq_col}")
        return seq_col
    for col in ["item_seq_dedup", "item_seq_raw", "item_seq"]:
        if col in df.columns:
            return col
    raise ValueError("找不到可用的历史序列列")


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从train.csv随机切分拟合集和验证集"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    val_idx = perm[:val_size]
    fit_idx = perm[val_size:]
    return df.iloc[fit_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def build_global_popularity(fit_df: pd.DataFrame) -> Counter:
    """统计全局target热门度"""
    return Counter(fit_df["target_iid"].astype(str).tolist())


def build_cooccur_stats(fit_df: pd.DataFrame, seq_col: str, recent_n: int) -> Dict[str, Counter]:
    """统计历史item到target item的共现关系

    例如某些训练样本中历史里出现过 i000832，且 target 是 i000481，
    那么就给 cooccur["i000832"]["i000481"] 加一票。推理时如果测试用户
    历史中也有 i000832，就可以优先推荐它常共同指向的 target。
    """
    stats: Dict[str, Counter] = defaultdict(Counter)
    for _, row in fit_df.iterrows():
        target = str(row["target_iid"])
        seq = parse_seq(row.get(seq_col))
        if recent_n > 0:
            seq = seq[-recent_n:]
        # 使用去重后的历史，避免同一行里重复item把共现票数放大太多。
        for item in dict.fromkeys(seq):
            stats[item][target] += 1
    return stats


def normalize_counter(counter: Counter) -> Dict[str, float]:
    """把计数转成0到1之间的相对分数"""
    if not counter:
        return {}
    max_count = max(counter.values())
    if max_count <= 0:
        return {}
    return {item: count / max_count for item, count in counter.items()}


def rank_items(
    row: pd.Series,
    seq_col: str,
    candidate_items: set,
    global_pop: Counter,
    cooccur_stats: Dict[str, Counter],
    topk: int,
    strategy: str,
    history_filter: str,
    recent_n: int,
) -> List[str]:
    """为一条验证样本生成TopK推荐"""
    seq = parse_seq(row.get(seq_col))
    recent_seq = seq[-recent_n:] if recent_n > 0 else seq
    history = set(seq)

    scores: Dict[str, float] = defaultdict(float)
    pop_scores = normalize_counter(global_pop)

    if strategy in {"popular", "hybrid"}:
        for item, score in pop_scores.items():
            scores[item] += score

    if strategy in {"last_item", "history", "hybrid"}:
        history_items = recent_seq[-1:] if strategy == "last_item" else recent_seq
        for hist_item in history_items:
            for target, count in cooccur_stats.get(hist_item, {}).items():
                scores[target] += math.log1p(count)

    if not scores:
        for item, score in pop_scores.items():
            scores[item] += score

    # 补全候选，避免某些短历史用户推荐不足TopK。
    for item, score in pop_scores.items():
        scores[item] += 1e-6 * score

    if history_filter == "hard":
        for item in history:
            scores.pop(item, None)
    elif history_filter == "soft":
        for item in history:
            if item in scores:
                scores[item] *= 0.5
    elif history_filter != "none":
        raise ValueError(f"未知history_filter: {history_filter}")

    ranked = [
        item for item, _ in sorted(
            scores.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if item in candidate_items
    ]
    return ranked[:topk]


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
) -> Dict:
    """评估一种推荐策略"""
    rows = []
    bucket_rows = defaultdict(list)

    for _, row in val_df.iterrows():
        target = str(row["target_iid"])
        preds = rank_items(
            row=row,
            seq_col=seq_col,
            candidate_items=candidate_items,
            global_pop=global_pop,
            cooccur_stats=cooccur_stats,
            topk=topk,
            strategy=strategy,
            history_filter=history_filter,
            recent_n=recent_n,
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
    parser.add_argument("--output_json", type=str, default="", help="指标JSON输出路径")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    train_path = os.path.join(args.data_path, "train.csv")
    item_path = os.path.join(args.data_path, "item.csv")

    train_df = pd.read_csv(train_path)
    item_df = pd.read_csv(item_path)
    candidate_items = set(item_df["iid"].astype(str).tolist())
    seq_col = choose_seq_col(train_df, args.seq_col)

    fit_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    global_pop = build_global_popularity(fit_df)
    cooccur_stats = build_cooccur_stats(fit_df, seq_col, args.recent_n)

    print("=" * 80)
    print("A2离线评估")
    print("=" * 80)
    print(f"数据目录: {args.data_path}")
    print(f"序列列: {seq_col}")
    print(f"拟合集: {len(fit_df)} 行, 验证集: {len(val_df)} 行")
    print(f"候选item: {len(candidate_items)} 个, target去重: {train_df['target_iid'].nunique()} 个")
    print(f"history_filter: {args.history_filter}, recent_n: {args.recent_n}")
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
        )
        all_metrics.append(metrics)
        print_metrics(metrics)
        print("-" * 80)

    best = max(all_metrics, key=lambda item: item["ndcg"])
    print(
        f"最佳策略: {best['strategy']}，NDCG@{args.topk}={best['ndcg']:.6f}，"
        f"Hit@{args.topk}={best['hit']:.6f}"
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"指标已保存: {args.output_json}")


if __name__ == "__main__":
    main()
