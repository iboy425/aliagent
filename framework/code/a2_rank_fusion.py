"""A2提交文件的按桶软排序融合

线上反馈说明：Exp045 整体替换会明显漂移，但在冷启动桶上有收益。
本工具把两个 A2 提交文件做 rank-level soft fusion，而不是整桶硬替换：

score(item) = 1 / (k + rank_base) + lambda_bucket / (k + rank_alt)

这样可以保留 Exp044 的稳定性，同时让 Exp045 在短历史用户上提供局部重排信号。
"""
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import pandas as pd

from rec_heuristics import build_global_popularity, normalize_counter, parse_seq


BUCKET_ORDER = ["len=0", "len=1", "len=2-3", "len=4-10", "len>10"]


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


def parse_prediction(value) -> List[str]:
    """解析prediction字段"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_bucket_float_map(value: str, default: float = 0.0) -> Dict[str, float]:
    """解析 `len=0:1.0,len=1:0.2` 形式的桶权重"""
    result = {bucket: float(default) for bucket in BUCKET_ORDER}
    if not value:
        return result
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"桶权重格式错误: {part}")
        bucket, raw_weight = part.split(":", 1)
        bucket = bucket.strip()
        if bucket not in result:
            raise ValueError(f"未知桶名: {bucket}, 可选: {BUCKET_ORDER}")
        result[bucket] = float(raw_weight)
    return result


def parse_bucket_set(value: str) -> Set[str]:
    """解析桶集合"""
    if not value:
        return set()
    buckets = {item.strip() for item in value.split(",") if item.strip()}
    unknown = buckets - set(BUCKET_ORDER)
    if unknown:
        raise ValueError(f"未知桶名: {sorted(unknown)}, 可选: {BUCKET_ORDER}")
    return buckets


def overlap_ratio(items_a: Sequence[str], items_b: Sequence[str], topk: int) -> float:
    """计算TopK集合重合率"""
    return len(set(items_a[:topk]) & set(items_b[:topk])) / topk if topk > 0 else 0.0


def add_rank_scores(
    scores: Dict[str, float],
    items: Sequence[str],
    weight: float,
    rrf_k: float,
):
    """把一个排序列表转为RRF分数"""
    if weight <= 0:
        return
    for rank, item in enumerate(items, start=1):
        if item:
            scores[item] = scores.get(item, 0.0) + weight / (rrf_k + rank)


def add_popularity_scores(
    scores: Dict[str, float],
    pop_scores: Mapping[str, float],
    weight: float,
):
    """加入热门度补充分数"""
    if weight <= 0:
        return
    for item, score in pop_scores.items():
            scores[item] = scores.get(item, 0.0) + weight * score


def build_suffix_transition_stats(
    train_df: pd.DataFrame,
    seq_col: str,
    target_col: str = "target_iid",
) -> Dict[int, Dict[Tuple[str, ...], Counter]]:
    """统计有序后缀到target的转移分布

    与无序共现不同，这里保留最后 n 个 item 的顺序：
    - n=1: P(target | last1)
    - n=2: P(target | last2 tuple)
    - n=3: P(target | last3 tuple)
    """
    stats = {1: defaultdict(Counter), 2: defaultdict(Counter), 3: defaultdict(Counter)}
    for _, row in train_df.iterrows():
        target = str(row[target_col])
        seq = parse_seq(row.get(seq_col))
        for size in (1, 2, 3):
            if len(seq) >= size:
                key = tuple(seq[-size:])
                stats[size][key][target] += 1
    return stats


def suffix_rank_lists(
    seq: Sequence[str],
    stats: Mapping[int, Mapping[Tuple[str, ...], Counter]],
    min_count2: int,
    min_count3: int,
    topn: int,
) -> Dict[int, List[str]]:
    """获取当前用户的 suffix transition 推荐列表"""
    result = {1: [], 2: [], 3: []}
    for size in (1, 2, 3):
        if len(seq) < size:
            continue
        key = tuple(seq[-size:])
        counter = stats.get(size, {}).get(key)
        if not counter:
            continue
        total = sum(counter.values())
        if size == 2 and total < min_count2:
            continue
        if size == 3 and total < min_count3:
            continue
        result[size] = [item for item, _ in counter.most_common(topn)]
    return result


def stable_rank_items(
    scores: Mapping[str, float],
    base_items: Sequence[str],
    alt_items: Sequence[str],
    popular_items: Sequence[str],
    topk: int,
) -> List[str]:
    """按分数排序并稳定补齐TopK"""
    base_rank = {item: rank for rank, item in enumerate(base_items, start=1)}
    alt_rank = {item: rank for rank, item in enumerate(alt_items, start=1)}
    ranked = sorted(
        scores,
        key=lambda item: (
            -scores[item],
            base_rank.get(item, 10_000),
            alt_rank.get(item, 10_000),
            item,
        ),
    )

    result = []
    seen = set()
    for source in [ranked, base_items, alt_items, popular_items]:
        for item in source:
            if item and item not in seen:
                result.append(item)
                seen.add(item)
                if len(result) >= topk:
                    return result
    return result


def fuse_one(
    base_items: Sequence[str],
    alt_items: Sequence[str],
    popular_items: Sequence[str],
    pop_scores: Mapping[str, float],
    lambda_weight: float,
    pop_weight: float,
    rrf_k: float,
    topk: int,
    suffix_lists: Mapping[int, Sequence[str]] = None,
    suffix_weights: Mapping[int, float] = None,
) -> List[str]:
    """融合单个用户的两个推荐列表"""
    scores: Dict[str, float] = {}
    add_rank_scores(scores, base_items, 1.0, rrf_k)
    add_rank_scores(scores, alt_items, lambda_weight, rrf_k)
    if suffix_lists and suffix_weights:
        for size in (1, 2, 3):
            add_rank_scores(
                scores,
                suffix_lists.get(size, []),
                float(suffix_weights.get(size, 0.0)),
                rrf_k,
            )
    add_popularity_scores(scores, pop_scores, pop_weight)
    return stable_rank_items(scores, base_items, alt_items, popular_items, topk)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="A2按历史长度桶做RRF软排序融合")
    parser.add_argument("--base_a2", required=True)
    parser.add_argument("--alt_a2", required=True)
    parser.add_argument("--test_csv", default="data/rec_data/test.csv")
    parser.add_argument("--train_csv", default="data/rec_data/train.csv")
    parser.add_argument("--seq_col", default="item_seq_raw")
    parser.add_argument("--bucket_lambdas", required=True,
                        help="如 len=0:1.0,len=1:0.2,len=2-3:0.05")
    parser.add_argument("--bucket_pop_weights", default="",
                        help="可选热门补充分数，如 len=0:0.001")
    parser.add_argument("--hard_buckets", default="",
                        help="这些桶直接使用alt，逗号分隔；默认全走软融合")
    parser.add_argument("--suffix_buckets", default="",
                        help="启用ordered suffix transition的桶，逗号分隔")
    parser.add_argument("--suffix_weight1", type=float, default=0.0)
    parser.add_argument("--suffix_weight2", type=float, default=0.0)
    parser.add_argument("--suffix_weight3", type=float, default=0.0)
    parser.add_argument("--suffix_min_count2", type=int, default=5)
    parser.add_argument("--suffix_min_count3", type=int, default=5)
    parser.add_argument("--suffix_topn", type=int, default=30)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    lambdas = parse_bucket_float_map(args.bucket_lambdas, default=0.0)
    pop_weights = parse_bucket_float_map(args.bucket_pop_weights, default=0.0)
    hard_buckets = parse_bucket_set(args.hard_buckets)
    suffix_buckets = parse_bucket_set(args.suffix_buckets)
    suffix_weights = {
        1: args.suffix_weight1,
        2: args.suffix_weight2,
        3: args.suffix_weight3,
    }

    base_df = pd.read_csv(args.base_a2)
    alt_df = pd.read_csv(args.alt_a2)
    test_df = pd.read_csv(args.test_csv)
    train_df = pd.read_csv(args.train_csv)

    base_uids = base_df["uid"].astype(str).tolist()
    alt_uids = alt_df["uid"].astype(str).tolist()
    test_uids = test_df["uid"].astype(str).tolist()
    if base_uids != alt_uids:
        raise ValueError("base_a2和alt_a2的uid顺序不一致")
    if base_uids != test_uids:
        raise ValueError("A2 uid顺序和test.csv不一致")

    global_pop = build_global_popularity(train_df)
    pop_scores = normalize_counter(global_pop)
    popular_items = [item for item, _ in global_pop.most_common()]
    suffix_stats = build_suffix_transition_stats(train_df, args.seq_col)

    out_rows = []
    stats = defaultdict(list)
    top1_counter = Counter()
    suffix_use_counter = Counter()
    for idx, uid in enumerate(base_uids):
        seq = parse_seq(test_df.iloc[idx].get(args.seq_col))
        bucket = bucket_seq_len(len(seq))
        base_items = parse_prediction(base_df.iloc[idx]["prediction"])
        alt_items = parse_prediction(alt_df.iloc[idx]["prediction"])
        suffix_lists = {}
        if bucket in suffix_buckets:
            suffix_lists = suffix_rank_lists(
                seq=seq,
                stats=suffix_stats,
                min_count2=args.suffix_min_count2,
                min_count3=args.suffix_min_count3,
                topn=args.suffix_topn,
            )
            for size, items in suffix_lists.items():
                if items and suffix_weights.get(size, 0.0) > 0:
                    suffix_use_counter[f"{bucket}:last{size}"] += 1

        if bucket in hard_buckets:
            pred = stable_rank_items({}, alt_items, base_items, popular_items, args.topk)
        else:
            pred = fuse_one(
                base_items=base_items,
                alt_items=alt_items,
                popular_items=popular_items,
                pop_scores=pop_scores,
                lambda_weight=lambdas[bucket],
                pop_weight=pop_weights[bucket],
                rrf_k=args.rrf_k,
                topk=args.topk,
                suffix_lists=suffix_lists,
                suffix_weights=suffix_weights,
            )

        top1_counter[pred[0] if pred else ""] += 1
        stats[bucket].append({
            "changed": float(base_items[:args.topk] != pred[:args.topk]),
            "top1_changed": float(base_items[:1] != pred[:1]),
            "overlap": overlap_ratio(base_items, pred, args.topk),
        })
        out_rows.append({"uid": uid, "prediction": ",".join(pred[:args.topk])})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.output_path, index=False)

    def avg(rows: List[Dict[str, float]], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    all_rows = [row for rows in stats.values() for row in rows]
    print("=" * 100)
    print("A2 RRF软融合完成")
    print("=" * 100)
    print(f"base={args.base_a2}")
    print(f"alt ={args.alt_a2}")
    print(f"output={args.output_path}")
    print(f"rrf_k={args.rrf_k}")
    print(f"bucket_lambdas={lambdas}")
    print(f"bucket_pop_weights={pop_weights}")
    print(f"hard_buckets={sorted(hard_buckets)}")
    print(f"suffix_buckets={sorted(suffix_buckets)}")
    print(
        "suffix_weights="
        f"last1:{args.suffix_weight1},last2:{args.suffix_weight2},last3:{args.suffix_weight3}, "
        f"min2={args.suffix_min_count2}, min3={args.suffix_min_count3}, topn={args.suffix_topn}"
    )
    print(
        f"总体 changed={avg(all_rows, 'changed'):.4%}, "
        f"top1_changed={avg(all_rows, 'top1_changed'):.4%}, "
        f"overlap={avg(all_rows, 'overlap'):.4%}"
    )
    for bucket in BUCKET_ORDER:
        rows = stats.get(bucket, [])
        print(
            f"  {bucket:<8} n={len(rows):<5} "
            f"lambda={lambdas[bucket]:<6.3f} "
            f"changed={avg(rows, 'changed'):.4%} "
            f"top1_changed={avg(rows, 'top1_changed'):.4%} "
            f"overlap={avg(rows, 'overlap'):.4%}"
        )

    print("\nTop1分布前10:")
    total = sum(top1_counter.values())
    for item, count in top1_counter.most_common(10):
        print(f"  {item}\t{count}\t{count / total:.4%}")

    if suffix_use_counter:
        print("\nSuffix使用统计:")
        for key, count in suffix_use_counter.most_common():
            print(f"  {key}\t{count}")


if __name__ == "__main__":
    main()
