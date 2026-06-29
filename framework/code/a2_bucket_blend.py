"""A2按历史长度桶混合两个提交文件

A2 线上反馈显示：离线分数更高的候选如果整体替换，可能因为分布漂移
导致线上下降。本工具用于做受控实验：

- 以线上最佳 A2 作为 base；
- 只在指定历史长度桶中使用 alt 的推荐；
- 其他用户完全保留 base。

这样可以用较低风险验证“冷启动/短历史用户是否适合新策略”。
"""
import argparse
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd

from rec_heuristics import parse_seq


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


def parse_buckets(value: str) -> List[str]:
    """解析桶列表"""
    buckets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in buckets if item not in BUCKET_ORDER]
    if unknown:
        raise ValueError(f"未知桶名: {unknown}, 可选: {BUCKET_ORDER}")
    return buckets


def parse_prediction(value) -> List[str]:
    """解析prediction字段"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def overlap_ratio(items_a: List[str], items_b: List[str], topk: int) -> float:
    """计算TopK集合重合率"""
    set_a = set(items_a[:topk])
    set_b = set(items_b[:topk])
    return len(set_a & set_b) / topk if topk > 0 else 0.0


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="A2按历史长度桶混合提交文件")
    parser.add_argument("--base_a2", required=True)
    parser.add_argument("--alt_a2", required=True)
    parser.add_argument("--test_csv", default="data/rec_data/test.csv")
    parser.add_argument("--seq_col", default="item_seq_raw")
    parser.add_argument("--buckets", required=True,
                        help="使用alt的桶，逗号分隔，如 len=0,len=1")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    selected_buckets = set(parse_buckets(args.buckets))
    base_df = pd.read_csv(args.base_a2)
    alt_df = pd.read_csv(args.alt_a2)
    test_df = pd.read_csv(args.test_csv)

    base_uids = base_df["uid"].astype(str).tolist()
    alt_uids = alt_df["uid"].astype(str).tolist()
    test_uids = test_df["uid"].astype(str).tolist()
    if base_uids != alt_uids:
        raise ValueError("base_a2和alt_a2的uid顺序不一致")
    if base_uids != test_uids:
        raise ValueError("A2 uid顺序和test.csv不一致")

    out_rows = []
    stats: Dict[str, List[Dict]] = defaultdict(list)
    top1_counter = Counter()
    for idx, uid in enumerate(base_uids):
        bucket = bucket_seq_len(len(parse_seq(test_df.iloc[idx].get(args.seq_col))))
        base_pred = parse_prediction(base_df.iloc[idx]["prediction"])
        alt_pred = parse_prediction(alt_df.iloc[idx]["prediction"])
        use_alt = bucket in selected_buckets
        pred = alt_pred if use_alt else base_pred
        top1_counter[pred[0] if pred else ""] += 1
        stats[bucket].append({
            "use_alt": float(use_alt),
            "changed": float(base_pred != pred),
            "top1_changed": float(base_pred[:1] != pred[:1]),
            "overlap": overlap_ratio(base_pred, pred, args.topk),
        })
        out_rows.append({"uid": uid, "prediction": ",".join(pred[:args.topk])})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_path, index=False)

    def avg(rows: List[Dict], key: str) -> float:
        if not rows:
            return 0.0
        return sum(row[key] for row in rows) / len(rows)

    all_rows = [row for rows in stats.values() for row in rows]
    print("=" * 100)
    print("A2桶混合完成")
    print("=" * 100)
    print(f"base={args.base_a2}")
    print(f"alt ={args.alt_a2}")
    print(f"buckets={','.join(selected_buckets)}")
    print(f"output={args.output_path}")
    print(
        f"总体 changed={avg(all_rows, 'changed'):.4%}, "
        f"top1_changed={avg(all_rows, 'top1_changed'):.4%}, "
        f"overlap={avg(all_rows, 'overlap'):.4%}"
    )
    for bucket in BUCKET_ORDER:
        rows = stats.get(bucket, [])
        print(
            f"  {bucket:<8} n={len(rows):<5} use_alt={avg(rows, 'use_alt'):.4%} "
            f"changed={avg(rows, 'changed'):.4%} "
            f"top1_changed={avg(rows, 'top1_changed'):.4%} "
            f"overlap={avg(rows, 'overlap'):.4%}"
        )
    print("\nTop1分布前10:")
    total = sum(top1_counter.values())
    for item, count in top1_counter.most_common(10):
        print(f"  {item}\t{count}\t{count / total:.4%}")


if __name__ == "__main__":
    main()
