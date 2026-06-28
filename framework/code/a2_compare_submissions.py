"""A2提交文件差异分析工具

线上结果反馈表明：离线 weighted_NDCG 高的候选不一定线上更好。
本工具用于比较两个 A2.csv 的推荐差异，重点观察：
- 总体和分桶变更率；
- Top1 分布是否明显漂移；
- TopK 平均重合度。

它不计算真实线上分数，只用于判断一个候选相对线上基线是否过激。
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


def parse_prediction(value) -> List[str]:
    """解析 prediction 字段"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def overlap_ratio(items_a: List[str], items_b: List[str], topk: int) -> float:
    """计算 TopK 集合重合率"""
    set_a = set(items_a[:topk])
    set_b = set(items_b[:topk])
    if topk <= 0:
        return 0.0
    return len(set_a & set_b) / topk


def summarize_top1(name: str, preds: List[List[str]], topn: int):
    """打印 Top1 分布"""
    counter = Counter(items[0] for items in preds if items)
    total = sum(counter.values())
    print(f"\n[{name}] Top1分布: unique={len(counter)}, total={total}")
    for item, count in counter.most_common(topn):
        ratio = count / total if total else 0.0
        print(f"  {item}\t{count}\t{ratio:.4%}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="A2提交文件差异分析")
    parser.add_argument("--base_a2", required=True, help="线上基线 A2.csv")
    parser.add_argument("--new_a2", required=True, help="候选 A2.csv")
    parser.add_argument("--test_csv", default="data/rec_data/test.csv", help="test.csv路径")
    parser.add_argument("--seq_col", default="item_seq_raw", help="历史序列列")
    parser.add_argument("--topk", type=int, default=10, help="比较TopK")
    parser.add_argument("--topn", type=int, default=15, help="Top1分布打印数量")
    args = parser.parse_args()

    base_df = pd.read_csv(args.base_a2)
    new_df = pd.read_csv(args.new_a2)
    test_df = pd.read_csv(args.test_csv)

    if base_df["uid"].astype(str).tolist() != new_df["uid"].astype(str).tolist():
        raise ValueError("base_a2 和 new_a2 的 uid 顺序不一致")
    if base_df["uid"].astype(str).tolist() != test_df["uid"].astype(str).tolist():
        raise ValueError("A2 uid 顺序和 test.csv 不一致")

    base_preds = [parse_prediction(v) for v in base_df["prediction"]]
    new_preds = [parse_prediction(v) for v in new_df["prediction"]]

    rows = []
    bucket_rows: Dict[str, List[Dict]] = defaultdict(list)
    for idx, (base_items, new_items) in enumerate(zip(base_preds, new_preds)):
        seq_len = len(parse_seq(test_df.iloc[idx].get(args.seq_col)))
        bucket = bucket_seq_len(seq_len)
        changed = base_items != new_items
        top1_changed = (base_items[:1] != new_items[:1])
        row = {
            "changed": float(changed),
            "top1_changed": float(top1_changed),
            "overlap": overlap_ratio(base_items, new_items, args.topk),
        }
        rows.append(row)
        bucket_rows[bucket].append(row)

    def avg(items: List[Dict], key: str) -> float:
        if not items:
            return 0.0
        return sum(item[key] for item in items) / len(items)

    print("=" * 100)
    print("A2提交差异分析")
    print("=" * 100)
    print(f"base={args.base_a2}")
    print(f"new ={args.new_a2}")
    print(f"样本数={len(rows)}, topk={args.topk}")
    print(
        f"总体: changed={avg(rows, 'changed'):.4%}, "
        f"top1_changed={avg(rows, 'top1_changed'):.4%}, "
        f"top{args.topk}_overlap={avg(rows, 'overlap'):.4%}"
    )

    print("\n按历史长度分桶:")
    for bucket in BUCKET_ORDER:
        items = bucket_rows.get(bucket, [])
        print(
            f"  {bucket:<8} n={len(items):<5} "
            f"changed={avg(items, 'changed'):.4%} "
            f"top1_changed={avg(items, 'top1_changed'):.4%} "
            f"overlap={avg(items, 'overlap'):.4%}"
        )

    summarize_top1("base", base_preds, args.topn)
    summarize_top1("new", new_preds, args.topn)


if __name__ == "__main__":
    main()
