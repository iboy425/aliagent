"""A2推荐启发式参数网格搜索

本脚本用于在本地验证集上批量比较 `seq_col`、`recent_n`、`strategy`
和 `history_filter`。它不训练模型，也不生成提交文件，只输出离线指标排序，
用于决定下一次线上提交候选。
"""
import argparse
import json
import os
from typing import List

import pandas as pd

from a2_offline_eval import evaluate_strategy, split_train_val
from rec_heuristics import build_cooccur_stats, build_global_popularity


def parse_csv_arg(value: str) -> List[str]:
    """解析逗号分隔参数"""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_arg(value: str) -> List[int]:
    """解析逗号分隔整数参数"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A2启发式推荐参数网格搜索")
    parser.add_argument("--data_path", type=str, default="data/rec_data", help="推荐数据目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--topk", type=int, default=10, help="推荐列表长度")
    parser.add_argument(
        "--seq_cols",
        type=str,
        default="item_seq_dedup,item_seq_raw",
        help="逗号分隔的序列列名",
    )
    parser.add_argument(
        "--recent_ns",
        type=str,
        default="1,3,5,10,20,50,0",
        help="逗号分隔的recent_n取值；0表示使用全部历史",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="popular,last_item,history,hybrid",
        help="逗号分隔的策略名",
    )
    parser.add_argument(
        "--history_filters",
        type=str,
        default="none",
        help="逗号分隔的历史item过滤策略",
    )
    parser.add_argument("--top_results", type=int, default=20, help="打印前多少个结果")
    parser.add_argument("--output_json", type=str, default="", help="可选JSON输出路径")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    candidate_items = set(item_df["iid"].astype(str).tolist())

    fit_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    global_pop = build_global_popularity(fit_df)

    seq_cols = parse_csv_arg(args.seq_cols)
    recent_ns = parse_int_csv_arg(args.recent_ns)
    strategies = parse_csv_arg(args.strategies)
    history_filters = parse_csv_arg(args.history_filters)

    print("=" * 100)
    print("A2启发式参数网格搜索")
    print("=" * 100)
    print(f"拟合集: {len(fit_df)} 行, 验证集: {len(val_df)} 行, 候选item: {len(candidate_items)}")
    print(f"seq_cols={seq_cols}")
    print(f"recent_ns={recent_ns}")
    print(f"strategies={strategies}")
    print(f"history_filters={history_filters}")
    print("-" * 100)

    results = []
    for seq_col in seq_cols:
        if seq_col not in train_df.columns:
            print(f"跳过不存在的序列列: {seq_col}")
            continue
        for recent_n in recent_ns:
            cooccur_stats = build_cooccur_stats(fit_df, seq_col, recent_n=recent_n)
            for history_filter in history_filters:
                for strategy in strategies:
                    metrics = evaluate_strategy(
                        val_df=val_df,
                        seq_col=seq_col,
                        candidate_items=candidate_items,
                        global_pop=global_pop,
                        cooccur_stats=cooccur_stats,
                        topk=args.topk,
                        strategy=strategy,
                        history_filter=history_filter,
                        recent_n=recent_n,
                    )
                    metrics["seq_col"] = seq_col
                    metrics["recent_n"] = recent_n
                    results.append(metrics)

    results = sorted(results, key=lambda item: item["ndcg"], reverse=True)

    print("排名 | seq_col        | recent_n | strategy  | filter | NDCG@10  | Hit@10   | MRR")
    print("-" * 100)
    for rank, item in enumerate(results[:args.top_results], start=1):
        print(
            f"{rank:>4} | {item['seq_col']:<14} | {item['recent_n']:>8} | "
            f"{item['strategy']:<9} | {item['history_filter']:<6} | "
            f"{item['ndcg']:.6f} | {item['hit']:.6f} | {item['mrr']:.6f}"
        )

    best = results[0]
    print("-" * 100)
    print(
        "最佳参数: "
        f"seq_col={best['seq_col']}, recent_n={best['recent_n']}, "
        f"strategy={best['strategy']}, history_filter={best['history_filter']}, "
        f"NDCG@{args.topk}={best['ndcg']:.6f}"
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"完整结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
