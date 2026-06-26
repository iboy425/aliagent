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

from a2_offline_eval import (
    add_weighted_metrics,
    apply_test_like_history_distribution,
    compute_bucket_weights,
    evaluate_strategy,
    split_train_val,
)
from rec_heuristics import (
    build_cooccur_stats,
    build_global_popularity,
    build_item_feature_transition_stats,
    build_user_combo_profile_stats,
    build_user_profile_stats,
    parse_item_feature_cols,
    parse_user_profile_cols,
)


def parse_csv_arg(value: str) -> List[str]:
    """解析逗号分隔参数"""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_arg(value: str) -> List[int]:
    """解析逗号分隔整数参数"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_csv_arg(value: str) -> List[float]:
    """解析逗号分隔浮点参数"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
        "--cooccur_decays",
        type=str,
        default="1.0",
        help="逗号分隔的历史共现近因衰减系数，1.0表示不衰减",
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
    parser.add_argument(
        "--user_weights",
        type=str,
        default="0.0",
        help="逗号分隔的用户画像权重",
    )
    parser.add_argument(
        "--user_combo_weights",
        type=str,
        default="0.0",
        help="逗号分隔的用户画像前缀组合权重",
    )
    parser.add_argument(
        "--user_combo_sizes",
        type=str,
        default="3,2,1",
        help="逗号分隔的用户画像前缀组合长度",
    )
    parser.add_argument("--user_combo_mode", type=str, default="prefix",
                        choices=["prefix", "all"],
                        help="用户画像组合模式: prefix=只用前缀组合，all=枚举所有指定长度组合")
    parser.add_argument("--user_combo_min_count", type=int, default=5,
                        help="画像组合最少训练样本数")
    parser.add_argument(
        "--item_feature_weights",
        type=str,
        default="0.0",
        help="逗号分隔的物品特征转移权重",
    )
    parser.add_argument(
        "--item_feature_cols",
        type=str,
        default="auto",
        help="物品特征列，auto表示使用item.csv中除iid外全部列",
    )
    parser.add_argument("--item_feature_min_count", type=int, default=20,
                        help="物品特征分组最少训练样本数")
    parser.add_argument("--item_feature_recent_n", type=int, default=-1,
                        help="物品特征转移使用最近多少个历史item；小于等于0时跟随recent_n")
    parser.add_argument(
        "--user_profile_cols",
        type=str,
        default="auto",
        help="用户画像列，auto表示使用user.csv中除uid外全部列",
    )
    parser.add_argument(
        "--pop_penalty_weights",
        type=str,
        default="0.0",
        help="逗号分隔的热门惩罚权重",
    )
    parser.add_argument(
        "--history_count_weights",
        type=str,
        default="0.0",
        help="逗号分隔的用户历史频次权重",
    )
    parser.add_argument("--test_like_eval", action="store_true",
                        help="把验证集历史截断为接近test.csv的长度分布")
    parser.add_argument("--sort_metric", type=str, default="ndcg",
                        choices=["ndcg", "weighted_ndcg"],
                        help="结果排序指标")
    parser.add_argument("--top_results", type=int, default=20, help="打印前多少个结果")
    parser.add_argument("--output_json", type=str, default="", help="可选JSON输出路径")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    candidate_items = set(item_df["iid"].astype(str).tolist())

    fit_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    global_pop = build_global_popularity(fit_df)
    user_col = "uid" if "uid" in train_df.columns else "user_id"

    seq_cols = parse_csv_arg(args.seq_cols)
    recent_ns = parse_int_csv_arg(args.recent_ns)
    cooccur_decays = parse_float_csv_arg(args.cooccur_decays)
    strategies = parse_csv_arg(args.strategies)
    history_filters = parse_csv_arg(args.history_filters)
    user_weights = parse_float_csv_arg(args.user_weights)
    user_combo_weights = parse_float_csv_arg(args.user_combo_weights)
    user_combo_sizes = parse_int_csv_arg(args.user_combo_sizes)
    item_feature_weights = parse_float_csv_arg(args.item_feature_weights)
    pop_penalty_weights = parse_float_csv_arg(args.pop_penalty_weights)
    history_count_weights = parse_float_csv_arg(args.history_count_weights)

    user_cols = []
    user_lookup = None
    user_profile_stats = None
    user_combo_profile_stats = None
    if max(user_weights or [0.0]) > 0 or max(user_combo_weights or [0.0]) > 0:
        user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
        user_cols = parse_user_profile_cols(user_df, user_col, args.user_profile_cols)
        if user_cols:
            if max(user_weights or [0.0]) > 0:
                user_profile_stats, user_lookup = build_user_profile_stats(
                    train_df=fit_df,
                    user_df=user_df,
                    user_cols=user_cols,
                    user_col=user_col,
                )
            else:
                user_lookup = user_df.set_index(user_col)
            if max(user_combo_weights or [0.0]) > 0:
                user_combo_profile_stats, user_lookup = build_user_combo_profile_stats(
                    train_df=fit_df,
                    user_df=user_df,
                    user_cols=user_cols,
                    combo_sizes=user_combo_sizes,
                    combo_mode=args.user_combo_mode,
                    min_count=args.user_combo_min_count,
                    user_col=user_col,
                )

    item_feature_cols = []
    if max(item_feature_weights or [0.0]) > 0:
        item_feature_cols = parse_item_feature_cols(item_df, args.item_feature_cols)

    print("=" * 100)
    print("A2启发式参数网格搜索")
    print("=" * 100)
    print(f"拟合集: {len(fit_df)} 行, 验证集: {len(val_df)} 行, 候选item: {len(candidate_items)}")
    print(f"seq_cols={seq_cols}")
    print(f"recent_ns={recent_ns}")
    print(f"cooccur_decays={cooccur_decays}")
    print(f"strategies={strategies}")
    print(f"history_filters={history_filters}")
    print(
        f"user_weights={user_weights}, user_combo_weights={user_combo_weights}, "
        f"user_combo_sizes={user_combo_sizes}, user_combo_mode={args.user_combo_mode}, "
        f"user_combo_min_count={args.user_combo_min_count}, "
        f"user_cols={user_cols}"
    )
    print(
        f"item_feature_weights={item_feature_weights}, item_feature_cols={item_feature_cols}, "
        f"item_feature_min_count={args.item_feature_min_count}, "
        f"item_feature_recent_n={args.item_feature_recent_n}"
    )
    print(f"pop_penalty_weights={pop_penalty_weights}")
    print(f"history_count_weights={history_count_weights}")
    print(f"test_like_eval={args.test_like_eval}, sort_metric={args.sort_metric}")
    print("-" * 100)

    results = []
    for seq_col in seq_cols:
        if seq_col not in train_df.columns:
            print(f"跳过不存在的序列列: {seq_col}")
            continue
        bucket_weights = compute_bucket_weights(test_df, seq_col)
        eval_val_df = (
            apply_test_like_history_distribution(val_df, test_df, seq_col, args.seed)
            if args.test_like_eval else val_df
        )
        for recent_n in recent_ns:
            cooccur_stats = build_cooccur_stats(fit_df, seq_col, recent_n=recent_n)
            item_lookup = None
            item_feature_stats = None
            item_feature_recent_n = args.item_feature_recent_n if args.item_feature_recent_n > 0 else recent_n
            if max(item_feature_weights or [0.0]) > 0 and item_feature_cols:
                item_feature_stats, item_lookup = build_item_feature_transition_stats(
                    train_df=fit_df,
                    item_df=item_df,
                    seq_col=seq_col,
                    feature_cols=item_feature_cols,
                    recent_n=item_feature_recent_n,
                    min_count=args.item_feature_min_count,
                )
            for history_filter in history_filters:
                for strategy in strategies:
                    for user_weight in user_weights:
                        for user_combo_weight in user_combo_weights:
                            for item_feature_weight in item_feature_weights:
                                for cooccur_decay in cooccur_decays:
                                    for pop_penalty_weight in pop_penalty_weights:
                                        for history_count_weight in history_count_weights:
                                            metrics = evaluate_strategy(
                                                val_df=eval_val_df,
                                                seq_col=seq_col,
                                                candidate_items=candidate_items,
                                                global_pop=global_pop,
                                                cooccur_stats=cooccur_stats,
                                                topk=args.topk,
                                                strategy=strategy,
                                                history_filter=history_filter,
                                                recent_n=recent_n,
                                                user_lookup=user_lookup,
                                                user_profile_stats=user_profile_stats,
                                                user_combo_profile_stats=user_combo_profile_stats,
                                                item_lookup=item_lookup,
                                                item_feature_stats=item_feature_stats,
                                                user_cols=user_cols,
                                                item_feature_cols=item_feature_cols,
                                                user_weight=user_weight,
                                                user_combo_weight=user_combo_weight,
                                                user_combo_mode=args.user_combo_mode,
                                                user_combo_min_count=args.user_combo_min_count,
                                                item_feature_weight=item_feature_weight,
                                                item_feature_recent_n=item_feature_recent_n,
                                                item_feature_min_count=args.item_feature_min_count,
                                                cooccur_decay=cooccur_decay,
                                                pop_penalty_weight=pop_penalty_weight,
                                                history_count_weight=history_count_weight,
                                            )
                                            add_weighted_metrics(metrics, bucket_weights)
                                            metrics["seq_col"] = seq_col
                                            metrics["recent_n"] = recent_n
                                            metrics["item_feature_recent_n"] = item_feature_recent_n
                                            metrics["test_like_eval"] = args.test_like_eval
                                            results.append(metrics)

    if not results:
        raise RuntimeError("没有产生任何网格搜索结果，请检查参数配置")

    results = sorted(results, key=lambda item: item[args.sort_metric], reverse=True)

    print("排名 | seq_col        | recent_n | strategy  | filter | decay | user_w | combo_w | item_w | hist_w | pop_pen | NDCG@10  | WNDCG@10 | Hit@10   | MRR")
    print("-" * 168)
    for rank, item in enumerate(results[:args.top_results], start=1):
        print(
            f"{rank:>4} | {item['seq_col']:<14} | {item['recent_n']:>8} | "
            f"{item['strategy']:<9} | {item['history_filter']:<6} | "
            f"{item.get('cooccur_decay', 1.0):<5.2f} | "
            f"{item['user_weight']:<6.3f} | {item.get('user_combo_weight', 0.0):<7.3f} | "
            f"{item.get('item_feature_weight', 0.0):<6.3f} | "
            f"{item.get('history_count_weight', 0.0):<6.3f} | "
            f"{item['pop_penalty_weight']:<7.3f} | "
            f"{item['ndcg']:.6f} | {item['weighted_ndcg']:.6f} | {item['hit']:.6f} | {item['mrr']:.6f}"
        )

    best = results[0]
    print("-" * 120)
    print(
        "最佳参数: "
        f"seq_col={best['seq_col']}, recent_n={best['recent_n']}, "
        f"strategy={best['strategy']}, history_filter={best['history_filter']}, "
        f"cooccur_decay={best.get('cooccur_decay', 1.0)}, "
        f"user_weight={best['user_weight']}, user_combo_weight={best.get('user_combo_weight', 0.0)}, "
        f"user_combo_mode={best.get('user_combo_mode', args.user_combo_mode)}, "
        f"item_feature_weight={best.get('item_feature_weight', 0.0)}, "
        f"item_feature_recent_n={best.get('item_feature_recent_n', args.item_feature_recent_n)}, "
        f"pop_penalty_weight={best['pop_penalty_weight']}, "
        f"history_count_weight={best.get('history_count_weight', 0.0)}, "
        f"NDCG@{args.topk}={best['ndcg']:.6f}, "
        f"weighted_NDCG@{args.topk}={best['weighted_ndcg']:.6f}"
    )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"完整结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
