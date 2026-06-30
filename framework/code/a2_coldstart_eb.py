"""A2空历史用户的经验贝叶斯画像先验

这个脚本只处理推荐任务里最难的一类用户：`item_seq_raw` 为空的冷启动用户。
这类用户没有历史序列，模型和共现规则都拿不到 item 行为信号，只能依赖：

1. 全局 target 热门度；
2. user.csv 里的用户画像类别；
3. 画像组合下的 target 分布。

旧做法通常把多个画像 counter 固定加权相加。问题是：细画像组合样本数可能
很少，直接相信会过拟合。这里使用经验贝叶斯回退：

    prior_d = lambda_d * P(target | prefix_d)
              + (1 - lambda_d) * prior_{d-1}

其中 lambda_d = n_d / (n_d + alpha_d)。组合样本 n_d 越多，越相信细画像；
样本越少，越退回粗画像或全局热门。
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
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


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """随机切分拟合集和验证集"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    val_idx = perm[:val_size]
    fit_idx = perm[val_size:]
    return df.iloc[fit_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def ndcg_at_k(preds: Sequence[str], target: str, k: int) -> float:
    """计算单样本 NDCG@K"""
    for rank, item in enumerate(preds[:k], start=1):
        if item == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点数"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_alpha_grid(value: str) -> List[List[float]]:
    """解析分号分隔的 alpha 网格"""
    return [parse_float_list(part) for part in value.split(";") if part.strip()]


def parse_prediction(value) -> List[str]:
    """解析提交文件中的 prediction 字段"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def profile_columns(user_df: pd.DataFrame, requested: str) -> List[str]:
    """确定使用哪些用户画像列"""
    if requested == "auto":
        return [col for col in user_df.columns if col != "uid"]
    return [col.strip() for col in requested.split(",") if col.strip()]


def make_key(row: pd.Series, cols: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """把多列画像转成可哈希 key"""
    values = []
    for col in cols:
        value = row.get(col)
        if pd.isna(value):
            return None
        values.append(str(value))
    return tuple(values)


def build_prefix_stats(
    train_df: pd.DataFrame,
    user_df: pd.DataFrame,
    user_cols: Sequence[str],
    max_depth: int,
) -> Tuple[Counter, Dict[int, Dict[Tuple[str, ...], Counter]], Dict[int, Counter], pd.DataFrame]:
    """统计全局热门度和画像前缀 target 分布"""
    max_depth = min(max_depth, len(user_cols))
    user_lookup = user_df.set_index("uid")
    feature_df = user_df[["uid"] + list(user_cols[:max_depth])].copy()
    merged = train_df[["uid", "target_iid"]].merge(feature_df, on="uid", how="left")

    global_counter = Counter(merged["target_iid"].astype(str).tolist())
    stats: Dict[int, Dict[Tuple[str, ...], Counter]] = {
        depth: defaultdict(Counter) for depth in range(1, max_depth + 1)
    }
    group_counts: Dict[int, Counter] = {
        depth: Counter() for depth in range(1, max_depth + 1)
    }

    for _, row in merged.iterrows():
        target = str(row["target_iid"])
        for depth in range(1, max_depth + 1):
            key = make_key(row, user_cols[:depth])
            if key is None:
                continue
            stats[depth][key][target] += 1
            group_counts[depth][key] += 1

    return global_counter, stats, group_counts, user_lookup


def normalize_counter(counter: Mapping[str, int]) -> Dict[str, float]:
    """把计数转为概率分布"""
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {str(item): float(count) / total for item, count in counter.items()}


def blend_distribution(
    base: Mapping[str, float],
    update: Mapping[str, float],
    lam: float,
) -> Dict[str, float]:
    """按 lambda 把两个分布融合"""
    if lam <= 0 or not update:
        return dict(base)
    if lam >= 1:
        return dict(update)
    keys = set(base) | set(update)
    return {
        item: (1.0 - lam) * float(base.get(item, 0.0)) + lam * float(update.get(item, 0.0))
        for item in keys
    }


def eb_scores_for_user(
    uid: str,
    user_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    global_counter: Counter,
    stats: Mapping[int, Mapping[Tuple[str, ...], Counter]],
    group_counts: Mapping[int, Counter],
    alphas: Sequence[float],
    depth: int,
) -> Dict[str, float]:
    """计算单个用户的经验贝叶斯 target 分布"""
    prior = normalize_counter(global_counter)
    if uid not in user_lookup.index:
        return prior

    row = user_lookup.loc[uid]
    max_depth = min(depth, len(user_cols), len(alphas))
    for level in range(1, max_depth + 1):
        key = make_key(row, user_cols[:level])
        if key is None:
            continue
        counter = stats.get(level, {}).get(key)
        n = float(group_counts.get(level, Counter()).get(key, 0))
        if not counter or n <= 0:
            continue
        alpha = max(float(alphas[level - 1]), 1e-9)
        lam = n / (n + alpha)
        prior = blend_distribution(prior, normalize_counter(counter), lam)
    return prior


def rank_from_scores(
    scores: Mapping[str, float],
    candidate_items: Sequence[str],
    global_counter: Counter,
    topk: int,
    temperature: float = 1.0,
) -> List[str]:
    """把分数转成合法 TopK 推荐"""
    if temperature <= 0:
        temperature = 1.0

    candidate_set = set(candidate_items)
    adjusted = {}
    for item, score in scores.items():
        if item in candidate_set:
            adjusted[item] = float(score) ** temperature

    ranked = [item for item, _ in sorted(adjusted.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set()
    result = []
    for item in ranked:
        if item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            return result

    for item, _ in global_counter.most_common():
        if item in candidate_set and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            return result

    for item in candidate_items:
        if item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            break
    return result


def rrf_fuse(
    base_items: Sequence[str],
    eb_items: Sequence[str],
    eb_weight: float,
    rrf_k: float,
    topk: int,
) -> List[str]:
    """对 base 列表和 EB 列表做轻量 RRF 融合"""
    scores = {}
    for rank, item in enumerate(base_items, start=1):
        scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
    for rank, item in enumerate(eb_items, start=1):
        scores[item] = scores.get(item, 0.0) + eb_weight / (rrf_k + rank)
    ranked = [item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked[:topk]


def evaluate_params(
    val_df: pd.DataFrame,
    user_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    candidate_items: Sequence[str],
    global_counter: Counter,
    stats: Mapping[int, Mapping[Tuple[str, ...], Counter]],
    group_counts: Mapping[int, Counter],
    alphas: Sequence[float],
    depth: int,
    topk: int,
    temperature: float,
) -> Dict:
    """评估一组 EB 参数"""
    scores = []
    for _, row in val_df.iterrows():
        uid = str(row["uid"])
        target = str(row["target_iid"])
        dist = eb_scores_for_user(
            uid=uid,
            user_lookup=user_lookup,
            user_cols=user_cols,
            global_counter=global_counter,
            stats=stats,
            group_counts=group_counts,
            alphas=alphas,
            depth=depth,
        )
        preds = rank_from_scores(
            dist,
            candidate_items=candidate_items,
            global_counter=global_counter,
            topk=topk,
            temperature=temperature,
        )
        scores.append(ndcg_at_k(preds, target, topk))
    return {
        "depth": depth,
        "alphas": list(alphas[:depth]),
        "temperature": temperature,
        "ndcg": float(np.mean(scores)) if scores else 0.0,
        "samples": len(scores),
    }


def iter_alpha_candidates(alpha_grid: Sequence[Sequence[float]], depth: int) -> Iterable[List[float]]:
    """枚举指定深度的 alpha 组合"""
    if depth <= 0:
        return
    grids = [list(alpha_grid[idx]) for idx in range(depth)]
    current = [0.0] * depth

    def dfs(pos: int):
        if pos == depth:
            yield list(current)
            return
        for value in grids[pos]:
            current[pos] = value
            yield from dfs(pos + 1)

    yield from dfs(0)


def run_search(args):
    """搜索经验贝叶斯冷启动参数"""
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    candidate_items = sorted(item_df["iid"].astype(str).tolist())
    user_cols = profile_columns(user_df, args.user_cols)
    max_depth = max(parse_int_list(args.depths))

    fit_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    if args.max_val_samples > 0 and len(val_df) > args.max_val_samples:
        val_df = val_df.sample(n=args.max_val_samples, random_state=args.seed).reset_index(drop=True)

    global_counter, stats, group_counts, user_lookup = build_prefix_stats(
        train_df=fit_df,
        user_df=user_df,
        user_cols=user_cols,
        max_depth=max_depth,
    )

    alpha_grid = parse_alpha_grid(args.alpha_grid)
    depths = parse_int_list(args.depths)
    temperatures = parse_float_list(args.temperatures)

    results = []
    for depth in depths:
        if depth > len(alpha_grid):
            continue
        for alphas in iter_alpha_candidates(alpha_grid, depth):
            for temperature in temperatures:
                results.append(evaluate_params(
                    val_df=val_df,
                    user_lookup=user_lookup,
                    user_cols=user_cols,
                    candidate_items=candidate_items,
                    global_counter=global_counter,
                    stats=stats,
                    group_counts=group_counts,
                    alphas=alphas,
                    depth=depth,
                    topk=args.topk,
                    temperature=temperature,
                ))

    results.sort(key=lambda item: item["ndcg"], reverse=True)
    output = {
        "best": results[0] if results else {},
        "results": results,
        "user_cols": user_cols,
        "fit_samples": len(fit_df),
        "val_samples": len(val_df),
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A2 冷启动 EB 参数搜索 Top-20")
    print("=" * 100)
    for item in results[:20]:
        print(
            f"ndcg={item['ndcg']:.6f}\tdepth={item['depth']}\t"
            f"alphas={item['alphas']}\ttemperature={item['temperature']}"
        )
    if args.output_json:
        print(f"\n搜索结果已保存: {args.output_json}")


def overlap_ratio(items_a: Sequence[str], items_b: Sequence[str], topk: int) -> float:
    """计算 TopK 集合重合率"""
    return len(set(items_a[:topk]) & set(items_b[:topk])) / topk


def load_best_params(args) -> Tuple[int, List[float], float]:
    """从命令行或搜索结果读取最佳参数"""
    if args.search_json:
        data = json.load(open(args.search_json, encoding="utf-8"))
        best = data["best"]
        return int(best["depth"]), [float(x) for x in best["alphas"]], float(best.get("temperature", 1.0))
    depth = int(args.depth)
    alphas = parse_float_list(args.alphas)
    if len(alphas) < depth:
        raise ValueError("alphas 数量不能小于 depth")
    return depth, alphas, float(args.temperature)


def run_predict(args):
    """生成只替换冷启动桶的 A2 候选"""
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    candidate_items = sorted(item_df["iid"].astype(str).tolist())
    user_cols = profile_columns(user_df, args.user_cols)
    depth, alphas, temperature = load_best_params(args)

    global_counter, stats, group_counts, user_lookup = build_prefix_stats(
        train_df=train_df,
        user_df=user_df,
        user_cols=user_cols,
        max_depth=depth,
    )

    base_df = pd.read_csv(args.base_a2) if args.base_a2 else None
    if base_df is not None and base_df["uid"].astype(str).tolist() != test_df["uid"].astype(str).tolist():
        raise ValueError("base_a2 的 uid 顺序和 test.csv 不一致")

    replace_buckets = set(item.strip() for item in args.replace_buckets.split(",") if item.strip())
    out_rows = []
    drift_rows = defaultdict(list)
    top1_counter = Counter()
    eb_top1_counter = Counter()

    for idx, row in test_df.iterrows():
        uid = str(row["uid"])
        bucket = bucket_seq_len(len(parse_seq(row.get(args.seq_col))))
        dist = eb_scores_for_user(
            uid=uid,
            user_lookup=user_lookup,
            user_cols=user_cols,
            global_counter=global_counter,
            stats=stats,
            group_counts=group_counts,
            alphas=alphas,
            depth=depth,
        )
        eb_pred = rank_from_scores(
            dist,
            candidate_items=candidate_items,
            global_counter=global_counter,
            topk=max(args.topk, args.eb_pool_size),
            temperature=temperature,
        )
        eb_top1_counter[eb_pred[0]] += 1

        if base_df is None:
            base_pred = []
            pred = eb_pred[:args.topk]
        else:
            base_pred = parse_prediction(base_df.iloc[idx]["prediction"])
            if bucket in replace_buckets:
                if args.eb_weight >= 999:
                    pred = eb_pred[:args.topk]
                else:
                    pred = rrf_fuse(
                        base_items=base_pred,
                        eb_items=eb_pred,
                        eb_weight=args.eb_weight,
                        rrf_k=args.rrf_k,
                        topk=args.topk,
                    )
            else:
                pred = base_pred[:args.topk]

        top1_counter[pred[0]] += 1
        if base_df is not None:
            drift_rows[bucket].append({
                "changed": float(base_pred[:args.topk] != pred[:args.topk]),
                "top1_changed": float(base_pred[:1] != pred[:1]),
                "overlap": overlap_ratio(base_pred, pred, args.topk),
                "replaced": float(bucket in replace_buckets),
            })
        out_rows.append({"uid": uid, "prediction": ",".join(pred[:args.topk])})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.output_path, index=False)

    print("=" * 100)
    print("A2 冷启动 EB 推理完成")
    print("=" * 100)
    print(f"output={args.output_path}")
    print(f"depth={depth}, alphas={alphas[:depth]}, temperature={temperature}")
    print(f"replace_buckets={sorted(replace_buckets)}, eb_weight={args.eb_weight}, rrf_k={args.rrf_k}")

    if base_df is not None:
        all_rows = [item for rows in drift_rows.values() for item in rows]

        def avg(rows: List[Dict], key: str) -> float:
            return float(np.mean([row[key] for row in rows])) if rows else 0.0

        print(
            f"总体 changed={avg(all_rows, 'changed'):.4%}, "
            f"top1_changed={avg(all_rows, 'top1_changed'):.4%}, "
            f"overlap={avg(all_rows, 'overlap'):.4%}"
        )
        for bucket in BUCKET_ORDER:
            rows = drift_rows.get(bucket, [])
            print(
                f"  {bucket:<8} n={len(rows):<5} replaced={avg(rows, 'replaced'):.4%} "
                f"changed={avg(rows, 'changed'):.4%} "
                f"top1_changed={avg(rows, 'top1_changed'):.4%} "
                f"overlap={avg(rows, 'overlap'):.4%}"
            )

    print("\n最终Top1分布前10:")
    total = sum(top1_counter.values())
    for item, count in top1_counter.most_common(10):
        print(f"  {item}\t{count}\t{count / total:.4%}")
    print("\nEB自身Top1分布前10:")
    total_eb = sum(eb_top1_counter.values())
    for item, count in eb_top1_counter.most_common(10):
        print(f"  {item}\t{count}\t{count / total_eb:.4%}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A2 冷启动经验贝叶斯画像先验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_path", default="data/rec_data")
    common.add_argument("--user_cols", default="auto")
    common.add_argument("--topk", type=int, default=10)

    search = subparsers.add_parser("search", parents=[common], help="搜索 EB 参数")
    search.add_argument("--val_ratio", type=float, default=0.1)
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--max_val_samples", type=int, default=8000)
    search.add_argument("--depths", default="1,2,3,4")
    search.add_argument(
        "--alpha_grid",
        default="20,50,100,200;10,30,50,100;5,10,20,50;5,10,20,50",
    )
    search.add_argument("--temperatures", default="0.7,1.0,1.3")
    search.add_argument("--output_json", default="")

    predict = subparsers.add_parser("predict", parents=[common], help="生成 A2 候选")
    predict.add_argument("--base_a2", default="")
    predict.add_argument("--seq_col", default="item_seq_raw")
    predict.add_argument("--replace_buckets", default="len=0")
    predict.add_argument("--search_json", default="")
    predict.add_argument("--depth", type=int, default=3)
    predict.add_argument("--alphas", default="50,30,10")
    predict.add_argument("--temperature", type=float, default=1.0)
    predict.add_argument("--eb_weight", type=float, default=999.0,
                         help=">=999 表示直接使用 EB；否则与 base 做 RRF 融合")
    predict.add_argument("--rrf_k", type=float, default=20.0)
    predict.add_argument("--eb_pool_size", type=int, default=50)
    predict.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    if args.command == "search":
        run_search(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
