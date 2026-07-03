"""A2 冷启动全组合经验贝叶斯画像先验

`a2_coldstart_eb.py` 使用用户画像前缀组合，例如
`u_cat_01 -> u_cat_01,u_cat_02 -> u_cat_01,u_cat_02,u_cat_03`。
这个脚本用于验证一个更强的假设：有效画像组合未必刚好出现在列顺序前缀里。

脚本只服务 A2 的冷启动/短历史用户，不加载神经模型，不改中长历史逻辑。
如果多 split 审计不稳定，应删除本脚本，避免把验证集微调带到线上。
"""
import argparse
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from a2_coldstart_eb import (
    BUCKET_ORDER,
    bucket_seq_len,
    ndcg_at_k,
    overlap_ratio,
    parse_prediction,
    rrf_fuse,
    split_train_val,
)
from rec_heuristics import parse_seq


Spec = Tuple[str, ...]


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数"""
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点数"""
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_weight_candidates(value: str) -> List[List[float]]:
    """解析权重候选，格式如 `1,0.5,0.25;1,1,0.5`"""
    rows = []
    for part in str(value).split(";"):
        part = part.strip()
        if part:
            rows.append(parse_float_list(part))
    return rows


def profile_columns(user_df: pd.DataFrame, requested: str) -> List[str]:
    """确定使用哪些用户画像列"""
    if requested == "auto":
        return [col for col in user_df.columns if col != "uid"]
    return [col.strip() for col in requested.split(",") if col.strip()]


def make_specs(user_cols: Sequence[str], combo_mode: str, combo_sizes: Sequence[int]) -> List[Spec]:
    """生成画像组合规格"""
    specs: List[Spec] = []
    for size in combo_sizes:
        if size <= 0 or size > len(user_cols):
            continue
        if combo_mode == "prefix":
            candidates = [tuple(user_cols[:size])]
        elif combo_mode == "all":
            candidates = [tuple(cols) for cols in itertools.combinations(user_cols, size)]
        else:
            raise ValueError(f"未知 combo_mode: {combo_mode}")
        for spec in candidates:
            if spec not in specs:
                specs.append(spec)
    # 细粒度组合先参与打分；同长度时按列名稳定排序。
    return sorted(specs, key=lambda item: (-len(item), item))


def make_key(row: pd.Series, spec: Spec) -> Optional[Tuple[str, ...]]:
    """把一组画像列转成 key"""
    values = []
    for col in spec:
        value = row.get(col)
        if pd.isna(value):
            return None
        values.append(str(value))
    return tuple(values)


def normalize_counter(counter: Mapping[str, int]) -> Dict[str, float]:
    """计数归一化为概率"""
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {str(item): float(count) / total for item, count in counter.items()}


def build_combo_stats(
    train_df: pd.DataFrame,
    user_df: pd.DataFrame,
    user_cols: Sequence[str],
    specs: Sequence[Spec],
) -> Tuple[Counter, Dict[Spec, Dict[Tuple[str, ...], Counter]], Dict[Spec, Counter], pd.DataFrame]:
    """统计每个画像组合下的 target 分布"""
    feature_cols = ["uid"] + sorted({col for spec in specs for col in spec})
    merged = train_df[["uid", "target_iid"]].merge(user_df[feature_cols], on="uid", how="left")
    global_counter = Counter(merged["target_iid"].astype(str).tolist())
    stats: Dict[Spec, Dict[Tuple[str, ...], Counter]] = {spec: defaultdict(Counter) for spec in specs}
    group_counts: Dict[Spec, Counter] = {spec: Counter() for spec in specs}

    for spec in specs:
        cols = list(spec)
        grouped = (
            merged.dropna(subset=cols)
            .groupby(cols + ["target_iid"], sort=False)
            .size()
        )
        for raw_key, count in grouped.items():
            if not isinstance(raw_key, tuple):
                raw_key = (raw_key,)
            key_values = tuple(str(value) for value in raw_key[:-1])
            target = str(raw_key[-1])
            count = int(count)
            stats[spec][key_values][target] += count
            group_counts[spec][key_values] += count

    return global_counter, stats, group_counts, user_df.set_index("uid")


def combo_scores_for_user(
    uid: str,
    user_lookup: pd.DataFrame,
    specs: Sequence[Spec],
    global_counter: Counter,
    stats: Mapping[Spec, Mapping[Tuple[str, ...], Counter]],
    group_counts: Mapping[Spec, Counter],
    alphas_by_size: Mapping[int, float],
    weights_by_size: Mapping[int, float],
    min_count: int,
    global_weight: float,
) -> Dict[str, float]:
    """计算一个用户的全组合 EB 分数"""
    scores = {
        item: global_weight * prob
        for item, prob in normalize_counter(global_counter).items()
    }
    if uid not in user_lookup.index:
        return scores

    row = user_lookup.loc[uid]
    for spec in specs:
        key = make_key(row, spec)
        if key is None:
            continue
        n = float(group_counts.get(spec, Counter()).get(key, 0))
        if n < min_count:
            continue
        counter = stats.get(spec, {}).get(key)
        if not counter:
            continue
        size = len(spec)
        alpha = max(float(alphas_by_size.get(size, 100.0)), 1e-9)
        weight = float(weights_by_size.get(size, 0.0))
        if weight <= 0:
            continue
        lam = n / (n + alpha)
        for item, prob in normalize_counter(counter).items():
            scores[item] = scores.get(item, 0.0) + weight * lam * prob
    return scores


def rank_from_scores(
    scores: Mapping[str, float],
    candidate_items: Sequence[str],
    global_counter: Counter,
    topk: int,
    temperature: float,
) -> List[str]:
    """根据分数生成合法推荐列表"""
    candidate_set = set(candidate_items)
    adjusted = {}
    power = max(float(temperature), 1e-9)
    for item, score in scores.items():
        if item in candidate_set:
            adjusted[item] = max(float(score), 0.0) ** power
    ranked = [item for item, _ in sorted(adjusted.items(), key=lambda kv: (-kv[1], kv[0]))]
    result = []
    seen = set()
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


def evaluate_combo(
    val_df: pd.DataFrame,
    user_lookup: pd.DataFrame,
    specs: Sequence[Spec],
    candidate_items: Sequence[str],
    global_counter: Counter,
    stats: Mapping[Spec, Mapping[Tuple[str, ...], Counter]],
    group_counts: Mapping[Spec, Counter],
    alphas_by_size: Mapping[int, float],
    weights_by_size: Mapping[int, float],
    min_count: int,
    global_weight: float,
    temperature: float,
    topk: int,
) -> Dict:
    """评估一组全组合 EB 参数"""
    scores = []
    bucket_scores = {bucket: [] for bucket in BUCKET_ORDER}
    for _, row in val_df.iterrows():
        uid = str(row["uid"])
        target = str(row["target_iid"])
        dist = combo_scores_for_user(
            uid=uid,
            user_lookup=user_lookup,
            specs=specs,
            global_counter=global_counter,
            stats=stats,
            group_counts=group_counts,
            alphas_by_size=alphas_by_size,
            weights_by_size=weights_by_size,
            min_count=min_count,
            global_weight=global_weight,
        )
        preds = rank_from_scores(dist, candidate_items, global_counter, topk, temperature)
        score = ndcg_at_k(preds, target, topk)
        scores.append(score)
        bucket_scores[bucket_seq_len(len(parse_seq(row.get("item_seq_raw"))))].append(score)

    return {
        "ndcg": float(np.mean(scores)) if scores else 0.0,
        "samples": len(scores),
        "bucket_ndcg": {
            bucket: float(np.mean(vals)) if vals else 0.0
            for bucket, vals in bucket_scores.items()
        },
        "bucket_samples": {bucket: len(vals) for bucket, vals in bucket_scores.items()},
    }


def prepare_eval_rows(
    val_df: pd.DataFrame,
    user_lookup: pd.DataFrame,
    specs: Sequence[Spec],
    stats: Mapping[Spec, Mapping[Tuple[str, ...], Counter]],
    group_counts: Mapping[Spec, Counter],
) -> List[Dict]:
    """预先缓存验证用户可命中的画像组合

    搜索阶段会反复评估不同 alpha/min_count/weight。若每次都重新根据 uid
    遍历所有画像组合，会非常慢。这里把每个验证用户对应的 counter 先查好。
    """
    rows = []
    for _, row in val_df.iterrows():
        uid = str(row["uid"])
        refs = []
        if uid in user_lookup.index:
            user_row = user_lookup.loc[uid]
            for spec in specs:
                key = make_key(user_row, spec)
                if key is None:
                    continue
                counter = stats.get(spec, {}).get(key)
                if not counter:
                    continue
                refs.append({
                    "size": len(spec),
                    "n": float(group_counts.get(spec, Counter()).get(key, 0)),
                    "probs": normalize_counter(counter),
                })
        rows.append({
            "target": str(row["target_iid"]),
            "bucket": bucket_seq_len(len(parse_seq(row.get("item_seq_raw")))),
            "refs": refs,
        })
    return rows


def evaluate_combo_cached(
    eval_rows: Sequence[Dict],
    candidate_items: Sequence[str],
    global_counter: Counter,
    global_probs: Mapping[str, float],
    alphas_by_size: Mapping[int, float],
    weights_by_size: Mapping[int, float],
    min_count: int,
    global_weight: float,
    temperature: float,
    topk: int,
) -> Dict:
    """使用缓存后的画像组合评估参数"""
    scores = []
    bucket_scores = {bucket: [] for bucket in BUCKET_ORDER}
    for row in eval_rows:
        dist = {
            item: global_weight * prob
            for item, prob in global_probs.items()
        }
        for ref in row["refs"]:
            n = float(ref["n"])
            if n < min_count:
                continue
            size = int(ref["size"])
            alpha = max(float(alphas_by_size.get(size, 100.0)), 1e-9)
            weight = float(weights_by_size.get(size, 0.0))
            if weight <= 0:
                continue
            lam = n / (n + alpha)
            for item, prob in ref["probs"].items():
                dist[item] = dist.get(item, 0.0) + weight * lam * prob

        preds = rank_from_scores(dist, candidate_items, global_counter, topk, temperature)
        score = ndcg_at_k(preds, row["target"], topk)
        scores.append(score)
        bucket_scores[row["bucket"]].append(score)

    return {
        "ndcg": float(np.mean(scores)) if scores else 0.0,
        "samples": len(scores),
        "bucket_ndcg": {
            bucket: float(np.mean(vals)) if vals else 0.0
            for bucket, vals in bucket_scores.items()
        },
        "bucket_samples": {bucket: len(vals) for bucket, vals in bucket_scores.items()},
    }


def build_param_rows(args) -> Iterable[Dict]:
    """生成参数候选"""
    alpha_values = parse_float_list(args.alpha_values)
    weight_rows = parse_weight_candidates(args.size_weight_grid)
    min_counts = parse_int_list(args.min_counts)
    temperatures = parse_float_list(args.temperatures)
    global_weights = parse_float_list(args.global_weights)
    sizes = parse_int_list(args.combo_sizes)
    for alpha in alpha_values:
        alphas_by_size = {size: alpha for size in sizes}
        for weights in weight_rows:
            weights_by_size = {
                size: (weights[idx] if idx < len(weights) else 0.0)
                for idx, size in enumerate(sizes)
            }
            for min_count in min_counts:
                for temperature in temperatures:
                    for global_weight in global_weights:
                        yield {
                            "alpha": alpha,
                            "alphas_by_size": alphas_by_size,
                            "weights_by_size": weights_by_size,
                            "min_count": min_count,
                            "temperature": temperature,
                            "global_weight": global_weight,
                        }


def run_search(args):
    """执行全组合 EB 多 split 搜索"""
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    candidate_items = sorted(item_df["iid"].astype(str).tolist())
    user_cols = profile_columns(user_df, args.user_cols)
    combo_sizes = parse_int_list(args.combo_sizes)
    specs = make_specs(user_cols, args.combo_mode, combo_sizes)
    split_seeds = parse_int_list(args.split_seeds)

    all_rows = []
    split_best = []
    for split_seed in split_seeds:
        fit_df, val_df = split_train_val(train_df, args.val_ratio, split_seed)
        if args.max_val_samples > 0 and len(val_df) > args.max_val_samples:
            val_df = val_df.sample(n=args.max_val_samples, random_state=split_seed).reset_index(drop=True)
        global_counter, stats, group_counts, user_lookup = build_combo_stats(
            fit_df, user_df, user_cols, specs
        )
        eval_rows = prepare_eval_rows(val_df, user_lookup, specs, stats, group_counts)
        global_probs = normalize_counter(global_counter)

        rows = []
        for params in build_param_rows(args):
            metrics = evaluate_combo_cached(
                eval_rows=eval_rows,
                candidate_items=candidate_items,
                global_counter=global_counter,
                global_probs=global_probs,
                alphas_by_size=params["alphas_by_size"],
                weights_by_size=params["weights_by_size"],
                min_count=params["min_count"],
                global_weight=params["global_weight"],
                temperature=params["temperature"],
                topk=args.topk,
            )
            row = {
                "split_seed": split_seed,
                "alpha": params["alpha"],
                "weights_by_size": params["weights_by_size"],
                "min_count": params["min_count"],
                "temperature": params["temperature"],
                "global_weight": params["global_weight"],
                **metrics,
            }
            rows.append(row)
            all_rows.append(row)
        rows.sort(key=lambda item: item["ndcg"], reverse=True)
        split_best.append(rows[0])
        print(
            f"[split={split_seed}] best_ndcg={rows[0]['ndcg']:.6f} "
            f"alpha={rows[0]['alpha']} min_count={rows[0]['min_count']} "
            f"weights={rows[0]['weights_by_size']}"
        )

    grouped = defaultdict(list)
    for row in all_rows:
        key = (
            row["alpha"],
            tuple(sorted((int(k), float(v)) for k, v in row["weights_by_size"].items())),
            row["min_count"],
            row["temperature"],
            row["global_weight"],
        )
        grouped[key].append(row)

    summary = []
    for key, rows in grouped.items():
        if len(rows) != len(split_seeds):
            continue
        summary.append({
            "alpha": rows[0]["alpha"],
            "weights_by_size": rows[0]["weights_by_size"],
            "min_count": rows[0]["min_count"],
            "temperature": rows[0]["temperature"],
            "global_weight": rows[0]["global_weight"],
            "mean_ndcg": float(np.mean([row["ndcg"] for row in rows])),
            "min_ndcg": float(np.min([row["ndcg"] for row in rows])),
            "std_ndcg": float(np.std([row["ndcg"] for row in rows])),
            "rows": rows,
        })
    summary.sort(key=lambda item: (item["mean_ndcg"], item["min_ndcg"]), reverse=True)

    output = {
        "best": summary[0] if summary else {},
        "summary": summary,
        "split_best": split_best,
        "user_cols": user_cols,
        "combo_mode": args.combo_mode,
        "combo_sizes": combo_sizes,
        "num_specs": len(specs),
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A2 全组合 EB 搜索 Top-20")
    print("=" * 100)
    for item in summary[:20]:
        print(
            f"mean={item['mean_ndcg']:.6f}\tmin={item['min_ndcg']:.6f}\t"
            f"std={item['std_ndcg']:.6f}\talpha={item['alpha']}\t"
            f"min_count={item['min_count']}\tgw={item['global_weight']}\t"
            f"weights={item['weights_by_size']}"
        )
    if args.output_json:
        print(f"\n搜索结果已保存: {args.output_json}")


def load_best_params(args) -> Dict:
    """读取搜索结果或命令行参数"""
    if args.search_json:
        data = json.load(open(args.search_json, encoding="utf-8"))
        best = data["best"]
        return {
            "alpha": float(best["alpha"]),
            "weights_by_size": {int(k): float(v) for k, v in best["weights_by_size"].items()},
            "min_count": int(best["min_count"]),
            "temperature": float(best["temperature"]),
            "global_weight": float(best["global_weight"]),
            "combo_mode": data.get("combo_mode", args.combo_mode),
            "combo_sizes": data.get("combo_sizes", parse_int_list(args.combo_sizes)),
        }
    sizes = parse_int_list(args.combo_sizes)
    weights = parse_float_list(args.size_weights)
    return {
        "alpha": float(args.alpha),
        "weights_by_size": {size: (weights[idx] if idx < len(weights) else 0.0) for idx, size in enumerate(sizes)},
        "min_count": int(args.min_count),
        "temperature": float(args.temperature),
        "global_weight": float(args.global_weight),
        "combo_mode": args.combo_mode,
        "combo_sizes": sizes,
    }


def run_predict(args):
    """用全组合 EB 生成候选 A2"""
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    candidate_items = sorted(item_df["iid"].astype(str).tolist())
    user_cols = profile_columns(user_df, args.user_cols)
    params = load_best_params(args)
    specs = make_specs(user_cols, params["combo_mode"], params["combo_sizes"])
    alphas_by_size = {size: params["alpha"] for size in params["combo_sizes"]}
    global_counter, stats, group_counts, user_lookup = build_combo_stats(
        train_df, user_df, user_cols, specs
    )
    base_df = pd.read_csv(args.base_a2) if args.base_a2 else None
    if base_df is not None and base_df["uid"].astype(str).tolist() != test_df["uid"].astype(str).tolist():
        raise ValueError("base_a2 的 uid 顺序和 test.csv 不一致")

    replace_buckets = set(item.strip() for item in args.replace_buckets.split(",") if item.strip())
    out_rows = []
    drift_rows = defaultdict(list)
    for idx, row in test_df.iterrows():
        uid = str(row["uid"])
        bucket = bucket_seq_len(len(parse_seq(row.get(args.seq_col))))
        dist = combo_scores_for_user(
            uid=uid,
            user_lookup=user_lookup,
            specs=specs,
            global_counter=global_counter,
            stats=stats,
            group_counts=group_counts,
            alphas_by_size=alphas_by_size,
            weights_by_size=params["weights_by_size"],
            min_count=params["min_count"],
            global_weight=params["global_weight"],
        )
        eb_pred = rank_from_scores(
            dist,
            candidate_items,
            global_counter,
            topk=max(args.topk, args.eb_pool_size),
            temperature=params["temperature"],
        )
        if base_df is None:
            base_pred = []
            pred = eb_pred[:args.topk]
        else:
            base_pred = parse_prediction(base_df.iloc[idx]["prediction"])
            if bucket in replace_buckets:
                if args.eb_weight >= 999:
                    pred = eb_pred[:args.topk]
                else:
                    pred = rrf_fuse(base_pred, eb_pred, args.eb_weight, args.rrf_k, args.topk)
            else:
                pred = base_pred[:args.topk]
        out_rows.append({"uid": uid, "prediction": ",".join(pred[:args.topk])})
        if base_df is not None:
            drift_rows[bucket].append({
                "changed": float(base_pred[:args.topk] != pred[:args.topk]),
                "top1_changed": float(base_pred[:1] != pred[:1]),
                "overlap": overlap_ratio(base_pred, pred, args.topk),
                "replaced": float(bucket in replace_buckets),
            })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.output_path, index=False)

    def avg(rows: Sequence[Mapping], key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    print("=" * 100)
    print("A2 全组合 EB 推理完成")
    print("=" * 100)
    print(f"output={args.output_path}")
    print(f"params={params}")
    print(f"replace_buckets={sorted(replace_buckets)}, eb_weight={args.eb_weight}, rrf_k={args.rrf_k}")
    if base_df is not None:
        all_rows = [item for rows in drift_rows.values() for item in rows]
        print(
            f"总体 changed={avg(all_rows, 'changed'):.4%}, "
            f"top1_changed={avg(all_rows, 'top1_changed'):.4%}, "
            f"overlap={avg(all_rows, 'overlap'):.4%}"
        )
        for bucket in BUCKET_ORDER:
            rows = drift_rows.get(bucket, [])
            print(
                f"{bucket:<8} n={len(rows):<5} replaced={avg(rows, 'replaced'):.4%} "
                f"changed={avg(rows, 'changed'):.4%} "
                f"top1_changed={avg(rows, 'top1_changed'):.4%} "
                f"overlap={avg(rows, 'overlap'):.4%}"
            )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A2 冷启动全组合 EB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_path", default="data/rec_data")
    common.add_argument("--user_cols", default="auto")
    common.add_argument("--topk", type=int, default=10)
    common.add_argument("--combo_mode", default="all", choices=["prefix", "all"])
    common.add_argument("--combo_sizes", default="1,2,3")

    search = subparsers.add_parser("search", parents=[common])
    search.add_argument("--val_ratio", type=float, default=0.1)
    search.add_argument("--split_seeds", default="42,777,2024,2026,3407")
    search.add_argument("--max_val_samples", type=int, default=4000)
    search.add_argument("--alpha_values", default="50,100,200,400")
    search.add_argument("--size_weight_grid", default="0.2,0.5,1.0;0.1,0.3,1.0;0.1,0.5,0.5")
    search.add_argument("--min_counts", default="10,20,50")
    search.add_argument("--global_weights", default="0.5,1.0")
    search.add_argument("--temperatures", default="1.0")
    search.add_argument("--output_json", default="")

    predict = subparsers.add_parser("predict", parents=[common])
    predict.add_argument("--base_a2", default="")
    predict.add_argument("--seq_col", default="item_seq_raw")
    predict.add_argument("--replace_buckets", default="len=0")
    predict.add_argument("--search_json", default="")
    predict.add_argument("--alpha", type=float, default=200.0)
    predict.add_argument("--size_weights", default="0.2,0.5,1.0")
    predict.add_argument("--min_count", type=int, default=20)
    predict.add_argument("--global_weight", type=float, default=1.0)
    predict.add_argument("--temperature", type=float, default=1.0)
    predict.add_argument("--eb_weight", type=float, default=1.0)
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
