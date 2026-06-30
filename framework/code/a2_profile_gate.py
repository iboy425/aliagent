"""A2按用户画像分群选择候选推荐

Exp045 的多 seed A2 在离线指标更强，但整体线上变差；Exp063/066 说明
只替换空历史用户有效。这个工具用于更细粒度地回答：

- 哪些历史长度桶、哪些用户画像分群适合使用 alt 推荐？
- 这些分群在多个验证切分上是否稳定？

注意：本工具只用训练集切分出的验证目标做审计，正式预测只读取
`base_a2` / `alt_a2` 两个提交文件，不读取任何测试标签。
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from a2_bucket_blend import parse_prediction
from a2_feature_ranker import (
    BUCKET_ORDER,
    apply_fixed_test_like_truncation,
    build_rule_context,
    bucket_seq_len,
    choose_seq_col,
    load_rankers,
    rank_with_fusion,
    score_dataframe_with_models,
    split_train_val,
    test_like_lengths,
)
from rec_heuristics import parse_seq


def ndcg_one(pred: Sequence[str], target: str, topk: int) -> float:
    """计算单样本 NDCG@K"""
    for rank, item in enumerate(pred[:topk], start=1):
        if item == target:
            return float(1.0 / np.log2(rank + 1))
    return 0.0


def parse_csv_list(value: str) -> List[str]:
    """解析逗号分隔字符串"""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_seed_list(value: str) -> List[int]:
    """解析随机种子列表"""
    return [int(item) for item in parse_csv_list(value)]


def profile_specs(user_cols: Sequence[str], spec_arg: str) -> List[Tuple[str, ...]]:
    """解析画像分群规格"""
    specs = []
    for part in parse_csv_list(spec_arg):
        if part == "single":
            specs.extend((col,) for col in user_cols)
        elif part.startswith("prefix"):
            size = int(part.replace("prefix", ""))
            if 1 <= size <= len(user_cols):
                specs.append(tuple(user_cols[:size]))
        elif "+" in part:
            cols = tuple(col.strip() for col in part.split("+") if col.strip())
            if cols:
                specs.append(cols)
        else:
            specs.append((part,))

    out = []
    seen = set()
    for spec in specs:
        if all(col in user_cols for col in spec) and spec not in seen:
            out.append(spec)
            seen.add(spec)
    return out


def make_key(row: pd.Series, bucket: str, spec: Sequence[str]) -> str:
    """生成分群 key"""
    values = []
    for col in spec:
        value = row.get(col)
        if pd.isna(value):
            return ""
        values.append(str(value))
    return f"{bucket}||{'+'.join(spec)}||{'|'.join(values)}"


def parse_key(key: str) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """解析分群 key"""
    bucket, cols, values = key.split("||", 2)
    return bucket, tuple(cols.split("+")), tuple(values.split("|"))


def iter_user_keys(
    row: pd.Series,
    bucket: str,
    specs: Sequence[Tuple[str, ...]],
) -> Iterable[str]:
    """生成某个用户所属的所有候选分群"""
    for spec in specs:
        key = make_key(row, bucket, spec)
        if key:
            yield key


def summarize_bucket(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """汇总一组样本的指标"""
    if not rows:
        return {"samples": 0, "base_ndcg": 0.0, "alt_ndcg": 0.0, "gain": 0.0}
    base = float(np.mean([row["base_ndcg"] for row in rows]))
    alt = float(np.mean([row["alt_ndcg"] for row in rows]))
    return {
        "samples": len(rows),
        "base_ndcg": base,
        "alt_ndcg": alt,
        "gain": alt - base,
    }


def collect_validation_rows(args) -> Tuple[List[Dict], List[str], List[Tuple[str, ...]]]:
    """生成多个验证切分上的 base/alt 对比样本"""
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_models, base_bundle = load_rankers(args.base_checkpoint, device)
    alt_models, alt_bundle = load_rankers(args.alt_checkpoint, device)
    if base_bundle.target_items != alt_bundle.target_items:
        raise ValueError("base/alt checkpoint 的 target_items 不一致")

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    seq_col = choose_seq_col(train_df, args.seq_col)
    test_seq_col = choose_seq_col(test_df, args.seq_col)
    user_cols = [col for col in user_df.columns if col != "uid"]
    specs = profile_specs(user_cols, args.profile_specs)
    selected_buckets = set(parse_csv_list(args.buckets))

    all_rows = []
    for split_seed in parse_seed_list(args.split_seeds):
        fit_df, val_df = split_train_val(train_df, args.val_ratio, split_seed)
        if args.test_like_val:
            lengths = test_like_lengths(test_df, test_seq_col)
            val_df = apply_fixed_test_like_truncation(val_df, lengths, seq_col, split_seed)
        if args.max_val_samples > 0 and len(val_df) > args.max_val_samples:
            val_df = val_df.sample(n=args.max_val_samples, random_state=split_seed).reset_index(drop=True)

        rule_context = build_rule_context(args, fit_df, user_df, item_df, seq_col)
        user_lookup = user_df.set_index("uid")
        base_scores = score_dataframe_with_models(
            val_df, base_models, base_bundle, user_lookup, seq_col,
            args.max_len, args.batch_size, device,
        )
        alt_scores = score_dataframe_with_models(
            val_df, alt_models, alt_bundle, user_lookup, seq_col,
            args.max_len, args.batch_size, device,
        )
        base_preds = rank_with_fusion(
            val_df, base_scores, seq_col, rule_context, args, args.base_model_weight,
        )
        alt_preds = rank_with_fusion(
            val_df, alt_scores, seq_col, rule_context, args, args.alt_model_weight,
        )

        user_feature_df = val_df[["uid", "target_iid", seq_col]].merge(user_df, on="uid", how="left")
        for idx, row in user_feature_df.iterrows():
            bucket = bucket_seq_len(len(parse_seq(row.get(seq_col))))
            if selected_buckets and bucket not in selected_buckets:
                continue
            target = str(row["target_iid"])
            all_rows.append({
                "split_seed": split_seed,
                "uid": str(row["uid"]),
                "bucket": bucket,
                "base_ndcg": ndcg_one(base_preds[idx], target, args.topk),
                "alt_ndcg": ndcg_one(alt_preds[idx], target, args.topk),
                "keys": list(iter_user_keys(row, bucket, specs)),
            })
        print(f"[split={split_seed}] rows={len(all_rows)}")
    return all_rows, user_cols, specs


def select_groups(args, rows: Sequence[Mapping]) -> Dict:
    """根据验证样本选择稳定正收益分群"""
    group_rows = defaultdict(list)
    bucket_rows = defaultdict(list)
    for row in rows:
        bucket_rows[row["bucket"]].append(row)
        for key in row["keys"]:
            group_rows[key].append(row)

    candidates = []
    for key, items in group_rows.items():
        if len(items) < args.min_samples:
            continue
        base = float(np.mean([item["base_ndcg"] for item in items]))
        alt = float(np.mean([item["alt_ndcg"] for item in items]))
        gain = alt - base
        if gain < args.min_gain:
            continue
        split_gains = {}
        for split_seed in sorted({item["split_seed"] for item in items}):
            split_items = [item for item in items if item["split_seed"] == split_seed]
            if len(split_items) < args.min_split_samples:
                continue
            split_base = float(np.mean([item["base_ndcg"] for item in split_items]))
            split_alt = float(np.mean([item["alt_ndcg"] for item in split_items]))
            split_gains[str(split_seed)] = split_alt - split_base
        positive_splits = sum(1 for val in split_gains.values() if val > 0)
        if positive_splits < args.min_positive_splits:
            continue
        bucket, cols, values = parse_key(key)
        candidates.append({
            "key": key,
            "bucket": bucket,
            "cols": list(cols),
            "values": list(values),
            "samples": len(items),
            "base_ndcg": base,
            "alt_ndcg": alt,
            "gain": gain,
            "split_gains": split_gains,
            "positive_splits": positive_splits,
        })

    candidates.sort(key=lambda item: (item["gain"], item["samples"]), reverse=True)
    selected = candidates[:args.max_groups]
    selected_keys = {item["key"] for item in selected}

    gated_rows = []
    for row in rows:
        use_alt = any(key in selected_keys for key in row["keys"])
        gated_rows.append({
            "bucket": row["bucket"],
            "base_ndcg": row["base_ndcg"],
            "alt_ndcg": row["alt_ndcg"],
            "gated_ndcg": row["alt_ndcg"] if use_alt else row["base_ndcg"],
            "use_alt": float(use_alt),
        })

    bucket_summary = {}
    for bucket in BUCKET_ORDER:
        items = [row for row in gated_rows if row["bucket"] == bucket]
        if not items:
            continue
        base = float(np.mean([row["base_ndcg"] for row in items]))
        alt = float(np.mean([row["alt_ndcg"] for row in items]))
        gated = float(np.mean([row["gated_ndcg"] for row in items]))
        bucket_summary[bucket] = {
            "samples": len(items),
            "base_ndcg": base,
            "alt_ndcg": alt,
            "gated_ndcg": gated,
            "gain_vs_base": gated - base,
            "use_alt_rate": float(np.mean([row["use_alt"] for row in items])),
        }

    base_all = float(np.mean([row["base_ndcg"] for row in gated_rows])) if gated_rows else 0.0
    alt_all = float(np.mean([row["alt_ndcg"] for row in gated_rows])) if gated_rows else 0.0
    gated_all = float(np.mean([row["gated_ndcg"] for row in gated_rows])) if gated_rows else 0.0
    return {
        "policy": {
            "selected_groups": selected,
            "selected_keys": list(selected_keys),
        },
        "summary": {
            "samples": len(gated_rows),
            "base_ndcg": base_all,
            "alt_ndcg": alt_all,
            "gated_ndcg": gated_all,
            "gain_vs_base": gated_all - base_all,
            "gain_vs_alt": gated_all - alt_all,
            "bucket_summary": bucket_summary,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
        },
        "candidates": candidates,
    }


def run_audit(args):
    """审计分群 gate"""
    rows, user_cols, specs = collect_validation_rows(args)
    result = select_groups(args, rows)
    result["config"] = {
        "user_cols": user_cols,
        "specs": [list(spec) for spec in specs],
        "buckets": parse_csv_list(args.buckets),
        "base_checkpoint": args.base_checkpoint,
        "alt_checkpoint": args.alt_checkpoint,
        "base_model_weight": args.base_model_weight,
        "alt_model_weight": args.alt_model_weight,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A2 画像分群 gate 审计")
    print("=" * 100)
    summary = result["summary"]
    print(
        f"base={summary['base_ndcg']:.6f}\talt={summary['alt_ndcg']:.6f}\t"
        f"gated={summary['gated_ndcg']:.6f}\tgain={summary['gain_vs_base']:+.6f}"
    )
    print(f"候选分群={summary['candidate_count']}, 选中分群={summary['selected_count']}")
    for bucket, item in summary["bucket_summary"].items():
        print(
            f"{bucket}\tn={item['samples']}\tbase={item['base_ndcg']:.6f}\t"
            f"alt={item['alt_ndcg']:.6f}\tgated={item['gated_ndcg']:.6f}\t"
            f"gain={item['gain_vs_base']:+.6f}\tuse_alt={item['use_alt_rate']:.2%}"
        )
    print("\nTop selected groups:")
    for item in result["policy"]["selected_groups"][:20]:
        print(
            f"gain={item['gain']:+.6f}\tn={item['samples']}\t"
            f"pos={item['positive_splits']}\t{item['key']}"
        )
    if args.output_json:
        print(f"\n结果已保存: {args.output_json}")


def match_policy(row: pd.Series, bucket: str, selected_groups: Sequence[Mapping]) -> bool:
    """判断测试用户是否命中策略"""
    for group in selected_groups:
        if group["bucket"] != bucket:
            continue
        ok = True
        for col, val in zip(group["cols"], group["values"]):
            cur = row.get(col)
            if pd.isna(cur) or str(cur) != str(val):
                ok = False
                break
        if ok:
            return True
    return False


def overlap_ratio(items_a: Sequence[str], items_b: Sequence[str], topk: int) -> float:
    """计算 TopK 重合率"""
    return len(set(items_a[:topk]) & set(items_b[:topk])) / max(topk, 1)


def run_predict(args):
    """根据审计策略生成 A2"""
    data = json.load(open(args.policy_json, encoding="utf-8"))
    selected_groups = data["policy"]["selected_groups"]
    base_df = pd.read_csv(args.base_a2)
    alt_df = pd.read_csv(args.alt_a2)
    test_df = pd.read_csv(args.test_csv)
    user_df = pd.read_csv(args.user_csv)
    merged = test_df.merge(user_df, on="uid", how="left")

    if base_df["uid"].astype(str).tolist() != alt_df["uid"].astype(str).tolist():
        raise ValueError("base_a2 和 alt_a2 的 uid 顺序不一致")
    if base_df["uid"].astype(str).tolist() != merged["uid"].astype(str).tolist():
        raise ValueError("A2 uid 顺序和 test/user 合并结果不一致")

    rows = []
    stats = defaultdict(list)
    for idx, row in merged.iterrows():
        bucket = bucket_seq_len(len(parse_seq(row.get(args.seq_col))))
        base_pred = parse_prediction(base_df.iloc[idx]["prediction"])
        alt_pred = parse_prediction(alt_df.iloc[idx]["prediction"])
        use_alt = match_policy(row, bucket, selected_groups)
        pred = alt_pred if use_alt else base_pred
        rows.append({"uid": str(row["uid"]), "prediction": ",".join(pred[:args.topk])})
        stats[bucket].append({
            "use_alt": float(use_alt),
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
    print("A2 画像分群 gate 推理完成")
    print("=" * 100)
    print(f"policy={args.policy_json}")
    print(f"output={args.output_path}")
    print(f"selected_groups={len(selected_groups)}")
    print(
        f"总体 use_alt={avg(all_rows, 'use_alt'):.4%}, "
        f"changed={avg(all_rows, 'changed'):.4%}, "
        f"top1_changed={avg(all_rows, 'top1_changed'):.4%}, "
        f"overlap={avg(all_rows, 'overlap'):.4%}"
    )
    for bucket in BUCKET_ORDER:
        items = stats.get(bucket, [])
        print(
            f"{bucket:<8} n={len(items):<5} use_alt={avg(items, 'use_alt'):.4%} "
            f"changed={avg(items, 'changed'):.4%} "
            f"top1_changed={avg(items, 'top1_changed'):.4%} "
            f"overlap={avg(items, 'overlap'):.4%}"
        )


def add_fusion_args(parser):
    """添加和 a2_feature_ranker 融合一致的参数"""
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A2画像分群gate")
    sub = parser.add_subparsers(dest="mode", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--data_path", default="data/rec_data")
    audit.add_argument("--base_checkpoint", required=True)
    audit.add_argument("--alt_checkpoint", required=True)
    audit.add_argument("--base_model_weight", type=float, default=1.0)
    audit.add_argument("--alt_model_weight", type=float, default=22.0)
    audit.add_argument("--split_seeds", default="42,777,2024")
    audit.add_argument("--val_ratio", type=float, default=0.1)
    audit.add_argument("--test_like_val", action="store_true")
    audit.add_argument("--max_val_samples", type=int, default=4000)
    audit.add_argument("--buckets", default="len=1,len=2-3,len=4-10,len>10")
    audit.add_argument("--profile_specs", default="single,prefix2,prefix3")
    audit.add_argument("--min_samples", type=int, default=80)
    audit.add_argument("--min_split_samples", type=int, default=20)
    audit.add_argument("--min_positive_splits", type=int, default=2)
    audit.add_argument("--min_gain", type=float, default=0.01)
    audit.add_argument("--max_groups", type=int, default=20)
    audit.add_argument("--output_json", default="")
    add_fusion_args(audit)

    predict = sub.add_parser("predict")
    predict.add_argument("--policy_json", required=True)
    predict.add_argument("--base_a2", required=True)
    predict.add_argument("--alt_a2", required=True)
    predict.add_argument("--test_csv", default="data/rec_data/test.csv")
    predict.add_argument("--user_csv", default="data/rec_data/user.csv")
    predict.add_argument("--seq_col", default="item_seq_raw")
    predict.add_argument("--topk", type=int, default=10)
    predict.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    if args.mode == "audit":
        run_audit(args)
    elif args.mode == "predict":
        run_predict(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
