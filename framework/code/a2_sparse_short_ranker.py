"""A2短历史稀疏特征排序器

当前A2的主要难点是训练集和测试集历史长度分布严重错位：
- train.csv 大多数样本有很长历史；
- test.csv 大量用户是空历史、1个历史或2-3个历史。

神经序列模型擅长利用长历史，但短历史场景下更依赖用户画像、最后几个
item、item类目和全局target先验。本脚本把这些离散信号转成稀疏特征，
训练 ComplementNB / MultinomialNB / SGDClassifier，作为和神经模型互补
的候选召回器。
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder

from a2_feature_ranker import (
    BUCKET_ORDER,
    apply_fixed_test_like_truncation,
    bucket_seq_len,
    choose_seq_col,
    compute_bucket_weights,
    evaluate_prediction_lists,
    split_train_val,
    test_like_lengths,
    truncate_seq_value,
)
from a2_bucket_blend import parse_prediction
from rec_heuristics import parse_item_counts, parse_seq


def parse_csv_list(value: str) -> List[str]:
    """解析逗号分隔字符串"""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数"""
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点数"""
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def value_token(value) -> str:
    """把缺失值和类别值统一转为字符串token"""
    if pd.isna(value):
        return "__NA__"
    return str(value)


def normalize_counter(counter: Counter) -> Dict[str, float]:
    """把计数归一化到0-1，避免大计数压倒其他稀疏特征"""
    if not counter:
        return {}
    max_count = max(counter.values())
    if max_count <= 0:
        return {}
    return {key: float(value) / float(max_count) for key, value in counter.items()}


def make_user_lookup(user_df: pd.DataFrame) -> pd.DataFrame:
    """构建用户画像索引"""
    out = user_df.copy()
    out["uid"] = out["uid"].astype(str)
    return out.set_index("uid")


def make_item_lookup(item_df: pd.DataFrame) -> pd.DataFrame:
    """构建物品特征索引"""
    out = item_df.copy()
    out["iid"] = out["iid"].astype(str)
    return out.set_index("iid")


def user_columns(user_df: pd.DataFrame, requested: str) -> List[str]:
    """解析用户画像列"""
    if requested == "auto":
        return [col for col in user_df.columns if col != "uid"]
    return parse_csv_list(requested)


def item_columns(item_df: pd.DataFrame, requested: str) -> List[str]:
    """解析物品特征列"""
    if requested == "auto":
        return [col for col in item_df.columns if col != "iid"]
    return parse_csv_list(requested)


def add_user_features(
    feats: Dict[str, float],
    uid: str,
    bucket: str,
    user_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    combo_sizes: Sequence[int],
):
    """加入用户画像及前缀组合特征"""
    if uid not in user_lookup.index:
        feats["uid_missing"] = 1.0
        return

    row = user_lookup.loc[uid]
    values = []
    for col in user_cols:
        value = value_token(row.get(col))
        values.append((col, value))
        feats[f"u:{col}={value}"] = 1.0
        feats[f"b+u:{bucket}|{col}={value}"] = 1.0

    for size in combo_sizes:
        if size <= 1 or size > len(values):
            continue
        combo = "|".join(f"{col}={value}" for col, value in values[:size])
        feats[f"u_prefix{size}:{combo}"] = 1.0
        feats[f"b+u_prefix{size}:{bucket}|{combo}"] = 1.0


def add_item_position_features(
    feats: Dict[str, float],
    seq: Sequence[str],
    bucket: str,
    item_lookup: pd.DataFrame,
    item_cols: Sequence[str],
    max_pos: int,
    user_values: Mapping[str, str],
    pair_user_item_cols: Sequence[str],
):
    """加入最近若干位置的item及item侧类别特征"""
    recent = list(reversed(seq[-max_pos:])) if max_pos > 0 else []
    for pos, item in enumerate(recent, start=1):
        weight = 1.0 / float(pos)
        feats[f"p{pos}:item={item}"] = weight
        feats[f"b+p{pos}:item={bucket}|{item}"] = weight

        if pos <= 2:
            for col in pair_user_item_cols:
                if col in user_values:
                    feats[f"uitem:{col}={user_values[col]}|p{pos}={item}"] = weight

        if item not in item_lookup.index:
            continue
        item_row = item_lookup.loc[item]
        for col in item_cols:
            value = value_token(item_row.get(col))
            feats[f"p{pos}:{col}={value}"] = weight
            feats[f"b+p{pos}:{col}={bucket}|{value}"] = weight
            if pos <= 2:
                for ucol in pair_user_item_cols:
                    if ucol in user_values:
                        feats[f"uicat:{ucol}={user_values[ucol]}|p{pos}:{col}={value}"] = weight


def add_recent_bag_features(
    feats: Dict[str, float],
    seq: Sequence[str],
    item_lookup: pd.DataFrame,
    item_cols: Sequence[str],
    recent_bag_n: int,
    decay: float,
):
    """加入最近历史item的无序bag特征"""
    if recent_bag_n <= 0:
        return
    seen = set()
    for age, item in enumerate(reversed(seq[-recent_bag_n:])):
        if item in seen:
            continue
        seen.add(item)
        weight = float(decay) ** age
        feats[f"hist:item={item}"] = max(feats.get(f"hist:item={item}", 0.0), weight)
        if item not in item_lookup.index:
            continue
        item_row = item_lookup.loc[item]
        for col in item_cols:
            value = value_token(item_row.get(col))
            key = f"hist:{col}={value}"
            feats[key] = max(feats.get(key, 0.0), weight)


def add_history_count_features(
    feats: Dict[str, float],
    count_value,
    topn: int,
):
    """加入当前用户历史高频item特征"""
    if topn <= 0:
        return
    for item, score in sorted(normalize_counter(parse_item_counts(count_value)).items(), key=lambda kv: -kv[1])[:topn]:
        feats[f"cnt:item={item}"] = score


def extract_features(
    row: pd.Series,
    seq_col: str,
    user_lookup: pd.DataFrame,
    item_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    item_cols: Sequence[str],
    combo_sizes: Sequence[int],
    max_pos: int,
    recent_bag_n: int,
    bag_decay: float,
    count_topn: int,
    pair_user_item_cols: Sequence[str],
) -> Dict[str, float]:
    """把一行样本转为稀疏特征字典"""
    uid = str(row["uid"])
    seq = parse_seq(row.get(seq_col))
    bucket = bucket_seq_len(len(seq))
    feats: Dict[str, float] = {
        f"bucket={bucket}": 1.0,
        f"len_exact={min(len(seq), 20)}": 1.0,
    }

    user_values = {}
    if uid in user_lookup.index:
        user_row = user_lookup.loc[uid]
        for col in user_cols:
            user_values[col] = value_token(user_row.get(col))

    add_user_features(feats, uid, bucket, user_lookup, user_cols, combo_sizes)
    add_item_position_features(
        feats,
        seq,
        bucket,
        item_lookup,
        item_cols,
        max_pos,
        user_values,
        pair_user_item_cols,
    )
    add_recent_bag_features(feats, seq, item_lookup, item_cols, recent_bag_n, bag_decay)
    add_history_count_features(feats, row.get("item_seq_counts"), count_topn)
    return feats


def truncate_history_counts(seq_value) -> str:
    """根据截断后的序列重建 item_seq_counts"""
    seq = parse_seq(seq_value)
    if not seq:
        return ""
    counter = Counter(seq)
    ordered = []
    for item in seq:
        if item not in ordered:
            ordered.append(item)
    return ",".join(f"{item}:{counter[item]}" for item in ordered)


def make_augmented_train(
    train_df: pd.DataFrame,
    test_lengths: np.ndarray,
    seq_col: str,
    seed: int,
    augment_repeats: int,
    include_original: bool,
) -> pd.DataFrame:
    """用测试集长度分布对训练集做短历史增强"""
    frames = []
    if include_original:
        frames.append(train_df.copy())
    if augment_repeats <= 0:
        return pd.concat(frames, ignore_index=True) if frames else train_df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    lengths = test_lengths if len(test_lengths) else np.array([0], dtype=np.int64)
    for repeat in range(augment_repeats):
        keep_lengths = rng.choice(lengths, size=len(train_df), replace=True)
        part = train_df.copy()
        part[seq_col] = [
            truncate_seq_value(value, int(keep_len))
            for value, keep_len in zip(part[seq_col], keep_lengths)
        ]
        if "item_seq_counts" in part.columns:
            part["item_seq_counts"] = [truncate_history_counts(value) for value in part[seq_col]]
        part["_augment_repeat"] = repeat + 1
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def build_feature_matrix(
    df: pd.DataFrame,
    args,
    seq_col: str,
    user_lookup: pd.DataFrame,
    item_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    item_cols: Sequence[str],
):
    """把DataFrame批量转为稀疏特征列表"""
    combo_sizes = parse_int_list(args.user_combo_sizes)
    pair_cols = parse_csv_list(args.pair_user_item_cols)
    return [
        extract_features(
            row,
            seq_col,
            user_lookup,
            item_lookup,
            user_cols,
            item_cols,
            combo_sizes,
            args.max_pos,
            args.recent_bag_n,
            args.bag_decay,
            args.count_topn,
            pair_cols,
        )
        for _, row in df.iterrows()
    ]


def make_model(model_type: str, alpha: float, max_iter: int, seed: int):
    """创建稀疏分类器"""
    if model_type == "cnb":
        return ComplementNB(alpha=alpha)
    if model_type == "mnb":
        return MultinomialNB(alpha=alpha)
    if model_type == "sgd":
        return SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol=1e-4,
            random_state=seed,
            n_jobs=-1,
            class_weight=None,
        )
    raise ValueError(f"未知模型类型: {model_type}")


def fit_sparse_model(
    train_df: pd.DataFrame,
    args,
    seq_col: str,
    user_lookup: pd.DataFrame,
    item_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    item_cols: Sequence[str],
    model_type: str,
    alpha: float,
    seed: int,
):
    """训练稀疏特征分类器"""
    feature_rows = build_feature_matrix(
        train_df, args, seq_col, user_lookup, item_lookup, user_cols, item_cols
    )
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(feature_rows)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["target_iid"].astype(str).tolist())
    model = make_model(model_type, alpha, args.max_iter, seed)
    model.fit(x_train, y_train)
    return model, vectorizer, label_encoder


def predict_lists(
    df: pd.DataFrame,
    args,
    seq_col: str,
    model,
    vectorizer: DictVectorizer,
    label_encoder: LabelEncoder,
    user_lookup: pd.DataFrame,
    item_lookup: pd.DataFrame,
    user_cols: Sequence[str],
    item_cols: Sequence[str],
    global_pop: Counter,
) -> List[List[str]]:
    """输出每行TopK预测列表"""
    feature_rows = build_feature_matrix(
        df, args, seq_col, user_lookup, item_lookup, user_cols, item_cols
    )
    x = vectorizer.transform(feature_rows)
    return predict_lists_from_matrix(x, args, model, label_encoder, global_pop)


def predict_lists_from_matrix(
    x,
    args,
    model,
    label_encoder: LabelEncoder,
    global_pop: Counter,
) -> List[List[str]]:
    """根据已经向量化的矩阵输出TopK预测列表"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
    else:
        scores = model.decision_function(x)
        scores = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)

    class_labels = label_encoder.inverse_transform(model.classes_.astype(int))
    fallback = [item for item, _ in global_pop.most_common()]
    preds = []
    topn = min(args.topk + args.extra_topn, proba.shape[1])
    for row_scores in proba:
        order = np.argpartition(-row_scores, kth=topn - 1)[:topn]
        order = sorted(order, key=lambda idx: (-row_scores[idx], class_labels[idx]))
        seen = set()
        items = []
        for idx in order:
            item = class_labels[idx]
            if item not in seen:
                items.append(item)
                seen.add(item)
            if len(items) >= args.topk:
                break
        for item in fallback:
            if item not in seen:
                items.append(item)
                seen.add(item)
            if len(items) >= args.topk:
                break
        preds.append(items[:args.topk])
    return preds


def summarize_bucket_metrics(metrics: Mapping) -> str:
    """压缩显示分桶NDCG"""
    parts = []
    for bucket in BUCKET_ORDER:
        item = metrics.get("buckets", {}).get(bucket, {})
        parts.append(f"{bucket}:{item.get('ndcg', 0.0):.4f}/{item.get('samples', 0)}")
    return " ".join(parts)


def audit(args):
    """多split审计稀疏短历史模型"""
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    seq_col = choose_seq_col(train_df, args.seq_col)
    test_seq_col = choose_seq_col(test_df, args.seq_col)
    user_lookup = make_user_lookup(user_df)
    item_lookup = make_item_lookup(item_df)
    user_cols = user_columns(user_df, args.user_cols)
    item_cols = item_columns(item_df, args.item_cols)
    test_lengths = test_like_lengths(test_df, test_seq_col)
    bucket_weights = compute_bucket_weights(test_df, test_seq_col)

    configs = []
    for model_type in parse_csv_list(args.model_types):
        alphas = parse_float_list(args.sgd_alphas if model_type == "sgd" else args.nb_alphas)
        for alpha in alphas:
            for augment_repeats in parse_int_list(args.augment_repeats_grid):
                configs.append((model_type, alpha, augment_repeats))

    all_rows = []
    split_details = []
    for split_seed in parse_int_list(args.split_seeds):
        fit_df, val_df = split_train_val(train_df, args.val_ratio, split_seed)
        if args.test_like_val:
            val_df = apply_fixed_test_like_truncation(val_df, test_lengths, seq_col, split_seed)
        if args.max_val_samples > 0 and len(val_df) > args.max_val_samples:
            val_df = val_df.sample(n=args.max_val_samples, random_state=split_seed).reset_index(drop=True)

        global_pop = Counter(fit_df["target_iid"].astype(str).tolist())
        targets = val_df["target_iid"].astype(str).tolist()
        val_feature_rows = build_feature_matrix(
            val_df, args, seq_col, user_lookup, item_lookup, user_cols, item_cols
        )
        print(f"[split={split_seed}] fit={len(fit_df)} val={len(val_df)} configs={len(configs)}", flush=True)

        split_rows = []
        configs_by_aug = defaultdict(list)
        for model_type, alpha, augment_repeats in configs:
            configs_by_aug[augment_repeats].append((model_type, alpha))

        for augment_repeats, aug_configs in sorted(configs_by_aug.items()):
            aug_df = make_augmented_train(
                fit_df,
                test_lengths,
                seq_col,
                seed=split_seed + augment_repeats * 17,
                augment_repeats=augment_repeats,
                include_original=args.include_original_train,
            )
            train_feature_rows = build_feature_matrix(
                aug_df, args, seq_col, user_lookup, item_lookup, user_cols, item_cols
            )
            vectorizer = DictVectorizer(sparse=True)
            x_train = vectorizer.fit_transform(train_feature_rows)
            x_val = vectorizer.transform(val_feature_rows)
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(aug_df["target_iid"].astype(str).tolist())
            print(
                f"  [aug={augment_repeats}] train_rows={len(aug_df)} "
                f"features={len(vectorizer.feature_names_)} configs={len(aug_configs)}",
                flush=True,
            )
            for model_type, alpha in aug_configs:
                model = make_model(model_type, alpha, args.max_iter, split_seed)
                model.fit(x_train, y_train)
                preds = predict_lists_from_matrix(x_val, args, model, label_encoder, global_pop)
                metrics = evaluate_prediction_lists(preds, targets, val_df, seq_col, args.topk, bucket_weights)
                row = {
                    "split_seed": split_seed,
                    "model_type": model_type,
                    "alpha": alpha,
                    "augment_repeats": augment_repeats,
                    "include_original_train": args.include_original_train,
                    "weighted_ndcg": metrics["weighted_ndcg"],
                    "ndcg": metrics["ndcg"],
                    "hit": metrics["hit"],
                    "mrr": metrics["mrr"],
                    "metrics": metrics,
                    "num_features": int(len(vectorizer.feature_names_)),
                    "train_rows": int(len(aug_df)),
                }
                split_rows.append(row)
                all_rows.append(row)
                print(
                    f"    {model_type:<3} alpha={alpha:<7g} "
                    f"weighted={metrics['weighted_ndcg']:.6f} ndcg={metrics['ndcg']:.6f} "
                    f"{summarize_bucket_metrics(metrics)}",
                    flush=True,
                )
        split_details.append({"split_seed": split_seed, "rows": split_rows})

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["model_type"], row["alpha"], row["augment_repeats"])].append(row)

    summary = []
    for (model_type, alpha, augment_repeats), rows in grouped.items():
        values = [row["weighted_ndcg"] for row in rows]
        summary.append({
            "model_type": model_type,
            "alpha": alpha,
            "augment_repeats": augment_repeats,
            "mean_weighted_ndcg": float(np.mean(values)),
            "min_weighted_ndcg": float(np.min(values)),
            "max_weighted_ndcg": float(np.max(values)),
            "std_weighted_ndcg": float(np.std(values)),
            "mean_ndcg": float(np.mean([row["ndcg"] for row in rows])),
            "rows": rows,
        })
    summary.sort(key=lambda item: (item["mean_weighted_ndcg"], item["min_weighted_ndcg"]), reverse=True)
    result = {
        "best": summary[0] if summary else {},
        "summary": summary,
        "split_details": split_details,
        "args": vars(args),
        "bucket_weights": bucket_weights,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("A2 短历史稀疏ranker审计 Top-20")
    print("=" * 100)
    for item in summary[:20]:
        print(
            f"{item['model_type']}\talpha={item['alpha']:<7g}\taug={item['augment_repeats']}\t"
            f"mean={item['mean_weighted_ndcg']:.6f}\tmin={item['min_weighted_ndcg']:.6f}\t"
            f"std={item['std_weighted_ndcg']:.6f}"
        )
    print(f"\n结果已保存: {args.output_json}")


def predict(args):
    """用全部训练集训练并生成A2"""
    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    seq_col = choose_seq_col(train_df, args.seq_col)
    test_seq_col = choose_seq_col(test_df, args.seq_col)
    user_lookup = make_user_lookup(user_df)
    item_lookup = make_item_lookup(item_df)
    user_cols = user_columns(user_df, args.user_cols)
    item_cols = item_columns(item_df, args.item_cols)
    test_lengths = test_like_lengths(test_df, test_seq_col)

    if args.audit_json:
        audit_data = json.load(open(args.audit_json, encoding="utf-8"))
        best = audit_data["best"]
        args.model_type = best["model_type"]
        args.alpha = float(best["alpha"])
        args.augment_repeats = int(best["augment_repeats"])

    train_aug = make_augmented_train(
        train_df,
        test_lengths,
        seq_col,
        seed=args.seed,
        augment_repeats=args.augment_repeats,
        include_original=args.include_original_train,
    )
    global_pop = Counter(train_df["target_iid"].astype(str).tolist())
    print("=" * 100)
    print("A2 短历史稀疏ranker推理")
    print("=" * 100)
    print(
        f"model={args.model_type}, alpha={args.alpha}, aug={args.augment_repeats}, "
        f"train_rows={len(train_aug)}, user_cols={user_cols}, item_cols={item_cols}"
    )

    model, vectorizer, label_encoder = fit_sparse_model(
        train_aug,
        args,
        seq_col,
        user_lookup,
        item_lookup,
        user_cols,
        item_cols,
        args.model_type,
        args.alpha,
        args.seed,
    )
    preds = predict_lists(
        test_df,
        args,
        test_seq_col,
        model,
        vectorizer,
        label_encoder,
        user_lookup,
        item_lookup,
        user_cols,
        item_cols,
        global_pop,
    )
    out_df = pd.DataFrame({
        "uid": test_df["uid"].astype(str).tolist(),
        "prediction": [",".join(items[:args.topk]) for items in preds],
    })
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"A2稀疏ranker预测已保存: {args.output_path}, rows={len(out_df)}")


def merge_lists(base_items: Sequence[str], alt_items: Sequence[str], keep_topn: int, topk: int) -> List[str]:
    """保留base前N个位置，其余用alt/base补齐"""
    result = []
    seen = set()
    for item in base_items[:keep_topn]:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    for item in alt_items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            return result[:topk]
    for item in base_items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            break
    return result[:topk]


def blend(args):
    """把稀疏ranker A2 和稳定A2做保护式融合"""
    base_df = pd.read_csv(args.base_a2)
    alt_df = pd.read_csv(args.alt_a2)
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    if base_df["uid"].astype(str).tolist() != alt_df["uid"].astype(str).tolist():
        raise ValueError("base_a2 和 alt_a2 uid 顺序不一致")
    if base_df["uid"].astype(str).tolist() != test_df["uid"].astype(str).tolist():
        raise ValueError("A2 uid 顺序和 test.csv 不一致")

    selected_buckets = set(parse_csv_list(args.buckets))
    rows = []
    stats = defaultdict(list)
    seq_col = choose_seq_col(test_df, args.seq_col)
    for idx, row in test_df.iterrows():
        bucket = bucket_seq_len(len(parse_seq(row.get(seq_col))))
        base_items = parse_prediction(base_df.iloc[idx]["prediction"])
        alt_items = parse_prediction(alt_df.iloc[idx]["prediction"])
        if bucket in selected_buckets:
            pred = merge_lists(base_items, alt_items, args.keep_topn, args.topk)
        else:
            pred = base_items[:args.topk]
        rows.append({"uid": str(row["uid"]), "prediction": ",".join(pred[:args.topk])})
        stats[bucket].append({
            "changed": float(pred[:args.topk] != base_items[:args.topk]),
            "top1_changed": float(pred[:1] != base_items[:1]),
            "overlap": len(set(pred[:args.topk]) & set(base_items[:args.topk])) / float(args.topk),
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_path, index=False)

    def avg(items, key):
        return float(np.mean([item[key] for item in items])) if items else 0.0

    all_stats = [item for values in stats.values() for item in values]
    print("=" * 100)
    print("A2稀疏ranker保护融合完成")
    print("=" * 100)
    print(f"base={args.base_a2}")
    print(f"alt={args.alt_a2}")
    print(f"keep_topn={args.keep_topn}, buckets={sorted(selected_buckets)}, output={args.output_path}")
    print(
        f"总体 changed={avg(all_stats, 'changed'):.4%}, "
        f"top1_changed={avg(all_stats, 'top1_changed'):.4%}, "
        f"overlap={avg(all_stats, 'overlap'):.4%}"
    )
    for bucket in BUCKET_ORDER:
        items = stats.get(bucket, [])
        print(
            f"{bucket:<8} n={len(items):<5} changed={avg(items, 'changed'):.4%} "
            f"top1_changed={avg(items, 'top1_changed'):.4%} overlap={avg(items, 'overlap'):.4%}"
        )


def add_common_args(parser: argparse.ArgumentParser):
    """添加公共特征参数"""
    parser.add_argument("--data_path", default="data/rec_data")
    parser.add_argument("--seq_col", default="item_seq_raw")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--extra_topn", type=int, default=20,
                        help="从分类器概率中多取一些候选，再去重补齐")
    parser.add_argument("--user_cols", default="auto")
    parser.add_argument("--item_cols", default="auto")
    parser.add_argument("--user_combo_sizes", default="2,3")
    parser.add_argument("--pair_user_item_cols", default="u_cat_01,u_cat_02,u_cat_03,u_cat_04,u_cat_05,u_cat_06,u_cat_07,u_cat_08")
    parser.add_argument("--max_pos", type=int, default=5)
    parser.add_argument("--recent_bag_n", type=int, default=20)
    parser.add_argument("--bag_decay", type=float, default=0.92)
    parser.add_argument("--count_topn", type=int, default=8)
    parser.add_argument("--include_original_train", action="store_true")
    parser.add_argument("--max_iter", type=int, default=80)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器"""
    parser = argparse.ArgumentParser(description="A2短历史稀疏特征排序器")
    sub = parser.add_subparsers(dest="mode", required=True)

    audit_parser = sub.add_parser("audit")
    add_common_args(audit_parser)
    audit_parser.add_argument("--split_seeds", default="42,777,2024")
    audit_parser.add_argument("--val_ratio", type=float, default=0.1)
    audit_parser.add_argument("--test_like_val", action="store_true")
    audit_parser.add_argument("--model_types", default="cnb,mnb")
    audit_parser.add_argument("--nb_alphas", default="0.01,0.03,0.1,0.3,1.0,3.0")
    audit_parser.add_argument("--sgd_alphas", default="0.0001,0.0003,0.001")
    audit_parser.add_argument("--augment_repeats_grid", default="1,2,3")
    audit_parser.add_argument("--max_val_samples", type=int, default=0)
    audit_parser.add_argument("--output_json", required=True)

    pred_parser = sub.add_parser("predict")
    add_common_args(pred_parser)
    pred_parser.add_argument("--audit_json", default="")
    pred_parser.add_argument("--model_type", default="cnb", choices=["cnb", "mnb", "sgd"])
    pred_parser.add_argument("--alpha", type=float, default=0.1)
    pred_parser.add_argument("--augment_repeats", type=int, default=2)
    pred_parser.add_argument("--seed", type=int, default=42)
    pred_parser.add_argument("--output_path", required=True)

    blend_parser = sub.add_parser("blend")
    add_common_args(blend_parser)
    blend_parser.add_argument("--base_a2", required=True)
    blend_parser.add_argument("--alt_a2", required=True)
    blend_parser.add_argument("--keep_topn", type=int, default=1)
    blend_parser.add_argument("--buckets", default="len=2-3,len=4-10,len>10")
    blend_parser.add_argument("--output_path", required=True)

    return parser


def main():
    """主入口"""
    args = build_parser().parse_args()
    if args.mode == "audit":
        audit(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "blend":
        blend(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
