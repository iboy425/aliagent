"""A2推荐启发式召回与重排工具

本模块存放推荐任务中可复用的轻量级规则：
- 解析用户历史序列
- 统计全局热门target
- 统计历史item到target item的共现关系
- 融合模型分数、热门度分数和共现分数
- 控制是否过滤历史item并补齐TopK

这些规则同时服务离线评估和正式推理，避免“本地测一套、提交跑另一套”。
"""
import math
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd


def parse_seq(value) -> List[str]:
    """解析逗号分隔的item序列"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def choose_seq_col(df: pd.DataFrame, seq_col: str = "auto") -> str:
    """确定使用哪个历史序列字段"""
    if seq_col != "auto":
        if seq_col not in df.columns:
            raise ValueError(f"指定的序列列不存在: {seq_col}")
        return seq_col

    for col in ["item_seq_dedup", "item_seq_raw", "item_seq"]:
        if col in df.columns:
            return col
    raise ValueError("找不到可用的历史序列列")


def build_global_popularity(df: pd.DataFrame, target_col: str = "target_iid") -> Counter:
    """统计全局target热门度"""
    return Counter(df[target_col].astype(str).tolist())


def build_cooccur_stats(
    df: pd.DataFrame,
    seq_col: str,
    target_col: str = "target_iid",
    recent_n: int = 20,
) -> Dict[str, Counter]:
    """统计历史item到target item的共现关系

    如果训练样本历史里出现过 item_a，且下一跳target是 item_b，
    则 `stats[item_a][item_b] += 1`。正式推理时，测试用户历史里
    出现 item_a，就优先召回训练中常跟它共同出现的 item_b。
    """
    stats: Dict[str, Counter] = defaultdict(Counter)
    for _, row in df.iterrows():
        target = str(row[target_col])
        seq = parse_seq(row.get(seq_col))
        if recent_n > 0:
            seq = seq[-recent_n:]

        # 单条样本内去重，避免重复点击把共现票数放得过大。
        for item in dict.fromkeys(seq):
            stats[item][target] += 1
    return stats


def normalize_counter(counter: Counter) -> Dict[str, float]:
    """把计数归一化为0到1之间的相对分数"""
    if not counter:
        return {}
    max_count = max(counter.values())
    if max_count <= 0:
        return {}
    return {item: count / max_count for item, count in counter.items()}


def normalize_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    """把任意实数分数压到0到1，便于和热门/共现分数融合"""
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return {item: 1.0 for item in scores}
    return {item: (float(score) - min_value) / (max_value - min_value) for item, score in scores.items()}


def _add_popularity_scores(
    scores: Dict[str, float],
    global_pop: Counter,
    weight: float,
):
    """加入全局热门度分数"""
    if weight <= 0:
        return
    for item, score in normalize_counter(global_pop).items():
        scores[item] = scores.get(item, 0.0) + weight * score


def _add_cooccur_scores(
    scores: Dict[str, float],
    history_items: Sequence[str],
    cooccur_stats: Mapping[str, Counter],
    weight: float,
):
    """加入历史共现分数"""
    if weight <= 0:
        return
    for hist_item in history_items:
        for target, count in cooccur_stats.get(hist_item, {}).items():
            scores[target] = scores.get(target, 0.0) + weight * math.log1p(count)


def _add_model_scores(
    scores: Dict[str, float],
    model_scores: Optional[Mapping[str, float]],
    weight: float,
):
    """加入模型分数"""
    if not model_scores or weight <= 0:
        return
    for item, score in normalize_scores(model_scores).items():
        scores[item] = scores.get(item, 0.0) + weight * score


def _apply_history_filter(
    scores: Dict[str, float],
    history: Set[str],
    history_filter: str,
    soft_factor: float,
):
    """按配置处理历史item"""
    if history_filter == "none":
        return
    if history_filter == "hard":
        for item in history:
            scores.pop(item, None)
        return
    if history_filter == "soft":
        for item in history:
            if item in scores:
                scores[item] *= soft_factor
        return
    raise ValueError(f"未知history_filter: {history_filter}")


def rank_items(
    seq: Sequence[str],
    candidate_items: Set[str],
    global_pop: Counter,
    cooccur_stats: Mapping[str, Counter],
    topk: int = 10,
    strategy: str = "history",
    history_filter: str = "none",
    recent_n: int = 20,
    model_scores: Optional[Mapping[str, float]] = None,
    model_weight: float = 1.0,
    pop_weight: float = 1.0,
    cooccur_weight: float = 1.0,
    history_soft_factor: float = 0.5,
) -> List[str]:
    """融合多种信号并返回TopK item

    Args:
        seq: 用户历史item字符串序列。
        candidate_items: 官方候选item集合。
        global_pop: 从训练集target统计出的全局热门度。
        cooccur_stats: 历史item到target item的共现计数。
        topk: 推荐列表长度。
        strategy: `model`、`popular`、`last_item`、`history` 或 `hybrid`。
        history_filter: `none` 保留历史item，`hard` 删除历史item，`soft` 降权历史item。
        recent_n: 共现召回使用最近多少个历史item。
        model_scores: 可选的模型输出分数，key为原始iid。
        model_weight/pop_weight/cooccur_weight: 三类信号的融合权重。
        history_soft_factor: soft过滤时历史item保留的分数比例。

    Returns:
        长度不超过topk的item id列表。正常情况下会用热门item补齐到topk。
    """
    if strategy not in {"model", "popular", "last_item", "history", "hybrid"}:
        raise ValueError(f"未知推荐策略: {strategy}")

    seq = [item for item in seq if item]
    recent_seq = seq[-recent_n:] if recent_n > 0 else list(seq)
    history = set(seq)
    scores: Dict[str, float] = {}

    if strategy in {"model", "hybrid"}:
        _add_model_scores(scores, model_scores, model_weight)

    if strategy in {"popular", "hybrid"}:
        _add_popularity_scores(scores, global_pop, pop_weight)

    if strategy in {"last_item", "history", "hybrid"}:
        history_items = recent_seq[-1:] if strategy == "last_item" else recent_seq
        _add_cooccur_scores(scores, history_items, cooccur_stats, cooccur_weight)

    # 空历史或冷启动时，任何策略都用热门target兜底。
    if not scores:
        _add_popularity_scores(scores, global_pop, max(pop_weight, 1.0))

    # 始终给热门item一个极小补全分，确保TopK够长且稳定。
    for item, score in normalize_counter(global_pop).items():
        scores[item] = scores.get(item, 0.0) + 1e-6 * score

    _apply_history_filter(scores, history, history_filter, history_soft_factor)

    ranked = [
        item for item, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if item in candidate_items
    ]

    # 如果过滤后不足TopK，用全局热门合法item补齐。
    seen = set()
    result = []
    for item in ranked:
        if item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            return result

    for item, _ in global_pop.most_common():
        if item in candidate_items and item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            return result

    # 极端情况下，target热门表不足时再用全部候选补齐。
    for item in sorted(candidate_items):
        if item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= topk:
            break
    return result
