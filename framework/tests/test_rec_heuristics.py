"""推荐启发式工具测试"""
import os
import sys
import unittest
from collections import Counter

import pandas as pd


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from rec_heuristics import (  # noqa: E402
    build_item_feature_transition_stats,
    build_user_combo_profile_stats,
    get_item_feature_counters,
    get_user_combo_profile_counters,
    rank_items,
)


class RecHeuristicsTest(unittest.TestCase):
    """验证推荐启发式统计函数"""

    def test_user_combo_profile_stats_returns_matching_combo_counter(self):
        """用户画像组合应返回同组合训练用户的target热门计数"""
        train_df = pd.DataFrame({
            "uid": ["u1", "u2", "u3"],
            "target_iid": ["i1", "i1", "i2"],
        })
        user_df = pd.DataFrame({
            "uid": ["u1", "u2", "u3", "t1"],
            "u_cat_01": [1, 1, 2, 1],
            "u_cat_02": [5, 5, 5, 5],
            "u_cat_03": [7, 7, 8, 7],
        })

        stats, lookup = build_user_combo_profile_stats(
            train_df=train_df,
            user_df=user_df,
            user_cols=["u_cat_01", "u_cat_02", "u_cat_03"],
            combo_sizes=[3, 2],
            min_count=1,
        )
        counters = get_user_combo_profile_counters(
            user_id="t1",
            user_lookup=lookup,
            combo_profile_stats=stats,
            min_count=1,
        )

        self.assertGreaterEqual(len(counters), 1)
        self.assertEqual(counters[0]["i1"], 2)

    def test_user_combo_profile_stats_respects_min_count(self):
        """样本数不足的画像组合应被过滤，避免冷启动过拟合"""
        train_df = pd.DataFrame({
            "uid": ["u1", "u2"],
            "target_iid": ["i1", "i2"],
        })
        user_df = pd.DataFrame({
            "uid": ["u1", "u2", "t1"],
            "u_cat_01": [1, 2, 1],
            "u_cat_02": [5, 5, 5],
        })

        stats, lookup = build_user_combo_profile_stats(
            train_df=train_df,
            user_df=user_df,
            user_cols=["u_cat_01", "u_cat_02"],
            combo_sizes=[2],
            min_count=2,
        )
        counters = get_user_combo_profile_counters(
            user_id="t1",
            user_lookup=lookup,
            combo_profile_stats=stats,
            min_count=2,
        )

        self.assertEqual(counters, [])

    def test_user_combo_profile_stats_all_mode_uses_non_prefix_columns(self):
        """all模式应能使用非前缀画像列，提升冷启动覆盖"""
        train_df = pd.DataFrame({
            "uid": ["u1", "u2"],
            "target_iid": ["i1", "i1"],
        })
        user_df = pd.DataFrame({
            "uid": ["u1", "u2", "t1"],
            "u_cat_01": [1, 2, 3],
            "u_cat_02": [9, 9, 9],
        })

        stats, lookup = build_user_combo_profile_stats(
            train_df=train_df,
            user_df=user_df,
            user_cols=["u_cat_01", "u_cat_02"],
            combo_sizes=[1],
            combo_mode="all",
            min_count=1,
        )
        counters = get_user_combo_profile_counters(
            user_id="t1",
            user_lookup=lookup,
            combo_profile_stats=stats,
            min_count=1,
        )

        self.assertGreaterEqual(len(counters), 1)
        self.assertEqual(counters[0]["i1"], 2)

    def test_item_feature_transition_stats_returns_category_counter(self):
        """历史item类目应召回同类目下常见target"""
        train_df = pd.DataFrame({
            "target_iid": ["i9", "i9", "i8"],
            "item_seq_raw": ["i1,i2", "i2", "i3"],
        })
        item_df = pd.DataFrame({
            "iid": ["i1", "i2", "i3"],
            "i_cat_01": [1, 1, 2],
        })

        stats, lookup = build_item_feature_transition_stats(
            train_df=train_df,
            item_df=item_df,
            seq_col="item_seq_raw",
            feature_cols=["i_cat_01"],
            recent_n=2,
            min_count=1,
        )
        counters = get_item_feature_counters(
            seq=["i1"],
            item_lookup=lookup,
            item_feature_stats=stats,
            feature_cols=["i_cat_01"],
            recent_n=1,
            min_count=1,
        )

        self.assertEqual(counters[0]["i9"], 2)

    def test_item_feature_transition_stats_respects_min_count(self):
        """样本数不足的item类目转移应被过滤"""
        train_df = pd.DataFrame({
            "target_iid": ["i9"],
            "item_seq_raw": ["i1"],
        })
        item_df = pd.DataFrame({
            "iid": ["i1"],
            "i_cat_01": [1],
        })

        stats, lookup = build_item_feature_transition_stats(
            train_df=train_df,
            item_df=item_df,
            seq_col="item_seq_raw",
            feature_cols=["i_cat_01"],
            recent_n=1,
            min_count=2,
        )
        counters = get_item_feature_counters(
            seq=["i1"],
            item_lookup=lookup,
            item_feature_stats=stats,
            feature_cols=["i_cat_01"],
            recent_n=1,
            min_count=2,
        )

        self.assertEqual(counters, [])

    def test_rank_items_cooccur_decay_prioritizes_recent_history(self):
        """共现衰减应让更近的历史item拥有更高解释权重"""
        cooccur_stats = {
            "old": {"i_old_target": 10},
            "new": {"i_new_target": 6},
        }
        candidate_items = {"i_old_target", "i_new_target"}

        no_decay = rank_items(
            seq=["old", "new"],
            candidate_items=candidate_items,
            global_pop=Counter(),
            cooccur_stats=cooccur_stats,
            strategy="history",
            recent_n=2,
            cooccur_decay=1.0,
            topk=2,
        )
        with_decay = rank_items(
            seq=["old", "new"],
            candidate_items=candidate_items,
            global_pop=Counter(),
            cooccur_stats=cooccur_stats,
            strategy="history",
            recent_n=2,
            cooccur_decay=0.1,
            topk=2,
        )

        self.assertEqual(no_decay[0], "i_old_target")
        self.assertEqual(with_decay[0], "i_new_target")


if __name__ == "__main__":
    unittest.main()
