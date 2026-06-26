"""推荐启发式工具测试"""
import os
import sys
import unittest

import pandas as pd


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from rec_heuristics import (  # noqa: E402
    build_user_combo_profile_stats,
    get_user_combo_profile_counters,
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


if __name__ == "__main__":
    unittest.main()
