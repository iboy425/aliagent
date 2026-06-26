"""A2 test-like 离线评估测试"""
import os
import sys
import unittest

import pandas as pd


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from a2_offline_eval import (  # noqa: E402
    add_weighted_metrics,
    apply_test_like_history_distribution,
    compute_bucket_weights,
    truncate_history_fields,
)


class A2TestLikeEvalTest(unittest.TestCase):
    """验证 A2 按测试集历史长度分布评估的辅助函数"""

    def test_compute_bucket_weights_uses_test_sequence_distribution(self):
        """桶权重应来自 test.csv 的历史长度分布"""
        test_df = pd.DataFrame({
            "uid": ["u1", "u2", "u3", "u4"],
            "item_seq_raw": ["", "i1", "i1,i2", "i1,i2,i3,i4"],
        })

        weights = compute_bucket_weights(test_df, "item_seq_raw")

        self.assertAlmostEqual(weights["len=0"], 0.25)
        self.assertAlmostEqual(weights["len=1"], 0.25)
        self.assertAlmostEqual(weights["len=2-3"], 0.25)
        self.assertAlmostEqual(weights["len=4-10"], 0.25)
        self.assertAlmostEqual(weights["len>10"], 0.0)

    def test_truncate_history_fields_updates_sequence_and_counts(self):
        """截断历史时应同步更新 item_seq_counts，避免频次信号偷看被截断历史"""
        val_df = pd.DataFrame({
            "uid": ["u1", "u2"],
            "target_iid": ["i9", "i8"],
            "item_seq_raw": ["i1,i2,i1,i3", "i4,i5"],
            "item_seq_counts": ["i1:2,i2:1,i3:1", "i4:1,i5:1"],
        })

        truncated = truncate_history_fields(val_df, "item_seq_raw", [2, 0])

        self.assertEqual(truncated.loc[0, "item_seq_raw"], "i1,i3")
        self.assertEqual(truncated.loc[0, "item_seq_counts"], "i1:1,i3:1")
        self.assertEqual(truncated.loc[1, "item_seq_raw"], "")
        self.assertEqual(truncated.loc[1, "item_seq_counts"], "")

    def test_apply_test_like_history_distribution_reuses_test_exact_lengths(self):
        """模拟验证集应从 test.csv 抽样历史长度并截断验证历史"""
        val_df = pd.DataFrame({
            "uid": ["u1", "u2", "u3"],
            "target_iid": ["i9", "i8", "i7"],
            "item_seq_raw": ["i1,i2,i3,i4", "i5,i6,i7,i8", "i9,i10,i11,i12"],
            "item_seq_counts": ["i1:1,i2:1,i3:1,i4:1", "i5:1,i6:1,i7:1,i8:1", "i9:1,i10:1,i11:1,i12:1"],
        })
        test_df = pd.DataFrame({
            "uid": ["t1"],
            "item_seq_raw": ["i100,i101"],
        })

        simulated = apply_test_like_history_distribution(
            val_df, test_df, "item_seq_raw", seed=42
        )

        self.assertTrue((simulated["item_seq_raw"].str.split(",").map(lambda x: 0 if x == [""] else len(x)) == 2).all())
        self.assertEqual(simulated.loc[0, "item_seq_raw"], "i3,i4")

    def test_add_weighted_metrics_uses_bucket_scores(self):
        """加权指标应按测试集桶权重聚合各桶指标"""
        metrics = {
            "buckets": {
                "len=0": {"ndcg": 0.2, "hit": 0.4, "mrr": 0.1},
                "len=1": {"ndcg": 0.6, "hit": 0.8, "mrr": 0.5},
            }
        }
        weights = {"len=0": 0.25, "len=1": 0.75}

        add_weighted_metrics(metrics, weights)

        self.assertAlmostEqual(metrics["weighted_ndcg"], 0.5)
        self.assertAlmostEqual(metrics["weighted_hit"], 0.7)
        self.assertAlmostEqual(metrics["weighted_mrr"], 0.4)


if __name__ == "__main__":
    unittest.main()
