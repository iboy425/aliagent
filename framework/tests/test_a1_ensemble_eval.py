"""A1 ensemble 评估工具测试"""
import os
import sys
import unittest

import torch


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from a1_ensemble_eval import greedy_weighted_ensemble  # noqa: E402


class A1EnsembleEvalTest(unittest.TestCase):
    """验证加权集成搜索行为"""

    def test_greedy_weighted_ensemble_adds_complementary_model(self):
        """互补模型能通过加权融合提升准确率"""
        labels = torch.tensor([0, 1, 1, 0])
        logits_a = torch.tensor([
            [3.0, 0.0],
            [0.0, 3.0],
            [2.0, 1.0],
            [3.0, 0.0],
        ])
        logits_b = torch.tensor([
            [0.0, 3.0],
            [0.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
        ])

        result = greedy_weighted_ensemble(
            logits_list=[logits_a, logits_b],
            labels=labels,
            names=["a", "b"],
            candidate_weights=[0.3],
            max_size=2,
        )

        self.assertEqual(result["selected_names"], ["a", "b"])
        self.assertAlmostEqual(result["val_acc"], 1.0)

    def test_greedy_weighted_ensemble_rejects_harmful_model(self):
        """有害模型不能为了凑数量被加入集成"""
        labels = torch.tensor([0, 1, 1, 0])
        logits_good = torch.tensor([
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
        ])
        logits_bad = -logits_good

        result = greedy_weighted_ensemble(
            logits_list=[logits_good, logits_bad],
            labels=labels,
            names=["good", "bad"],
            candidate_weights=[0.5],
            max_size=2,
        )

        self.assertEqual(result["selected_names"], ["good"])
        self.assertAlmostEqual(result["val_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
