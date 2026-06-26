"""A1 Correct and Smooth 核心逻辑测试"""
import os
import sys
import unittest

import torch


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from a1_correct_smooth import (  # noqa: E402
    _parse_checkpoint_weights,
    correct_predictions,
    normalize_score_rows,
    propagate_with_restart,
    smooth_predictions,
)


class CorrectSmoothTest(unittest.TestCase):
    """验证C&S后处理的数学行为"""

    def test_correct_predictions_spreads_training_residual_to_neighbor(self):
        """训练节点预测残差应沿边传播并修正邻居类别分数"""
        adj = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        probs = torch.tensor(
            [
                [0.20, 0.80],
                [0.35, 0.65],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, -1], dtype=torch.long)
        train_idx = torch.tensor([0], dtype=torch.long)

        corrected = correct_predictions(
            model_probs=probs,
            labels=labels,
            train_idx=train_idx,
            adj=adj,
            alpha=1.0,
            num_iter=1,
            correction_weight=1.0,
        )

        self.assertGreater(corrected[1, 0].item(), probs[1, 0].item())
        self.assertLess(corrected[1, 1].item(), probs[1, 1].item())
        self.assertTrue(torch.allclose(corrected.sum(dim=1), torch.ones(2)))

    def test_smooth_predictions_uses_known_labels_as_anchors(self):
        """平滑阶段应把训练标签作为锚点传播到相邻未标注节点"""
        adj = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        corrected = torch.tensor(
            [
                [0.20, 0.80],
                [0.30, 0.70],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, -1], dtype=torch.long)
        train_idx = torch.tensor([0], dtype=torch.long)

        smoothed = smooth_predictions(
            corrected_probs=corrected,
            labels=labels,
            train_idx=train_idx,
            adj=adj,
            alpha=1.0,
            num_iter=1,
            smooth_weight=1.0,
        )

        self.assertGreater(smoothed[1, 0].item(), corrected[1, 0].item())
        self.assertLess(smoothed[1, 1].item(), corrected[1, 1].item())
        self.assertTrue(torch.allclose(smoothed.sum(dim=1), torch.ones(2)))

    def test_smooth_predictions_uses_pseudo_labels_as_soft_anchors(self):
        """高置信伪标签应能作为软锚点参与平滑"""
        adj = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        corrected = torch.tensor(
            [
                [0.90, 0.10],
                [0.60, 0.40],
                [0.20, 0.80],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, -1, -1], dtype=torch.long)
        train_idx = torch.tensor([0], dtype=torch.long)
        pseudo_idx = torch.tensor([2], dtype=torch.long)
        pseudo_labels = torch.tensor([1], dtype=torch.long)

        smoothed = smooth_predictions(
            corrected_probs=corrected,
            labels=labels,
            train_idx=train_idx,
            adj=adj,
            alpha=1.0,
            num_iter=1,
            smooth_weight=1.0,
            pseudo_idx=pseudo_idx,
            pseudo_labels=pseudo_labels,
            pseudo_weight=1.0,
        )

        self.assertGreater(smoothed[1, 1].item(), smoothed[1, 0].item())
        self.assertTrue(torch.allclose(smoothed.sum(dim=1), torch.ones(3)))

    def test_propagate_with_restart_keeps_seed_when_alpha_is_zero(self):
        """alpha为0时传播结果应完全等于初始种子"""
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        seed = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

        propagated = propagate_with_restart(adj, seed, alpha=0.0, num_iter=5)

        self.assertTrue(torch.equal(propagated, seed))

    def test_normalize_score_rows_repairs_zero_rows(self):
        """全零分数行应被修复为均匀分布，避免后续argmax异常"""
        scores = torch.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=torch.float32)

        normalized = normalize_score_rows(scores)

        self.assertTrue(torch.allclose(normalized[0], torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(normalized[1], torch.tensor([2 / 3, 1 / 3])))

    def test_parse_checkpoint_weights_normalizes_explicit_weights(self):
        """显式checkpoint权重应按数量校验并归一化"""
        weights = _parse_checkpoint_weights("95,5", 2)

        self.assertEqual(weights, [0.95, 0.05])

    def test_parse_checkpoint_weights_defaults_to_equal_weights(self):
        """未提供checkpoint权重时应退化为等权平均"""
        weights = _parse_checkpoint_weights("", 4)

        self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25])

    def test_parse_checkpoint_weights_rejects_mismatched_count(self):
        """权重数量必须和checkpoint数量一致，避免静默错配"""
        with self.assertRaises(ValueError):
            _parse_checkpoint_weights("0.8,0.2", 3)


if __name__ == "__main__":
    unittest.main()
