"""模型结构测试"""
import os
import sys
import unittest

import torch


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models import GNNClassifier  # noqa: E402


class ModelTest(unittest.TestCase):
    """验证模型前向传播的基本行为"""

    def test_sparse_gat_forward_shape(self):
        """稀疏GAT应在小图上输出节点分类logits"""
        indices = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ])
        values = torch.ones(indices.size(1))
        adj = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
        x = torch.randn(4, 5)

        model = GNNClassifier(
            in_dim=5,
            hidden_dim=8,
            num_classes=3,
            num_layers=2,
            dropout=0.0,
            model_type="gat_sparse",
            gat_heads=2,
        )
        logits = model(x, adj)

        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertFalse(torch.isnan(logits).any().item())


if __name__ == "__main__":
    unittest.main()
