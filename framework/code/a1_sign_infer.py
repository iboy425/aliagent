"""A1 SIGN/MLP checkpoint 集成推理

本脚本用于把 Exp-050 中训练出的多个 SIGN/MLP checkpoint 做概率平均，
再使用全部训练标签执行 Correct and Smooth，生成提交用 A1.csv。
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import GraphDataset
from utils import get_device
from a1_correct_smooth import correct_and_smooth, normalize_score_rows, _prepare_cs_adj
from a1_sign_mlp import SignMLP, build_model_features


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 SIGN/MLP集成推理")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--checkpoint_weights", type=str, default="",
                        help="逗号分隔checkpoint权重；为空时等权平均")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")

    parser.add_argument("--cs_normalize", type=str, default="random_walk",
                        choices=["random_walk", "symmetric", "none"])
    parser.add_argument("--correct_alpha", type=float, default=0.3)
    parser.add_argument("--correct_iter", type=int, default=5)
    parser.add_argument("--correct_weight", type=float, default=0.0)
    parser.add_argument("--smooth_alpha", type=float, default=0.7)
    parser.add_argument("--smooth_iter", type=int, default=5)
    parser.add_argument("--smooth_weight", type=float, default=0.75)
    return parser.parse_args()


def _parse_weights(value: str, count: int) -> List[float]:
    """解析并归一化 checkpoint 权重"""
    if not value:
        return [1.0 / count for _ in range(count)]
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError("checkpoint权重数量必须和checkpoint数量一致")
    if any(weight < 0 for weight in weights):
        raise ValueError("checkpoint权重不能为负数")
    total = sum(weights)
    if total <= 0:
        raise ValueError("checkpoint权重总和必须大于0")
    return [weight / total for weight in weights]


def _namespace_from_dict(values: Dict):
    """把checkpoint中的args字典转为可属性访问对象"""
    class Namespace:
        pass

    obj = Namespace()
    for key, value in values.items():
        setattr(obj, key, value)
    return obj


def load_sign_probs(data: Dict, checkpoint_path: str, device: torch.device) -> torch.Tensor:
    """加载单个SIGN checkpoint并输出全节点概率"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = _namespace_from_dict(checkpoint["args"])
    args.device = str(device)

    x = build_model_features(data, args, data["train_idx"], device)
    model = SignMLP(
        in_dim=int(checkpoint.get("input_dim", x.size(1))),
        hidden_dim=int(args.hidden_dim),
        num_classes=int(checkpoint.get("num_classes", data["num_classes"])),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        use_batchnorm=not bool(getattr(args, "no_batchnorm", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probs = normalize_score_rows(F.softmax(model(x), dim=1))
    return probs


def main():
    """主入口"""
    args = parse_args()
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    weights = _parse_weights(args.checkpoint_weights, len(args.checkpoints))

    avg_probs = None
    model_records = []
    for idx, (checkpoint_path, weight) in enumerate(zip(args.checkpoints, weights), start=1):
        print(f"[SIGN推理] {idx}/{len(args.checkpoints)} weight={weight:.6f} {checkpoint_path}")
        probs = load_sign_probs(data, checkpoint_path, device)
        avg_probs = probs.mul(weight) if avg_probs is None else avg_probs + probs.mul(weight)
        model_records.append({"checkpoint": checkpoint_path, "weight": weight})

    labels = torch.LongTensor(data["labels"]).to(device)
    adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)
    params = {
        "correct_alpha": args.correct_alpha,
        "correct_iter": args.correct_iter,
        "correct_weight": args.correct_weight,
        "smooth_alpha": args.smooth_alpha,
        "smooth_iter": args.smooth_iter,
        "smooth_weight": args.smooth_weight,
    }
    scores = correct_and_smooth(
        model_probs=avg_probs,
        labels=labels,
        train_idx=torch.LongTensor(data["train_idx"]).to(device),
        adj=adj,
        params=params,
    )

    test_idx_t = torch.LongTensor(data["test_idx"]).to(device)
    predictions = torch.argmax(scores[test_idx_t], dim=1).cpu().numpy()
    result_df = pd.DataFrame({"test_idx": data["test_idx"], "label": predictions})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)

    unique, counts = np.unique(predictions, return_counts=True)
    distribution = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
    print("\nA1 SIGN集成推理完成，类别分布:")
    for cls, cnt in distribution.items():
        print(f"  类别 {cls}: {cnt}")
    print(f"C&S参数: {params}")
    print(f"结果已保存: {args.output_path}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "models": model_records,
                "cs_params": params,
                "class_distribution": distribution,
                "output_path": args.output_path,
            }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
