"""A1多检查点评估工具

用途：
1. 在固定验证集上比较多个A1 checkpoint。
2. 评估Top-K checkpoint的logits平均集成效果。
3. 辅助决定哪些A1模型进入最终提交包。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from datasets import GraphDataset
from infer import _prepare_task1_tensors, load_model_from_checkpoint
from utils import compute_accuracy, get_device, split_train_val, stratified_split


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 checkpoint集成验证工具")
    parser.add_argument("--data_path", type=str, required=True,
                        help="A1.npz数据路径")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="待评估的checkpoint列表")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备，如 cuda 或 cpu")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例，需与训练命令保持一致")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="固定验证集划分随机种子")
    parser.add_argument("--stratified_split", action="store_true",
                        help="使用分层验证集划分")
    parser.add_argument("--topks", type=str, default="1,3,5",
                        help="按单模型验证准确率排序后，评估哪些Top-K集成")
    parser.add_argument("--greedy_max_size", type=int, default=0,
                        help="若大于0，则执行贪心加权集成搜索，最多选择多少个checkpoint")
    parser.add_argument("--greedy_weights", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5",
                        help="贪心搜索中新加入模型的候选权重")
    parser.add_argument("--greedy_min_gain", type=float, default=0.0,
                        help="贪心搜索每步至少需要提升多少准确率才加入新模型")
    parser.add_argument("--output_json", type=str, default="",
                        help="可选：保存评估结果JSON")
    return parser.parse_args()


def _split_eval_indices(data, args):
    """复现A1训练脚本中的验证集划分"""
    if args.stratified_split:
        _, val_idx = stratified_split(
            data["labels"], data["train_idx"], args.val_ratio, args.split_seed
        )
    else:
        _, val_idx = split_train_val(data["train_idx"], args.val_ratio, args.split_seed)
    return val_idx


def _checkpoint_name(path):
    """生成便于阅读的checkpoint名称"""
    parent = os.path.basename(os.path.dirname(path))
    if parent:
        return parent
    return os.path.basename(path)


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算logits准确率"""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def greedy_weighted_ensemble(
    logits_list,
    labels: torch.Tensor,
    names,
    candidate_weights,
    max_size: int,
    min_gain: float = 0.0,
):
    """贪心搜索加权集成

    每一步尝试加入一个尚未选择的模型，并搜索新模型权重 `w`：
    `new_logits = (1 - w) * current_logits + w * candidate_logits`。
    只有验证准确率提升超过 `min_gain` 时才接受该模型。
    """
    if not logits_list:
        raise ValueError("logits_list不能为空")
    if len(logits_list) != len(names):
        raise ValueError("logits_list和names长度必须一致")

    labels = labels.to(logits_list[0].device)
    single_accs = [_accuracy_from_logits(logits, labels) for logits in logits_list]
    first_idx = int(np.argmax(single_accs))
    selected = [first_idx]
    weights = {first_idx: 1.0}
    current_logits = logits_list[first_idx].clone()
    current_acc = single_accs[first_idx]
    steps = [{
        "step": 1,
        "added": names[first_idx],
        "added_weight": 1.0,
        "val_acc": current_acc,
        "selected_names": [names[first_idx]],
        "weights": {names[first_idx]: 1.0},
    }]

    max_size = max(1, min(max_size, len(logits_list)))
    while len(selected) < max_size:
        best = None
        for idx, logits in enumerate(logits_list):
            if idx in selected:
                continue
            for new_weight in candidate_weights:
                if not 0 < new_weight <= 1:
                    continue
                trial_logits = (1.0 - new_weight) * current_logits + new_weight * logits
                trial_acc = _accuracy_from_logits(trial_logits, labels)
                if best is None or trial_acc > best["val_acc"]:
                    best = {
                        "idx": idx,
                        "new_weight": float(new_weight),
                        "logits": trial_logits,
                        "val_acc": float(trial_acc),
                    }

        if best is None or best["val_acc"] <= current_acc + min_gain:
            break

        for idx in list(weights):
            weights[idx] *= 1.0 - best["new_weight"]
        weights[best["idx"]] = best["new_weight"]
        selected.append(best["idx"])
        current_logits = best["logits"]
        current_acc = best["val_acc"]
        steps.append({
            "step": len(selected),
            "added": names[best["idx"]],
            "added_weight": best["new_weight"],
            "val_acc": current_acc,
            "selected_names": [names[idx] for idx in selected],
            "weights": {names[idx]: float(weights[idx]) for idx in selected},
        })

    return {
        "val_acc": float(current_acc),
        "selected_indices": selected,
        "selected_names": [names[idx] for idx in selected],
        "weights": {names[idx]: float(weights[idx]) for idx in selected},
        "steps": steps,
    }


def main():
    """主入口"""
    args = parse_args()
    device = get_device(args.device)

    data = GraphDataset.load(args.data_path)
    val_idx = _split_eval_indices(data, args)
    labels = torch.LongTensor(data["labels"]).to(device)
    val_idx_t = torch.LongTensor(val_idx).to(device)

    rows = []
    logits_by_path = {}

    with torch.no_grad():
        for path in args.checkpoints:
            checkpoint_meta = torch.load(path, map_location="cpu", weights_only=False)
            model, model_args = load_model_from_checkpoint(path, device)
            features, adj = _prepare_task1_tensors(data, model_args, device)
            logits = model(features, adj)[val_idx_t]
            acc = compute_accuracy(logits, labels[val_idx_t])
            logits_by_path[path] = logits.detach().clone()
            rows.append({
                "name": _checkpoint_name(path),
                "checkpoint": path,
                "val_acc": acc,
                "saved_epoch": checkpoint_meta.get("epoch"),
                "saved_val_acc": checkpoint_meta.get("val_acc"),
                "model_type": model_args.get("model_type", "unknown"),
                "seed": model_args.get("seed"),
                "split_seed": model_args.get("split_seed", model_args.get("seed")),
            })

            del model, features, adj, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda x: x["val_acc"], reverse=True)
    print("\n单模型验证准确率（固定验证集）")
    for row in rows:
        print(
            f"{row['val_acc']:.6f}\tseed={row['seed']}\tsplit_seed={row['split_seed']}\t"
            f"saved={row['saved_val_acc']}\t"
            f"{row['name']}\t{row['checkpoint']}"
        )

    topks = sorted({int(x) for x in args.topks.split(",") if x.strip()})
    ensemble_rows = []
    for topk in topks:
        selected = rows[:topk]
        if not selected:
            continue
        logits_sum = None
        for row in selected:
            logits = logits_by_path[row["checkpoint"]]
            logits_sum = logits if logits_sum is None else logits_sum + logits
        avg_logits = logits_sum / len(selected)
        acc = compute_accuracy(avg_logits, labels[val_idx_t])
        ensemble_rows.append({
            "topk": topk,
            "val_acc": acc,
            "checkpoints": [row["checkpoint"] for row in selected],
        })

    print("\n集成验证准确率（按单模型准确率排序取Top-K）")
    for row in ensemble_rows:
        print(f"Top-{row['topk']}\t{row['val_acc']:.6f}")

    greedy_result = None
    if args.greedy_max_size > 0:
        candidate_weights = [float(x) for x in args.greedy_weights.split(",") if x.strip()]
        logits_list = [logits_by_path[row["checkpoint"]] for row in rows]
        names = [row["name"] for row in rows]
        greedy_result = greedy_weighted_ensemble(
            logits_list=logits_list,
            labels=labels[val_idx_t],
            names=names,
            candidate_weights=candidate_weights,
            max_size=args.greedy_max_size,
            min_gain=args.greedy_min_gain,
        )

        print("\n贪心加权集成搜索")
        for step in greedy_result["steps"]:
            print(
                f"Step-{step['step']}\tval_acc={step['val_acc']:.6f}\t"
                f"added={step['added']}\tadded_weight={step['added_weight']:.3f}\t"
                f"weights={step['weights']}"
            )
        print(f"Greedy best\t{greedy_result['val_acc']:.6f}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "single": rows,
                "ensemble": ensemble_rows,
                "greedy": greedy_result,
                "val_ratio": args.val_ratio,
                "split_seed": args.split_seed,
                "stratified_split": args.stratified_split,
            }, f, ensure_ascii=False, indent=2)

        # 额外保存CSV，方便人工查看。
        csv_path = os.path.splitext(args.output_json)[0] + "_single.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n评估结果已保存: {args.output_json}")
        print(f"单模型表格已保存: {csv_path}")


if __name__ == "__main__":
    main()
