"""A1 SIGN标签传播特征伪标签自训练审计

流程：

1. 加载第一阶段 SIGN-label checkpoint；
2. 用当前 split 的训练标签构造输入，得到全节点预测；
3. 选取未标注节点中的高置信预测作为伪标签；
4. 用真实标签 + 伪标签构造标签传播特征；
5. 只在真实训练标签上训练第二阶段 MLP；
6. 在验证节点上评估，避免真实标签泄漏。
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import GraphDataset
from utils import compute_accuracy, get_device, set_seed, stratified_split
from a1_correct_smooth import normalize_score_rows
from a1_sign_mlp import (
    SignMLP,
    _make_adj,
    build_model_features,
    build_sign_features,
    run_cs_search,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 SIGN伪标签自训练审计")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_checkpoint_dir", type=str, required=True,
                        help="第一阶段checkpoint目录，内部应包含split*/best_model.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split_seeds", type=str, default="42,777,2024,2026,3407")
    parser.add_argument("--thresholds", type=str, default="0.9,0.95")
    parser.add_argument("--pseudo_weights", type=str, default="0.5,1.0")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smooth_weights", type=str, default="0,0.5,0.75")
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    """解析逗号分隔整数列表"""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点列表"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def namespace_from_dict(values: Dict):
    """把字典转为属性对象"""
    class Namespace:
        pass

    obj = Namespace()
    for key, value in values.items():
        setattr(obj, key, value)
    return obj


def split_indices(data: Dict, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用分层划分得到训练/验证节点"""
    return stratified_split(data["labels"], data["train_idx"], 0.1, split_seed)


def load_base_probs(
    data: Dict,
    checkpoint_path: str,
    fit_idx: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, object]:
    """加载第一阶段模型并返回全节点概率"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = namespace_from_dict(checkpoint["args"])
    model_args.device = str(device)
    x = build_model_features(data, model_args, fit_idx, device)
    model = SignMLP(
        in_dim=int(checkpoint.get("input_dim", x.size(1))),
        hidden_dim=int(model_args.hidden_dim),
        num_classes=int(checkpoint.get("num_classes", data["num_classes"])),
        num_layers=int(model_args.num_layers),
        dropout=float(model_args.dropout),
        use_batchnorm=not bool(getattr(model_args, "no_batchnorm", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probs = normalize_score_rows(F.softmax(model(x), dim=1))
    return probs, model_args


def build_pseudo_label_features(
    data: Dict,
    args_obj,
    fit_idx: np.ndarray,
    pseudo_idx: np.ndarray,
    pseudo_labels: np.ndarray,
    pseudo_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """用真实标签和伪标签构造标签传播特征"""
    label_feature_hops = int(getattr(args_obj, "label_feature_hops", 0))
    if label_feature_hops <= 0:
        return None

    labels_np = data["labels"]
    num_nodes = labels_np.shape[0]
    num_classes = data["num_classes"]
    seed = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)

    fit_idx_t = torch.LongTensor(fit_idx).to(device)
    fit_labels_t = torch.LongTensor(labels_np[fit_idx]).to(device)
    seed[fit_idx_t] = F.one_hot(fit_labels_t, num_classes=num_classes).float()

    if len(pseudo_idx) > 0 and pseudo_weight > 0:
        pseudo_idx_t = torch.LongTensor(pseudo_idx).to(device)
        pseudo_labels_t = torch.LongTensor(pseudo_labels).to(device)
        seed[pseudo_idx_t] = (
            F.one_hot(pseudo_labels_t, num_classes=num_classes).float()
            * float(pseudo_weight)
        )

    adj = _make_adj(data, args_obj, device, norm_type=getattr(args_obj, "label_feature_norm", "random_walk"))
    current = seed
    blocks = []
    if bool(getattr(args_obj, "label_feature_include_seed", False)):
        blocks.append(seed)
    for _ in range(label_feature_hops):
        current = torch.sparse.mm(adj, current)
        block = current
        if bool(getattr(args_obj, "label_feature_row_norm", False)):
            block = normalize_score_rows(block)
        blocks.append(block)
    if not blocks:
        return None
    return torch.cat(blocks, dim=1) * float(getattr(args_obj, "label_feature_weight", 1.0))


def build_stage2_features(
    data: Dict,
    args_obj,
    fit_idx: np.ndarray,
    pseudo_idx: np.ndarray,
    pseudo_labels: np.ndarray,
    pseudo_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """构造第二阶段输入特征"""
    x_attr = build_sign_features(data, args_obj, device)
    x_label = build_pseudo_label_features(
        data, args_obj, fit_idx, pseudo_idx, pseudo_labels, pseudo_weight, device
    )
    if x_label is None:
        return x_attr
    return torch.cat([x_attr, x_label], dim=1)


def train_stage2(
    data: Dict,
    args_obj,
    x: torch.Tensor,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    train_args,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict], int, float]:
    """训练第二阶段MLP并返回最佳概率"""
    labels = torch.LongTensor(data["labels"]).to(device)
    fit_idx_t = torch.LongTensor(fit_idx).to(device)
    val_idx_t = torch.LongTensor(val_idx).to(device)
    model = SignMLP(
        in_dim=x.size(1),
        hidden_dim=int(args_obj.hidden_dim),
        num_classes=data["num_classes"],
        num_layers=int(args_obj.num_layers),
        dropout=float(args_obj.dropout),
        use_batchnorm=not bool(getattr(args_obj, "no_batchnorm", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(args_obj, "lr", 0.001)),
        weight_decay=float(getattr(args_obj, "weight_decay", 0.0005)),
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, train_args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits[fit_idx_t], labels[fit_idx_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            train_acc = compute_accuracy(logits[fit_idx_t], labels[fit_idx_t])
            val_acc = compute_accuracy(logits[val_idx_t], labels[val_idx_t])
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= train_args.patience:
            break

    if best_state is None:
        raise RuntimeError("第二阶段训练失败")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        probs = normalize_score_rows(F.softmax(model(x), dim=1))
    return probs, history, best_epoch, best_acc


def run_one_split(data: Dict, split_seed: int, args, device: torch.device) -> List[Dict]:
    """运行单个split的伪标签审计"""
    fit_idx, val_idx = split_indices(data, split_seed)
    checkpoint_path = os.path.join(args.base_checkpoint_dir, f"split{split_seed}", "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"缺少第一阶段checkpoint: {checkpoint_path}")

    base_probs, model_args = load_base_probs(data, checkpoint_path, fit_idx, device)
    base_conf, base_pred = torch.max(base_probs, dim=1)
    fit_mask = np.zeros(data["labels"].shape[0], dtype=bool)
    fit_mask[fit_idx] = True
    candidate_idx = np.where(~fit_mask)[0]

    rows = []
    labels = torch.LongTensor(data["labels"]).to(device)
    for threshold in parse_float_list(args.thresholds):
        selected_mask = base_conf[candidate_idx].detach().cpu().numpy() >= threshold
        pseudo_idx = candidate_idx[selected_mask]
        pseudo_labels = base_pred[pseudo_idx].detach().cpu().numpy()
        for pseudo_weight in parse_float_list(args.pseudo_weights):
            print(
                f"[split={split_seed}] threshold={threshold} "
                f"pseudo_weight={pseudo_weight} pseudo_count={len(pseudo_idx)}"
            )
            x = build_stage2_features(
                data, model_args, fit_idx, pseudo_idx, pseudo_labels, pseudo_weight, device
            )
            probs, history, best_epoch, best_acc = train_stage2(
                data, model_args, x, fit_idx, val_idx, args, device
            )
            # 复用SIGN脚本的轻量C&S搜索，通常model_only已很强，但保留检查。
            model_args.smooth_weights = args.smooth_weights
            cs_rows = run_cs_search(data, probs, labels, val_idx, fit_idx, model_args, device)
            best = cs_rows[0]
            row = {
                "split_seed": split_seed,
                "threshold": threshold,
                "pseudo_weight": pseudo_weight,
                "pseudo_count": int(len(pseudo_idx)),
                "best_epoch": int(best_epoch),
                "stage2_best_val_acc": float(best_acc),
                **best,
            }
            rows.append(row)
    rows.sort(key=lambda item: item["val_acc"], reverse=True)
    return rows


def summarize(all_rows: List[Dict]) -> List[Dict]:
    """按伪标签参数汇总"""
    groups = {}
    for row in all_rows:
        key = (row["threshold"], row["pseudo_weight"], row["kind"], row.get("smooth_weight", 0.0))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        values = [float(row["val_acc"]) for row in rows]
        threshold, pseudo_weight, kind, smooth_weight = key
        summary.append({
            "threshold": threshold,
            "pseudo_weight": pseudo_weight,
            "kind": kind,
            "smooth_weight": smooth_weight,
            "n": len(values),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "avg_pseudo_count": float(np.mean([row["pseudo_count"] for row in rows])),
            "details": rows,
        })
    summary.sort(key=lambda item: (item["mean"], item["min"]), reverse=True)
    return summary


def main():
    """主入口"""
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for split_seed in parse_int_list(args.split_seeds):
        print("\n" + "=" * 100)
        print(f"[Split] {split_seed}")
        print("=" * 100)
        rows = run_one_split(data, split_seed, args, device)
        all_rows.extend(rows)
        with open(os.path.join(args.output_dir, f"split{split_seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"results": rows}, f, ensure_ascii=False, indent=2)
        for row in rows[:8]:
            print(
                f"val_acc={row['val_acc']:.6f}\tthreshold={row['threshold']}\t"
                f"pseudo_weight={row['pseudo_weight']}\tpseudo_count={row['pseudo_count']}\t"
                f"{row['kind']}\tsmooth={row.get('smooth_weight', 0.0)}"
            )

    summary = summarize(all_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[汇总] A1 SIGN伪标签自训练审计 Top-20")
    print("=" * 100)
    for row in summary[:20]:
        print(
            f"mean={row['mean']:.6f}\tmin={row['min']:.6f}\tstd={row['std']:.6f}\t"
            f"n={row['n']}\tthreshold={row['threshold']}\t"
            f"pseudo_weight={row['pseudo_weight']}\tavg_pseudo={row['avg_pseudo_count']:.1f}\t"
            f"{row['kind']}\tsmooth={row['smooth_weight']}"
        )


if __name__ == "__main__":
    main()
