"""A1 SIGN/MLP 多跳传播特征分类器

SIGN 的核心思想是先预计算多跳图传播特征：

    X_sign = concat(X, A X, A^2 X, ..., A^K X)

再用普通 MLP 做节点分类。它和 GCN/GAT 的差异在于：

- 传播和非线性解耦，能显式利用更远的 K 跳邻域；
- 训练阶段只有 MLP，速度快，适合多 split 审计；
- 对当前 A1 这种训练边同质性较高的图，可能和 GAT 形成互补。
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import GraphDataset
from utils import (
    compute_accuracy,
    get_device,
    normalize_adj_sparse,
    preprocess_features,
    random_walk_normalize_sparse,
    set_seed,
    stratified_split,
    split_train_val,
)
from a1_correct_smooth import (
    correct_and_smooth,
    normalize_score_rows,
    _prepare_cs_adj,
    _score_accuracy,
)


class SignMLP(nn.Module):
    """SIGN拼接特征上的多层感知机"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.4,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers必须至少为1")

        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="A1 SIGN/MLP训练、评估与推理")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--stratified_split", action="store_true")
    parser.add_argument("--train_all_labels", action="store_true",
                        help="最终训练模式：使用全部train_idx标签训练，不保留验证集")
    parser.add_argument("--disable_early_stop", action="store_true",
                        help="关闭早停，完整训练到--epochs；适合最终固定epoch重训")

    parser.add_argument("--hops", type=int, default=3,
                        help="SIGN传播跳数K，最终拼接K+1组特征")
    parser.add_argument("--prop_norm", type=str, default="symmetric",
                        choices=["symmetric", "random_walk"],
                        help="传播矩阵归一化方式")
    parser.add_argument("--graph_mode", type=str, default="undirected",
                        choices=["directed", "undirected"],
                        help="传播时使用原始有向图还是无向化图")
    parser.add_argument("--feature_norm", type=str, default="none",
                        choices=["none", "row", "l2"],
                        help="原始节点特征归一化方式")
    parser.add_argument("--feature_transform", type=str, default="none",
                        choices=[
                            "none",
                            "standard",
                            "svd",
                            "raw_plus_svd",
                            "raw_plus_standard_svd",
                        ],
                        help="原始节点特征变换：svd用于低秩降噪，raw_plus_*保留原始特征并追加降维特征")
    parser.add_argument("--svd_dim", type=int, default=128,
                        help="feature_transform包含svd时的降维维度")
    parser.add_argument("--svd_weight", type=float, default=1.0,
                        help="SVD降维特征整体缩放权重")
    parser.add_argument("--svd_seed", type=int, default=42,
                        help="TruncatedSVD随机种子")
    parser.add_argument("--block_norm", action="store_true",
                        help="对每一跳传播特征做行L2归一化，降低高阶传播尺度差异")
    parser.add_argument("--label_feature_hops", type=int, default=0,
                        help="标签传播特征跳数；0表示关闭")
    parser.add_argument("--label_feature_norm", type=str, default="random_walk",
                        choices=["symmetric", "random_walk"],
                        help="标签传播特征使用的邻接矩阵归一化方式")
    parser.add_argument("--label_feature_graph_modes", type=str, default="",
                        help="标签传播使用的图方向，逗号分隔；空值表示沿用--graph_mode，可选undirected/directed/reverse")
    parser.add_argument("--label_feature_norms", type=str, default="",
                        help="标签传播归一化方式，逗号分隔；空值表示沿用--label_feature_norm")
    parser.add_argument("--label_feature_include_seed", action="store_true",
                        help="是否把原始标签one-hot种子Y0拼入特征；默认只拼1..H跳传播结果")
    parser.add_argument("--label_feature_row_norm", action="store_true",
                        help="是否对每一跳标签传播特征做行归一化")
    parser.add_argument("--label_feature_weight", type=float, default=1.0,
                        help="标签传播特征整体缩放权重")
    parser.add_argument("--structure_feature_mode", type=str, default="none",
                        choices=["none", "basic", "label"],
                        help="结构特征模式：basic=度数/低度标记，label=额外加入训练邻居标签统计")
    parser.add_argument("--structure_feature_weight", type=float, default=1.0,
                        help="结构特征整体缩放权重")

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--no_batchnorm", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--log_interval", type=int, default=25)

    parser.add_argument("--cs_normalize", type=str, default="random_walk",
                        choices=["random_walk", "symmetric", "none"])
    parser.add_argument("--smooth_weights", type=str, default="0,0.5,0.75,1.0")
    parser.add_argument("--output_path", type=str, default="",
                        help="若提供，则用全部train_idx标签做C&S并生成A1.csv")
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    """解析逗号分隔浮点列表"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> List[str]:
    """解析逗号分隔字符串列表"""
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _make_adj(
    data: Dict,
    args,
    device: torch.device,
    norm_type: str = None,
    graph_mode: str = None,
) -> torch.Tensor:
    """构造传播矩阵"""
    adj = data["adj"].tocsr().astype(np.float32)
    graph_mode = getattr(args, "graph_mode", "undirected") if graph_mode is None else graph_mode
    graph_mode = str(graph_mode).lower()
    if graph_mode == "undirected":
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    elif graph_mode in {"directed", "forward"}:
        adj = adj.tocsr()
    elif graph_mode in {"reverse", "backward"}:
        adj = adj.T.tocsr()
    else:
        raise ValueError(f"未知图方向模式: {graph_mode}")
    norm_type = args.prop_norm if norm_type is None else norm_type
    if norm_type == "symmetric":
        return normalize_adj_sparse(adj, device=device)
    return random_walk_normalize_sparse(adj, device=device)


def _row_l2_normalize_dense(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """对dense特征做逐行L2归一化"""
    norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    return x / norm.clamp(min=eps)


def _to_numpy_dense(features) -> np.ndarray:
    """把特征矩阵转换为float32 dense数组"""
    if sp.issparse(features):
        return features.toarray().astype(np.float32)
    if isinstance(features, torch.Tensor):
        return features.detach().cpu().numpy().astype(np.float32)
    return np.asarray(features, dtype=np.float32)


def _standardize_dense(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对dense特征按列标准化"""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, eps)).astype(np.float32)


def _build_transformed_base_features(features, args) -> np.ndarray:
    """构造进入SIGN传播前的基础特征

    官方资料建议“PCA降维、标准化”来减少原始属性噪声。这里用
    `TruncatedSVD` 替代传统PCA，因为原始输入是稀疏矩阵，SVD可以直接
    处理稀疏输入，避免先转dense带来的额外开销。
    """
    mode = getattr(args, "feature_transform", "none")
    if mode == "none":
        return _to_numpy_dense(features)

    raw_dense = _to_numpy_dense(features)
    if mode == "standard":
        return _standardize_dense(raw_dense)

    if "svd" not in mode:
        raise ValueError(f"未知特征变换方式: {mode}")

    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError as exc:
        raise ImportError("feature_transform使用SVD时需要安装scikit-learn") from exc

    svd_dim = int(getattr(args, "svd_dim", 128))
    max_dim = min(features.shape[0], features.shape[1]) - 1
    svd_dim = max(1, min(svd_dim, max_dim))
    svd = TruncatedSVD(n_components=svd_dim, random_state=int(getattr(args, "svd_seed", 42)))
    svd_features = svd.fit_transform(features).astype(np.float32)
    svd_features = _standardize_dense(svd_features)
    svd_features *= float(getattr(args, "svd_weight", 1.0))

    if mode == "svd":
        return svd_features
    if mode == "raw_plus_svd":
        return np.concatenate([raw_dense, svd_features], axis=1).astype(np.float32)
    if mode == "raw_plus_standard_svd":
        return np.concatenate([_standardize_dense(raw_dense), svd_features], axis=1).astype(np.float32)
    raise ValueError(f"未知特征变换方式: {mode}")


def build_sign_features(data: Dict, args, device: torch.device) -> torch.Tensor:
    """预计算并拼接多跳传播特征"""
    features = preprocess_features(data["features"], method=args.feature_norm)
    current_np = _build_transformed_base_features(features, args)
    current = torch.tensor(current_np, dtype=torch.float32, device=device)

    adj = _make_adj(data, args, device)
    blocks = []
    for hop in range(args.hops + 1):
        block = current
        if args.block_norm:
            block = _row_l2_normalize_dense(block)
        blocks.append(block)
        if hop < args.hops:
            current = torch.sparse.mm(adj, current)
    return torch.cat(blocks, dim=1)


def build_label_features(
    data: Dict,
    args,
    label_idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """构造训练标签传播特征

    训练/验证审计时，`label_idx` 必须是当前 split 的训练节点，不能包含验证节点。
    正式推理时才使用全部 `train_idx`。
    """
    label_feature_hops = int(getattr(args, "label_feature_hops", 0))
    if label_feature_hops <= 0:
        return None

    labels_np = data["labels"]
    num_nodes = labels_np.shape[0]
    num_classes = data["num_classes"]
    seed = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=device)
    label_idx_t = torch.LongTensor(label_idx).to(device)
    label_values = torch.LongTensor(labels_np[label_idx]).to(device)
    seed[label_idx_t] = F.one_hot(label_values, num_classes=num_classes).float()

    blocks = []
    if bool(getattr(args, "label_feature_include_seed", False)):
        blocks.append(seed)

    graph_modes = _parse_str_list(getattr(args, "label_feature_graph_modes", ""))
    if not graph_modes:
        graph_modes = [getattr(args, "graph_mode", "undirected")]
    norm_types = _parse_str_list(getattr(args, "label_feature_norms", ""))
    if not norm_types:
        norm_types = [getattr(args, "label_feature_norm", "random_walk")]

    for graph_mode in graph_modes:
        for label_feature_norm in norm_types:
            current = seed
            adj = _make_adj(data, args, device, norm_type=label_feature_norm, graph_mode=graph_mode)
            for _ in range(label_feature_hops):
                current = torch.sparse.mm(adj, current)
                block = current
                if bool(getattr(args, "label_feature_row_norm", False)):
                    block = normalize_score_rows(block)
                blocks.append(block)
    if not blocks:
        return None
    return torch.cat(blocks, dim=1) * float(getattr(args, "label_feature_weight", 1.0))


def _standardize_columns(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对结构特征按列标准化"""
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, eps)


def build_structure_features(
    data: Dict,
    args,
    label_idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """构造图结构特征

    `basic` 只使用图结构，不使用标签。
    `label` 在 basic 基础上加入当前训练标签邻居统计；验证节点标签不会进入统计。
    """
    mode = getattr(args, "structure_feature_mode", "none")
    if mode == "none":
        return None

    adj = data["adj"].tocsr().astype(np.float32)
    undir = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    in_degree = np.asarray(adj.sum(axis=0)).reshape(-1).astype(np.float32)
    out_degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    degree = np.asarray(undir.sum(axis=1)).reshape(-1).astype(np.float32)
    n = degree.shape[0]

    basic = np.column_stack([
        np.log1p(in_degree),
        np.log1p(out_degree),
        np.log1p(degree),
        np.sqrt(degree),
        (degree == 0).astype(np.float32),
        (degree <= 1).astype(np.float32),
        (degree <= 2).astype(np.float32),
        (degree <= 5).astype(np.float32),
        (in_degree == 0).astype(np.float32),
        (out_degree == 0).astype(np.float32),
    ]).astype(np.float32)
    blocks = [_standardize_columns(basic)]

    if mode == "label":
        num_classes = data["num_classes"]
        label_seed = np.zeros((n, num_classes), dtype=np.float32)
        labels_np = data["labels"]
        label_seed[label_idx, labels_np[label_idx]] = 1.0
        neigh_label_counts = undir.dot(label_seed).astype(np.float32)
        labeled_neigh = neigh_label_counts.sum(axis=1, keepdims=True)
        degree_den = np.maximum(degree.reshape(-1, 1), 1.0)
        neigh_label_ratio = labeled_neigh / degree_den
        neigh_label_dist = neigh_label_counts / np.maximum(labeled_neigh, 1.0)
        label_stats = np.concatenate([
            np.log1p(labeled_neigh),
            neigh_label_ratio,
            neigh_label_dist,
        ], axis=1).astype(np.float32)
        blocks.append(_standardize_columns(label_stats[:, :2]))
        blocks.append(neigh_label_dist)

    features = np.concatenate(blocks, axis=1).astype(np.float32)
    features *= float(getattr(args, "structure_feature_weight", 1.0))
    return torch.tensor(features, dtype=torch.float32, device=device)


def build_model_features(
    data: Dict,
    args,
    label_idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """构造最终送入MLP的特征"""
    x = build_sign_features(data, args, device)
    label_features = build_label_features(data, args, label_idx, device)
    if label_features is not None:
        x = torch.cat([x, label_features], dim=1)
    structure_features = build_structure_features(data, args, label_idx, device)
    if structure_features is not None:
        x = torch.cat([x, structure_features], dim=1)
    return x


def split_indices(data: Dict, args) -> Tuple[np.ndarray, np.ndarray]:
    """划分训练/验证节点"""
    if getattr(args, "train_all_labels", False):
        return data["train_idx"], data["train_idx"]
    if args.stratified_split:
        return stratified_split(data["labels"], data["train_idx"], args.val_ratio, args.split_seed)
    return split_train_val(data["train_idx"], args.val_ratio, args.split_seed)


def run_cs_search(
    data: Dict,
    probs: torch.Tensor,
    labels: torch.Tensor,
    val_idx: np.ndarray,
    fit_idx: np.ndarray,
    args,
    device: torch.device,
) -> List[Dict]:
    """对SIGN模型输出做轻量C&S搜索"""
    adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)
    fit_idx_t = torch.LongTensor(fit_idx).to(device)
    rows = [{
        "kind": "model_only",
        "val_acc": _score_accuracy(probs, labels, val_idx, device),
        "correct_alpha": None,
        "correct_iter": None,
        "correct_weight": 0.0,
        "smooth_alpha": None,
        "smooth_iter": None,
        "smooth_weight": 0.0,
    }]

    for smooth_weight in _parse_float_list(args.smooth_weights):
        params = {
            "correct_alpha": 0.3,
            "correct_iter": 5,
            "correct_weight": 0.0,
            "smooth_alpha": 0.7,
            "smooth_iter": 5,
            "smooth_weight": smooth_weight,
        }
        scores = correct_and_smooth(
            model_probs=probs,
            labels=labels,
            train_idx=fit_idx_t,
            adj=adj,
            params=params,
        )
        rows.append({
            "kind": "correct_smooth",
            "val_acc": _score_accuracy(scores, labels, val_idx, device),
            **params,
        })

    rows.sort(key=lambda item: item["val_acc"], reverse=True)
    return rows


def save_outputs(output_dir: str, history: List[Dict], cs_rows: List[Dict], args):
    """保存训练历史和评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "cs.json"), "w", encoding="utf-8") as f:
        json.dump({"results": cs_rows, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(cs_rows).to_csv(os.path.join(output_dir, "cs.csv"), index=False)


def train_and_eval(args):
    """训练SIGN/MLP并执行无泄漏评估"""
    set_seed(args.seed)
    device = get_device(args.device)
    data = GraphDataset.load(args.data_path)
    train_idx, val_idx = split_indices(data, args)
    validation_disabled = bool(getattr(args, "train_all_labels", False))

    print("=" * 100)
    print("A1 SIGN/MLP")
    print("=" * 100)
    print(
        f"hops={args.hops}, prop_norm={args.prop_norm}, graph_mode={args.graph_mode}, "
        f"feature_norm={args.feature_norm}, feature_transform={args.feature_transform}, "
        f"svd_dim={args.svd_dim}, block_norm={args.block_norm}"
    )
    print(
        f"label_feature_hops={args.label_feature_hops}, "
        f"label_feature_norm={args.label_feature_norm}, "
        f"label_feature_graph_modes={args.label_feature_graph_modes}, "
        f"label_feature_norms={args.label_feature_norms}, "
        f"label_feature_include_seed={args.label_feature_include_seed}, "
        f"label_feature_row_norm={args.label_feature_row_norm}, "
        f"label_feature_weight={args.label_feature_weight}"
    )
    print(f"train={len(train_idx)}, val={len(val_idx)}, split_seed={args.split_seed}, seed={args.seed}")
    if validation_disabled:
        print("启用最终全标签训练：验证指标仅作训练监控，不作为泛化指标")

    labels = torch.LongTensor(data["labels"]).to(device)
    x = build_model_features(data, args, train_idx, device)
    train_idx_t = torch.LongTensor(train_idx).to(device)
    val_idx_t = torch.LongTensor(val_idx).to(device)

    model = SignMLP(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=data["num_classes"],
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batchnorm=not args.no_batchnorm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    wait = 0
    history: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits[train_idx_t], labels[train_idx_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            train_acc = compute_accuracy(logits[train_idx_t], labels[train_idx_t])
            val_acc = train_acc if validation_disabled else compute_accuracy(logits[val_idx_t], labels[val_idx_t])
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })
        if epoch % args.log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d}/{args.epochs} "
                f"loss={loss.item():.5f} train_acc={train_acc:.6f} val_acc={val_acc:.6f}"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (not args.disable_early_stop) and wait >= args.patience:
            print(f"早停触发: best_acc={best_acc:.6f}, best_epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("训练未产生有效模型")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
        probs = normalize_score_rows(probs)

    cs_rows = run_cs_search(data, probs, labels, val_idx, train_idx, args, device)
    print("\nSIGN/MLP C&S Top结果")
    for row in cs_rows[:8]:
        print(
            f"val_acc={row['val_acc']:.6f}\t{row['kind']}\t"
            f"smooth=({row['smooth_alpha']},{row['smooth_iter']},{row['smooth_weight']})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "args": vars(args),
        "input_dim": x.size(1),
        "num_classes": data["num_classes"],
    }, os.path.join(args.output_dir, "best_model.pt"))
    save_outputs(args.output_dir, history, cs_rows, args)

    if args.output_path:
        best = cs_rows[0]
        infer_with_full_labels(data, model, labels, best, args, device)


def infer_with_full_labels(
    data: Dict,
    model: nn.Module,
    labels: torch.Tensor,
    params: Dict,
    args,
    device: torch.device,
):
    """使用全部训练标签C&S并生成A1.csv"""
    x = build_model_features(data, args, data["train_idx"], device)
    model.eval()
    with torch.no_grad():
        probs = normalize_score_rows(F.softmax(model(x), dim=1))
    if params["kind"] == "correct_smooth":
        adj = _prepare_cs_adj(data["adj"], args.cs_normalize, device)
        scores = correct_and_smooth(
            model_probs=probs,
            labels=labels,
            train_idx=torch.LongTensor(data["train_idx"]).to(device),
            adj=adj,
            params=params,
        )
    else:
        scores = probs

    test_idx_t = torch.LongTensor(data["test_idx"]).to(device)
    predictions = torch.argmax(scores[test_idx_t], dim=1).cpu().numpy()
    result_df = pd.DataFrame({"test_idx": data["test_idx"], "label": predictions})
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nA1 SIGN/MLP 推理完成，类别分布:")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt}")
    print(f"结果已保存: {args.output_path}")


def main():
    """主入口"""
    args = parse_args()
    train_and_eval(args)


if __name__ == "__main__":
    main()
