"""A2用户画像序列排序模型

这个脚本用于训练一个轻量级神经排序器。它和原始 GRU4Rec/SASRec 的区别是：
1. 只预测训练集中出现过的 target item，减少无效类别；
2. 同时使用用户画像类别特征，增强测试新用户的泛化能力；
3. 支持按测试集历史长度分布裁剪训练/验证历史，模拟线上稀疏反馈。
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rec_heuristics import parse_seq
from utils import compute_hit_rate, compute_mrr, compute_ndcg, set_seed


@dataclass
class A2FeatureBundle:
    """A2特征映射集合"""

    item2idx: Dict[str, int]
    idx2item: Dict[int, str]
    target_items: List[str]
    target2class: Dict[str, int]
    user_cols: List[str]
    user_value_maps: Dict[str, Dict[str, int]]


def choose_seq_col(df: pd.DataFrame, seq_col: str) -> str:
    """确定历史序列列名"""
    if seq_col != "auto":
        if seq_col not in df.columns:
            raise ValueError(f"指定的序列列不存在: {seq_col}")
        return seq_col
    for col in ["item_seq_raw", "item_seq_dedup", "item_seq"]:
        if col in df.columns:
            return col
    raise ValueError("找不到历史序列列")


def make_feature_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    user_cols_arg: str,
) -> A2FeatureBundle:
    """根据训练/测试数据建立离散特征映射"""
    all_items = sorted(item_df["iid"].astype(str).unique().tolist())
    item2idx = {iid: idx + 1 for idx, iid in enumerate(all_items)}
    idx2item = {idx + 1: iid for idx, iid in enumerate(all_items)}

    # 线上target通常来自训练target集合。只预测这些类可以显著降低学习难度。
    target_items = sorted(train_df["target_iid"].astype(str).unique().tolist())
    target2class = {iid: idx for idx, iid in enumerate(target_items)}

    if user_cols_arg == "auto":
        user_cols = [col for col in user_df.columns if col != "uid"]
    elif user_cols_arg:
        user_cols = [col.strip() for col in user_cols_arg.split(",") if col.strip()]
    else:
        user_cols = []

    user_value_maps = {}
    for col in user_cols:
        values = sorted(user_df[col].astype(str).fillna("__NA__").unique().tolist())
        user_value_maps[col] = {value: idx + 1 for idx, value in enumerate(values)}

    return A2FeatureBundle(
        item2idx=item2idx,
        idx2item=idx2item,
        target_items=target_items,
        target2class=target2class,
        user_cols=user_cols,
        user_value_maps=user_value_maps,
    )


def parse_item_seq_to_indices(seq_value, item2idx: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    """把 item 字符串序列转为左侧 padding 的索引序列"""
    items = parse_seq(seq_value)
    if max_len > 0:
        items = items[-max_len:]
    indices = [item2idx.get(item, 0) for item in items]
    seq_len = len(indices)
    indices = [0] * max(max_len - seq_len, 0) + indices
    return indices, seq_len


def user_features_to_indices(
    uid: str,
    user_lookup: pd.DataFrame,
    bundle: A2FeatureBundle,
) -> List[int]:
    """把用户画像转为每列一个类别索引"""
    if uid not in user_lookup.index:
        return [0] * len(bundle.user_cols)
    row = user_lookup.loc[uid]
    values = []
    for col in bundle.user_cols:
        value = "__NA__" if pd.isna(row[col]) else str(row[col])
        values.append(bundle.user_value_maps[col].get(value, 0))
    return values


def test_like_lengths(test_df: pd.DataFrame, seq_col: str) -> np.ndarray:
    """提取测试集历史长度分布"""
    lengths = np.array([len(parse_seq(row.get(seq_col))) for _, row in test_df.iterrows()], dtype=np.int64)
    return lengths if len(lengths) else np.array([0], dtype=np.int64)


def truncate_seq_value(seq_value, keep_len: int) -> str:
    """按保留长度截断历史序列"""
    items = parse_seq(seq_value)
    if keep_len <= 0:
        return ""
    return ",".join(items[-keep_len:])


def apply_fixed_test_like_truncation(
    df: pd.DataFrame,
    test_lengths: np.ndarray,
    seq_col: str,
    seed: int,
) -> pd.DataFrame:
    """给验证集做固定的 test-like 历史截断"""
    rng = np.random.default_rng(seed)
    sampled = rng.choice(test_lengths, size=len(df), replace=True)
    out = df.copy()
    out[seq_col] = [truncate_seq_value(value, int(k)) for value, k in zip(out[seq_col], sampled)]
    return out


class A2FeatureDataset(Dataset):
    """A2神经排序训练数据集"""

    def __init__(
        self,
        df: pd.DataFrame,
        user_lookup: pd.DataFrame,
        bundle: A2FeatureBundle,
        seq_col: str,
        max_len: int,
        test_lengths: np.ndarray = None,
        random_test_like: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.user_lookup = user_lookup
        self.bundle = bundle
        self.seq_col = seq_col
        self.max_len = max_len
        self.test_lengths = test_lengths
        self.random_test_like = random_test_like

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_value = row.get(self.seq_col)
        if self.random_test_like and self.test_lengths is not None:
            keep_len = int(np.random.choice(self.test_lengths))
            seq_value = truncate_seq_value(seq_value, keep_len)

        seq, seq_len = parse_item_seq_to_indices(seq_value, self.bundle.item2idx, self.max_len)
        user_feats = user_features_to_indices(str(row["uid"]), self.user_lookup, self.bundle)
        target = self.bundle.target2class[str(row["target_iid"])]

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(user_feats, dtype=torch.long),
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class A2FeatureRanker(nn.Module):
    """用户画像 + 历史序列的轻量神经排序模型"""

    def __init__(
        self,
        num_items: int,
        num_targets: int,
        user_cardinalities: Sequence[int],
        embedding_dim: int = 128,
        user_embedding_dim: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, user_embedding_dim, padding_idx=0)
            for cardinality in user_cardinalities
        ])
        self.length_embedding = nn.Embedding(8, user_embedding_dim)

        input_dim = embedding_dim * 2 + user_embedding_dim * len(user_cardinalities) + user_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

        nn.init.xavier_uniform_(self.item_embedding.weight)

    @staticmethod
    def length_bucket(seq_len: torch.Tensor) -> torch.Tensor:
        """把历史长度离散成桶"""
        buckets = torch.zeros_like(seq_len)
        buckets = torch.where(seq_len >= 1, torch.ones_like(buckets), buckets)
        buckets = torch.where(seq_len >= 2, torch.full_like(buckets, 2), buckets)
        buckets = torch.where(seq_len >= 4, torch.full_like(buckets, 3), buckets)
        buckets = torch.where(seq_len >= 8, torch.full_like(buckets, 4), buckets)
        buckets = torch.where(seq_len >= 16, torch.full_like(buckets, 5), buckets)
        buckets = torch.where(seq_len >= 32, torch.full_like(buckets, 6), buckets)
        buckets = torch.where(seq_len >= 64, torch.full_like(buckets, 7), buckets)
        return buckets

    def forward(self, seq: torch.Tensor, user_feats: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """前向传播，输出 target 类别 logits"""
        item_emb = self.item_embedding(seq)
        mask = (seq != 0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_emb = (item_emb * mask).sum(dim=1) / denom

        last_pos = (seq.size(1) - 1).to(seq.device) if isinstance(seq.size(1), torch.Tensor) else seq.size(1) - 1
        last_emb = item_emb[:, last_pos, :]

        user_parts = []
        for i, emb in enumerate(self.user_embeddings):
            user_parts.append(emb(user_feats[:, i]))
        user_repr = torch.cat(user_parts, dim=-1) if user_parts else mean_emb.new_zeros((seq.size(0), 0))

        len_repr = self.length_embedding(self.length_bucket(seq_len))
        features = torch.cat([mean_emb, last_emb, user_repr, len_repr], dim=-1)
        return self.mlp(features)


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """随机划分训练/验证集"""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def evaluate_model(model, loader, target_items: Sequence[str], device: torch.device, topk: int) -> Dict[str, float]:
    """评估模型 NDCG/Hit/MRR"""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for seq, user_feats, seq_len, y in loader:
            seq = seq.to(device)
            user_feats = user_feats.to(device)
            seq_len = seq_len.to(device)
            logits = model(seq, user_feats, seq_len)
            top_idx = torch.topk(logits, k=topk, dim=-1).indices.cpu().numpy()
            preds.extend([[target_items[int(i)] for i in row] for row in top_idx])
            targets.extend([target_items[int(i)] for i in y.numpy()])
    return {
        "ndcg": float(compute_ndcg(preds, targets, k=topk)),
        "hit": float(compute_hit_rate(preds, targets, k=topk)),
        "mrr": float(compute_mrr(preds, targets)),
    }


def save_checkpoint(path: str, model: nn.Module, bundle: A2FeatureBundle, args, metrics: Dict):
    """保存模型和特征映射"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "bundle": {
                "item2idx": bundle.item2idx,
                "idx2item": bundle.idx2item,
                "target_items": bundle.target_items,
                "target2class": bundle.target2class,
                "user_cols": bundle.user_cols,
                "user_value_maps": bundle.user_value_maps,
            },
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def train(args):
    """训练入口"""
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    item_df = pd.read_csv(os.path.join(args.data_path, "item.csv"))
    seq_col = choose_seq_col(train_df, args.seq_col)
    test_seq_col = choose_seq_col(test_df, args.seq_col)

    bundle = make_feature_bundle(train_df, test_df, user_df, item_df, args.user_cols)
    user_lookup = user_df.set_index("uid")
    lengths = test_like_lengths(test_df, test_seq_col)

    trn_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    if args.test_like_val:
        val_df = apply_fixed_test_like_truncation(val_df, lengths, seq_col, args.seed)

    trn_dataset = A2FeatureDataset(
        trn_df,
        user_lookup,
        bundle,
        seq_col,
        args.max_len,
        test_lengths=lengths,
        random_test_like=args.random_test_like_train,
    )
    val_dataset = A2FeatureDataset(val_df, user_lookup, bundle, seq_col, args.max_len)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    pin_memory = device.type == "cuda"
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    user_cardinalities = [len(bundle.user_value_maps[col]) for col in bundle.user_cols]
    model = A2FeatureRanker(
        num_items=len(bundle.item2idx),
        num_targets=len(bundle.target_items),
        user_cardinalities=user_cardinalities,
        embedding_dim=args.embedding_dim,
        user_embedding_dim=args.user_embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(args.patience // 3, 1)
    )

    best_ndcg = -1.0
    best_metrics = {}
    bad_epochs = 0
    history = []

    print("=" * 100)
    print("A2用户画像序列排序模型训练")
    print(f"device={device}, seq_col={seq_col}, targets={len(bundle.target_items)}, user_cols={bundle.user_cols}")
    print(f"train={len(trn_dataset)}, val={len(val_dataset)}, test_like_val={args.test_like_val}")
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for seq, user_feats, seq_len, y in trn_loader:
            seq = seq.to(device, non_blocking=pin_memory)
            user_feats = user_feats.to(device, non_blocking=pin_memory)
            seq_len = seq_len.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            logits = model(seq, user_feats, seq_len)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * len(y)
            total_count += len(y)

        metrics = evaluate_model(model, val_loader, bundle.target_items, device, args.topk)
        train_loss = total_loss / max(total_count, 1)
        scheduler.step(metrics["ndcg"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(row)

        if epoch == 1 or epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:03d} loss={train_loss:.5f} "
                f"NDCG@{args.topk}={metrics['ndcg']:.6f} "
                f"Hit={metrics['hit']:.6f} MRR={metrics['mrr']:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}"
            )

        if metrics["ndcg"] > best_ndcg:
            best_ndcg = metrics["ndcg"]
            best_metrics = dict(row)
            bad_epochs = 0
            save_checkpoint(os.path.join(args.output_dir, "best_model.pt"), model, bundle, args, best_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"早停触发: best_ndcg={best_ndcg:.6f}, best_epoch={best_metrics.get('epoch')}")
                break

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best_metrics, "history": history}, f, ensure_ascii=False, indent=2)
    print(f"训练完成: best={best_metrics}")


def load_bundle(raw: Dict) -> A2FeatureBundle:
    """从checkpoint恢复特征映射"""
    idx2item = {int(k): v for k, v in raw["idx2item"].items()}
    return A2FeatureBundle(
        item2idx=raw["item2idx"],
        idx2item=idx2item,
        target_items=raw["target_items"],
        target2class=raw["target2class"],
        user_cols=raw["user_cols"],
        user_value_maps=raw["user_value_maps"],
    )


def predict(args):
    """生成A2提交文件"""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bundle = load_bundle(checkpoint["bundle"])

    model_args = checkpoint["args"]
    user_cardinalities = [len(bundle.user_value_maps[col]) for col in bundle.user_cols]
    model = A2FeatureRanker(
        num_items=len(bundle.item2idx),
        num_targets=len(bundle.target_items),
        user_cardinalities=user_cardinalities,
        embedding_dim=model_args.get("embedding_dim", 128),
        user_embedding_dim=model_args.get("user_embedding_dim", 12),
        hidden_dim=model_args.get("hidden_dim", 256),
        dropout=model_args.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device)
    model.eval()

    test_df = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    user_df = pd.read_csv(os.path.join(args.data_path, "user.csv"))
    user_lookup = user_df.set_index("uid")
    seq_col = choose_seq_col(test_df, args.seq_col)

    rows = []
    with torch.no_grad():
        for start in range(0, len(test_df), args.batch_size):
            batch = test_df.iloc[start:start + args.batch_size]
            seqs, user_feats, lens = [], [], []
            for _, row in batch.iterrows():
                seq, seq_len = parse_item_seq_to_indices(row.get(seq_col), bundle.item2idx, args.max_len)
                seqs.append(seq)
                lens.append(seq_len)
                user_feats.append(user_features_to_indices(str(row["uid"]), user_lookup, bundle))

            seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
            user_t = torch.tensor(user_feats, dtype=torch.long, device=device)
            len_t = torch.tensor(lens, dtype=torch.long, device=device)
            logits = model(seq_t, user_t, len_t)
            top_idx = torch.topk(logits, k=args.topk, dim=-1).indices.cpu().numpy()

            for uid, indices in zip(batch["uid"].astype(str).tolist(), top_idx):
                pred = [bundle.target_items[int(i)] for i in indices]
                rows.append({"uid": uid, "prediction": ",".join(pred)})

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"A2预测已保存: {args.output_path}, rows={len(out_df)}")


def build_parser():
    """命令行参数"""
    parser = argparse.ArgumentParser(description="A2用户画像序列排序模型")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_path", type=str, default="data/rec_data")
    common.add_argument("--seq_col", type=str, default="item_seq_raw")
    common.add_argument("--max_len", type=int, default=80)
    common.add_argument("--device", type=str, default="cuda")
    common.add_argument("--batch_size", type=int, default=2048)
    common.add_argument("--topk", type=int, default=10)

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--output_dir", type=str, required=True)
    train_parser.add_argument("--user_cols", type=str, default="auto")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--val_ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--embedding_dim", type=int, default=128)
    train_parser.add_argument("--user_embedding_dim", type=int, default=12)
    train_parser.add_argument("--hidden_dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=12)
    train_parser.add_argument("--grad_clip", type=float, default=5.0)
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--log_interval", type=int, default=1)
    train_parser.add_argument("--test_like_val", action="store_true")
    train_parser.add_argument("--random_test_like_train", action="store_true")

    pred_parser = subparsers.add_parser("predict", parents=[common])
    pred_parser.add_argument("--checkpoint", type=str, required=True)
    pred_parser.add_argument("--output_path", type=str, required=True)

    return parser


def main():
    """主入口"""
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
