#!/usr/bin/env bash
set -euo pipefail

# Exp-073：A1 all_h2 配置全标签重训
#
# all_h2 来自 Exp-059：
# - 标签传播方向：undirected,directed,reverse
# - 标签传播跳数：2
# - OOF 均值不一定最高，但和当前线上主力 undir / undir_reverse 有互补性。
#
# 训练完成后用于 Exp-074 的二层元模型 stacking。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
BASE_DIR="${BASE_DIR:-output/exp059_a1_sign_directed_label_feature_audit/standard_struct_label_all_h2_rw}"
OUT_DIR="${OUT_DIR:-output/exp073_a1_fulltrain_all_h2}"
RETRAIN="${RETRAIN:-0}"

mkdir -p "$OUT_DIR"

read_best_epoch() {
  local ckpt="$1"
  local fallback="$2"
  CKPT_PATH="$ckpt" FALLBACK_EPOCH="$fallback" python3 - <<'PY'
import os
import torch

path = os.environ["CKPT_PATH"]
fallback = int(os.environ["FALLBACK_EPOCH"])
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(int(ckpt.get("best_epoch") or fallback))
except Exception:
    print(fallback)
PY
}

train_one() {
  local split_seed="$1"
  local seed="$2"
  local epochs="$3"
  local out_dir="$OUT_DIR/all_h2_seed${seed}_e${epochs}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/best_model.pt" && "$RETRAIN" != "1" ]]; then
    echo "[跳过训练] $out_dir 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[全标签训练] all_h2 split=$split_seed seed=$seed epochs=$epochs"
  echo "============================================================"

  python3 code/a1_sign_mlp.py \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --device "$DEVICE" \
    --seed "$seed" \
    --split_seed 42 \
    --val_ratio 0.1 \
    --stratified_split \
    --train_all_labels \
    --disable_early_stop \
    --hops 5 \
    --prop_norm random_walk \
    --graph_mode undirected \
    --feature_norm none \
    --feature_transform standard \
    --block_norm \
    --label_feature_hops 2 \
    --label_feature_norm random_walk \
    --label_feature_graph_modes undirected,directed,reverse \
    --label_feature_norms random_walk \
    --label_feature_row_norm \
    --label_feature_weight 1.0 \
    --structure_feature_mode basic \
    --structure_feature_weight 1.0 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.4 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --epochs "$epochs" \
    --patience "$epochs" \
    --log_interval 10 \
    --cs_normalize random_walk \
    --smooth_weights 0
}

SPLITS=(42 777 2024 2026 3407)
SEEDS=(42 777 2024 2026 3407)

echo
echo "============================================================"
echo "[训练] Exp-073 A1 all_h2 全标签模型"
echo "============================================================"

for i in "${!SPLITS[@]}"; do
  split="${SPLITS[$i]}"
  seed="${SEEDS[$i]}"
  epoch="$(read_best_epoch "$BASE_DIR/split${split}/best_model.pt" 15)"
  train_one "$split" "$seed" "$epoch"
done

echo
echo "============================================================"
echo "[检查] Exp-073 checkpoint"
echo "============================================================"
find "$OUT_DIR" -maxdepth 2 -name best_model.pt -print | sort

count="$(find "$OUT_DIR" -maxdepth 2 -name best_model.pt | wc -l)"
if [[ "$count" -lt 5 ]]; then
  echo "checkpoint数量不足: $count，需要5个" >&2
  exit 1
fi

echo
echo "Exp-073 all_h2 全标签训练完成：$OUT_DIR"
