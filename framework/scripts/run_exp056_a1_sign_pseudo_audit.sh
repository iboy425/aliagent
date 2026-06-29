#!/usr/bin/env bash
set -euo pipefail

# Exp-056：A1 SIGN标签传播特征伪标签自训练审计
#
# 复用 Exp-053 最强 split checkpoint 作为第一阶段模型。
# 每个 split：
# 1. 第一阶段模型预测未标注节点；
# 2. 选择高置信预测作为伪标签；
# 3. 用真实标签 + 伪标签重建标签传播特征；
# 4. 第二阶段只在真实训练标签上训练；
# 5. 验证集只用于评估和早停，不参与伪标签真值。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp056_a1_sign_pseudo_audit}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-output/exp053_a1_sign_label_feature_audit/sign_rw_k5_block_labelrw_h3_rownorm_w1}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"

mkdir -p "$OUT_DIR"

python3 code/a1_sign_pseudo_audit.py \
  --data_path data/cls_data/A1.npz \
  --output_dir "$OUT_DIR" \
  --base_checkpoint_dir "$BASE_CKPT_DIR" \
  --device "$DEVICE" \
  --split_seeds "$SPLIT_SEEDS" \
  --thresholds "${THRESHOLDS:-0.6,0.7,0.75}" \
  --pseudo_weights "${PSEUDO_WEIGHTS:-0.5,1.0}" \
  --epochs "${EPOCHS:-180}" \
  --patience "${PATIENCE:-60}" \
  --seed "${SEED:-2026}" \
  --smooth_weights 0,0.5,0.75
