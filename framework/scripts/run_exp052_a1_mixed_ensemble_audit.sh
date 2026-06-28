#!/usr/bin/env bash
set -euo pipefail

# Exp-052：A1 异构模型融合无泄漏审计
#
# 目标：
# - Exp-051 线上 A1=0.7117，确认 SIGN 有效但仍未接近第一名。
# - 本实验复用已有 checkpoint，不重训；在每个 split 内融合：
#   SIGN-rw、SIGN-sym、SIGN-row、GAT、GCN。
# - 每个 split 只评估同 split 训练出的模型，避免训练泄漏。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp052_a1_mixed_ensemble_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"

mkdir -p "$OUT_DIR"

python3 code/a1_mixed_ensemble_audit.py \
  --data_path data/cls_data/A1.npz \
  --output_dir "$OUT_DIR" \
  --device "$DEVICE" \
  --split_seeds "$SPLIT_SEEDS" \
  --val_ratio 0.1 \
  --stratified_split \
  --cs_normalize random_walk \
  --smooth_weights 0.5,0.75,1.0 \
  --base_dir output
