#!/usr/bin/env bash
set -euo pipefail

# Exp-060：A1 SIGN不同标签传播配置概率融合审计
#
# Exp-059 结论：
# - `undirected+reverse` 均值最高，但在 split2024/2026 输给纯 `undirected`；
# - 这说明反向标签通道有明显增益，但不是每个验证切分都稳定。
#
# 本实验不重新训练，只融合 Exp-059 已训练好的 checkpoint 概率：
# - undir：稳定基线；
# - undir_reverse：上限更高；
# - all_h2：保留一点 directed 信息，检查是否有互补收益。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
BASE_DIR="${BASE_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
OUT_DIR="${OUT_DIR:-output/exp060_a1_sign_config_ensemble_audit}"

python3 code/a1_sign_config_ensemble_audit.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT_DIR" \
  --device "$DEVICE" \
  --split_seeds 42,777,2024,2026,3407 \
  --val_ratio 0.1 \
  --stratified_split \
  --sources \
    "undir=$BASE_DIR/standard_struct_label_undir_h3_rw" \
    "undir_reverse=$BASE_DIR/standard_struct_label_undir_reverse_h3_rw" \
    "all_h2=$BASE_DIR/standard_struct_label_all_h2_rw" \
  --grid_step 0.1 \
  --cs_normalize random_walk \
  --smooth_weights 0
