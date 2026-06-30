#!/usr/bin/env bash
set -euo pipefail

# Exp-077：A2画像分群 gate 审计
#
# 目的：
# - Exp045 多 seed A2 整体线上变差，但 len=0 替换有效；
# - 本实验不再按历史长度整桶替换，而是用训练集验证切分找出
#   “哪些用户画像分群适合使用 Exp045 alt 推荐”；
# - 只做审计，不直接提交。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

BASE_CHECKPOINT="${BASE_CHECKPOINT:-output/exp044_a2_feature_ranker_seed42/best_model.pt}"
ALT_CHECKPOINT="${ALT_CHECKPOINT:-output/exp045_a2_feature_multiseed/seed42/best_model.pt,output/exp045_a2_feature_multiseed/seed777/best_model.pt,output/exp045_a2_feature_multiseed/seed2024/best_model.pt}"
OUT_DIR="${OUT_DIR:-output/exp077_a2_profile_gate_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4000}"

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[Exp-077] A2画像分群 gate 审计"
echo "============================================================"
echo "base_checkpoint=$BASE_CHECKPOINT"
echo "alt_checkpoint=$ALT_CHECKPOINT"
echo "split_seeds=$SPLIT_SEEDS"
echo "out_dir=$OUT_DIR"

python3 code/a2_profile_gate.py audit \
  --data_path data/rec_data \
  --base_checkpoint "$BASE_CHECKPOINT" \
  --alt_checkpoint "$ALT_CHECKPOINT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len 120 \
  --test_like_val \
  --split_seeds "$SPLIT_SEEDS" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --base_model_weight 1.0 \
  --alt_model_weight 22.0 \
  --buckets "len=1,len=2-3,len=4-10,len>10" \
  --profile_specs "single,prefix2,prefix3" \
  --min_samples 80 \
  --min_split_samples 20 \
  --min_positive_splits 2 \
  --min_gain 0.008 \
  --max_groups 30 \
  --output_json "$OUT_DIR/policy.json"

echo
echo "============================================================"
echo "[Exp-077完成]"
echo "============================================================"
echo "请把上面摘要和这个文件内容发回来：$OUT_DIR/policy.json"
