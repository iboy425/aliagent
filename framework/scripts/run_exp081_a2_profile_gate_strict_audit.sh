#!/usr/bin/env bash
set -euo pipefail

# Exp-081：A2 严格画像分群 gate 审计
#
# 修正 Exp078 的两个问题：
# 1. 不再审计 len=1，因为正式 base=Exp066 在 len=1 已经替换过 alt；
# 2. 只保留 3 个 split 全部为正收益的高置信分群，减少同集选择偏差。
#
# 审计目标：
# - base 对应 Exp044/Exp066 在非冷启动桶上的稳定推荐；
# - alt 对应 Exp045 多 seed 推荐；
# - 只在 len=2-3,len=4-10,len>10 中寻找极少量高置信替换人群。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

BASE_CHECKPOINT="${BASE_CHECKPOINT:-output/exp044_a2_feature_ranker_seed42/best_model.pt}"
ALT_CHECKPOINT="${ALT_CHECKPOINT:-output/exp045_a2_feature_multiseed/seed42/best_model.pt,output/exp045_a2_feature_multiseed/seed777/best_model.pt,output/exp045_a2_feature_multiseed/seed2024/best_model.pt}"
OUT_DIR="${OUT_DIR:-output/exp081_a2_profile_gate_strict_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4000}"

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[Exp-081] A2严格画像分群 gate 审计"
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
  --buckets "len=2-3,len=4-10,len>10" \
  --profile_specs "single,prefix2,prefix3" \
  --min_samples 100 \
  --min_split_samples 25 \
  --min_positive_splits 3 \
  --min_gain 0.015 \
  --max_groups 12 \
  --output_json "$OUT_DIR/policy.json"

echo
echo "============================================================"
echo "[Exp-081完成]"
echo "============================================================"
echo "请把上面摘要和这个文件内容发回来：$OUT_DIR/policy.json"
