#!/usr/bin/env bash
set -euo pipefail

# Exp-083：A2 保护TopN融合审计
#
# 核心假设：
# - Exp066 的 Top1 是线上稳定信号，不能轻易改；
# - Exp045 多 seed 模型离线强，可以用来补强第2-10位；
# - 先审计 keep_topn=1/2 和不同历史长度桶，再决定是否生成候选。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

BASE_CHECKPOINT="${BASE_CHECKPOINT:-output/exp044_a2_feature_ranker_seed42/best_model.pt}"
ALT_CHECKPOINT="${ALT_CHECKPOINT:-output/exp045_a2_feature_multiseed/seed42/best_model.pt,output/exp045_a2_feature_multiseed/seed777/best_model.pt,output/exp045_a2_feature_multiseed/seed2024/best_model.pt}"
OUT_DIR="${OUT_DIR:-output/exp083_a2_protected_blend_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4000}"

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[Exp-083] A2保护TopN融合审计"
echo "============================================================"
echo "base_checkpoint=$BASE_CHECKPOINT"
echo "alt_checkpoint=$ALT_CHECKPOINT"
echo "split_seeds=$SPLIT_SEEDS"
echo "out_dir=$OUT_DIR"

python3 code/a2_protected_blend.py eval \
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
  --keep_topns "1,2" \
  --bucket_sets "len=2-3,len>10,len=2-3+len>10,len=2-3+len=4-10+len>10" \
  --output_json "$OUT_DIR/results.json"

echo
echo "============================================================"
echo "[Exp-083完成]"
echo "============================================================"
echo "请把上面摘要和这个文件内容发回来：$OUT_DIR/results.json"
