#!/usr/bin/env bash
set -euo pipefail

# Exp-103：审计 Exp102 bucket-head 模型是否适合接入线上稳定 A2
#
# 先用 checkpoint 级别多 split 验证：
# - base：Exp044 单 seed线上有效模型；
# - alt：Exp102 bucket-head 多 seed模型；
# - 比较不同 keep_topn 和历史长度桶组合。
#
# 只有 mean_gain、min_gain 同时可靠时，才进入候选打包。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

BASE_CHECKPOINT="${BASE_CHECKPOINT:-output/exp044_a2_feature_ranker_seed42/best_model.pt}"
ALT_CHECKPOINT="${ALT_CHECKPOINT:-}"
if [[ -z "$ALT_CHECKPOINT" ]]; then
  if [[ -f output/exp102_a2_bucket_head_multiseed/checkpoints.txt ]]; then
    ALT_CHECKPOINT="$(cat output/exp102_a2_bucket_head_multiseed/checkpoints.txt)"
  else
    echo "ALT_CHECKPOINT为空，且找不到 Exp102 checkpoints.txt" >&2
    exit 1
  fi
fi

OUT_DIR="${OUT_DIR:-output/exp103_a2_bucket_head_protected_audit}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4000}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024}"
ALT_MODEL_WEIGHT="${ALT_MODEL_WEIGHT:-}"
if [[ -z "$ALT_MODEL_WEIGHT" ]]; then
  if [[ -f output/exp102_a2_bucket_head_multiseed/fusion_eval.json ]]; then
    ALT_MODEL_WEIGHT="$(python3 -c "import json; d=json.load(open('output/exp102_a2_bucket_head_multiseed/fusion_eval.json')); print(d['best']['model_weight'])")"
  else
    ALT_MODEL_WEIGHT="3.0"
  fi
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[Exp-103] A2 bucket-head 保护融合审计"
echo "============================================================"
echo "base_checkpoint=$BASE_CHECKPOINT"
echo "alt_checkpoint=$ALT_CHECKPOINT"
echo "alt_model_weight=$ALT_MODEL_WEIGHT"

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
  --alt_model_weight "$ALT_MODEL_WEIGHT" \
  --keep_topns 0,1,2,3 \
  --bucket_sets "len=0,len=1,len=2-3,len=4-10,len>10,len=0+len=1,len=0+len=1+len=2-3,len=2-3+len=4-10+len>10,len=4-10+len>10,len=0+len=1+len=2-3+len=4-10+len>10" \
  --output_json "$OUT_DIR/results.json"

echo
echo "============================================================"
echo "[Exp-103完成]"
echo "============================================================"
echo "请把摘要和 $OUT_DIR/results.json 发回来"
