#!/usr/bin/env bash
set -euo pipefail

# Exp-051：A1 SIGN五折集成 + A2线上最佳候选包
#
# 依据：
# - Exp-050 最强配置 `sign_rw_undir_k5_h512_l2_do04_block`
#   无泄漏5 split均值 0.725297，显著高于此前 GAT 稳定水平 0.698935。
# - A2 继续沿用当前线上最佳 Exp-044，避免两个任务同时变化。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
SRC_DIR="${SRC_DIR:-output/exp050_a1_sign_audit/sign_rw_undir_k5_h512_l2_do04_block}"
OUT_DIR="${OUT_DIR:-output/exp051_submit_a1_sign_ensemble_a2_exp044}"
A2_SOURCE="${A2_SOURCE:-output/exp044_a2_feature_fusion/A2.csv}"

mkdir -p "$OUT_DIR"

CHECKPOINTS=(
  "$SRC_DIR/split42/best_model.pt"
  "$SRC_DIR/split777/best_model.pt"
  "$SRC_DIR/split2024/best_model.pt"
  "$SRC_DIR/split2026/best_model.pt"
  "$SRC_DIR/split3407/best_model.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "缺少SIGN checkpoint: $ckpt"
    echo "请先运行：CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp050_a1_sign_audit.sh"
    exit 1
  fi
done

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE"
  exit 1
fi

echo
echo "============================================================"
echo "[推理] A1 SIGN五折集成"
echo "============================================================"

python3 code/a1_sign_infer.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --device "$DEVICE" \
  --cs_normalize random_walk \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight "${SMOOTH_WEIGHT:-0.75}" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_sign_ensemble_infer.json"

echo
echo "============================================================"
echo "[打包] prediction.zip"
echo "============================================================"

cp "$A2_SOURCE" "$OUT_DIR/A2.csv"
(
  cd "$OUT_DIR"
  rm -f prediction.zip
  zip -q prediction.zip A1.csv A2.csv
)

python3 code/validate_submission.py \
  --zip_path "$OUT_DIR/prediction.zip" \
  --cls_data_path data/cls_data/A1.npz \
  --rec_data_dir data/rec_data \
  --topk 10

echo
echo "候选提交包：$OUT_DIR/prediction.zip"
