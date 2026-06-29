#!/usr/bin/env bash
set -euo pipefail

# Exp-060 候选提交包：A1 SIGN配置级概率融合 + A2线上最佳
#
# 离线依据：
# - Exp-060 修复标签泄漏后的最佳组合：
#   `undir:0.3 + undir_reverse:0.7`
# - 5 split 均值 0.764384，高于 Exp-059 单配置 0.763653。
#
# 实现方式：
# - 5个 pure undirected checkpoint 总权重 0.3，每个 0.06；
# - 5个 undirected+reverse checkpoint 总权重 0.7，每个 0.14；
# - A2 沿用当前线上最佳 Exp-044。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
BASE_DIR="${BASE_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
OUT_DIR="${OUT_DIR:-output/exp060_submit_a1_config_ensemble_a2_exp044}"
A2_SOURCE="${A2_SOURCE:-output/exp044_a2_feature_fusion/A2.csv}"

UNDIR_DIR="$BASE_DIR/standard_struct_label_undir_h3_rw"
UNDIR_REVERSE_DIR="$BASE_DIR/standard_struct_label_undir_reverse_h3_rw"

mkdir -p "$OUT_DIR"

CHECKPOINTS=(
  "$UNDIR_DIR/split42/best_model.pt"
  "$UNDIR_DIR/split777/best_model.pt"
  "$UNDIR_DIR/split2024/best_model.pt"
  "$UNDIR_DIR/split2026/best_model.pt"
  "$UNDIR_DIR/split3407/best_model.pt"
  "$UNDIR_REVERSE_DIR/split42/best_model.pt"
  "$UNDIR_REVERSE_DIR/split777/best_model.pt"
  "$UNDIR_REVERSE_DIR/split2024/best_model.pt"
  "$UNDIR_REVERSE_DIR/split2026/best_model.pt"
  "$UNDIR_REVERSE_DIR/split3407/best_model.pt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "缺少checkpoint: $ckpt"
    echo "请先运行：CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp059_a1_sign_directed_label_feature_audit.sh"
    exit 1
  fi
done

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE"
  exit 1
fi

echo
echo "============================================================"
echo "[推理] A1 Exp-060 配置融合 undir:0.3 + undir_reverse:0.7"
echo "============================================================"

python3 code/a1_sign_infer.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --checkpoint_weights "0.06,0.06,0.06,0.06,0.06,0.14,0.14,0.14,0.14,0.14" \
  --device "$DEVICE" \
  --cs_normalize random_walk \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight "${SMOOTH_WEIGHT:-0.0}" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_exp060_config_ensemble_infer.json"

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
