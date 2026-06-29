#!/usr/bin/env bash
set -euo pipefail

# Exp-065：A1 全标签/分折模型融合 + A2 len0+len1 联合候选
#
# 同步推进：
# - A1 不再固定 Exp-061，而是把 Exp-061 全标签模型与 Exp-060 分折模型融合；
# - A2 从 Exp-063 的 len0 扩展到 len0+len1。
#
# A1 融合直觉：
# - Exp-061 全标签重训线上更强，但没有验证集，可能有一点偏置；
# - Exp-060 分折模型线上也有效，保留少量权重可增强稳健性；
# - 默认权重：Exp061 0.85，Exp060 0.15。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp065_submit_a1_fulltrain_split_blend_a2_len0_len1}"

mkdir -p "$OUT_DIR"

EXP060_UNDIR_DIR="$EXP059_DIR/standard_struct_label_undir_h3_rw"
EXP060_UNDIR_REVERSE_DIR="$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw"

CHECKPOINTS=(
  "$EXP061_DIR/undir_seed42_e15/best_model.pt"
  "$EXP061_DIR/undir_seed777_e8/best_model.pt"
  "$EXP061_DIR/undir_seed2024_e21/best_model.pt"
  "$EXP061_DIR/undir_seed2026_e21/best_model.pt"
  "$EXP061_DIR/undir_seed3407_e12/best_model.pt"
  "$EXP061_DIR/undir_reverse_seed42_e18/best_model.pt"
  "$EXP061_DIR/undir_reverse_seed777_e15/best_model.pt"
  "$EXP061_DIR/undir_reverse_seed2024_e10/best_model.pt"
  "$EXP061_DIR/undir_reverse_seed2026_e13/best_model.pt"
  "$EXP061_DIR/undir_reverse_seed3407_e11/best_model.pt"
  "$EXP060_UNDIR_DIR/split42/best_model.pt"
  "$EXP060_UNDIR_DIR/split777/best_model.pt"
  "$EXP060_UNDIR_DIR/split2024/best_model.pt"
  "$EXP060_UNDIR_DIR/split2026/best_model.pt"
  "$EXP060_UNDIR_DIR/split3407/best_model.pt"
  "$EXP060_UNDIR_REVERSE_DIR/split42/best_model.pt"
  "$EXP060_UNDIR_REVERSE_DIR/split777/best_model.pt"
  "$EXP060_UNDIR_REVERSE_DIR/split2024/best_model.pt"
  "$EXP060_UNDIR_REVERSE_DIR/split2026/best_model.pt"
  "$EXP060_UNDIR_REVERSE_DIR/split3407/best_model.pt"
)

# Exp061内部仍按 0.3/0.7；总权重0.85：
# - Exp061 undir 每个 0.85*0.3/5 = 0.051
# - Exp061 undir_reverse 每个 0.85*0.7/5 = 0.119
# Exp060总权重0.15：
# - Exp060 undir 每个 0.15*0.3/5 = 0.009
# - Exp060 undir_reverse 每个 0.15*0.7/5 = 0.021
CHECKPOINT_WEIGHTS="0.051,0.051,0.051,0.051,0.051,0.119,0.119,0.119,0.119,0.119,0.009,0.009,0.009,0.009,0.009,0.021,0.021,0.021,0.021,0.021"

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "缺少checkpoint: $ckpt"
    echo "请先运行 Exp-059 和 Exp-061。"
    exit 1
  fi
done
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2"
  exit 1
fi
if [[ ! -f "$ALT_A2" ]]; then
  echo "ALT_A2不存在: $ALT_A2"
  exit 1
fi

echo
echo "============================================================"
echo "[A1] Exp061全标签 + Exp060分折模型融合"
echo "============================================================"

python3 code/a1_sign_infer.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --checkpoint_weights "$CHECKPOINT_WEIGHTS" \
  --device "$DEVICE" \
  --cs_normalize random_walk \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight "${SMOOTH_WEIGHT:-0.0}" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_exp065_fulltrain_split_blend_infer.json"

echo
echo "============================================================"
echo "[A2] 生成 len=0,len=1 桶替换"
echo "============================================================"

python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=0,len=1" \
  --output_path "$OUT_DIR/A2.csv" \
  --topk 10

echo
echo "============================================================"
echo "[打包] Exp-065"
echo "============================================================"

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
