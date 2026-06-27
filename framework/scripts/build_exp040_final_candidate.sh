#!/usr/bin/env bash
set -euo pipefail

# Exp-040：最后一次提交候选包
#
# A1：Exp-039 最优方案
# - Exp-038 贪心加权 GAT 集成：64/16/20
# - Correct and Smooth：correct=(0.3,5,0.0), smooth=(0.75,5,0.75)
#
# A2：沿用 Exp-030 线上最佳方案
# - history 共现启发式
# - cooccur_decay=0.96
# - user_weight=0.02
# - user_combo_weight=0.18

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="output/exp040_submit_a1_exp039_greedy_a2_exp030"
mkdir -p "$OUT_DIR"

A1_TOP1="output/exp038_a1_gat_grid/h256_heads4_d040_lr005_wd5e4_seed42/best_model.pt"
A1_TOP2="output/exp038_a1_gat_grid/h224_heads4_d045_lr005_wd5e4_seed42/best_model.pt"
A1_TOP3="output/exp038_a1_gat_grid/h256_heads8_d045_lr005_wd5e4_seed3407/best_model.pt"

echo
echo "============================================================"
echo "[A1] 生成 Exp-039 greedy C&S 预测"
echo "============================================================"

python3 code/a1_correct_smooth.py \
  --data_path data/cls_data/A1.npz \
  --checkpoints "$A1_TOP1" "$A1_TOP2" "$A1_TOP3" \
  --checkpoint_weights 64,16,20 \
  --device "$DEVICE" \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --cs_normalize random_walk \
  --correct_alphas 0.3 \
  --correct_iters 5 \
  --correct_weights 0 \
  --smooth_alphas 0.75 \
  --smooth_iters 5 \
  --smooth_weights 0.75 \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0 \
  --smooth_alpha 0.75 \
  --smooth_iter 5 \
  --smooth_weight 0.75 \
  --output_json "$OUT_DIR/a1_exp039_cs.json" \
  --output_path "$OUT_DIR/A1.csv"

echo
echo "============================================================"
echo "[A2] 生成 Exp-030 推荐预测"
echo "============================================================"

python3 code/infer.py \
  --task task2 \
  --data_path data/rec_data \
  --output_path "$OUT_DIR/A2.csv" \
  --rec_strategy history \
  --seq_col item_seq_raw \
  --recent_n 10 \
  --cooccur_decay 0.96 \
  --history_filter none \
  --user_weight 0.02 \
  --user_combo_weight 0.18 \
  --user_combo_sizes 3,2,1 \
  --user_combo_mode prefix \
  --user_combo_min_count 5 \
  --history_count_weight 0 \
  --user_profile_cols auto \
  --topk 10 \
  --device "$DEVICE"

echo
echo "============================================================"
echo "[打包] prediction.zip"
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
echo "最终候选提交包：$OUT_DIR/prediction.zip"
