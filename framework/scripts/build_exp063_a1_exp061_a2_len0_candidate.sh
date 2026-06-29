#!/usr/bin/env bash
set -euo pipefail

# Exp-063：A1 Exp-061 + A2 len0冷启动替换联合候选
#
# 目的：
# - 一次线上提交同时获得 A1 和 A2 反馈；
# - A1 使用 Exp-061 全标签重训候选；
# - A2 使用低风险 len=0 冷启动桶替换候选。
#
# 依赖：
# - 已运行 Exp-061，生成 A1.csv；
# - A2 len0 若不存在，本脚本会自动用 Exp-044/Exp-045 生成。

cd "$(dirname "$0")/.."

A1_SOURCE="${A1_SOURCE:-output/exp061_a1_fulltrain_config_ensemble_candidate/A1.csv}"
BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp063_submit_a1_exp061_a2_len0}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  echo "请先运行：CUDA_VISIBLE_DEVICES=0 ./scripts/run_exp061_a1_fulltrain_config_ensemble_candidate.sh"
  exit 1
fi
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2"
  exit 1
fi
if [[ ! -f "$ALT_A2" ]]; then
  echo "ALT_A2不存在: $ALT_A2"
  echo "如GPU机器没有Exp-045输出，请从本地同步 output/exp045_a2_feature_multiseed/A2.csv，或先运行Exp-045。"
  exit 1
fi

echo
echo "============================================================"
echo "[A2] 生成 len=0 冷启动桶替换"
echo "============================================================"

python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=0" \
  --output_path "$OUT_DIR/A2.csv" \
  --topk 10

echo
echo "============================================================"
echo "[打包] A1 Exp-061 + A2 len0"
echo "============================================================"

cp "$A1_SOURCE" "$OUT_DIR/A1.csv"
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
