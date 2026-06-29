#!/usr/bin/env bash
set -euo pipefail

# Exp-064：A1 Exp-061 + A2 len0+len1 冷启动/短历史替换联合候选
#
# 背景：
# - Exp-063 线上 A1/A2 均小幅提升；
# - A2 len0 替换从 0.5023 提到 0.5033，说明冷启动桶方向有效；
# - 本实验在保持 A1 不变的前提下，把 A2 替换桶从 len=0 扩展到 len=0,len=1。

cd "$(dirname "$0")/.."

A1_SOURCE="${A1_SOURCE:-output/exp061_a1_fulltrain_config_ensemble_candidate/A1.csv}"
BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp064_submit_a1_exp061_a2_len0_len1}"

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
echo "[A2] 生成 len=0,len=1 冷启动/短历史桶替换"
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
echo "[打包] A1 Exp-061 + A2 len0_len1"
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
