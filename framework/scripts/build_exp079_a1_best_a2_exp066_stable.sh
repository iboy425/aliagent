#!/usr/bin/env bash
set -euo pipefail

# Exp-079：A1 保留 Exp075/078 最优结果，A2 回滚到 Exp066 稳定线上结果
#
# 线上反馈：
# - Exp078 A1=0.7601，是当前最好 A1；
# - Exp078 A2=0.5014，低于 Exp066 A2=0.5033；
# - 因此本候选只保留 A1 正收益，撤销 A2 的 EB/profile gate 改动。

cd "$(dirname "$0")/.."

A1_SOURCE="${A1_SOURCE:-output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv}"
if [[ ! -f "$A1_SOURCE" ]]; then
  A1_SOURCE="output/exp075_submit_a1_meta_threshold_a2_eb/A1.csv"
fi
A2_SOURCE="${A2_SOURCE:-output/exp066_submit_a1_blend90_a2_len0_len1/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp079_submit_a1_best_a2_exp066_stable}"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  exit 1
fi
if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE"
  exit 1
fi

mkdir -p "$OUT_DIR"

cp "$A1_SOURCE" "$OUT_DIR/A1.csv"
cp "$A2_SOURCE" "$OUT_DIR/A2.csv"

echo
echo "============================================================"
echo "[Exp-079] A1 best + A2 Exp066 stable"
echo "============================================================"
echo "A1_SOURCE=$A1_SOURCE"
echo "A2_SOURCE=$A2_SOURCE"
echo "OUT_DIR=$OUT_DIR"

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
