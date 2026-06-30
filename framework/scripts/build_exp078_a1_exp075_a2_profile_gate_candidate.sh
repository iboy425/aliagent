#!/usr/bin/env bash
set -euo pipefail

# Exp-078：A1沿用Exp075 + A2画像分群 gate 候选
#
# 只有当 Exp077 审计显示 gated 相比 base 有稳定明显提升时才运行。

cd "$(dirname "$0")/.."

POLICY_JSON="${POLICY_JSON:-output/exp077_a2_profile_gate_audit/policy.json}"
A1_SOURCE="${A1_SOURCE:-output/exp075_submit_a1_meta_threshold_a2_eb/A1.csv}"
BASE_A2="${BASE_A2:-output/exp075_submit_a1_meta_threshold_a2_eb/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp078_submit_a1_exp075_a2_profile_gate}"

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "POLICY_JSON不存在: $POLICY_JSON"
  echo "请先运行 scripts/run_exp077_a2_profile_gate_audit.sh"
  exit 1
fi
if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  exit 1
fi
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2"
  exit 1
fi
if [[ ! -f "$ALT_A2" ]]; then
  echo "ALT_A2不存在: $ALT_A2"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A2] 根据画像分群 gate 生成候选"
echo "============================================================"
echo "policy=$POLICY_JSON"
echo "base=$BASE_A2"
echo "alt=$ALT_A2"

python3 code/a2_profile_gate.py predict \
  --policy_json "$POLICY_JSON" \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --test_csv data/rec_data/test.csv \
  --user_csv data/rec_data/user.csv \
  --seq_col item_seq_raw \
  --topk 10 \
  --output_path "$OUT_DIR/A2.csv"

cp "$A1_SOURCE" "$OUT_DIR/A1.csv"

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
echo "候选提交包：$OUT_DIR/prediction.zip"
