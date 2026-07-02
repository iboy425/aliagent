#!/usr/bin/env bash
set -euo pipefail

# Exp-084：A1 best + A2保护TopN融合候选
#
# 只有当 Exp083 审计显示 mean_gain>0 且 min_gain 不明显为负时才运行。

cd "$(dirname "$0")/.."

AUDIT_JSON="${AUDIT_JSON:-output/exp083_a2_protected_blend_audit/results.json}"
A1_SOURCE="${A1_SOURCE:-output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv}"
if [[ ! -f "$A1_SOURCE" ]]; then
  A1_SOURCE="output/exp075_submit_a1_meta_threshold_a2_eb/A1.csv"
fi
BASE_A2="${BASE_A2:-output/exp066_submit_a1_blend90_a2_len0_len1/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp084_submit_a1_best_a2_protected_blend}"

if [[ ! -f "$AUDIT_JSON" ]]; then
  echo "AUDIT_JSON不存在: $AUDIT_JSON"
  echo "请先运行 scripts/run_exp083_a2_protected_blend_audit.sh"
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
echo "[A2] 保护TopN融合"
echo "============================================================"
echo "audit=$AUDIT_JSON"
echo "base=$BASE_A2"
echo "alt=$ALT_A2"

python3 code/a2_protected_blend.py predict \
  --audit_json "$AUDIT_JSON" \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --test_csv data/rec_data/test.csv \
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
