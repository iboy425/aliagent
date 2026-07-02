#!/usr/bin/env bash
set -euo pipefail

# Exp-086：A1 best + A2严格分群gate，仅用于中长历史用户，并保护base Top1
#
# 相比 Exp085：
# - 继续使用 Exp081 严格筛出的画像分群；
# - 只启用 len=4-10 和 len>10，排除人数较多且收益较弱的 len=2-3；
# - 命中分群时保留 Exp066 的 Top1，只调整第 2-10 位。

cd "$(dirname "$0")/.."

POLICY_JSON="${POLICY_JSON:-output/exp081_a2_profile_gate_strict_audit/policy.json}"
A1_SOURCE="${A1_SOURCE:-output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv}"
if [[ ! -f "$A1_SOURCE" ]]; then
  A1_SOURCE="output/exp075_submit_a1_meta_threshold_a2_eb/A1.csv"
fi
BASE_A2="${BASE_A2:-output/exp066_submit_a1_blend90_a2_len0_len1/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
PREDICT_BUCKETS="${PREDICT_BUCKETS:-len=4-10,len>10}"
OUT_DIR="${OUT_DIR:-output/exp086_submit_a1_best_a2_strict_gate_midlong_keep_top1}"

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "POLICY_JSON不存在: $POLICY_JSON"
  echo "请先运行 scripts/run_exp081_a2_profile_gate_strict_audit.sh"
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
echo "[A2] 严格分群gate + 中长历史 + 保护Top1"
echo "============================================================"
echo "policy=$POLICY_JSON"
echo "base=$BASE_A2"
echo "alt=$ALT_A2"
echo "predict_buckets=$PREDICT_BUCKETS"

python3 code/a2_profile_gate.py predict \
  --policy_json "$POLICY_JSON" \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --test_csv data/rec_data/test.csv \
  --user_csv data/rec_data/user.csv \
  --seq_col item_seq_raw \
  --keep_topn 1 \
  --predict_buckets "$PREDICT_BUCKETS" \
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
