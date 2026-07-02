#!/usr/bin/env bash
set -euo pipefail

# Exp-085：A1 best + A2严格分群gate + 保护base Top1
#
# 结合 Exp082 和 Exp084：
# - 使用 Exp081 严格筛出的 7 个画像分群，避免 Exp084 全桶大范围改第2-10位；
# - 命中分群时保护 Exp066 的 Top1，避免 Exp082 改 Top1 带来的风险；
# - 目标是低覆盖、小漂移、保留 A2 稳定 Top1，同时补强后续排序。

cd "$(dirname "$0")/.."

POLICY_JSON="${POLICY_JSON:-output/exp081_a2_profile_gate_strict_audit/policy.json}"
A1_SOURCE="${A1_SOURCE:-output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv}"
if [[ ! -f "$A1_SOURCE" ]]; then
  A1_SOURCE="output/exp075_submit_a1_meta_threshold_a2_eb/A1.csv"
fi
BASE_A2="${BASE_A2:-output/exp066_submit_a1_blend90_a2_len0_len1/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp085_submit_a1_best_a2_strict_gate_keep_top1}"

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
echo "[A2] 严格分群gate + 保护Top1"
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
  --keep_topn 1 \
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
