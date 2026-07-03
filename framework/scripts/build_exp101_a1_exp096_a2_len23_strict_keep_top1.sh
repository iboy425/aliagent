#!/usr/bin/env bash
set -euo pipefail

# Exp-101：A1 使用 Exp096，A2 在 Exp090 稳定底座上只启用 len=2-3 严格分群
#
# 背景：
# - Exp090 是当前线上 A2 最优稳定底座，A2=0.5035；
# - Exp086/090 已经启用了中长历史严格分群，并保护 Top1；
# - Exp081 严格策略里还有两个 len=2-3 分群离线稳定正收益，但此前没有单独提交；
# - 本候选只在这两个分群上改第 2-10 位，Top1 完全保留，降低线上漂移风险。

cd "$(dirname "$0")/.."

POLICY_JSON="${POLICY_JSON:-output/exp081_a2_profile_gate_strict_audit/policy.json}"
A1_SOURCE="${A1_SOURCE:-output/exp096_submit_a1_label_neighbor_meta_a2_stable/A1.csv}"
BASE_A2="${BASE_A2:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/exp101_submit_a1_exp096_a2_len23_strict_keep_top1}"

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "POLICY_JSON不存在: $POLICY_JSON" >&2
  exit 1
fi
if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  exit 1
fi
if [[ ! -f "$ALT_A2" ]]; then
  echo "ALT_A2不存在: $ALT_A2" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A2] Exp090 + len=2-3 严格分群 + 保护Top1"
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
  --predict_buckets "len=2-3" \
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
