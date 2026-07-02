#!/usr/bin/env bash
set -euo pipefail

# Exp-090：A1 稳定 meta-bias + A2 Exp086 联合候选
#
# Exp088 审计结果中：
# - c8_x1.05 平均提升最高，但 min_gain 为负；
# - c2_x0.95 平均提升为正、min_gain=0、改动比例更小。
#
# 因此 Exp090 强制选择 c2_x0.95，用作更稳的 A1+A2 联合提交候选。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

AUDIT_JSON="${AUDIT_JSON:-output/exp088_a1_meta_bias_audit/summary.json}"
SELECT_NAME="${SELECT_NAME:-c2_x0.95}"
OUT_DIR="${OUT_DIR:-output/exp090_submit_a1_safe_bias_a2_exp086}"
EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP073_DIR="${EXP073_DIR:-output/exp073_a1_fulltrain_all_h2}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
A2_SOURCE="${A2_SOURCE:-output/exp086_submit_a1_best_a2_strict_gate_midlong_keep_top1/A2.csv}"

if [[ ! -f "$AUDIT_JSON" ]]; then
  echo "AUDIT_JSON不存在: $AUDIT_JSON" >&2
  echo "请先运行 scripts/run_exp088_a1_meta_bias_audit.sh" >&2
  exit 1
fi
if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE" >&2
  echo "请先运行 scripts/build_exp086_a1_best_a2_strict_gate_midlong_keep_top1.sh" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A1] 稳定 meta bias 推理"
echo "============================================================"
echo "select_name=$SELECT_NAME"

python3 code/a1_meta_bias_search.py infer \
  --data_path data/cls_data/A1.npz \
  --audit_json "$AUDIT_JSON" \
  --select_name "$SELECT_NAME" \
  --device "$DEVICE" \
  --split_seeds 42,777,2024,2026,3407 \
  --val_ratio 0.1 \
  --stratified_split \
  --sources \
    "undir=$EXP061_DIR/undir_seed*_e*/best_model.pt" \
    "undir_reverse=$EXP061_DIR/undir_reverse_seed*_e*/best_model.pt" \
    "all_h2=$EXP073_DIR/all_h2_seed*_e*/best_model.pt" \
  --oof_sources \
    "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw" \
    "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw" \
    "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_meta_bias_infer.json"

echo
echo "============================================================"
echo "[A2] 复用 Exp086"
echo "============================================================"
cp "$A2_SOURCE" "$OUT_DIR/A2.csv"

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
