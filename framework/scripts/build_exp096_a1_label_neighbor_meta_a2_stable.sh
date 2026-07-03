#!/usr/bin/env bash
set -euo pipefail

# Exp-096：A1 无泄漏邻居标签元特征 meta-stack + A2 稳定底座
#
# Exp096 的 A1 审计结果：
# - C=0.02, threshold=0.5
# - 5 split LOO mean_gain=+0.003470
# - min_gain=+0.000913
#
# A2 不在本实验里冒险，默认复用 Exp090/Exp086 的线上稳定版本。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp096_submit_a1_label_neighbor_meta_a2_stable}"
AUDIT_JSON="${AUDIT_JSON:-output/exp096_a1_label_neighbor_meta_audit/summary.json}"
EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP073_DIR="${EXP073_DIR:-output/exp073_a1_fulltrain_all_h2}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
A2_SOURCE="${A2_SOURCE:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"

if [[ ! -f "$AUDIT_JSON" ]]; then
  echo "AUDIT_JSON不存在: $AUDIT_JSON" >&2
  echo "请先运行 A1 Exp096 审计。" >&2
  exit 1
fi
if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A1] 邻居标签元特征 meta-stack 推理"
echo "============================================================"
echo "audit=$AUDIT_JSON"

python3 code/a1_sign_meta_stack.py infer \
  --data_path data/cls_data/A1.npz \
  --audit_json "$AUDIT_JSON" \
  --device "$DEVICE" \
  --split_seeds 42,777,2024,2026,3407 \
  --val_ratio 0.1 \
  --stratified_split \
  --use_label_neighbor_meta \
  --label_neighbor_hops 2 \
  --sources \
    "undir=$EXP061_DIR/undir_seed*_e*/best_model.pt" \
    "undir_reverse=$EXP061_DIR/undir_reverse_seed*_e*/best_model.pt" \
    "all_h2=$EXP073_DIR/all_h2_seed*_e*/best_model.pt" \
  --oof_sources \
    "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw" \
    "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw" \
    "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_label_neighbor_meta_infer.json"

echo
echo "============================================================"
echo "[A2] 复用线上稳定 A2"
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
