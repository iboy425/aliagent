#!/usr/bin/env bash
set -euo pipefail

# Exp-075：A1 置信度回退 meta-stack + A2 冷启动 EB 联合候选
#
# 相比 Exp-074：
# - A1 元模型不再无条件覆盖 base；
# - 只有元模型置信度达到阈值，或者与 base 预测一致时，才使用元模型；
# - OOF 审计最佳参数：C=0.005, threshold=0.3，平均提升更高且改动比例更小。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp075_submit_a1_meta_threshold_a2_eb}"
EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP073_DIR="${EXP073_DIR:-output/exp073_a1_fulltrain_all_h2}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
A2_SOURCE="${A2_SOURCE:-output/exp070_submit_a1_exp066_a2_coldstart_eb/A2.csv}"
AUDIT_DIR="${AUDIT_DIR:-output/exp075_a1_meta_stack_threshold_audit}"
AUDIT_JSON="$AUDIT_DIR/summary.json"

mkdir -p "$OUT_DIR" "$AUDIT_DIR"

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "缺少A2来源: $A2_SOURCE" >&2
  echo "请先运行：bash scripts/build_exp070_a1_exp066_a2_coldstart_eb_candidate.sh" >&2
  exit 1
fi

all_h2_count="$(find "$EXP073_DIR" -maxdepth 2 -name best_model.pt 2>/dev/null | wc -l || true)"
if [[ "$all_h2_count" -lt 5 ]]; then
  echo "Exp073 all_h2 checkpoint不足: $all_h2_count，需要5个" >&2
  echo "请先运行：CUDA_VISIBLE_DEVICES=0 DEVICE=cuda bash scripts/run_exp073_a1_fulltrain_all_h2.sh" >&2
  exit 1
fi

if [[ ! -f "$AUDIT_JSON" ]]; then
  echo
  echo "============================================================"
  echo "[A1] OOF元模型阈值审计"
  echo "============================================================"
  python3 code/a1_sign_meta_stack.py audit \
    --data_path data/cls_data/A1.npz \
    --output_dir "$AUDIT_DIR" \
    --device "$DEVICE" \
    --split_seeds 42,777,2024,2026,3407 \
    --val_ratio 0.1 \
    --stratified_split \
    --sources \
      "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw" \
      "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw" \
      "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw" \
    --c_values 0.003,0.005,0.01 \
    --class_weights none \
    --thresholds 0,0.3,0.4,0.5,0.6,0.7,0.8,0.9
fi

echo
echo "============================================================"
echo "[A1] 置信度回退 meta-stack 推理"
echo "============================================================"

python3 code/a1_sign_meta_stack.py infer \
  --data_path data/cls_data/A1.npz \
  --audit_json "$AUDIT_JSON" \
  --device "$DEVICE" \
  --sources \
    "undir=$EXP061_DIR/undir_seed*_e*/best_model.pt" \
    "undir_reverse=$EXP061_DIR/undir_reverse_seed*_e*/best_model.pt" \
    "all_h2=$EXP073_DIR/all_h2_seed*_e*/best_model.pt" \
  --oof_sources \
    "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw" \
    "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw" \
    "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw" \
  --split_seeds 42,777,2024,2026,3407 \
  --val_ratio 0.1 \
  --stratified_split \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_meta_threshold_infer.json"

echo
echo "============================================================"
echo "[A2] 复用 Exp070 冷启动EB"
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
