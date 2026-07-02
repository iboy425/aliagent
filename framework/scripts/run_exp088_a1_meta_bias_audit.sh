#!/usr/bin/env bash
set -euo pipefail

# Exp-088：A1 当前最强 meta-stack 后的类别概率微调审计
#
# 目的：
# - 不重训底层 SIGN 模型；
# - 基于 Exp075 的 OOF 元模型输出搜索低改动 class-bias；
# - 只有当审计显示 mean_gain 为正且改动比例可控时，才进入联合提交候选。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp088_a1_meta_bias_audit}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
BASE_AUDIT_JSON="${BASE_AUDIT_JSON:-output/exp075_a1_meta_stack_threshold_audit/summary.json}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$BASE_AUDIT_JSON" ]]; then
  echo "缺少 Exp075 审计文件: $BASE_AUDIT_JSON" >&2
  echo "请先运行 scripts/build_exp075_a1_meta_threshold_a2_eb_candidate.sh" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[Exp-088] A1 meta bias 审计"
echo "============================================================"

python3 code/a1_meta_bias_search.py audit \
  --data_path data/cls_data/A1.npz \
  --output_dir "$OUT_DIR" \
  --device "$DEVICE" \
  --base_audit_json "$BASE_AUDIT_JSON" \
  --split_seeds 42,777,2024,2026,3407 \
  --val_ratio 0.1 \
  --stratified_split \
  --sources \
    "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw" \
    "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw" \
    "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw" \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --factors 0.85,0.9,0.95,1.0,1.03,1.05,1.08,1.1,1.15,1.2,1.3

echo
echo "============================================================"
echo "[Exp-088完成]"
echo "============================================================"
echo "请把摘要和文件内容发回来：$OUT_DIR/summary.json"
