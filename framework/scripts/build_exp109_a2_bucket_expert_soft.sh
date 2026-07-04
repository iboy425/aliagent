#!/usr/bin/env bash
set -euo pipefail

# Exp-109：A2 分桶专家软融合候选
#
# Exp107 的 hard / keep1 / mixed 都改动过大：
# - hard/mixed 会移动短历史 Top1；
# - keep1 虽保 Top1，但第2-10位整体漂移很大。
#
# 本脚本只把专家作为弱排序信号，用 RRF 小权重重排对应桶。
# 目标是：Top1 基本不变，Top10 集合尽量不变，只微调排序。

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-output/exp109_a2_bucket_expert_soft}"
BASE_A2="${BASE_A2:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"
A1_SOURCE="${A1_SOURCE:-output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv}"
EXP107_DIR="${EXP107_DIR:-output/exp107_a2_bucket_experts}"

if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  exit 1
fi
if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE" >&2
  exit 1
fi
for expert in expert_len0 expert_len1 expert_len3; do
  if [[ ! -f "$EXP107_DIR/$expert/A2.csv" ]]; then
    echo "缺少 $EXP107_DIR/$expert/A2.csv，请先运行 scripts/run_exp107_a2_bucket_experts.sh" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

build_soft() {
  local name="$1"
  local l0="$2"
  local l1="$3"
  local l23="$4"

  echo
  echo "============================================================"
  echo "[Exp-109] 构造 $name: len0=$l0 len1=$l1 len2-3=$l23"
  echo "============================================================"

  python3 code/a2_rank_fusion.py \
    --base_a2 "$BASE_A2" \
    --alt_a2 "$EXP107_DIR/expert_len0/A2.csv" \
    --test_csv data/rec_data/test.csv \
    --train_csv data/rec_data/train.csv \
    --seq_col item_seq_raw \
    --bucket_lambdas "len=0:${l0},len=1:0,len=2-3:0,len=4-10:0,len>10:0" \
    --rrf_k 60 \
    --output_path "$OUT_DIR/tmp_${name}_len0.csv"

  python3 code/a2_rank_fusion.py \
    --base_a2 "$OUT_DIR/tmp_${name}_len0.csv" \
    --alt_a2 "$EXP107_DIR/expert_len1/A2.csv" \
    --test_csv data/rec_data/test.csv \
    --train_csv data/rec_data/train.csv \
    --seq_col item_seq_raw \
    --bucket_lambdas "len=0:0,len=1:${l1},len=2-3:0,len=4-10:0,len>10:0" \
    --rrf_k 60 \
    --output_path "$OUT_DIR/tmp_${name}_len1.csv"

  python3 code/a2_rank_fusion.py \
    --base_a2 "$OUT_DIR/tmp_${name}_len1.csv" \
    --alt_a2 "$EXP107_DIR/expert_len3/A2.csv" \
    --test_csv data/rec_data/test.csv \
    --train_csv data/rec_data/train.csv \
    --seq_col item_seq_raw \
    --bucket_lambdas "len=0:0,len=1:0,len=2-3:${l23},len=4-10:0,len>10:0" \
    --rrf_k 60 \
    --output_path "$OUT_DIR/A2_${name}.csv"

  echo
  echo "[差异分析] $name vs Exp090 A2"
  python3 code/a2_compare_submissions.py \
    --base_a2 "$BASE_A2" \
    --new_a2 "$OUT_DIR/A2_${name}.csv" \
    --test_csv data/rec_data/test.csv \
    --seq_col item_seq_raw \
    --topk 10 \
    --topn 10

  local submit_dir="$OUT_DIR/submit_${name}"
  mkdir -p "$submit_dir"
  cp "$A1_SOURCE" "$submit_dir/A1.csv"
  cp "$OUT_DIR/A2_${name}.csv" "$submit_dir/A2.csv"
  (
    cd "$submit_dir"
    rm -f prediction.zip
    zip -q prediction.zip A1.csv A2.csv
  )
  python3 code/validate_submission.py \
    --zip_path "$submit_dir/prediction.zip" \
    --cls_data_path data/cls_data/A1.npz \
    --rec_data_dir data/rec_data \
    --topk 10
  echo "候选包：$submit_dir/prediction.zip"
}

build_soft "safe" 0.03 0.03 0.02
build_soft "light" 0.05 0.05 0.03
build_soft "mid" 0.08 0.08 0.05
build_soft "len23_only" 0.0 0.0 0.04

echo
echo "Exp-109 完成。提交前优先看 changed/top1_changed/overlap："
echo "- safe/light：保守试探；"
echo "- mid：更激进；"
echo "- len23_only：只动最大短历史桶，便于判断 len=2-3 专家是否有用。"
