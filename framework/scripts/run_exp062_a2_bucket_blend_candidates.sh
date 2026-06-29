#!/usr/bin/env bash
set -euo pipefail

# Exp-062：A2受控桶替换候选
#
# 背景：
# - Exp-044 是当前线上最佳 A2=0.5023；
# - Exp-045 多seed模型离线更强，但整体提交线上降到0.4957；
# - 因此不能整体替换，需要只对冷启动/短历史桶做受控替换。
#
# 本脚本以 Exp-044 为 base，以 Exp-045 为 alt，生成不同历史长度桶的混合候选。

cd "$(dirname "$0")/.."

BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
A1_SOURCE="${A1_SOURCE:-output/exp060_submit_a1_config_ensemble_a2_exp044/A1.csv}"
OUT_DIR="${OUT_DIR:-output/exp062_a2_bucket_blend_candidates}"

if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2"
  exit 1
fi
if [[ ! -f "$ALT_A2" ]]; then
  echo "ALT_A2不存在: $ALT_A2"
  exit 1
fi
if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  exit 1
fi

mkdir -p "$OUT_DIR"

build_one() {
  local name="$1"
  local buckets="$2"
  local dir="$OUT_DIR/$name"
  mkdir -p "$dir"

  echo
  echo "============================================================"
  echo "[A2桶混合] $name buckets=$buckets"
  echo "============================================================"

  python3 code/a2_bucket_blend.py \
    --base_a2 "$BASE_A2" \
    --alt_a2 "$ALT_A2" \
    --test_csv data/rec_data/test.csv \
    --seq_col item_seq_raw \
    --buckets "$buckets" \
    --output_path "$dir/A2.csv" \
    --topk 10

  cp "$A1_SOURCE" "$dir/A1.csv"
  (
    cd "$dir"
    rm -f prediction.zip
    zip -q prediction.zip A1.csv A2.csv
  )

  python3 code/validate_submission.py \
    --zip_path "$dir/prediction.zip" \
    --cls_data_path data/cls_data/A1.npz \
    --rec_data_dir data/rec_data \
    --topk 10

  echo "[候选包] $dir/prediction.zip"
}

build_one "len0" "len=0"
build_one "len1" "len=1"
build_one "len2_3" "len=2-3"
build_one "len0_len1" "len=0,len=1"
build_one "len0_len1_len2_3" "len=0,len=1,len=2-3"

echo
echo "============================================================"
echo "[Exp-062完成]"
echo "============================================================"
echo "建议提交优先级："
echo "1. len0：低风险，只替换空历史用户。"
echo "2. len0_len1：中等风险，替换空历史和单历史用户。"
echo "3. len0_len1_len2_3：高风险，影响约90%用户，仅在线上次数充足时尝试。"
