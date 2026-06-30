#!/usr/bin/env bash
set -euo pipefail

# Exp-070：保留线上最佳 A1，A2 只在空历史桶加入经验贝叶斯画像先验
#
# 线上反馈：
# - Exp066：A1=0.7594, A2=0.5033，是当前最佳稳定线；
# - Exp069：A1 gate + A2 RRF/suffix 同时下降，说明不能再动非冷启动桶。
#
# 本候选只改 A2 的 len=0：
# - base 使用 Exp066 的 A2；
# - 对 len=0 用户，用用户画像 EB 先验和 base 列表做 RRF；
# - len>=1 用户完全不变。

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-output/exp070_submit_a1_exp066_a2_coldstart_eb}"
BASE_DIR="${BASE_DIR:-output/exp066_submit_a1_blend90_a2_len0_len1}"
SEARCH_JSON="${SEARCH_JSON:-output/exp070_a2_coldstart_eb/search_seed42_confirm.json}"
EB_WEIGHT="${EB_WEIGHT:-1.0}"
RRF_K="${RRF_K:-20}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$BASE_DIR/A1.csv" ]]; then
  echo "缺少 A1 基准文件: $BASE_DIR/A1.csv" >&2
  exit 1
fi
if [[ ! -f "$BASE_DIR/A2.csv" ]]; then
  echo "缺少 A2 基准文件: $BASE_DIR/A2.csv" >&2
  exit 1
fi

if [[ ! -f "$SEARCH_JSON" ]]; then
  echo
  echo "============================================================"
  echo "[A2] 搜索冷启动 EB 参数"
  echo "============================================================"
  python3 code/a2_coldstart_eb.py search \
    --data_path data/rec_data \
    --val_ratio 0.1 \
    --seed 42 \
    --max_val_samples 4000 \
    --depths 3 \
    --alpha_grid "200;120;60" \
    --temperatures 1.0 \
    --output_json "$SEARCH_JSON"
fi

echo
echo "============================================================"
echo "[A1] 复用 Exp066 线上最佳 A1"
echo "============================================================"
cp "$BASE_DIR/A1.csv" "$OUT_DIR/A1.csv"

echo
echo "============================================================"
echo "[A2] len=0 冷启动 EB + RRF"
echo "============================================================"
python3 code/a2_coldstart_eb.py predict \
  --data_path data/rec_data \
  --base_a2 "$BASE_DIR/A2.csv" \
  --search_json "$SEARCH_JSON" \
  --replace_buckets "len=0" \
  --eb_weight "$EB_WEIGHT" \
  --rrf_k "$RRF_K" \
  --output_path "$OUT_DIR/A2.csv"

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
