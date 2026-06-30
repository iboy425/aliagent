#!/usr/bin/env bash
set -euo pipefail

# Exp-068：A1 Exp066路线 + A2 Exp044/Exp045按桶RRF软融合
#
# 背景：
# - Exp063证明 len=0 硬替换有效；
# - Exp066证明 A1 全标签权重0.90小幅有效，A2 len0+len1没有伤分；
# - Exp045整体替换线上下降，因此不再做 len2-3 硬替换，而是用RRF软融合。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp068_submit_a1_blend90_a2_rrf_soft}"
BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
FULLTRAIN_WEIGHT="${FULLTRAIN_WEIGHT:-0.90}"
RRF_K="${RRF_K:-60}"

# 默认权重比较保守：
# - len=0：Exp045已被线上证明在冷启动有局部价值，给较高权重；
# - len=1：Exp066没有明显增益也没有伤分，给中低权重；
# - len=2-3：只给很小权重，避免大面积漂移；
# - 长历史桶样本少，保留极小探索权重。
BUCKET_LAMBDAS="${BUCKET_LAMBDAS:-len=0:1.20,len=1:0.35,len=2-3:0.20,len=4-10:0.05,len>10:0.05}"
BUCKET_POP_WEIGHTS="${BUCKET_POP_WEIGHTS:-}"

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A1] 生成/复用 Exp066 路线 A1"
echo "============================================================"

FULLTRAIN_WEIGHT="$FULLTRAIN_WEIGHT" \
A2_BUCKETS="base" \
OUT_DIR="$OUT_DIR" \
DEVICE="$DEVICE" \
./scripts/build_joint_a1_sign_a2_bucket_candidate.sh

echo
echo "============================================================"
echo "[A2] Exp044/Exp045 RRF按桶软融合"
echo "============================================================"

python3 code/a2_rank_fusion.py \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --train_csv data/rec_data/train.csv \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --bucket_lambdas "$BUCKET_LAMBDAS" \
  --bucket_pop_weights "$BUCKET_POP_WEIGHTS" \
  --rrf_k "$RRF_K" \
  --topk 10 \
  --output_path "$OUT_DIR/A2.csv"

echo
echo "============================================================"
echo "[打包] Exp-068"
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
