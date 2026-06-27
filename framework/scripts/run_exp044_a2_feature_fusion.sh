#!/usr/bin/env bash
set -euo pipefail

# Exp-044：A2用户画像序列模型训练后的融合评估与候选生成
#
# 用法：
#   ./scripts/run_exp044_a2_feature_fusion.sh output/exp044_a2_feature_ranker_seed42/best_model.pt
#
# 逻辑：
# 1. 使用 checkpoint 对验证集生成模型分数；
# 2. 与 Exp-043 最稳的 jaccard 规则融合，搜索 model_weight；
# 3. 如果最佳 model_weight > 0，则生成 A2_fusion.csv 和 prediction.zip；
# 4. A1 默认沿用 Exp-030 线上更稳版本。

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-${CHECKPOINT:-}}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "用法: $0 <a2_feature_ranker_checkpoint>"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint不存在: $CHECKPOINT"
  exit 1
fi

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

MAX_LEN="${MAX_LEN:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-0,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0}"
OUT_DIR="${OUT_DIR:-output/exp044_a2_feature_fusion}"
A1_SOURCE="${A1_SOURCE:-output/exp030_submit_a1cs_a2_decay_combo018/A1.csv}"
FORCE_BUILD="${FORCE_BUILD:-0}"

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[评估] A2模型 + jaccard规则融合"
echo "============================================================"
echo "checkpoint=$CHECKPOINT"
echo "out_dir=$OUT_DIR"

python3 code/a2_feature_ranker.py eval_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --test_like_val \
  --model_weights "$MODEL_WEIGHTS" \
  --output_json "$OUT_DIR/fusion_eval.json"

BEST_WEIGHT="$(python3 -c "import json; d=json.load(open('$OUT_DIR/fusion_eval.json')); print(d['best']['model_weight'])")"
BEST_WEIGHTED_NDCG="$(python3 -c "import json; d=json.load(open('$OUT_DIR/fusion_eval.json')); print(d['best']['weighted_ndcg'])")"
BASE_WEIGHTED_NDCG="$(python3 -c "import json; d=json.load(open('$OUT_DIR/fusion_eval.json')); print(next(x['weighted_ndcg'] for x in d['results'] if x['model_weight'] == 0))")"

echo
echo "============================================================"
echo "[结论] 融合权重搜索"
echo "============================================================"
echo "baseline(model_weight=0) weighted_NDCG=$BASE_WEIGHTED_NDCG"
echo "best_model_weight=$BEST_WEIGHT"
echo "best weighted_NDCG=$BEST_WEIGHTED_NDCG"

SHOULD_BUILD="$(python3 -c "w=float('$BEST_WEIGHT'); force=int('$FORCE_BUILD'); print(1 if force or w > 0 else 0)")"
if [[ "$SHOULD_BUILD" != "1" ]]; then
  echo
  echo "最佳权重为0，说明当前模型没有给jaccard规则带来增益；默认不生成提交包。"
  echo "如需强制生成，设置 FORCE_BUILD=1 后重跑。"
  exit 0
fi

echo
echo "============================================================"
echo "[推理] 生成 A2_fusion.csv"
echo "============================================================"

python3 code/a2_feature_ranker.py predict_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT" \
  --output_path "$OUT_DIR/A2.csv" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --model_weight "$BEST_WEIGHT"

echo
echo "============================================================"
echo "[打包] prediction.zip"
echo "============================================================"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  exit 1
fi

cp "$A1_SOURCE" "$OUT_DIR/A1.csv"
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
