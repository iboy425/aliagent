#!/usr/bin/env bash
set -euo pipefail

# Exp-045：A2 feature ranker 多 seed 训练 + logits 平均融合
#
# 目标：
# - 验证 Exp-044 单 seed 模型融合在线上有效后，继续降低 seed 偶然性；
# - 多个 feature ranker checkpoint 的 logits 先平均，再与 jaccard 规则融合；
# - 如果离线融合优于单 seed，则生成新的 prediction.zip 候选。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SEEDS="${SEEDS:-42,777,2024}"
BASE_OUT="${BASE_OUT:-output/exp045_a2_feature_multiseed}"
MAX_LEN="${MAX_LEN:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-120}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
USER_EMBEDDING_DIM="${USER_EMBEDDING_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-0,0.01,0.02,0.05,0.1,0.2,0.5,0.8,1.0,1.2,1.5,2.0,3.0}"
A1_SOURCE="${A1_SOURCE:-output/exp030_submit_a1cs_a2_decay_combo018/A1.csv}"
FORCE_BUILD="${FORCE_BUILD:-0}"

mkdir -p "$BASE_OUT"

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
CHECKPOINTS=()

echo
echo "============================================================"
echo "[训练] A2 feature ranker 多 seed"
echo "============================================================"
echo "seeds=$SEEDS"
echo "base_out=$BASE_OUT"

for seed in "${SEED_LIST[@]}"; do
  seed="$(echo "$seed" | xargs)"
  out_dir="$BASE_OUT/seed${seed}"
  ckpt="$out_dir/best_model.pt"
  mkdir -p "$out_dir"

  if [[ -f "$ckpt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] seed=$seed 已存在 $ckpt"
  else
    echo
    echo "------------------------------------------------------------"
    echo "[训练] seed=$seed"
    echo "------------------------------------------------------------"
    python3 code/a2_feature_ranker.py train \
      --data_path data/rec_data \
      --output_dir "$out_dir" \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --max_len "$MAX_LEN" \
      --embedding_dim "$EMBEDDING_DIM" \
      --user_embedding_dim "$USER_EMBEDDING_DIM" \
      --hidden_dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --patience "$PATIENCE" \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed" \
      --test_like_val \
      --random_test_like_train
  fi
  CHECKPOINTS+=("$ckpt")
done

CHECKPOINT_ARG="$(IFS=','; echo "${CHECKPOINTS[*]}")"
echo "$CHECKPOINT_ARG" > "$BASE_OUT/checkpoints.txt"

echo
echo "============================================================"
echo "[汇总] 单 seed 训练指标"
echo "============================================================"
BASE_OUT_FOR_SUMMARY="$BASE_OUT" python3 - <<'PY'
import glob
import json
import os

rows = []
base_out = os.environ["BASE_OUT_FOR_SUMMARY"]
for path in sorted(glob.glob(os.path.join(base_out, "seed*", "metrics.json"))):
    data = json.load(open(path, encoding="utf-8"))
    best = data.get("best", {})
    rows.append((
        best.get("ndcg", 0.0),
        best.get("hit", 0.0),
        best.get("mrr", 0.0),
        best.get("epoch"),
        os.path.dirname(path),
    ))

for ndcg, hit, mrr, epoch, dirname in sorted(rows, reverse=True):
    print(f"{dirname}\tepoch={epoch}\tndcg={ndcg:.6f}\thit={hit:.6f}\tmrr={mrr:.6f}")
PY

echo
echo "============================================================"
echo "[评估] 多 seed logits 平均 + jaccard 融合"
echo "============================================================"
echo "checkpoints=$CHECKPOINT_ARG"

python3 code/a2_feature_ranker.py eval_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT_ARG" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --test_like_val \
  --model_weights "$MODEL_WEIGHTS" \
  --output_json "$BASE_OUT/fusion_eval.json"

BEST_WEIGHT="$(python3 -c "import json; d=json.load(open('$BASE_OUT/fusion_eval.json')); print(d['best']['model_weight'])")"
BEST_WEIGHTED_NDCG="$(python3 -c "import json; d=json.load(open('$BASE_OUT/fusion_eval.json')); print(d['best']['weighted_ndcg'])")"
BASE_WEIGHTED_NDCG="$(python3 -c "import json; d=json.load(open('$BASE_OUT/fusion_eval.json')); print(next(x['weighted_ndcg'] for x in d['results'] if x['model_weight'] == 0))")"

echo
echo "============================================================"
echo "[结论] 多 seed 融合"
echo "============================================================"
echo "baseline(model_weight=0) weighted_NDCG=$BASE_WEIGHTED_NDCG"
echo "best_model_weight=$BEST_WEIGHT"
echo "best weighted_NDCG=$BEST_WEIGHTED_NDCG"

SHOULD_BUILD="$(python3 -c "w=float('$BEST_WEIGHT'); force=int('$FORCE_BUILD'); print(1 if force or w > 0 else 0)")"
if [[ "$SHOULD_BUILD" != "1" ]]; then
  echo "最佳权重为0，默认不生成提交包。"
  exit 0
fi

echo
echo "============================================================"
echo "[推理] 生成多 seed 融合 A2"
echo "============================================================"

python3 code/a2_feature_ranker.py predict_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT_ARG" \
  --output_path "$BASE_OUT/A2.csv" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --model_weight "$BEST_WEIGHT"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  exit 1
fi

cp "$A1_SOURCE" "$BASE_OUT/A1.csv"
(
  cd "$BASE_OUT"
  rm -f prediction.zip
  zip -q prediction.zip A1.csv A2.csv
)

python3 code/validate_submission.py \
  --zip_path "$BASE_OUT/prediction.zip" \
  --cls_data_path data/cls_data/A1.npz \
  --rec_data_dir data/rec_data \
  --topk 10

echo
echo "候选提交包：$BASE_OUT/prediction.zip"
