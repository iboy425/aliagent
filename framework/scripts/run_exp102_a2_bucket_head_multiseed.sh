#!/usr/bin/env bash
set -euo pipefail

# Exp-102：A2 历史长度桶专用输出头多 seed 训练
#
# 现有 Exp045/Exp090 的 A2 模型使用共享输出头。测试集却高度偏向空历史和
# 短历史，和训练集原始长历史分布严重错位。本实验打开 --bucket_heads，
# 让不同历史长度桶学习专用 target 分类头。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SEEDS="${SEEDS:-42,777,2024}"
BASE_OUT="${BASE_OUT:-output/exp102_a2_bucket_head_multiseed}"
MAX_LEN="${MAX_LEN:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-140}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
USER_EMBEDDING_DIM="${USER_EMBEDDING_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-0,0.5,1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0,12.0,15.0,20.0,25.0,30.0,40.0,50.0}"

mkdir -p "$BASE_OUT"

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
CHECKPOINTS=()

echo
echo "============================================================"
echo "[Exp-102] 训练 A2 bucket-head ranker 多 seed"
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
      --random_test_like_train \
      --bucket_heads
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

base_out = os.environ["BASE_OUT_FOR_SUMMARY"]
rows = []
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
echo "[评估] bucket-head 多 seed logits 平均 + jaccard 融合"
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
echo "[结论] Exp-102 多 seed 融合"
echo "============================================================"
echo "baseline(model_weight=0) weighted_NDCG=$BASE_WEIGHTED_NDCG"
echo "best_model_weight=$BEST_WEIGHT"
echo "best weighted_NDCG=$BEST_WEIGHTED_NDCG"

echo
echo "============================================================"
echo "[推理] 生成 Exp-102 A2_alt"
echo "============================================================"

python3 code/a2_feature_ranker.py predict_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT_ARG" \
  --output_path "$BASE_OUT/A2_alt.csv" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --model_weight "$BEST_WEIGHT"

echo
echo "Exp-102 A2_alt：$BASE_OUT/A2_alt.csv"
echo "下一步运行：CUDA_VISIBLE_DEVICES=0 DEVICE=cuda bash scripts/run_exp103_a2_bucket_head_protected_audit.sh"
