#!/usr/bin/env bash
set -euo pipefail

# Exp-094：A1回退线上最优 + A2 feature ranker 全量重训
#
# 失败教训：
# - Exp093 的 leaderboard 反推同时伤害 A1/A2，说明线上稀疏分数不能硬当标签反推。
#
# 这轮改法：
# - A1 先回退到线上最稳的 Exp090/Exp078，避免继续扩大损失；
# - A2 把 Exp045 的 feature ranker 从“90%训练+10%验证”改为“全部train.csv重训”；
# - 每个 seed 使用之前离线最佳 epoch 附近的固定轮数，减少过拟合和随机试错。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp094_submit_a1_best_a2_fulltrain_ranker}"
MODEL_DIR="${MODEL_DIR:-output/exp094_a2_feature_fulltrain}"
A1_SOURCE="${A1_SOURCE:-output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv}"
if [[ ! -f "$A1_SOURCE" ]]; then
  A1_SOURCE="output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv"
fi

MAX_LEN="${MAX_LEN:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
USER_EMBEDDING_DIM="${USER_EMBEDDING_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# 来自 Exp045 多 seed 的离线最佳 epoch：
# seed42=38, seed777=54, seed2024=28。
# 全量训练没有验证集，固定 epoch 比重新早停更可复现。
SEED_EPOCHS="${SEED_EPOCHS:-42:38,777:54,2024:28}"
MODEL_WEIGHT="${MODEL_WEIGHT:-3.0}"

mkdir -p "$MODEL_DIR" "$OUT_DIR"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE"
  exit 1
fi

CHECKPOINTS=()
IFS=',' read -ra PAIRS <<< "$SEED_EPOCHS"

echo
echo "============================================================"
echo "[A2] feature ranker 全量重训"
echo "============================================================"
echo "seed_epochs=$SEED_EPOCHS"
echo "model_dir=$MODEL_DIR"

for pair in "${PAIRS[@]}"; do
  seed="${pair%%:*}"
  epochs="${pair##*:}"
  out_seed="$MODEL_DIR/seed${seed}_e${epochs}"
  ckpt="$out_seed/best_model.pt"
  mkdir -p "$out_seed"

  if [[ -f "$ckpt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] seed=$seed epochs=$epochs 已存在 $ckpt"
  else
    echo
    echo "------------------------------------------------------------"
    echo "[训练全量] seed=$seed epochs=$epochs"
    echo "------------------------------------------------------------"
    python3 code/a2_feature_ranker.py train \
      --data_path data/rec_data \
      --output_dir "$out_seed" \
      --device "$DEVICE" \
      --epochs "$epochs" \
      --batch_size "$BATCH_SIZE" \
      --max_len "$MAX_LEN" \
      --embedding_dim "$EMBEDDING_DIM" \
      --user_embedding_dim "$USER_EMBEDDING_DIM" \
      --hidden_dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --patience "$epochs" \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed" \
      --random_test_like_train \
      --train_all
  fi
  CHECKPOINTS+=("$ckpt")
done

CHECKPOINT_ARG="$(IFS=','; echo "${CHECKPOINTS[*]}")"
echo "$CHECKPOINT_ARG" > "$MODEL_DIR/checkpoints.txt"

echo
echo "============================================================"
echo "[A2] 全量模型 + 规则融合推理"
echo "============================================================"
echo "checkpoints=$CHECKPOINT_ARG"
echo "model_weight=$MODEL_WEIGHT"

python3 code/a2_feature_ranker.py predict_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT_ARG" \
  --output_path "$OUT_DIR/A2.csv" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --model_weight "$MODEL_WEIGHT"

cp "$A1_SOURCE" "$OUT_DIR/A1.csv"

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
