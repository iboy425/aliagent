#!/usr/bin/env bash
set -euo pipefail

# Exp-061：A1 Exp-060有效配置的全标签最终重训候选
#
# 背景：
# - Exp-060 线上 A1=0.7550，证明 `undirected + reverse` 标签传播方向有效；
# - 但 Exp-060 提交用的是 5 折 split checkpoint，每个模型训练时只看到约90%训练标签；
# - 正式比赛预测时，我们可以使用全部 train_idx 标签训练模型。
#
# 本脚本做两套配置的全标签重训：
# - undir：稳定基线，总权重0.3；
# - undir_reverse：线上有效的高上限配置，总权重0.7；
# 每套训练5个 seed，epoch 读取对应 Exp-059 split checkpoint 的 best_epoch。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
BASE_DIR="${BASE_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
OUT_DIR="${OUT_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
A2_SOURCE="${A2_SOURCE:-output/exp044_a2_feature_fusion/A2.csv}"

mkdir -p "$OUT_DIR"

read_best_epoch() {
  local ckpt="$1"
  local fallback="$2"
  if [[ ! -f "$ckpt" ]]; then
    echo "$fallback"
    return
  fi
  CKPT_PATH="$ckpt" FALLBACK_EPOCH="$fallback" python3 - <<'PY'
import os
import torch

path = os.environ["CKPT_PATH"]
fallback = int(os.environ["FALLBACK_EPOCH"])
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(int(ckpt.get("best_epoch") or fallback))
except Exception:
    print(fallback)
PY
}

train_one() {
  local config_name="$1"
  local label_modes="$2"
  local seed="$3"
  local epochs="$4"
  local out_dir="$OUT_DIR/${config_name}_seed${seed}_e${epochs}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/best_model.pt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过训练] $out_dir 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[全标签训练] config=$config_name seed=$seed epochs=$epochs label_modes=$label_modes"
  echo "============================================================"

  python3 code/a1_sign_mlp.py \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --device "$DEVICE" \
    --seed "$seed" \
    --split_seed 42 \
    --val_ratio 0.1 \
    --stratified_split \
    --train_all_labels \
    --disable_early_stop \
    --hops 5 \
    --prop_norm random_walk \
    --graph_mode undirected \
    --feature_norm none \
    --feature_transform standard \
    --block_norm \
    --label_feature_hops 3 \
    --label_feature_norm random_walk \
    --label_feature_graph_modes "$label_modes" \
    --label_feature_norms random_walk \
    --label_feature_row_norm \
    --label_feature_weight 1.0 \
    --structure_feature_mode basic \
    --structure_feature_weight 1.0 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.4 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --epochs "$epochs" \
    --patience "$epochs" \
    --log_interval 10 \
    --cs_normalize random_walk \
    --smooth_weights 0
}

SPLITS=(42 777 2024 2026 3407)
SEEDS=(42 777 2024 2026 3407)

UNDIR_DIR="$BASE_DIR/standard_struct_label_undir_h3_rw"
UNDIR_REVERSE_DIR="$BASE_DIR/standard_struct_label_undir_reverse_h3_rw"

echo
echo "============================================================"
echo "[训练] A1 Exp-061 全标签模型"
echo "============================================================"

CHECKPOINTS=()
WEIGHTS=()
for i in "${!SPLITS[@]}"; do
  split="${SPLITS[$i]}"
  seed="${SEEDS[$i]}"
  epoch="$(read_best_epoch "$UNDIR_DIR/split${split}/best_model.pt" 40)"
  train_one "undir" "undirected" "$seed" "$epoch"
  CHECKPOINTS+=("$OUT_DIR/undir_seed${seed}_e${epoch}/best_model.pt")
  WEIGHTS+=("0.06")
done

for i in "${!SPLITS[@]}"; do
  split="${SPLITS[$i]}"
  seed="${SEEDS[$i]}"
  epoch="$(read_best_epoch "$UNDIR_REVERSE_DIR/split${split}/best_model.pt" 40)"
  train_one "undir_reverse" "undirected,reverse" "$seed" "$epoch"
  CHECKPOINTS+=("$OUT_DIR/undir_reverse_seed${seed}_e${epoch}/best_model.pt")
  WEIGHTS+=("0.14")
done

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "缺少checkpoint: $ckpt"
    exit 1
  fi
done

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE"
  exit 1
fi

CHECKPOINT_WEIGHTS="$(IFS=','; echo "${WEIGHTS[*]}")"

echo
echo "============================================================"
echo "[推理] A1 Exp-061 全标签配置融合"
echo "============================================================"
echo "checkpoint_weights=$CHECKPOINT_WEIGHTS"

python3 code/a1_sign_infer.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --checkpoint_weights "$CHECKPOINT_WEIGHTS" \
  --device "$DEVICE" \
  --cs_normalize random_walk \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight "${SMOOTH_WEIGHT:-0.0}" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_exp061_fulltrain_config_ensemble_infer.json"

echo
echo "============================================================"
echo "[打包] prediction.zip"
echo "============================================================"

cp "$A2_SOURCE" "$OUT_DIR/A2.csv"
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
