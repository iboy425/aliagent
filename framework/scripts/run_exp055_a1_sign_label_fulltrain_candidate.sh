#!/usr/bin/env bash
set -euo pipefail

# Exp-055：A1 SIGN标签传播特征全标签最终训练候选
#
# 背景：
# - Exp-054 线上 A1=0.7459，继续提升空间主要来自 A1。
# - Exp-054 使用的是5个split checkpoint，每个模型训练时只见过约90%标签。
# - Exp-053最佳模型的最优epoch集中在34-45。
#
# 思路：
# - 使用全部 train_idx 标签训练最终模型；
# - 标签传播特征也用全部 train_idx 构造；
# - 固定epoch，不用验证早停；
# - 多 seed / 多 epoch 形成小集成，降低单次训练随机性。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp055_a1_sign_label_fulltrain_candidate}"
A2_SOURCE="${A2_SOURCE:-output/exp044_a2_feature_fusion/A2.csv}"

mkdir -p "$OUT_DIR"

train_one() {
  local name="$1"
  local seed="$2"
  local epochs="$3"
  local out_dir="$OUT_DIR/$name"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/best_model.pt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过训练] $name 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[全标签训练] $name seed=$seed epochs=$epochs"
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
    --block_norm \
    --label_feature_hops 3 \
    --label_feature_norm random_walk \
    --label_feature_row_norm \
    --label_feature_weight 1.0 \
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

echo
echo "============================================================"
echo "[训练] A1 SIGN标签传播特征全标签模型"
echo "============================================================"

# epoch选择来自Exp-053最佳配置的各split最优epoch：34,34,36,40,45。
train_one "seed42_e34" 42 34
train_one "seed777_e34" 777 34
train_one "seed3407_e36" 3407 36
train_one "seed2026_e40" 2026 40
train_one "seed2024_e45" 2024 45

CHECKPOINTS=(
  "$OUT_DIR/seed42_e34/best_model.pt"
  "$OUT_DIR/seed777_e34/best_model.pt"
  "$OUT_DIR/seed3407_e36/best_model.pt"
  "$OUT_DIR/seed2026_e40/best_model.pt"
  "$OUT_DIR/seed2024_e45/best_model.pt"
)

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

echo
echo "============================================================"
echo "[推理] A1 全标签SIGN标签特征集成"
echo "============================================================"

python3 code/a1_sign_infer.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --device "$DEVICE" \
  --cs_normalize random_walk \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight 0.0 \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_fulltrain_infer.json"

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
