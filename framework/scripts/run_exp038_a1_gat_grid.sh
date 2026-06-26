#!/usr/bin/env bash
set -euo pipefail

# Exp-038：A1 稀疏GAT小网格搜索
#
# 目的：
# - 在 Exp-031 的有效配置基础上，系统搜索 GAT 容量和正则化。
# - 所有模型使用固定验证集：val_ratio=0.1, split_seed=42, stratified_split。
# - 训练完成后自动汇总单模型准确率，并做贪心加权集成评估。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="data/cls_data/A1.npz"
BASE_OUT="output/exp038_a1_gat_grid"
mkdir -p "$BASE_OUT"

run_train() {
  local name="$1"
  local hidden="$2"
  local heads="$3"
  local dropout="$4"
  local lr="$5"
  local weight_decay="$6"
  local seed="$7"
  local out_dir="$BASE_OUT/$name"

  if [[ -f "$out_dir/best_model.pt" ]]; then
    echo "[跳过] $name 已存在 best_model.pt"
    return
  fi

  echo
  echo "============================================================"
  echo "[训练] $name"
  echo "hidden=$hidden heads=$heads dropout=$dropout lr=$lr wd=$weight_decay seed=$seed"
  echo "============================================================"

  python3 code/train.py \
    --task task1 \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --model_type gat_sparse \
    --hidden_dim "$hidden" \
    --num_layers 2 \
    --num_heads "$heads" \
    --lr "$lr" \
    --dropout "$dropout" \
    --weight_decay "$weight_decay" \
    --epochs 700 \
    --patience 90 \
    --normalize none \
    --adj_format sparse \
    --feature_norm none \
    --dropedge_rate 0 \
    --feature_mask_rate 0 \
    --class_weight none \
    --val_ratio 0.1 \
    --split_seed 42 \
    --stratified_split \
    --seed "$seed" \
    --device "$DEVICE" \
    --log_interval 20
}

# 先搜围绕当前最优 h256/head4 的低风险邻域。
for seed in 42 777 2026 3407; do
  run_train "h224_heads4_d045_lr005_wd5e4_seed${seed}" 224 4 0.45 0.005 0.0005 "$seed"
  run_train "h256_heads4_d040_lr005_wd5e4_seed${seed}" 256 4 0.40 0.005 0.0005 "$seed"
  run_train "h256_heads4_d050_lr003_wd5e4_seed${seed}" 256 4 0.50 0.003 0.0005 "$seed"
  run_train "h288_heads4_d045_lr005_wd5e4_seed${seed}" 288 4 0.45 0.005 0.0005 "$seed"
done

# 少量探索多头数。hidden_dim 必须能被 heads 整除。
for seed in 2026 3407; do
  run_train "h256_heads8_d045_lr005_wd5e4_seed${seed}" 256 8 0.45 0.005 0.0005 "$seed"
  run_train "h320_heads8_d050_lr003_wd5e4_seed${seed}" 320 8 0.50 0.003 0.0005 "$seed"
done

mapfile -t checkpoints < <(find "$BASE_OUT" -mindepth 2 -maxdepth 2 -name best_model.pt | sort)
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "没有找到任何 checkpoint"
  exit 1
fi

echo
echo "============================================================"
echo "[评估] Exp-038 单模型、等权集成、贪心加权集成"
echo "============================================================"

python3 code/a1_ensemble_eval.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${checkpoints[@]}" \
  --device "$DEVICE" \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --topks 1,2,3,5,8,12,20 \
  --greedy_max_size 5 \
  --greedy_weights 0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3 \
  --greedy_min_gain 0 \
  --output_json "$BASE_OUT/ensemble_eval.json"

echo
echo "Exp-038 完成。结果：$BASE_OUT/ensemble_eval.json"
