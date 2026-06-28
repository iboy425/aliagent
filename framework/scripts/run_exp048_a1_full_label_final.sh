#!/usr/bin/env bash
set -euo pipefail

# Exp-048：A1 全标签最终重训候选
#
# 背景：
# - 前面的 A1 checkpoint 都为验证集早停服务，只使用约 90% 的 train_idx 训练。
# - 正式提交时，模型结构和训练轮数已经由多 split 审计确定，应该把全部
#   11001 个已知标签用于训练，再用 C&S 生成测试节点预测。
#
# 设计：
# - A2 沿用当前线上最强的 Exp-044，避免两个任务同时变化。
# - A1 训练两个互补模型：
#   1. 稀疏 GAT：Exp-047 中最稳定，真多 split 平均约 0.699。
#   2. GCN：作为少量多样性补充，权重较低。
# - 默认生成一个 GAT:GCN = 0.95:0.05 的 C&S 候选包。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp048_a1_full_label_final}"
A2_SOURCE="${A2_SOURCE:-output/exp044_a2_feature_fusion/A2.csv}"

mkdir -p "$OUT_DIR"

train_full() {
  local name="$1"
  local model_type="$2"
  local hidden="$3"
  local heads="$4"
  local dropout="$5"
  local lr="$6"
  local wd="$7"
  local seed="$8"
  local normalize="$9"
  local epochs="${10}"
  local out_dir="$OUT_DIR/$name"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/final_model.pt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过训练] $name 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[全标签训练] $name"
  echo "model=$model_type hidden=$hidden heads=$heads dropout=$dropout lr=$lr wd=$wd seed=$seed normalize=$normalize epochs=$epochs"
  echo "============================================================"

  python3 code/train.py \
    --task task1 \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --model_type "$model_type" \
    --hidden_dim "$hidden" \
    --num_layers 2 \
    --num_heads "$heads" \
    --lr "$lr" \
    --dropout "$dropout" \
    --weight_decay "$wd" \
    --epochs "$epochs" \
    --patience "$epochs" \
    --normalize "$normalize" \
    --adj_format sparse \
    --feature_norm none \
    --dropedge_rate 0 \
    --feature_mask_rate 0 \
    --class_weight none \
    --train_all_labels \
    --disable_early_stop \
    --seed "$seed" \
    --device "$DEVICE" \
    --log_interval 20
}

echo
echo "============================================================"
echo "[训练] A1 全标签模型"
echo "============================================================"

# GAT最佳 epoch 在真多 split 中集中在 244-283，取 270 作为中位附近的正式训练轮数。
train_full "gat_h256_heads4_seed2026_e270" gat_sparse 256 4 0.4 0.005 0.0005 2026 none 270

# GCN作为低权重多样性补充，取此前有效区间附近的固定轮数。
train_full "gcn_h256_seed777_e120" gcn 256 2 0.5 0.01 0.0005 777 symmetric 120

GAT_CKPT="$OUT_DIR/gat_h256_heads4_seed2026_e270/final_model.pt"
GCN_CKPT="$OUT_DIR/gcn_h256_seed777_e120/final_model.pt"

echo
echo "============================================================"
echo "[推理] A1 C&S 全标签候选"
echo "============================================================"

python3 code/a1_correct_smooth.py \
  --data_path "$DATA_PATH" \
  --checkpoints "$GAT_CKPT" "$GCN_CKPT" \
  --checkpoint_weights "0.95,0.05" \
  --device "$DEVICE" \
  --val_ratio 0.1 \
  --split_seed 42 \
  --stratified_split \
  --cs_normalize random_walk \
  --correct_alphas 0.3 \
  --correct_iters 5 \
  --correct_weights 0.0 \
  --smooth_alphas 0.7 \
  --smooth_iters 5 \
  --smooth_weights 0.75 \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight 0.75 \
  --output_json "$OUT_DIR/a1_cs_eval.json" \
  --output_path "$OUT_DIR/A1.csv"

echo
echo "============================================================"
echo "[打包] prediction.zip"
echo "============================================================"

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE"
  exit 1
fi

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
