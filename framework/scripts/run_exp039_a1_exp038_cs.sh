#!/usr/bin/env bash
set -euo pipefail

# Exp-039：对 Exp-038 的有效集成候选做 Correct and Smooth 搜索
#
# 依据：
# - Exp-038 Top-3 等权集成原始验证准确率达到 0.709589。
# - Exp-033 当前最优 C&S 为 0.711416。
# - 本脚本只做后处理搜索，不重新训练模型。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="data/cls_data/A1.npz"
OUT_DIR="output/exp039_a1_exp038_cs"
mkdir -p "$OUT_DIR"

TOP1="output/exp038_a1_gat_grid/h256_heads4_d040_lr005_wd5e4_seed42/best_model.pt"
TOP2="output/exp038_a1_gat_grid/h256_heads4_d050_lr003_wd5e4_seed2026/best_model.pt"
TOP3="output/exp038_a1_gat_grid/h288_heads4_d045_lr005_wd5e4_seed777/best_model.pt"
GREEDY2="output/exp038_a1_gat_grid/h224_heads4_d045_lr005_wd5e4_seed42/best_model.pt"
GREEDY3="output/exp038_a1_gat_grid/h256_heads8_d045_lr005_wd5e4_seed3407/best_model.pt"
OLD_GCN="output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt"

run_cs() {
  local name="$1"
  local weights="$2"
  shift 2

  echo
  echo "============================================================"
  echo "[C&S] $name"
  echo "weights=${weights:-equal}"
  echo "============================================================"

  local weight_args=()
  if [[ -n "$weights" ]]; then
    weight_args=(--checkpoint_weights "$weights")
  fi

  python3 code/a1_correct_smooth.py \
    --data_path "$DATA_PATH" \
    --checkpoints "$@" \
    "${weight_args[@]}" \
    --device "$DEVICE" \
    --val_ratio 0.1 \
    --split_seed 42 \
    --stratified_split \
    --cs_normalize random_walk \
    --correct_alphas 0.3 \
    --correct_iters 5 \
    --correct_weights 0 \
    --smooth_alphas 0.6,0.65,0.7,0.75,0.8 \
    --smooth_iters 3,5,7,10,15 \
    --smooth_weights 0.5,0.6,0.7,0.75,0.8,0.9,1.0 \
    --pseudo_thresholds 0.95,0.97 \
    --pseudo_weights 0.5,1.0 \
    --output_json "$OUT_DIR/${name}.json"
}

# Exp-038 Top-K 等权候选。Top-3 原始验证 0.709589，是当前最值得优先验证的组合。
run_cs "top2_equal" "" "$TOP1" "$TOP2"
run_cs "top3_equal" "" "$TOP1" "$TOP2" "$TOP3"

# Exp-038 贪心权重候选：0.64/0.16/0.20。
run_cs "greedy_weighted" "64,16,20" "$TOP1" "$GREEDY2" "$GREEDY3"

# 在 Top-3 GAT 基础上加少量旧 GCN，测试跨模型族互补是否仍然有效。
if [[ -f "$OLD_GCN" ]]; then
  run_cs "top3_plus_old_gcn_95_5" "31.6667,31.6667,31.6667,5" "$TOP1" "$TOP2" "$TOP3" "$OLD_GCN"
fi

python3 - <<'PY'
import glob
import json
import os

rows = []
for path in sorted(glob.glob("output/exp039_a1_exp038_cs/*.json")):
    data = json.load(open(path, encoding="utf-8"))
    best = data["results"][0]
    rows.append((best["val_acc"], os.path.basename(path), best))

print("\nExp-039 C&S 汇总")
for acc, name, best in sorted(rows, reverse=True):
    print(
        f"{acc:.6f}\t{name}\t"
        f"correct=({best['correct_alpha']},{best['correct_iter']},{best['correct_weight']})\t"
        f"smooth=({best['smooth_alpha']},{best['smooth_iter']},{best['smooth_weight']})\t"
        f"pseudo=({best.get('pseudo_threshold')},{best.get('pseudo_weight')},{best.get('pseudo_count', 0)})"
    )
PY

echo
echo "Exp-039 完成。结果目录：$OUT_DIR"
