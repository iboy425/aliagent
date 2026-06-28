#!/usr/bin/env bash
set -euo pipefail

# Exp-046：A1 多 split 稳定性审计
#
# 背景：
# - Exp-039 在 split_seed=42 上本地达到 0.713242，但线上 A1 只有 0.6848；
# - Exp-019/030 A1 本地不一定最高，但线上稳定在 0.6874；
# - 因此 A1 下一步必须先做多 split 稳定性审计，而不是继续追单 split 高分。
#
# 本脚本评估多个 A1 候选在多个验证集划分上的固定 C&S 表现。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp046_a1_multisplit_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"

mkdir -p "$OUT_DIR"

run_candidate() {
  local name="$1"
  local weights="$2"
  local correct_alpha="$3"
  local correct_iter="$4"
  local correct_weight="$5"
  local smooth_alpha="$6"
  local smooth_iter="$7"
  local smooth_weight="$8"
  shift 8
  local checkpoints=("$@")

  local missing=0
  for ckpt in "${checkpoints[@]}"; do
    if [[ ! -f "$ckpt" ]]; then
      echo "[跳过] $name 缺少 checkpoint: $ckpt"
      missing=1
    fi
  done
  if [[ "$missing" == "1" ]]; then
    return
  fi

  local candidate_dir="$OUT_DIR/$name"
  mkdir -p "$candidate_dir"

  IFS=',' read -ra seed_list <<< "$SPLIT_SEEDS"
  for split_seed in "${seed_list[@]}"; do
    split_seed="$(echo "$split_seed" | xargs)"
    local output_json="$candidate_dir/split${split_seed}.json"

    if [[ -f "$output_json" && "${RERUN:-0}" != "1" ]]; then
      echo "[跳过] $name split_seed=$split_seed 已存在"
      continue
    fi

    local weight_args=()
    if [[ -n "$weights" ]]; then
      weight_args=(--checkpoint_weights "$weights")
    fi

    echo
    echo "============================================================"
    echo "[A1审计] $name split_seed=$split_seed"
    echo "C&S correct=($correct_alpha,$correct_iter,$correct_weight) smooth=($smooth_alpha,$smooth_iter,$smooth_weight)"
    echo "============================================================"

    python3 code/a1_correct_smooth.py \
      --data_path "$DATA_PATH" \
      --checkpoints "${checkpoints[@]}" \
      "${weight_args[@]}" \
      --device "$DEVICE" \
      --val_ratio 0.1 \
      --split_seed "$split_seed" \
      --stratified_split \
      --cs_normalize random_walk \
      --correct_alphas "$correct_alpha" \
      --correct_iters "$correct_iter" \
      --correct_weights "$correct_weight" \
      --smooth_alphas "$smooth_alpha" \
      --smooth_iters "$smooth_iter" \
      --smooth_weights "$smooth_weight" \
      --output_json "$output_json"
  done
}

# Exp-019/030：线上最稳 A1，Exp-010 Top-5 + C&S。
run_candidate \
  "exp019_top5_fixed_cs" \
  "" \
  0.3 5 0.0 \
  0.7 7 1.0 \
  output/exp010_a1_grid_c6_gcn_h384_l2_symmetric_none_seed777/best_model.pt \
  output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt \
  output/exp010_a1_grid_c1_sage_h384_l2_symmetric_none_seed3407/best_model.pt \
  output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed42/best_model.pt \
  output/exp010_a1_grid_c7_sage_h256_l2_random_walk_none_seed777/best_model.pt

# Exp-033：GAT 0.95 + GCN 0.05 + C&S。
run_candidate \
  "exp033_gat_gcn_95_5_fixed_cs" \
  "95,5" \
  0.3 5 0.0 \
  0.7 5 0.75 \
  output/exp031_a1_gat_sparse_h256_heads4_seed2026/best_model.pt \
  output/exp010_a1_grid_c5_gcn_h256_l2_symmetric_none_seed777/best_model.pt

# Exp-039/040：本地最高但线上回落的 GAT 贪心加权方案。
run_candidate \
  "exp039_greedy_fixed_cs" \
  "64,16,20" \
  0.3 5 0.0 \
  0.75 5 0.75 \
  output/exp038_a1_gat_grid/h256_heads4_d040_lr005_wd5e4_seed42/best_model.pt \
  output/exp038_a1_gat_grid/h224_heads4_d045_lr005_wd5e4_seed42/best_model.pt \
  output/exp038_a1_gat_grid/h256_heads8_d045_lr005_wd5e4_seed3407/best_model.pt

echo
echo "============================================================"
echo "[汇总] A1 多 split 稳定性"
echo "============================================================"

OUT_DIR_FOR_SUMMARY="$OUT_DIR" python3 - <<'PY'
import glob
import json
import os
import statistics

base = os.environ["OUT_DIR_FOR_SUMMARY"]
rows = []
for candidate_dir in sorted(glob.glob(os.path.join(base, "*"))):
    if not os.path.isdir(candidate_dir):
        continue
    vals = []
    bases = []
    for path in sorted(glob.glob(os.path.join(candidate_dir, "split*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        vals.append(float(data["results"][0]["val_acc"]))
        bases.append(float(data["baseline_acc"]))
    if not vals:
        continue
    rows.append({
        "candidate": os.path.basename(candidate_dir),
        "n": len(vals),
        "mean_cs": statistics.mean(vals),
        "min_cs": min(vals),
        "max_cs": max(vals),
        "std_cs": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "mean_model": statistics.mean(bases),
        "mean_gain": statistics.mean(vals) - statistics.mean(bases),
    })

rows.sort(key=lambda item: (item["mean_cs"], item["min_cs"]), reverse=True)
for item in rows:
    print(
        f"{item['candidate']}\t"
        f"n={item['n']}\t"
        f"mean_cs={item['mean_cs']:.6f}\t"
        f"min_cs={item['min_cs']:.6f}\t"
        f"std={item['std_cs']:.6f}\t"
        f"mean_model={item['mean_model']:.6f}\t"
        f"mean_gain={item['mean_gain']:+.6f}"
    )

with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
PY

echo
echo "审计结果：$OUT_DIR/summary.json"
