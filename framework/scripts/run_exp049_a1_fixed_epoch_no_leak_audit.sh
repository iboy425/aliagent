#!/usr/bin/env bash
set -euo pipefail

# Exp-049：A1 固定 epoch 无泄漏审计
#
# 目的：
# - Exp-048 用全部 train_idx 重训后，验证集已经参与训练，`0.97+` 指标不能代表线上。
# - 本脚本保留 10% 验证节点不参与训练，但训练策略尽量模拟 Exp-048：
#   固定 epoch、关闭早停、关闭学习率调度器。
# - 若该无泄漏审计明显好于 Exp-047，才说明全标签最终重训值得提交。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp049_a1_fixed_epoch_no_leak_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"

mkdir -p "$OUT_DIR"

train_fixed() {
  local name="$1"
  local split_seed="$2"
  local model_type="$3"
  local hidden="$4"
  local heads="$5"
  local dropout="$6"
  local lr="$7"
  local wd="$8"
  local seed="$9"
  local normalize="${10}"
  local epochs="${11}"
  local out_dir="$OUT_DIR/$name/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/final_model.pt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过训练] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[固定epoch训练] $name split_seed=$split_seed"
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
    --val_ratio 0.1 \
    --split_seed "$split_seed" \
    --stratified_split \
    --disable_early_stop \
    --scheduler none \
    --seed "$seed" \
    --device "$DEVICE" \
    --log_interval 25
}

eval_variant() {
  local variant="$1"
  local split_seed="$2"
  local output_json="$OUT_DIR/eval/${variant}/split${split_seed}.json"
  mkdir -p "$(dirname "$output_json")"
  if [[ -f "$output_json" && "${RERUN_EVAL:-0}" != "1" ]]; then
    echo "[跳过评估] $variant split_seed=$split_seed 已存在"
    return
  fi

  local gat="$OUT_DIR/gat_h256_heads4_seed2026_e270/split${split_seed}/final_model.pt"
  local gcn="$OUT_DIR/gcn_h256_seed777_e120/split${split_seed}/final_model.pt"

  echo
  echo "============================================================"
  echo "[评估] $variant split_seed=$split_seed"
  echo "============================================================"

  case "$variant" in
    gat_only)
      python3 code/a1_correct_smooth.py \
        --data_path "$DATA_PATH" \
        --checkpoints "$gat" \
        --device "$DEVICE" \
        --val_ratio 0.1 \
        --split_seed "$split_seed" \
        --stratified_split \
        --cs_normalize random_walk \
        --correct_alphas 0.3 \
        --correct_iters 5 \
        --correct_weights 0.0 \
        --smooth_alphas 0.7 \
        --smooth_iters 5 \
        --smooth_weights 0,0.5,0.75,1.0 \
        --output_json "$output_json"
      ;;
    gat_gcn_95_5)
      python3 code/a1_correct_smooth.py \
        --data_path "$DATA_PATH" \
        --checkpoints "$gat" "$gcn" \
        --checkpoint_weights "0.95,0.05" \
        --device "$DEVICE" \
        --val_ratio 0.1 \
        --split_seed "$split_seed" \
        --stratified_split \
        --cs_normalize random_walk \
        --correct_alphas 0.3 \
        --correct_iters 5 \
        --correct_weights 0.0 \
        --smooth_alphas 0.7 \
        --smooth_iters 5 \
        --smooth_weights 0,0.5,0.75,1.0 \
        --output_json "$output_json"
      ;;
    *)
      echo "未知variant: $variant"
      exit 1
      ;;
  esac
}

IFS=',' read -ra seed_list <<< "$SPLIT_SEEDS"
for split_seed in "${seed_list[@]}"; do
  split_seed="$(echo "$split_seed" | xargs)"

  train_fixed "gat_h256_heads4_seed2026_e270" "$split_seed" gat_sparse 256 4 0.4 0.005 0.0005 2026 none 270
  train_fixed "gcn_h256_seed777_e120" "$split_seed" gcn 256 2 0.5 0.01 0.0005 777 symmetric 120

  eval_variant "gat_only" "$split_seed"
  eval_variant "gat_gcn_95_5" "$split_seed"
done

echo
echo "============================================================"
echo "[汇总] Exp-049 固定epoch无泄漏审计"
echo "============================================================"

OUT_DIR_FOR_SUMMARY="$OUT_DIR" python3 - <<'PY'
import csv
import glob
import json
import os
import statistics

base = os.environ["OUT_DIR_FOR_SUMMARY"]
rows = []
for variant_dir in sorted(glob.glob(os.path.join(base, "eval", "*"))):
    values = []
    detail = []
    for path in sorted(glob.glob(os.path.join(variant_dir, "split*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        best = data["results"][0]
        value = float(best["val_acc"])
        values.append(value)
        detail.append((os.path.basename(path), value, best))
    if not values:
        continue
    rows.append({
        "variant": os.path.basename(variant_dir),
        "n": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "detail": detail,
    })

rows.sort(key=lambda item: (item["mean"], item["min"]), reverse=True)
for row in rows:
    print(
        f"{row['variant']}\tn={row['n']}\t"
        f"mean={row['mean']:.6f}\tmin={row['min']:.6f}\t"
        f"max={row['max']:.6f}\tstd={row['std']:.6f}"
    )
    for split_name, value, best in row["detail"]:
        print(
            f"  {split_name}\t{value:.6f}\t"
            f"correct=({best['correct_alpha']},{best['correct_iter']},{best['correct_weight']})\t"
            f"smooth=({best['smooth_alpha']},{best['smooth_iter']},{best['smooth_weight']})"
        )

summary_json = os.path.join(base, "summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

summary_csv = os.path.join(base, "summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "n", "mean", "min", "max", "std"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in ["variant", "n", "mean", "min", "max", "std"]})

print(f"\n汇总已保存: {summary_json}")
PY
