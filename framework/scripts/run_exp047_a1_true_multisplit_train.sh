#!/usr/bin/env bash
set -euo pipefail

# Exp-047：A1 真多 split 重训练审计
#
# Exp-046 复用了 split_seed=42 训练出的 checkpoint 去评估其他 split，
# 导致其他 split 的验证节点可能已经参与过训练，分数虚高。
# 本脚本对每个 split_seed 重新训练模型，再在同一个 split 上做固定 C&S。
#
# 默认只跑两个代表单模型：
# - gcn_h384：Exp-010 中稳定有效的 GCN 家族代表；
# - gat_h256：Exp-031/038 中最强的 GAT 家族代表。
#
# 如果代表单模型多 split 稳定，再扩展到完整 ensemble。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp047_a1_true_multisplit_train}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024}"
EPOCHS="${EPOCHS:-700}"
PATIENCE="${PATIENCE:-90}"

mkdir -p "$OUT_DIR"

train_one() {
  local name="$1"
  local split_seed="$2"
  local model_type="$3"
  local hidden="$4"
  local heads="$5"
  local dropout="$6"
  local lr="$7"
  local wd="$8"
  local model_seed="$9"
  local normalize="${10}"
  local out_dir="$OUT_DIR/$name/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/best_model.pt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过训练] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[训练] $name split_seed=$split_seed"
  echo "model=$model_type hidden=$hidden heads=$heads dropout=$dropout lr=$lr wd=$wd model_seed=$model_seed normalize=$normalize"
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
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --normalize "$normalize" \
    --adj_format sparse \
    --feature_norm none \
    --dropedge_rate 0 \
    --feature_mask_rate 0 \
    --class_weight none \
    --val_ratio 0.1 \
    --split_seed "$split_seed" \
    --stratified_split \
    --seed "$model_seed" \
    --device "$DEVICE" \
    --log_interval 20
}

cs_one() {
  local name="$1"
  local split_seed="$2"
  local correct_alpha="$3"
  local correct_iter="$4"
  local correct_weight="$5"
  local smooth_alpha="$6"
  local smooth_iter="$7"
  local smooth_weight="$8"
  local ckpt="$OUT_DIR/$name/split${split_seed}/best_model.pt"
  local output_json="$OUT_DIR/$name/split${split_seed}/cs.json"

  if [[ ! -f "$ckpt" ]]; then
    echo "[跳过C&S] 缺少 checkpoint: $ckpt"
    return
  fi
  if [[ -f "$output_json" && "${RERUN_CS:-0}" != "1" ]]; then
    echo "[跳过C&S] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[C&S] $name split_seed=$split_seed"
  echo "============================================================"

  python3 code/a1_correct_smooth.py \
    --data_path "$DATA_PATH" \
    --checkpoints "$ckpt" \
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
}

IFS=',' read -ra seed_list <<< "$SPLIT_SEEDS"
for split_seed in "${seed_list[@]}"; do
  split_seed="$(echo "$split_seed" | xargs)"

  # GCN代表：Exp-010 最强单模型配置。
  train_one "gcn_h384_seed777" "$split_seed" gcn 384 2 0.5 0.01 0.0005 777 symmetric
  cs_one "gcn_h384_seed777" "$split_seed" 0.3 5 0.0 0.7 7 1.0

  # GAT代表：Exp-031/038 有效配置。
  train_one "gat_h256_heads4_seed2026" "$split_seed" gat_sparse 256 4 0.4 0.005 0.0005 2026 none
  cs_one "gat_h256_heads4_seed2026" "$split_seed" 0.3 5 0.0 0.7 5 0.75
done

echo
echo "============================================================"
echo "[汇总] A1 真多 split 重训练"
echo "============================================================"

OUT_DIR_FOR_SUMMARY="$OUT_DIR" python3 - <<'PY'
import csv
import glob
import json
import os
import statistics

base = os.environ["OUT_DIR_FOR_SUMMARY"]
rows = []
for cand_dir in sorted(glob.glob(os.path.join(base, "*"))):
    if not os.path.isdir(cand_dir):
        continue
    train_vals = []
    cs_vals = []
    for split_dir in sorted(glob.glob(os.path.join(cand_dir, "split*"))):
        metrics_path = os.path.join(split_dir, "metrics.json")
        cs_path = os.path.join(split_dir, "cs.json")
        if os.path.exists(metrics_path):
            metrics = json.load(open(metrics_path, encoding="utf-8"))
            vals = metrics.get("val_acc", [])
            if vals:
                train_vals.append(max(vals))
        if os.path.exists(cs_path):
            cs = json.load(open(cs_path, encoding="utf-8"))
            cs_vals.append(float(cs["results"][0]["val_acc"]))
    if not train_vals and not cs_vals:
        continue
    values = cs_vals if cs_vals else train_vals
    rows.append({
        "candidate": os.path.basename(cand_dir),
        "n": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "mean_train": statistics.mean(train_vals) if train_vals else 0.0,
        "mean_cs": statistics.mean(cs_vals) if cs_vals else 0.0,
    })

rows.sort(key=lambda item: (item["mean"], item["min"]), reverse=True)
for item in rows:
    print(
        f"{item['candidate']}\t"
        f"n={item['n']}\t"
        f"mean={item['mean']:.6f}\t"
        f"min={item['min']:.6f}\t"
        f"max={item['max']:.6f}\t"
        f"std={item['std']:.6f}\t"
        f"mean_train={item['mean_train']:.6f}\t"
        f"mean_cs={item['mean_cs']:.6f}"
    )

with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
with open(os.path.join(base, "summary.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "candidate", "n", "mean", "min", "max", "std", "mean_train", "mean_cs",
    ])
    writer.writeheader()
    writer.writerows(rows)
PY

echo
echo "结果：$OUT_DIR/summary.json"
