#!/usr/bin/env bash
set -euo pipefail

# Exp-057：A1 SIGN标签传播特征 + 图结构特征审计
#
# 官方提分资料强调：
# - 节点度特征拼接；
# - 低度节点额外标记；
# - 特征归一化。
#
# 本实验在 Exp-053 最强 SIGN-label 配置上加入结构特征：
# - basic：入度、出度、无向度、低度/孤立标记；
# - label：basic + 当前训练标签邻居类别分布。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp057_a1_sign_structure_feature_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"
EPOCHS="${EPOCHS:-700}"
PATIENCE="${PATIENCE:-90}"

mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  local split_seed="$2"
  local structure_mode="$3"
  local structure_weight="$4"
  local out_dir="$OUT_DIR/$name/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/cs.json" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[SIGN+结构特征] $name split_seed=$split_seed"
  echo "structure_mode=$structure_mode structure_weight=$structure_weight"
  echo "============================================================"

  python3 code/a1_sign_mlp.py \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --device "$DEVICE" \
    --seed 2026 \
    --split_seed "$split_seed" \
    --val_ratio 0.1 \
    --stratified_split \
    --hops 5 \
    --prop_norm random_walk \
    --graph_mode undirected \
    --feature_norm none \
    --block_norm \
    --label_feature_hops 3 \
    --label_feature_norm random_walk \
    --label_feature_row_norm \
    --label_feature_weight 1.0 \
    --structure_feature_mode "$structure_mode" \
    --structure_feature_weight "$structure_weight" \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.4 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --log_interval 25 \
    --cs_normalize random_walk \
    --smooth_weights 0,0.5,0.75
}

IFS=',' read -ra seed_list <<< "$SPLIT_SEEDS"
for split_seed in "${seed_list[@]}"; do
  split_seed="$(echo "$split_seed" | xargs)"

  run_one "sign_label_struct_basic_w1" "$split_seed" basic 1.0
  run_one "sign_label_struct_basic_w05" "$split_seed" basic 0.5
  run_one "sign_label_struct_label_w1" "$split_seed" label 1.0
  run_one "sign_label_struct_label_w05" "$split_seed" label 0.5
done

echo
echo "============================================================"
echo "[汇总] Exp-057 A1 SIGN+结构特征审计"
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
    values = []
    details = []
    for cs_path in sorted(glob.glob(os.path.join(cand_dir, "split*", "cs.json"))):
        data = json.load(open(cs_path, encoding="utf-8"))
        best = data["results"][0]
        value = float(best["val_acc"])
        values.append(value)
        details.append((os.path.basename(os.path.dirname(cs_path)), value, best))
    if not values:
        continue
    rows.append({
        "candidate": os.path.basename(cand_dir),
        "n": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "details": details,
    })

rows.sort(key=lambda item: (item["mean"], item["min"]), reverse=True)
for row in rows:
    print(
        f"{row['candidate']}\tn={row['n']}\tmean={row['mean']:.6f}\t"
        f"min={row['min']:.6f}\tmax={row['max']:.6f}\tstd={row['std']:.6f}"
    )
    for split_name, value, best in row["details"]:
        print(
            f"  {split_name}\t{value:.6f}\t{best['kind']}\t"
            f"smooth=({best['smooth_alpha']},{best['smooth_iter']},{best['smooth_weight']})"
        )

summary_json = os.path.join(base, "summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

summary_csv = os.path.join(base, "summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["candidate", "n", "mean", "min", "max", "std"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in ["candidate", "n", "mean", "min", "max", "std"]})

print(f"\n汇总已保存: {summary_json}")
PY
