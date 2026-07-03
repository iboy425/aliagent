#!/usr/bin/env bash
set -euo pipefail

# Exp-106：A1 SIGN 标签平滑审计
#
# 目的：
# - 当前 A1 最强路线是 SIGN + 多方向标签传播特征。
# - 这类模型很容易对训练标签过度自信，线上有轻微过拟合风险。
# - 官方提分建议中也明确提到 Label Smoothing。本实验只改训练损失，
#   不改特征、不改推理后处理，用 5 split 判断是否稳定提升。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp106_a1_label_smoothing_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"
LABEL_SMOOTHINGS="${LABEL_SMOOTHINGS:-0.03,0.05}"
EPOCHS="${EPOCHS:-700}"
PATIENCE="${PATIENCE:-90}"

mkdir -p "$OUT_DIR"

sanitize_float() {
  echo "$1" | sed 's/\./p/g'
}

run_one() {
  local config_name="$1"
  local label_modes="$2"
  local label_norms="$3"
  local label_hops="$4"
  local split_seed="$5"
  local label_smoothing="$6"
  local label_smoothing_tag
  label_smoothing_tag="$(sanitize_float "$label_smoothing")"
  local out_dir="$OUT_DIR/${config_name}_ls${label_smoothing_tag}/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/cs.json" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] $config_name label_smoothing=$label_smoothing split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[A1 Label Smoothing] config=$config_name split_seed=$split_seed label_smoothing=$label_smoothing"
  echo "label_modes=$label_modes label_norms=$label_norms label_hops=$label_hops"
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
    --feature_transform standard \
    --block_norm \
    --label_feature_hops "$label_hops" \
    --label_feature_norm random_walk \
    --label_feature_graph_modes "$label_modes" \
    --label_feature_norms "$label_norms" \
    --label_feature_row_norm \
    --label_feature_weight 1.0 \
    --structure_feature_mode basic \
    --structure_feature_weight 1.0 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.4 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --label_smoothing "$label_smoothing" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --log_interval 25 \
    --cs_normalize random_walk \
    --smooth_weights 0,0.5,0.75
}

IFS=',' read -ra split_list <<< "$SPLIT_SEEDS"
IFS=',' read -ra smoothing_list <<< "$LABEL_SMOOTHINGS"

for label_smoothing in "${smoothing_list[@]}"; do
  label_smoothing="$(echo "$label_smoothing" | xargs)"
  for split_seed in "${split_list[@]}"; do
    split_seed="$(echo "$split_seed" | xargs)"

    # Exp059/Exp060 线上有效的两个核心配置：
    # - undirected：稳定基底；
    # - undirected+reverse：当前 A1 高上限主力。
    run_one "undir_h3_rw" "undirected" "random_walk" 3 "$split_seed" "$label_smoothing"
    run_one "undir_reverse_h3_rw" "undirected,reverse" "random_walk" 3 "$split_seed" "$label_smoothing"
  done
done

echo
echo "============================================================"
echo "[汇总] Exp-106 A1 Label Smoothing 审计"
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
