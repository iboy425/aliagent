#!/usr/bin/env bash
set -euo pipefail

# Exp-053：A1 SIGN + 标签传播特征审计
#
# 背景：
# - Exp-051线上 A1=0.7117，证明SIGN有效。
# - Exp-052异构融合没有超过单SIGN无泄漏均值。
# - 本实验把训练标签传播后的分布作为MLP输入，让模型在训练阶段直接学习
#   “属性传播特征 + 标签传播特征”的组合，而不是只在最后做C&S。
#
# 防泄漏：
# - 每个split只用该split训练节点构造标签传播特征。
# - 验证节点标签不参与标签特征构造。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp053_a1_sign_label_feature_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"
EPOCHS="${EPOCHS:-700}"
PATIENCE="${PATIENCE:-90}"

mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  local split_seed="$2"
  local label_hops="$3"
  local label_norm="$4"
  local label_row_norm="$5"
  local label_weight="$6"
  local include_seed="$7"
  local out_dir="$OUT_DIR/$name/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/cs.json" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[SIGN+标签特征] $name split_seed=$split_seed"
  echo "label_hops=$label_hops label_norm=$label_norm label_row_norm=$label_row_norm label_weight=$label_weight include_seed=$include_seed"
  echo "============================================================"

  local label_row_arg=()
  if [[ "$label_row_norm" == "1" ]]; then
    label_row_arg=(--label_feature_row_norm)
  fi
  local include_seed_arg=()
  if [[ "$include_seed" == "1" ]]; then
    include_seed_arg=(--label_feature_include_seed)
  fi

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
    --label_feature_hops "$label_hops" \
    --label_feature_norm "$label_norm" \
    "${label_row_arg[@]}" \
    "${include_seed_arg[@]}" \
    --label_feature_weight "$label_weight" \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.4 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --log_interval 25 \
    --cs_normalize random_walk \
    --smooth_weights 0,0.5,0.75,1.0
}

IFS=',' read -ra seed_list <<< "$SPLIT_SEEDS"
for split_seed in "${seed_list[@]}"; do
  split_seed="$(echo "$split_seed" | xargs)"

  # 不拼Y0，只拼标签传播后的1..H跳结果，避免训练节点直接看到自己标签。
  run_one "sign_rw_k5_block_labelrw_h2_w1" "$split_seed" 2 random_walk 0 1.0 0
  run_one "sign_rw_k5_block_labelrw_h3_w1" "$split_seed" 3 random_walk 0 1.0 0
  run_one "sign_rw_k5_block_labelrw_h5_w1" "$split_seed" 5 random_walk 0 1.0 0

  # 行归一化标签传播分布，降低度数带来的幅度差。
  run_one "sign_rw_k5_block_labelrw_h3_rownorm_w1" "$split_seed" 3 random_walk 1 1.0 0

  # 对称归一化标签传播，和随机游走形成对照。
  run_one "sign_rw_k5_block_labelsym_h3_w1" "$split_seed" 3 symmetric 0 1.0 0

  # 降低标签特征权重，检查是否比强标签信号更稳。
  run_one "sign_rw_k5_block_labelrw_h3_w05" "$split_seed" 3 random_walk 0 0.5 0
done

echo
echo "============================================================"
echo "[汇总] Exp-053 A1 SIGN+标签传播特征审计"
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
