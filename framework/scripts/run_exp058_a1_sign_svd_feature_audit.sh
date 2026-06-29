#!/usr/bin/env bash
set -euo pipefail

# Exp-058：A1 SIGN标签传播特征 + 结构特征 + SVD降噪特征审计
#
# 官方提分资料提到：
# - PCA降维：767维 -> 128维，减少噪声；
# - 特征归一化：L2 / StandardScaler；
# - 节点度特征拼接。
#
# 本实验在 Exp-057 最强配置 basic_w1 上继续加“低秩降噪特征”。
# 这里用 TruncatedSVD 替代 PCA，原因是 A1 原始特征是稀疏矩阵，
# TruncatedSVD 可以直接处理稀疏输入，更适合当前数据。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp058_a1_sign_svd_feature_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"
EPOCHS="${EPOCHS:-700}"
PATIENCE="${PATIENCE:-90}"

mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  local split_seed="$2"
  local feature_transform="$3"
  local svd_dim="$4"
  local svd_weight="$5"
  local out_dir="$OUT_DIR/$name/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/cs.json" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[SIGN+SVD特征] $name split_seed=$split_seed"
  echo "feature_transform=$feature_transform svd_dim=$svd_dim svd_weight=$svd_weight"
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
    --feature_transform "$feature_transform" \
    --svd_dim "$svd_dim" \
    --svd_weight "$svd_weight" \
    --svd_seed 42 \
    --block_norm \
    --label_feature_hops 3 \
    --label_feature_norm random_walk \
    --label_feature_row_norm \
    --label_feature_weight 1.0 \
    --structure_feature_mode basic \
    --structure_feature_weight 1.0 \
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

  run_one "sign_label_struct_basic_standard" "$split_seed" standard 128 1.0
  run_one "sign_label_struct_basic_svd128_w1" "$split_seed" svd 128 1.0
  run_one "sign_label_struct_basic_raw_svd64_w05" "$split_seed" raw_plus_svd 64 0.5
  run_one "sign_label_struct_basic_raw_svd128_w05" "$split_seed" raw_plus_svd 128 0.5
  run_one "sign_label_struct_basic_raw_svd128_w1" "$split_seed" raw_plus_svd 128 1.0
  run_one "sign_label_struct_basic_raw_svd256_w05" "$split_seed" raw_plus_svd 256 0.5
done

echo
echo "============================================================"
echo "[汇总] Exp-058 A1 SIGN+SVD降噪特征审计"
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
