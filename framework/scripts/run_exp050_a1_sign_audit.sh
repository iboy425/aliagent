#!/usr/bin/env bash
set -euo pipefail

# Exp-050：A1 SIGN/MLP 多跳传播特征审计
#
# 目的：
# - 当前 GAT/GCN 真多 split 稳定水平约 0.699，距离第一名 A1 仍远。
# - A1训练边同质性约0.79，说明多跳图传播有价值。
# - SIGN把 X, AX, A^2X... 直接拼接，再训练MLP，和GAT/GCN互补。
#
# 判断标准：
# - 若某组 mean 明显超过 0.70，且 min 不低于当前稳定候选，再扩展训练/集成。
# - 若所有配置仍停在 0.69 左右，说明需要换更强图模型或额外特征工程。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
OUT_DIR="${OUT_DIR:-output/exp050_a1_sign_audit}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"
EPOCHS="${EPOCHS:-700}"
PATIENCE="${PATIENCE:-90}"

mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  local split_seed="$2"
  local hops="$3"
  local prop_norm="$4"
  local graph_mode="$5"
  local feature_norm="$6"
  local block_norm="$7"
  local hidden="$8"
  local layers="$9"
  local dropout="${10}"
  local lr="${11}"
  local wd="${12}"
  local seed="${13}"
  local out_dir="$OUT_DIR/$name/split${split_seed}"

  mkdir -p "$out_dir"
  if [[ -f "$out_dir/cs.json" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] $name split_seed=$split_seed 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[SIGN审计] $name split_seed=$split_seed"
  echo "hops=$hops prop_norm=$prop_norm graph_mode=$graph_mode feature_norm=$feature_norm block_norm=$block_norm hidden=$hidden layers=$layers dropout=$dropout lr=$lr wd=$wd seed=$seed"
  echo "============================================================"

  local block_arg=()
  if [[ "$block_norm" == "1" ]]; then
    block_arg=(--block_norm)
  fi

  python3 code/a1_sign_mlp.py \
    --data_path "$DATA_PATH" \
    --output_dir "$out_dir" \
    --device "$DEVICE" \
    --seed "$seed" \
    --split_seed "$split_seed" \
    --val_ratio 0.1 \
    --stratified_split \
    --hops "$hops" \
    --prop_norm "$prop_norm" \
    --graph_mode "$graph_mode" \
    --feature_norm "$feature_norm" \
    "${block_arg[@]}" \
    --hidden_dim "$hidden" \
    --num_layers "$layers" \
    --dropout "$dropout" \
    --lr "$lr" \
    --weight_decay "$wd" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --log_interval 25 \
    --cs_normalize random_walk \
    --smooth_weights 0,0.5,0.75,1.0
}

IFS=',' read -ra seed_list <<< "$SPLIT_SEEDS"
for split_seed in "${seed_list[@]}"; do
  split_seed="$(echo "$split_seed" | xargs)"

  # 基础SIGN：无向对称归一化，3跳拼接。
  run_one "sign_sym_undir_k3_h512_l2_do04_none" "$split_seed" 3 symmetric undirected none 0 512 2 0.4 0.001 0.0005 2026

  # 更深传播：看5跳是否补足远距离测试节点。
  run_one "sign_sym_undir_k5_h512_l2_do04_none" "$split_seed" 5 symmetric undirected none 0 512 2 0.4 0.001 0.0005 2026

  # 每跳行归一化：降低高阶传播尺度漂移。
  run_one "sign_sym_undir_k5_h512_l2_do04_block" "$split_seed" 5 symmetric undirected none 1 512 2 0.4 0.001 0.0005 2026

  # 随机游走传播：和C&S一致，可能更适合同质图标签扩散。
  run_one "sign_rw_undir_k5_h512_l2_do04_block" "$split_seed" 5 random_walk undirected none 1 512 2 0.4 0.001 0.0005 2026

  # row特征归一化：官方提分建议之一，单独验证是否改善SIGN。
  run_one "sign_sym_undir_k5_h512_l2_do04_rowblock" "$split_seed" 5 symmetric undirected row 1 512 2 0.4 0.001 0.0005 2026
done

echo
echo "============================================================"
echo "[汇总] Exp-050 A1 SIGN/MLP审计"
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
