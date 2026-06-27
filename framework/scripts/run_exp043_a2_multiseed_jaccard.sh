#!/usr/bin/env bash
set -euo pipefail

# Exp-043：A2 jaccard 多seed稳定性评估
#
# 目的：
# - 对比旧线上方案 log_count 与新 jaccard 方案在多个随机验证切分上的表现。
# - 防止只因为 seed=42 的验证集偶然偏好 jaccard 就贸然提交。

cd "$(dirname "$0")/.."

OUT_DIR="output/exp043_a2_multiseed_jaccard"
mkdir -p "$OUT_DIR"

run_eval() {
  local name="$1"
  local seed="$2"
  local recent_n="$3"
  local formula="$4"
  local decay="$5"
  local user_w="$6"
  local combo_w="$7"

  python3 code/a2_grid_search.py \
    --data_path data/rec_data \
    --val_ratio 0.1 \
    --seed "$seed" \
    --topk 10 \
    --seq_cols item_seq_raw \
    --recent_ns "$recent_n" \
    --cooccur_decays "$decay" \
    --cooccur_formulas "$formula" \
    --strategies history \
    --history_filters none \
    --user_weights "$user_w" \
    --user_combo_weights "$combo_w" \
    --user_combo_sizes 3,2,1 \
    --user_combo_mode prefix \
    --user_combo_min_count 5 \
    --item_feature_weights 0 \
    --pop_penalty_weights 0 \
    --history_count_weights 0 \
    --test_like_eval \
    --sort_metric weighted_ndcg \
    --top_results 3 \
    --output_json "$OUT_DIR/${name}_seed${seed}.json"
}

for seed in 42 777 2024 2026 3407; do
  echo
  echo "============================================================"
  echo "[seed=$seed] 旧方案 log_count"
  echo "============================================================"
  run_eval "old_log_count" "$seed" 10 log_count 0.96 0.02 0.18

  echo
  echo "============================================================"
  echo "[seed=$seed] 新方案 jaccard"
  echo "============================================================"
  run_eval "new_jaccard" "$seed" 18 jaccard 1.0 0.01 0.1
done

python3 - <<'PY'
import glob
import json
import os
from statistics import mean

rows = []
for path in sorted(glob.glob("output/exp043_a2_multiseed_jaccard/*.json")):
    data = json.load(open(path, encoding="utf-8"))
    best = data[0]
    base = os.path.basename(path)
    kind = "new_jaccard" if base.startswith("new_jaccard") else "old_log_count"
    seed = int(base.rsplit("seed", 1)[1].split(".")[0])
    rows.append({
        "kind": kind,
        "seed": seed,
        "weighted_ndcg": best["weighted_ndcg"],
        "ndcg": best["ndcg"],
        "hit": best["hit"],
        "mrr": best["mrr"],
    })

by_seed = {}
for row in rows:
    by_seed.setdefault(row["seed"], {})[row["kind"]] = row

print("\nExp-043 多seed对比")
gains = []
for seed in sorted(by_seed):
    old = by_seed[seed]["old_log_count"]
    new = by_seed[seed]["new_jaccard"]
    gain = new["weighted_ndcg"] - old["weighted_ndcg"]
    gains.append(gain)
    print(
        f"seed={seed}\t"
        f"old={old['weighted_ndcg']:.6f}\t"
        f"new={new['weighted_ndcg']:.6f}\t"
        f"gain={gain:+.6f}"
    )

print("-" * 72)
print(f"avg_gain={mean(gains):+.6f}")
print(f"win_count={sum(g > 0 for g in gains)}/{len(gains)}")
print(f"min_gain={min(gains):+.6f}")
print(f"max_gain={max(gains):+.6f}")
PY
