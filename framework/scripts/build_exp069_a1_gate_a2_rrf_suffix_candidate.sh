#!/usr/bin/env bash
set -euo pipefail

# Exp-069：A1邻居多数类轻门控 + A2 RRF软融合 + len2-3有序后缀转移
#
# 相比 Exp-068：
# - A1：加入无泄漏审计后只有小幅正收益的 majority gate，参数保持保守；
# - A2：在 RRF 软融合基础上，对最大短历史桶 len=2-3 加 ordered suffix transition。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
OUT_DIR="${OUT_DIR:-output/exp069_submit_a1_gate_a2_rrf_suffix}"
BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"

mkdir -p "$OUT_DIR"

find_one_checkpoint() {
  local pattern="$1"
  local matches=()
  while IFS= read -r match; do
    matches+=("$match")
  done < <(compgen -G "$pattern" || true)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "checkpoint匹配数量异常: pattern=$pattern, count=${#matches[@]}" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
  fi
  echo "${matches[0]}"
}

SPLITS=(42 777 2024 2026 3407)
SEEDS=(42 777 2024 2026 3407)
EXP059_UNDIR_DIR="$EXP059_DIR/standard_struct_label_undir_h3_rw"
EXP059_REVERSE_DIR="$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw"

CHECKPOINTS=()
WEIGHTS=()

for seed in "${SEEDS[@]}"; do
  CHECKPOINTS+=("$(find_one_checkpoint "$EXP061_DIR/undir_seed${seed}_e*/best_model.pt")")
  WEIGHTS+=("0.054")
done
for seed in "${SEEDS[@]}"; do
  CHECKPOINTS+=("$(find_one_checkpoint "$EXP061_DIR/undir_reverse_seed${seed}_e*/best_model.pt")")
  WEIGHTS+=("0.126")
done
for split in "${SPLITS[@]}"; do
  CHECKPOINTS+=("$EXP059_UNDIR_DIR/split${split}/best_model.pt")
  WEIGHTS+=("0.006")
done
for split in "${SPLITS[@]}"; do
  CHECKPOINTS+=("$EXP059_REVERSE_DIR/split${split}/best_model.pt")
  WEIGHTS+=("0.014")
done

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "缺少checkpoint: $ckpt"
    exit 1
  fi
done

CHECKPOINT_WEIGHTS="$(IFS=','; echo "${WEIGHTS[*]}")"

echo
echo "============================================================"
echo "[A1] Exp066概率 + 邻居多数类轻门控"
echo "============================================================"
echo "checkpoint_weights=$CHECKPOINT_WEIGHTS"

python3 code/a1_neighbor_gate.py infer \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --checkpoint_weights "$CHECKPOINT_WEIGHTS" \
  --device "$DEVICE" \
  --expert majority \
  --min_neighbors_value 2 \
  --purity_threshold 0.85 \
  --max_model_conf 0.65 \
  --expert_weight 0.35 \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_gate_infer.json"

echo
echo "============================================================"
echo "[A2] RRF软融合 + len2-3 ordered suffix transition"
echo "============================================================"

python3 code/a2_rank_fusion.py \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$ALT_A2" \
  --train_csv data/rec_data/train.csv \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --bucket_lambdas "len=0:1.20,len=1:0.35,len=2-3:0.20,len=4-10:0.05,len>10:0.05" \
  --suffix_buckets "len=2-3" \
  --suffix_weight1 0.04 \
  --suffix_weight2 0.08 \
  --suffix_weight3 0.12 \
  --suffix_min_count2 10 \
  --suffix_min_count3 10 \
  --rrf_k 60 \
  --topk 10 \
  --output_path "$OUT_DIR/A2.csv"

echo
echo "============================================================"
echo "[打包] Exp-069"
echo "============================================================"

(
  cd "$OUT_DIR"
  rm -f prediction.zip
  zip -q prediction.zip A1.csv A2.csv
)

python3 code/validate_submission.py \
  --zip_path "$OUT_DIR/prediction.zip" \
  --cls_data_path data/cls_data/A1.npz \
  --rec_data_dir data/rec_data \
  --topk 10

echo
echo "候选提交包：$OUT_DIR/prediction.zip"
