#!/usr/bin/env bash
set -euo pipefail

# 通用联合候选生成器：A1 SIGN配置融合 + A2历史长度桶替换
#
# 设计目的：
# - 每次提交同时改变 A1 和 A2，平台返回两边分数后可以同步归因；
# - A1 通过 FULLTRAIN_WEIGHT 控制“全标签重训模型”和“分折模型”的比例；
# - A2 通过 A2_BUCKETS 控制哪些历史长度用户使用备用推荐结果。
#
# 常用参数示例：
#   FULLTRAIN_WEIGHT=0.90 A2_BUCKETS="len=0,len=1" \
#   OUT_DIR=output/my_candidate ./scripts/build_joint_a1_sign_a2_bucket_candidate.sh
#
# A2_BUCKETS 可取：
# - base：直接复制 BASE_A2，不做桶替换；
# - len=0
# - len=0,len=1
# - len=0,len=1,len=2-3

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
BASE_A2="${BASE_A2:-output/exp044_a2_feature_fusion/A2.csv}"
ALT_A2="${ALT_A2:-output/exp045_a2_feature_multiseed/A2.csv}"
OUT_DIR="${OUT_DIR:-output/joint_a1_sign_a2_bucket_candidate}"

FULLTRAIN_WEIGHT="${FULLTRAIN_WEIGHT:-0.85}"
UNDER_WEIGHT="${UNDER_WEIGHT:-0.30}"
REVERSE_WEIGHT="${REVERSE_WEIGHT:-0.70}"
A2_BUCKETS="${A2_BUCKETS:-len=0,len=1}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.0}"

mkdir -p "$OUT_DIR"

float_expr() {
  awk "BEGIN { printf \"%.12g\", $* }"
}

is_positive() {
  awk "BEGIN { exit !($1 > 1e-12) }"
}

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

add_group() {
  local group_name="$1"
  local total_weight="$2"
  local per_count="$3"
  shift 3

  if ! is_positive "$total_weight"; then
    echo "[A1权重] 跳过 $group_name total_weight=$total_weight"
    return
  fi

  local per_weight
  per_weight="$(float_expr "$total_weight / $per_count")"
  echo "[A1权重] $group_name total_weight=$total_weight per_weight=$per_weight"

  for checkpoint_path in "$@"; do
    if [[ ! -f "$checkpoint_path" ]]; then
      echo "缺少checkpoint: $checkpoint_path" >&2
      exit 1
    fi
    CHECKPOINTS+=("$checkpoint_path")
    WEIGHTS+=("$per_weight")
  done
}

SPLITS=(42 777 2024 2026 3407)
SEEDS=(42 777 2024 2026 3407)

EXP059_UNDIR_DIR="$EXP059_DIR/standard_struct_label_undir_h3_rw"
EXP059_UNDIR_REVERSE_DIR="$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw"

FULL_UNDIR_CHECKPOINTS=()
FULL_REVERSE_CHECKPOINTS=()
SPLIT_UNDIR_CHECKPOINTS=()
SPLIT_REVERSE_CHECKPOINTS=()

for seed in "${SEEDS[@]}"; do
  FULL_UNDIR_CHECKPOINTS+=("$(find_one_checkpoint "$EXP061_DIR/undir_seed${seed}_e*/best_model.pt")")
  FULL_REVERSE_CHECKPOINTS+=("$(find_one_checkpoint "$EXP061_DIR/undir_reverse_seed${seed}_e*/best_model.pt")")
done

for split in "${SPLITS[@]}"; do
  SPLIT_UNDIR_CHECKPOINTS+=("$EXP059_UNDIR_DIR/split${split}/best_model.pt")
  SPLIT_REVERSE_CHECKPOINTS+=("$EXP059_UNDIR_REVERSE_DIR/split${split}/best_model.pt")
done

SPLIT_WEIGHT="$(float_expr "1.0 - $FULLTRAIN_WEIGHT")"
FULL_UNDIR_WEIGHT="$(float_expr "$FULLTRAIN_WEIGHT * $UNDER_WEIGHT")"
FULL_REVERSE_WEIGHT="$(float_expr "$FULLTRAIN_WEIGHT * $REVERSE_WEIGHT")"
SPLIT_UNDIR_WEIGHT="$(float_expr "$SPLIT_WEIGHT * $UNDER_WEIGHT")"
SPLIT_REVERSE_WEIGHT="$(float_expr "$SPLIT_WEIGHT * $REVERSE_WEIGHT")"

CHECKPOINTS=()
WEIGHTS=()
add_group "Exp061全标签 undir" "$FULL_UNDIR_WEIGHT" 5 "${FULL_UNDIR_CHECKPOINTS[@]}"
add_group "Exp061全标签 undir_reverse" "$FULL_REVERSE_WEIGHT" 5 "${FULL_REVERSE_CHECKPOINTS[@]}"
add_group "Exp059分折 undir" "$SPLIT_UNDIR_WEIGHT" 5 "${SPLIT_UNDIR_CHECKPOINTS[@]}"
add_group "Exp059分折 undir_reverse" "$SPLIT_REVERSE_WEIGHT" 5 "${SPLIT_REVERSE_CHECKPOINTS[@]}"

CHECKPOINT_WEIGHTS="$(IFS=','; echo "${WEIGHTS[*]}")"

if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  exit 1
fi
if [[ "$A2_BUCKETS" != "base" && ! -f "$ALT_A2" ]]; then
  echo "ALT_A2不存在: $ALT_A2" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[A1] SIGN配置融合"
echo "============================================================"
echo "FULLTRAIN_WEIGHT=$FULLTRAIN_WEIGHT"
echo "SPLIT_WEIGHT=$SPLIT_WEIGHT"
echo "UNDER_WEIGHT=$UNDER_WEIGHT"
echo "REVERSE_WEIGHT=$REVERSE_WEIGHT"
echo "checkpoint_count=${#CHECKPOINTS[@]}"
echo "checkpoint_weights=$CHECKPOINT_WEIGHTS"

python3 code/a1_sign_infer.py \
  --data_path "$DATA_PATH" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --checkpoint_weights "$CHECKPOINT_WEIGHTS" \
  --device "$DEVICE" \
  --cs_normalize random_walk \
  --correct_alpha 0.3 \
  --correct_iter 5 \
  --correct_weight 0.0 \
  --smooth_alpha 0.7 \
  --smooth_iter 5 \
  --smooth_weight "$SMOOTH_WEIGHT" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_sign_joint_infer.json"

echo
echo "============================================================"
echo "[A2] 历史长度桶替换"
echo "============================================================"
echo "A2_BUCKETS=$A2_BUCKETS"

if [[ "$A2_BUCKETS" == "base" ]]; then
  cp "$BASE_A2" "$OUT_DIR/A2.csv"
else
  python3 code/a2_bucket_blend.py \
    --base_a2 "$BASE_A2" \
    --alt_a2 "$ALT_A2" \
    --test_csv data/rec_data/test.csv \
    --seq_col item_seq_raw \
    --buckets "$A2_BUCKETS" \
    --output_path "$OUT_DIR/A2.csv" \
    --topk 10
fi

echo
echo "============================================================"
echo "[打包] 联合候选"
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
