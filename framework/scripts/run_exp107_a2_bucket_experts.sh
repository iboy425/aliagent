#!/usr/bin/env bash
set -euo pipefail

# Exp-107：A2 按历史长度训练分桶专家模型
#
# 背景：
# - A2 测试集以 len=0 / len=1 / len=2-3 为主；
# - 训练集原始历史大多很长，单一模型容易学到“长历史用户”的决策边界；
# - 之前 Exp045 整体替换线上会掉分，说明强模型不能无脑接管全部用户。
#
# 本实验把训练输入固定截断成 0、1、3 三种长度，分别训练专家：
# - expert_len0：只看用户画像和全局/画像先验；
# - expert_len1：学习单个最近 item + 用户画像；
# - expert_len3：学习 2-3 个最近 item 的短序列模式。
#
# 输出三个 A2 候选：
# - A2_hard_short.csv：len=0/1/2-3 直接用专家，冲击大但风险高；
# - A2_keep1_short.csv：对应桶保留 base Top1，专家只补第2-10位，风险低；
# - A2_mixed_short.csv：len=0/1 直接用专家，len=2-3 保护 Top1，折中。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SEEDS="${SEEDS:-42,777,2024}"
BASE_OUT="${BASE_OUT:-output/exp107_a2_bucket_experts}"
BASE_A2="${BASE_A2:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"
A1_SOURCE="${A1_SOURCE:-output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv}"

MAX_LEN="${MAX_LEN:-32}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-120}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
USER_EMBEDDING_DIM="${USER_EMBEDDING_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-0,0.2,0.5,1.0,2.0,3.0,5.0,8.0,10.0,15.0,20.0,30.0,50.0,75.0}"

if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  exit 1
fi
if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE" >&2
  exit 1
fi

mkdir -p "$BASE_OUT"

train_expert() {
  local name="$1"
  local truncate_len="$2"
  local expert_dir="$BASE_OUT/$name"
  mkdir -p "$expert_dir"

  IFS=',' read -ra seed_list <<< "$SEEDS"
  local checkpoints=()

  echo
  echo "============================================================"
  echo "[训练] $name fixed_truncate_len=$truncate_len"
  echo "============================================================"

  for seed in "${seed_list[@]}"; do
    seed="$(echo "$seed" | xargs)"
    local out_dir="$expert_dir/seed${seed}"
    local ckpt="$out_dir/best_model.pt"
    mkdir -p "$out_dir"

    if [[ -f "$ckpt" && "${RETRAIN:-0}" != "1" ]]; then
      echo "[跳过] $name seed=$seed 已存在 $ckpt"
    else
      python3 code/a2_feature_ranker.py train \
        --data_path data/rec_data \
        --output_dir "$out_dir" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --max_len "$MAX_LEN" \
        --embedding_dim "$EMBEDDING_DIM" \
        --user_embedding_dim "$USER_EMBEDDING_DIM" \
        --hidden_dim "$HIDDEN_DIM" \
        --dropout "$DROPOUT" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --patience "$PATIENCE" \
        --num_workers "$NUM_WORKERS" \
        --seed "$seed" \
        --fixed_truncate_len "$truncate_len" \
        --fixed_val_truncate_len "$truncate_len"
    fi
    checkpoints+=("$ckpt")
  done

  local checkpoint_arg
  checkpoint_arg="$(IFS=','; echo "${checkpoints[*]}")"
  echo "$checkpoint_arg" > "$expert_dir/checkpoints.txt"

  echo
  echo "============================================================"
  echo "[汇总] $name 单 seed 指标"
  echo "============================================================"
  EXPERT_DIR="$expert_dir" python3 - <<'PY'
import glob
import json
import os

rows = []
for path in sorted(glob.glob(os.path.join(os.environ["EXPERT_DIR"], "seed*", "metrics.json"))):
    data = json.load(open(path, encoding="utf-8"))
    best = data.get("best", {})
    rows.append((best.get("ndcg", 0), best.get("hit", 0), best.get("mrr", 0), best.get("epoch"), os.path.dirname(path)))

for ndcg, hit, mrr, epoch, dirname in sorted(rows, reverse=True):
    print(f"{dirname}\tepoch={epoch}\tndcg={ndcg:.6f}\thit={hit:.6f}\tmrr={mrr:.6f}")
PY

  echo
  echo "============================================================"
  echo "[评估] $name logits平均 + 规则融合"
  echo "============================================================"
  python3 code/a2_feature_ranker.py eval_fusion \
    --data_path data/rec_data \
    --checkpoint "$checkpoint_arg" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --test_like_val \
    --predict_truncate_len "$truncate_len" \
    --model_weights "$MODEL_WEIGHTS" \
    --output_json "$expert_dir/fusion_eval.json"

  local best_weight
  best_weight="$(python3 -c "import json; d=json.load(open('$expert_dir/fusion_eval.json')); print(d['best']['model_weight'])")"
  echo "[结论] $name best_model_weight=$best_weight"

  python3 code/a2_feature_ranker.py predict_fusion \
    --data_path data/rec_data \
    --checkpoint "$checkpoint_arg" \
    --output_path "$expert_dir/A2.csv" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --predict_truncate_len "$truncate_len" \
    --model_weight "$best_weight"
}

train_expert "expert_len0" 0
train_expert "expert_len1" 1
train_expert "expert_len3" 3

echo
echo "============================================================"
echo "[融合] 生成 hard / keep1 / mixed 三个 A2 候选"
echo "============================================================"

python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$BASE_OUT/expert_len0/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=0" \
  --output_path "$BASE_OUT/tmp_hard_len0.csv"
python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_OUT/tmp_hard_len0.csv" \
  --alt_a2 "$BASE_OUT/expert_len1/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=1" \
  --output_path "$BASE_OUT/tmp_hard_len1.csv"
python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_OUT/tmp_hard_len1.csv" \
  --alt_a2 "$BASE_OUT/expert_len3/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=2-3" \
  --output_path "$BASE_OUT/A2_hard_short.csv"

python3 code/a2_protected_blend.py predict \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$BASE_OUT/expert_len0/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --keep_topn 1 \
  --buckets "len=0" \
  --output_path "$BASE_OUT/tmp_keep1_len0.csv"
python3 code/a2_protected_blend.py predict \
  --base_a2 "$BASE_OUT/tmp_keep1_len0.csv" \
  --alt_a2 "$BASE_OUT/expert_len1/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --keep_topn 1 \
  --buckets "len=1" \
  --output_path "$BASE_OUT/tmp_keep1_len1.csv"
python3 code/a2_protected_blend.py predict \
  --base_a2 "$BASE_OUT/tmp_keep1_len1.csv" \
  --alt_a2 "$BASE_OUT/expert_len3/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --keep_topn 1 \
  --buckets "len=2-3" \
  --output_path "$BASE_OUT/A2_keep1_short.csv"

python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$BASE_OUT/expert_len0/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=0" \
  --output_path "$BASE_OUT/tmp_mixed_len0.csv"
python3 code/a2_bucket_blend.py \
  --base_a2 "$BASE_OUT/tmp_mixed_len0.csv" \
  --alt_a2 "$BASE_OUT/expert_len1/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --buckets "len=1" \
  --output_path "$BASE_OUT/tmp_mixed_len1.csv"
python3 code/a2_protected_blend.py predict \
  --base_a2 "$BASE_OUT/tmp_mixed_len1.csv" \
  --alt_a2 "$BASE_OUT/expert_len3/A2.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --keep_topn 1 \
  --buckets "len=2-3" \
  --output_path "$BASE_OUT/A2_mixed_short.csv"

echo
echo "============================================================"
echo "[打包] 三个候选，仅供评估选择"
echo "============================================================"

for name in hard_short keep1_short mixed_short; do
  out_dir="$BASE_OUT/submit_${name}"
  mkdir -p "$out_dir"
  cp "$A1_SOURCE" "$out_dir/A1.csv"
  cp "$BASE_OUT/A2_${name}.csv" "$out_dir/A2.csv"
  (
    cd "$out_dir"
    rm -f prediction.zip
    zip -q prediction.zip A1.csv A2.csv
  )
  python3 code/validate_submission.py \
    --zip_path "$out_dir/prediction.zip" \
    --cls_data_path data/cls_data/A1.npz \
    --rec_data_dir data/rec_data \
    --topk 10
  echo "候选包：$out_dir/prediction.zip"
done

echo
echo "Exp-107 完成。请把三类候选的输出摘要发回来，尤其是 changed/top1_changed/overlap。"
