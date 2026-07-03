#!/usr/bin/env bash
set -euo pipefail

# Exp-099：A2 item.csv 特征嵌入模型多 seed 训练
#
# 与 Exp045 的区别：
# - Exp045 只用 item id 序列 + user.csv 画像；
# - Exp099 在每个历史 item 的 id embedding 外，额外拼接 item.csv 的
#   i_cat_*/i_bucket_* 特征 embedding，再投影回序列表示。
#
# 目标：
# - 验证官方建议“将 item.csv 特征嵌入模型”是否带来 A2 大幅提升；
# - 若离线优于 Exp045/Exp090，再与 A1 Exp096 组成联合候选。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SEEDS="${SEEDS:-42,777,2024}"
BASE_OUT="${BASE_OUT:-output/exp099_a2_item_feature_model_multiseed}"
MAX_LEN="${MAX_LEN:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-120}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
USER_EMBEDDING_DIM="${USER_EMBEDDING_DIM:-16}"
ITEM_FEATURE_EMBEDDING_DIM="${ITEM_FEATURE_EMBEDDING_DIM:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0,15.0,20.0,22.0,25.0,30.0,40.0,50.0}"
A1_SOURCE="${A1_SOURCE:-output/exp096_submit_a1_label_neighbor_meta_a2_stable/A1.csv}"
BASE_A2="${BASE_A2:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"
FORCE_BUILD="${FORCE_BUILD:-1}"

mkdir -p "$BASE_OUT"

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
CHECKPOINTS=()

echo
echo "============================================================"
echo "[Exp-099] 训练 A2 item-feature ranker 多 seed"
echo "============================================================"
echo "seeds=$SEEDS"
echo "base_out=$BASE_OUT"

for seed in "${SEED_LIST[@]}"; do
  seed="$(echo "$seed" | xargs)"
  out_dir="$BASE_OUT/seed${seed}"
  ckpt="$out_dir/best_model.pt"
  mkdir -p "$out_dir"

  if [[ -f "$ckpt" && "${RETRAIN:-0}" != "1" ]]; then
    echo "[跳过] seed=$seed 已存在 $ckpt"
  else
    echo
    echo "------------------------------------------------------------"
    echo "[训练] seed=$seed"
    echo "------------------------------------------------------------"
    python3 code/a2_feature_ranker.py train \
      --data_path data/rec_data \
      --output_dir "$out_dir" \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --max_len "$MAX_LEN" \
      --embedding_dim "$EMBEDDING_DIM" \
      --user_embedding_dim "$USER_EMBEDDING_DIM" \
      --item_feature_embedding_dim "$ITEM_FEATURE_EMBEDDING_DIM" \
      --hidden_dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --patience "$PATIENCE" \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed" \
      --test_like_val \
      --random_test_like_train \
      --item_cols auto
  fi
  CHECKPOINTS+=("$ckpt")
done

CHECKPOINT_ARG="$(IFS=','; echo "${CHECKPOINTS[*]}")"
echo "$CHECKPOINT_ARG" > "$BASE_OUT/checkpoints.txt"

echo
echo "============================================================"
echo "[汇总] 单 seed 训练指标"
echo "============================================================"
BASE_OUT_FOR_SUMMARY="$BASE_OUT" python3 - <<'PY'
import glob
import json
import os

rows = []
base_out = os.environ["BASE_OUT_FOR_SUMMARY"]
for path in sorted(glob.glob(os.path.join(base_out, "seed*", "metrics.json"))):
    data = json.load(open(path, encoding="utf-8"))
    best = data.get("best", {})
    rows.append((
        best.get("ndcg", 0.0),
        best.get("hit", 0.0),
        best.get("mrr", 0.0),
        best.get("epoch"),
        os.path.dirname(path),
    ))

for ndcg, hit, mrr, epoch, dirname in sorted(rows, reverse=True):
    print(f"{dirname}\tepoch={epoch}\tndcg={ndcg:.6f}\thit={hit:.6f}\tmrr={mrr:.6f}")
PY

echo
echo "============================================================"
echo "[评估] item-feature 多 seed logits 平均 + 规则融合"
echo "============================================================"

python3 code/a2_feature_ranker.py eval_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT_ARG" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --test_like_val \
  --model_weights "$MODEL_WEIGHTS" \
  --output_json "$BASE_OUT/fusion_eval.json"

BEST_WEIGHT="$(python3 -c "import json; d=json.load(open('$BASE_OUT/fusion_eval.json')); print(d['best']['model_weight'])")"
BEST_WEIGHTED_NDCG="$(python3 -c "import json; d=json.load(open('$BASE_OUT/fusion_eval.json')); print(d['best']['weighted_ndcg'])")"

echo
echo "============================================================"
echo "[结论] Exp-099 多 seed 融合"
echo "============================================================"
echo "best_model_weight=$BEST_WEIGHT"
echo "best weighted_NDCG=$BEST_WEIGHTED_NDCG"

echo
echo "============================================================"
echo "[推理] 生成 Exp-099 A2"
echo "============================================================"

python3 code/a2_feature_ranker.py predict_fusion \
  --data_path data/rec_data \
  --checkpoint "$CHECKPOINT_ARG" \
  --output_path "$BASE_OUT/A2_alt.csv" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --model_weight "$BEST_WEIGHT"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  exit 1
fi

echo
echo "============================================================"
echo "[A2] 保护Top1，中长历史接入 Exp-099"
echo "============================================================"

python3 code/a2_protected_blend.py predict \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$BASE_OUT/A2_alt.csv" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --keep_topn 1 \
  --buckets "len=4-10,len>10" \
  --topk 10 \
  --output_path "$BASE_OUT/A2.csv"

cp "$A1_SOURCE" "$BASE_OUT/A1.csv"

(
  cd "$BASE_OUT"
  rm -f prediction.zip
  zip -q prediction.zip A1.csv A2.csv
)

python3 code/validate_submission.py \
  --zip_path "$BASE_OUT/prediction.zip" \
  --cls_data_path data/cls_data/A1.npz \
  --rec_data_dir data/rec_data \
  --topk 10

echo
echo "候选提交包：$BASE_OUT/prediction.zip"
