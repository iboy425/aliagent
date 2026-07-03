#!/usr/bin/env bash
set -euo pipefail

# Exp-095：A1复用线上稳定版本，A2尝试 item.csv 物品侧特征转移 + 严格gate
#
# 背景：
# - Exp094 说明“全量重训ranker”没有线上收益；
# - 官方提分建议强调 item.csv 物品侧特征，但主力融合链路之前没有真正接入；
# - 直接替换整份 A2 风险高，所以本脚本先生成 item-feature alt，再用画像分群 gate
#   和 Top1 保护把它接到当前线上最稳的 Exp086/Exp090 底座上。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp095_submit_a1_best_a2_item_feature_gate}"
A1_SOURCE="${A1_SOURCE:-output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv}"
BASE_A2="${BASE_A2:-output/exp086_submit_a1_best_a2_strict_gate_midlong_keep_top1/A2.csv}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-output/exp044_a2_feature_ranker_seed42/best_model.pt}"
ALT_CHECKPOINT="${ALT_CHECKPOINT:-}"
if [[ -z "$ALT_CHECKPOINT" ]]; then
  if [[ -f output/exp045_a2_feature_multiseed/checkpoints.txt ]]; then
    ALT_CHECKPOINT="$(cat output/exp045_a2_feature_multiseed/checkpoints.txt)"
  else
    ALT_CHECKPOINT="output/exp045_a2_feature_multiseed/seed42/best_model.pt,output/exp045_a2_feature_multiseed/seed777/best_model.pt,output/exp045_a2_feature_multiseed/seed2024/best_model.pt"
  fi
fi

BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_LEN="${MAX_LEN:-120}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0,15.0,20.0,22.0,25.0,30.0}"
ITEM_FEATURE_WEIGHTS="${ITEM_FEATURE_WEIGHTS:-0,0.002,0.005,0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2}"
ITEM_FEATURE_RECENT_N="${ITEM_FEATURE_RECENT_N:-10}"
ITEM_FEATURE_MIN_COUNT="${ITEM_FEATURE_MIN_COUNT:-20}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4000}"
PREDICT_BUCKETS="${PREDICT_BUCKETS:-len=4-10,len>10}"
KEEP_TOPN="${KEEP_TOPN:-1}"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  exit 1
fi
if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "BASE_CHECKPOINT不存在: $BASE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[Exp-095] A2 item-feature 融合审计"
echo "============================================================"
echo "out_dir=$OUT_DIR"
echo "base_a2=$BASE_A2"
echo "alt_checkpoint=$ALT_CHECKPOINT"
echo "model_weights=$MODEL_WEIGHTS"
echo "item_feature_weights=$ITEM_FEATURE_WEIGHTS"

python3 code/a2_feature_ranker.py eval_fusion \
  --data_path data/rec_data \
  --checkpoint "$ALT_CHECKPOINT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --test_like_val \
  --model_weights "$MODEL_WEIGHTS" \
  --item_feature_weights "$ITEM_FEATURE_WEIGHTS" \
  --item_feature_recent_n "$ITEM_FEATURE_RECENT_N" \
  --item_feature_min_count "$ITEM_FEATURE_MIN_COUNT" \
  --output_json "$OUT_DIR/item_feature_fusion_eval.json"

read -r BEST_MODEL_WEIGHT BEST_ITEM_FEATURE_WEIGHT BEST_WEIGHTED_NDCG <<< "$(
  python3 - <<PY
import json
d = json.load(open("$OUT_DIR/item_feature_fusion_eval.json", encoding="utf-8"))
b = d["best"]
print(b["model_weight"], b["item_feature_weight"], b["weighted_ndcg"])
PY
)"

python3 - <<PY
import json
d = json.load(open("$OUT_DIR/item_feature_fusion_eval.json", encoding="utf-8"))
b = d["best"]
baseline = max(
    row["weighted_ndcg"]
    for row in d["results"]
    if float(row.get("item_feature_weight", 0.0)) == 0.0
)
out = {
    "best": b,
    "best_no_item_feature_weighted_ndcg": baseline,
    "gain_vs_no_item_feature": b["weighted_ndcg"] - baseline,
}
json.dump(out, open("$OUT_DIR/item_feature_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("best_model_weight=", b["model_weight"])
print("best_item_feature_weight=", b["item_feature_weight"])
print("best_weighted_ndcg=", b["weighted_ndcg"])
print("best_no_item_feature_weighted_ndcg=", baseline)
print("gain_vs_no_item_feature=", b["weighted_ndcg"] - baseline)
PY

echo
echo "============================================================"
echo "[Exp-095] 生成 item-feature alt A2"
echo "============================================================"

python3 code/a2_feature_ranker.py predict_fusion \
  --data_path data/rec_data \
  --checkpoint "$ALT_CHECKPOINT" \
  --output_path "$OUT_DIR/A2_item_feature_alt.csv" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --model_weight "$BEST_MODEL_WEIGHT" \
  --item_feature_weight "$BEST_ITEM_FEATURE_WEIGHT" \
  --item_feature_recent_n "$ITEM_FEATURE_RECENT_N" \
  --item_feature_min_count "$ITEM_FEATURE_MIN_COUNT"

echo
echo "============================================================"
echo "[Exp-095] 严格画像分群 gate 审计"
echo "============================================================"

python3 code/a2_profile_gate.py audit \
  --data_path data/rec_data \
  --base_checkpoint "$BASE_CHECKPOINT" \
  --alt_checkpoint "$ALT_CHECKPOINT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --test_like_val \
  --split_seeds "$SPLIT_SEEDS" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --base_model_weight 1.0 \
  --alt_model_weight "$BEST_MODEL_WEIGHT" \
  --item_feature_weight "$BEST_ITEM_FEATURE_WEIGHT" \
  --item_feature_recent_n "$ITEM_FEATURE_RECENT_N" \
  --item_feature_min_count "$ITEM_FEATURE_MIN_COUNT" \
  --buckets "len=2-3,len=4-10,len>10" \
  --profile_specs "single,prefix2,prefix3" \
  --min_samples 100 \
  --min_split_samples 25 \
  --min_positive_splits 3 \
  --min_gain 0.015 \
  --max_groups 12 \
  --output_json "$OUT_DIR/profile_gate_policy.json"

echo
echo "============================================================"
echo "[Exp-095] 生成 gate 后 A2"
echo "============================================================"

python3 code/a2_profile_gate.py predict \
  --policy_json "$OUT_DIR/profile_gate_policy.json" \
  --base_a2 "$BASE_A2" \
  --alt_a2 "$OUT_DIR/A2_item_feature_alt.csv" \
  --test_csv data/rec_data/test.csv \
  --user_csv data/rec_data/user.csv \
  --seq_col item_seq_raw \
  --keep_topn "$KEEP_TOPN" \
  --predict_buckets "$PREDICT_BUCKETS" \
  --topk 10 \
  --output_path "$OUT_DIR/A2.csv"

cp "$A1_SOURCE" "$OUT_DIR/A1.csv"

echo
echo "============================================================"
echo "[Exp-095] 打包 prediction.zip"
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
echo "审计摘要：$OUT_DIR/item_feature_summary.json"
echo "gate策略：$OUT_DIR/profile_gate_policy.json"
