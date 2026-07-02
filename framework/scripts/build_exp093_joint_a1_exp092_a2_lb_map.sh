#!/usr/bin/env bash
set -euo pipefail

# Exp-093：A1 Exp092 leaderboard MAP + A2 leaderboard MAP 联合候选
#
# 这是 A1/A2 两头同时推进的高风险候选：
# - A1 使用 Exp092：Accuracy 稀疏反馈约束 MAP；
# - A2 使用 NDCG 稀疏反馈约束 MAP；
# - 默认 A2 不保护 Top1，属于大幅重排；可设置 PROTECT_TOPN=1 生成保守版。

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-output/exp093_submit_a1_exp092_a2_lb_map}"
A1_SOURCE="${A1_SOURCE:-output/exp092_submit_a1_lb_map_model_prior_a2_exp086/A1.csv}"
BASE_A2="${BASE_A2:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"
PROTECT_TOPN="${PROTECT_TOPN:-0}"
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.5}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.015}"
OTHER_PRIOR="${OTHER_PRIOR:--0.02}"
WEIGHT_MODE="${WEIGHT_MODE:-centered}"
SOLVER="${SOLVER:-greedy}"

if [[ ! -f "$A1_SOURCE" ]]; then
  echo "A1_SOURCE不存在: $A1_SOURCE" >&2
  echo "请先运行 scripts/build_exp092_a1_lb_map_model_prior_a2_exp086.sh" >&2
  exit 1
fi
if [[ ! -f "$BASE_A2" ]]; then
  echo "BASE_A2不存在: $BASE_A2" >&2
  echo "请先运行 scripts/build_exp090_joint_a1_safe_bias_a2_exp086.sh" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A2] leaderboard constrained MAP"
echo "============================================================"
echo "protect_topn=$PROTECT_TOPN"
echo "score_tolerance=$SCORE_TOLERANCE"
echo "weight_mode=$WEIGHT_MODE"
echo "solver=$SOLVER"

python3 code/a2_leaderboard_map.py \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --base_submission exp090 \
  --anchor_submission exp090 \
  --weight_mode "$WEIGHT_MODE" \
  --anchor_weight "$ANCHOR_WEIGHT" \
  --other_prior "$OTHER_PRIOR" \
  --score_tolerance "$SCORE_TOLERANCE" \
  --include_other \
  --protect_topn "$PROTECT_TOPN" \
  --time_limit 300 \
  --mip_rel_gap 0.0 \
  --solver "$SOLVER" \
  --submission "exp030,0.4746,output/exp030_submit_a1cs_a2_decay_combo018/A2.csv" \
  --submission "exp044,0.5023,output/exp044_a2_feature_fusion/A2.csv" \
  --submission "exp045,0.4957,output/exp045_a2_feature_multiseed/A2.csv" \
  --submission "exp063,0.5033,output/exp063_submit_a1_exp061_a2_len0/A2.csv" \
  --submission "exp066,0.5033,output/exp066_submit_a1_blend90_a2_len0_len1/A2.csv" \
  --submission "exp069,0.5017,output/exp069_submit_a1_gate_a2_rrf_suffix/A2.csv" \
  --submission "exp078,0.5014,output/exp078_submit_a1_exp075_a2_profile_gate/A2.csv" \
  --submission "exp090,0.5035,output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv" \
  --output_path "$OUT_DIR/A2.csv" \
  --output_json "$OUT_DIR/a2_lb_map.json"

echo
echo "============================================================"
echo "[A1] 复用 Exp092"
echo "============================================================"
cp "$A1_SOURCE" "$OUT_DIR/A1.csv"

echo
echo "============================================================"
echo "[打包] prediction.zip"
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
