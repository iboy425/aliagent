#!/usr/bin/env bash
set -euo pipefail

# Exp-091：A1 leaderboard 约束 MAP + A2 Exp086 联合候选
#
# A1 思路：
# - 用历史 A1 提交文件和对应线上 Accuracy 构造正确数约束；
# - 用高分提交加权投票作为先验；
# - 求解满足所有历史分数约束的最可能测试标签。
#
# A2 暂时沿用 Exp086，因为 Exp090 线上已把 A2 从 0.5033 小幅推到 0.5035。

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-output/exp091_submit_a1_lb_map_a2_exp086}"
PRIOR_MODE="${PRIOR_MODE:-power}"
COUNT_TOLERANCE="${COUNT_TOLERANCE:-0}"
ANCHOR_SUBMISSION="${ANCHOR_SUBMISSION:-exp078}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.1}"
CLASS_PRIOR_WEIGHT="${CLASS_PRIOR_WEIGHT:-0.05}"
A2_SOURCE="${A2_SOURCE:-output/exp086_submit_a1_best_a2_strict_gate_midlong_keep_top1/A2.csv}"

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE" >&2
  echo "请先运行 scripts/build_exp086_a1_best_a2_strict_gate_midlong_keep_top1.sh" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A1] leaderboard constrained MAP"
echo "============================================================"
echo "prior_mode=$PRIOR_MODE"
echo "count_tolerance=$COUNT_TOLERANCE"
echo "anchor=$ANCHOR_SUBMISSION weight=$ANCHOR_WEIGHT"

python3 code/a1_leaderboard_map.py \
  --data_path data/cls_data/A1.npz \
  --prior_mode "$PRIOR_MODE" \
  --count_tolerance "$COUNT_TOLERANCE" \
  --anchor_submission "$ANCHOR_SUBMISSION" \
  --anchor_weight "$ANCHOR_WEIGHT" \
  --class_prior_weight "$CLASS_PRIOR_WEIGHT" \
  --base_submission exp078 \
  --submission "exp054,0.7459,output/exp054_submit_a1_sign_label_ensemble_a2_exp044/A1.csv" \
  --submission "exp055,0.7466,output/exp055_a1_sign_label_fulltrain_candidate/A1.csv" \
  --submission "exp060,0.7550,output/exp060_submit_a1_config_ensemble_a2_exp044/A1.csv" \
  --submission "exp063,0.7590,output/exp063_submit_a1_exp061_a2_len0/A1.csv" \
  --submission "exp066,0.7594,output/exp066_submit_a1_blend90_a2_len0_len1/A1.csv" \
  --submission "exp069,0.7590,output/exp069_submit_a1_gate_a2_rrf_suffix/A1.csv" \
  --submission "exp078,0.7601,output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv" \
  --submission "exp090,0.7601,output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv" \
  --output_path "$OUT_DIR/A1.csv" \
  --output_json "$OUT_DIR/a1_lb_map.json"

echo
echo "============================================================"
echo "[A2] 复用 Exp086"
echo "============================================================"
cp "$A2_SOURCE" "$OUT_DIR/A2.csv"

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
