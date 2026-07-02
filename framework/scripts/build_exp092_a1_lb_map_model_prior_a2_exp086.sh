#!/usr/bin/env bash
set -euo pipefail

# Exp-092：A1 leaderboard MAP + 模型概率强先验 + 类别分布约束
#
# Exp091 只用提交投票先验，虽然满足历史分数约束，但类别分布坍缩。
# Exp092 加入两层防护：
# - 使用当前最强 Exp075/078 meta-stack 概率作为强先验；
# - 限制最终类别数量不能偏离 Exp078 太多。
#
# 这是一次真正的大幅 A1 尝试，但仍然保留可解释约束。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp092_submit_a1_lb_map_model_prior_a2_exp086}"
PRIOR_MODE="${PRIOR_MODE:-power}"
COUNT_TOLERANCE="${COUNT_TOLERANCE:-0}"
MODEL_PRIOR_WEIGHT="${MODEL_PRIOR_WEIGHT:-40.0}"
CLASS_COUNT_SLACK="${CLASS_COUNT_SLACK:-30}"
ANCHOR_SUBMISSION="${ANCHOR_SUBMISSION:-exp078}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.2}"
CLASS_PRIOR_WEIGHT="${CLASS_PRIOR_WEIGHT:-0.02}"
A2_SOURCE="${A2_SOURCE:-output/exp086_submit_a1_best_a2_strict_gate_midlong_keep_top1/A2.csv}"

EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP073_DIR="${EXP073_DIR:-output/exp073_a1_fulltrain_all_h2}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"
BASE_AUDIT_JSON="${BASE_AUDIT_JSON:-output/exp075_a1_meta_stack_threshold_audit/summary.json}"
SCORES_NPY="$OUT_DIR/a1_meta_scores.npy"

if [[ ! -f "$A2_SOURCE" ]]; then
  echo "A2_SOURCE不存在: $A2_SOURCE" >&2
  echo "请先运行 scripts/build_exp086_a1_best_a2_strict_gate_midlong_keep_top1.sh" >&2
  exit 1
fi
if [[ ! -f "$BASE_AUDIT_JSON" ]]; then
  echo "BASE_AUDIT_JSON不存在: $BASE_AUDIT_JSON" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo
echo "============================================================"
echo "[A1] 导出 Exp075/078 meta-stack 概率先验"
echo "============================================================"

python3 code/a1_meta_bias_search.py infer \
  --data_path data/cls_data/A1.npz \
  --audit_json "$BASE_AUDIT_JSON" \
  --device "$DEVICE" \
  --split_seeds 42,777,2024,2026,3407 \
  --val_ratio 0.1 \
  --stratified_split \
  --sources \
    "undir=$EXP061_DIR/undir_seed*_e*/best_model.pt" \
    "undir_reverse=$EXP061_DIR/undir_reverse_seed*_e*/best_model.pt" \
    "all_h2=$EXP073_DIR/all_h2_seed*_e*/best_model.pt" \
  --oof_sources \
    "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw" \
    "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw" \
    "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw" \
  --output_path "$OUT_DIR/A1_meta_identity.csv" \
  --output_json "$OUT_DIR/a1_meta_identity.json" \
  --output_scores_npy "$SCORES_NPY"

echo
echo "============================================================"
echo "[A1] leaderboard constrained MAP with model prior"
echo "============================================================"
echo "model_prior_weight=$MODEL_PRIOR_WEIGHT"
echo "class_count_slack=$CLASS_COUNT_SLACK"

python3 code/a1_leaderboard_map.py \
  --data_path data/cls_data/A1.npz \
  --prior_mode "$PRIOR_MODE" \
  --count_tolerance "$COUNT_TOLERANCE" \
  --prior_scores_npy "$SCORES_NPY" \
  --model_prior_weight "$MODEL_PRIOR_WEIGHT" \
  --class_count_reference output/exp078_submit_a1_exp075_a2_profile_gate/A1.csv \
  --class_count_slack "$CLASS_COUNT_SLACK" \
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
