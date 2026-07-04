#!/usr/bin/env bash
set -euo pipefail

# Exp-110：A1 按预测类别选择 SIGN 专家组合
#
# 现有 A1 最强路线是：
# - 多个 SIGN/标签传播配置产生概率；
# - OOF meta-stack / bias 做全局融合。
#
# 但不同类别的最优图传播方向可能不同：
# - 类别4样本多，可能更相信稳定主模型；
# - 小类别可能更依赖某个方向的标签传播；
# - 全局平均会把这些差异抹平。
#
# 本实验使用 a1_sign_classwise_stack.py 做 leave-one-split-out 审计：
# 用 4 个 split 学“预测为某类时该用哪个专家组合”，在剩下 1 个 split 验证。

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUT_DIR="${OUT_DIR:-output/exp110_a1_classwise_stack_audit}"
DATA_PATH="${DATA_PATH:-data/cls_data/A1.npz}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42,777,2024,2026,3407}"
A2_SOURCE="${A2_SOURCE:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"

EXP061_DIR="${EXP061_DIR:-output/exp061_a1_fulltrain_config_ensemble_candidate}"
EXP073_DIR="${EXP073_DIR:-output/exp073_a1_fulltrain_all_h2}"
EXP059_DIR="${EXP059_DIR:-output/exp059_a1_sign_directed_label_feature_audit}"

SOURCES=(
  "undir=$EXP059_DIR/standard_struct_label_undir_h3_rw"
  "undir_reverse=$EXP059_DIR/standard_struct_label_undir_reverse_h3_rw"
  "all_h2=$EXP059_DIR/standard_struct_label_all_h2_rw"
)

INFER_SOURCES=(
  "undir=$EXP061_DIR/undir_seed*_e*/best_model.pt"
  "undir_reverse=$EXP061_DIR/undir_reverse_seed*_e*/best_model.pt"
  "all_h2=$EXP073_DIR/all_h2_seed*_e*/best_model.pt"
)

CANDIDATES="${CANDIDATES:-base=undir:0.3,undir_reverse:0.7|reverse=undir_reverse:1.0|undir=undir:1.0|u7r3=undir:0.7,undir_reverse:0.3|u2r8=undir:0.2,undir_reverse:0.8|u3r6a1=undir:0.3,undir_reverse:0.6,all_h2:0.1|u3r5a2=undir:0.3,undir_reverse:0.5,all_h2:0.2|u3r4a3=undir:0.3,undir_reverse:0.4,all_h2:0.3|u2r6a2=undir:0.2,undir_reverse:0.6,all_h2:0.2|a1=all_h2:1.0}"

mkdir -p "$OUT_DIR"

run_audit() {
  local name="$1"
  local min_bucket_count="$2"
  local min_gain="$3"
  local out="$OUT_DIR/$name"
  mkdir -p "$out"

  if [[ -f "$out/summary.json" && "${RERUN:-0}" != "1" ]]; then
    echo "[跳过] $name 已存在"
    return
  fi

  echo
  echo "============================================================"
  echo "[Exp-110] $name min_bucket_count=$min_bucket_count min_gain=$min_gain"
  echo "============================================================"

  python3 code/a1_sign_classwise_stack.py audit \
    --data_path "$DATA_PATH" \
    --output_dir "$out" \
    --device "$DEVICE" \
    --split_seeds "$SPLIT_SEEDS" \
    --val_ratio 0.1 \
    --stratified_split \
    --sources "${SOURCES[@]}" \
    --candidates "$CANDIDATES" \
    --base_candidate base \
    --min_bucket_count "$min_bucket_count" \
    --min_gain "$min_gain"
}

run_audit "mb60_gain0" 60 0.0
run_audit "mb80_gain0005" 80 0.0005
run_audit "mb120_gain001" 120 0.001
run_audit "mb160_gain001" 160 0.001
run_audit "mb200_gain002" 200 0.002

echo
echo "============================================================"
echo "[汇总] Exp-110 A1 类别 stacking 审计"
echo "============================================================"

OUT_DIR_FOR_SUMMARY="$OUT_DIR" python3 - <<'PY'
import glob
import json
import os

rows = []
base = os.environ["OUT_DIR_FOR_SUMMARY"]
for path in sorted(glob.glob(os.path.join(base, "*", "summary.json"))):
    data = json.load(open(path, encoding="utf-8"))
    rows.append({
        "name": os.path.basename(os.path.dirname(path)),
        "base_mean": data["base_mean"],
        "loo_mean": data["loo_mean"],
        "loo_min": data["loo_min"],
        "gain": data["loo_mean"] - data["base_mean"],
        "path": path,
        "final_mapping": data["final_mapping"],
    })

rows.sort(key=lambda item: (item["loo_mean"], item["loo_min"], item["gain"]), reverse=True)
for row in rows:
    print(
        f"{row['name']}\tbase={row['base_mean']:.6f}\tloo={row['loo_mean']:.6f}\t"
        f"gain={row['gain']:+.6f}\tmin={row['loo_min']:.6f}\tpath={row['path']}"
    )
    print(f"  mapping={row['final_mapping']}")

with open(os.path.join(base, "summary_all.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\n汇总已保存: {os.path.join(base, 'summary_all.json')}")
PY

BEST_AUDIT="$(python3 - <<'PY'
import json
rows=json.load(open('output/exp110_a1_classwise_stack_audit/summary_all.json', encoding='utf-8'))
print(rows[0]['path'] if rows else '')
PY
)"

if [[ -n "$BEST_AUDIT" ]]; then
  echo
  echo "============================================================"
  echo "[推理] 使用最佳审计生成 A1 候选"
  echo "============================================================"
  echo "best_audit=$BEST_AUDIT"

  python3 code/a1_sign_classwise_stack.py infer \
    --data_path "$DATA_PATH" \
    --audit_json "$BEST_AUDIT" \
    --device "$DEVICE" \
    --sources "${INFER_SOURCES[@]}" \
    --candidates "$CANDIDATES" \
    --output_path "$OUT_DIR/A1.csv" \
    --output_json "$OUT_DIR/a1_infer.json"

  if [[ -f "$A2_SOURCE" ]]; then
    cp "$A2_SOURCE" "$OUT_DIR/A2.csv"
    (
      cd "$OUT_DIR"
      rm -f prediction.zip
      zip -q prediction.zip A1.csv A2.csv
    )
    python3 code/validate_submission.py \
      --zip_path "$OUT_DIR/prediction.zip" \
      --cls_data_path "$DATA_PATH" \
      --rec_data_dir data/rec_data \
      --topk 10
    echo "A1类别stack候选包：$OUT_DIR/prediction.zip"
  else
    echo "A2_SOURCE不存在，仅生成 A1.csv：$A2_SOURCE"
  fi
fi

echo
echo "Exp-110 完成。请把汇总和 a1_infer.json 摘要发回来。"
