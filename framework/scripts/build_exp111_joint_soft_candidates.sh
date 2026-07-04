#!/usr/bin/env bash
set -euo pipefail

# Exp-111：A1/A2 联合候选组装
#
# 生成两个候选：
# 1. stable_a1_a2_len23_soft：
#    - A1 使用线上已验证最强 Exp090；
#    - A2 使用 Exp109 的 len=2-3 专家软融合；
#    - 风险最低，主要押 A2 微调。
# 2. classwise_a1_a2_len23_soft：
#    - A1 使用 Exp110 类别级专家选择；
#    - A2 同上；
#    - A1 有 OOF 小幅正收益，但未线上验证，风险更高。

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-output/exp111_joint_soft_candidates}"
STABLE_A1="${STABLE_A1:-output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv}"
CLASSWISE_A1="${CLASSWISE_A1:-output/exp110_a1_classwise_stack_audit/A1.csv}"
A2_LEN23="${A2_LEN23:-output/exp109_a2_bucket_expert_soft/A2_len23_only.csv}"
A2_BASE="${A2_BASE:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"

mkdir -p "$OUT_DIR"

make_pkg() {
  local name="$1"
  local a1="$2"
  local a2="$3"
  local out="$OUT_DIR/$name"
  mkdir -p "$out"
  if [[ ! -f "$a1" ]]; then
    echo "A1不存在: $a1" >&2
    exit 1
  fi
  if [[ ! -f "$a2" ]]; then
    echo "A2不存在: $a2" >&2
    exit 1
  fi
  cp "$a1" "$out/A1.csv"
  cp "$a2" "$out/A2.csv"
  (
    cd "$out"
    rm -f prediction.zip
    zip -q prediction.zip A1.csv A2.csv
  )
  python3 code/validate_submission.py \
    --zip_path "$out/prediction.zip" \
    --cls_data_path data/cls_data/A1.npz \
    --rec_data_dir data/rec_data \
    --topk 10
  echo "候选包：$out/prediction.zip"
}

make_pkg "stable_a1_a2_len23_soft" "$STABLE_A1" "$A2_LEN23"
make_pkg "classwise_a1_a2_len23_soft" "$CLASSWISE_A1" "$A2_LEN23"
make_pkg "classwise_a1_a2_base" "$CLASSWISE_A1" "$A2_BASE"

echo
echo "[A1差异] Exp110 classwise vs Exp090 stable"
python3 - <<'PY'
import pandas as pd
from collections import Counter

base = pd.read_csv("output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv")
new = pd.read_csv("output/exp110_a1_classwise_stack_audit/A1.csv")
changed = (base["label"].to_numpy() != new["label"].to_numpy())
print(f"changed={changed.mean():.4%}, count={changed.sum()}")
print("stable", dict(Counter(base["label"].astype(int))))
print("classwise", dict(Counter(new["label"].astype(int))))
PY

echo
echo "[A2差异] Exp109 len23_only vs Exp090"
python3 code/a2_compare_submissions.py \
  --base_a2 "$A2_BASE" \
  --new_a2 "$A2_LEN23" \
  --test_csv data/rec_data/test.csv \
  --seq_col item_seq_raw \
  --topk 10 \
  --topn 10
