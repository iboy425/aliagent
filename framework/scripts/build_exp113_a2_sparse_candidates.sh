#!/usr/bin/env bash
set -euo pipefail

# Exp-113：A2短历史稀疏ranker候选组装
#
# 原理：
# - Exp112 的 ComplementNB 稀疏模型在 test-like 多split验证中达到
#   mean weighted NDCG≈0.520，高于此前稳定A2的离线量级；
# - 但线上 A2 对 Top1 漂移敏感，所以这里同时生成纯替换和保护Top1
#   的融合版本，提交前优先选择保护版本。

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-output/exp113_a2_sparse_short_ranker_candidates}"
AUDIT_JSON="${AUDIT_JSON:-output/exp112_a2_sparse_short_ranker_audit/results.json}"
STABLE_A1="${STABLE_A1:-output/exp090_submit_a1_safe_bias_a2_exp086/A1.csv}"
STABLE_A2="${STABLE_A2:-output/exp090_submit_a1_safe_bias_a2_exp086/A2.csv}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$AUDIT_JSON" ]]; then
  echo "缺少Exp112审计结果: $AUDIT_JSON" >&2
  echo "请先运行: python3 code/a2_sparse_short_ranker.py audit ..." >&2
  exit 1
fi
if [[ ! -f "$STABLE_A1" ]]; then
  echo "缺少稳定A1: $STABLE_A1" >&2
  exit 1
fi
if [[ ! -f "$STABLE_A2" ]]; then
  echo "缺少稳定A2: $STABLE_A2" >&2
  exit 1
fi

echo "============================================================"
echo "[A2] 训练全量短历史稀疏ranker"
echo "============================================================"
python3 code/a2_sparse_short_ranker.py predict \
  --data_path data/rec_data \
  --seq_col item_seq_raw \
  --audit_json "$AUDIT_JSON" \
  --include_original_train \
  --output_path "$OUT_DIR/A2_sparse.csv"

echo
echo "============================================================"
echo "[A2] 保护Top1融合候选"
echo "============================================================"
python3 code/a2_sparse_short_ranker.py blend \
  --data_path data/rec_data \
  --seq_col item_seq_raw \
  --base_a2 "$STABLE_A2" \
  --alt_a2 "$OUT_DIR/A2_sparse.csv" \
  --keep_topn 1 \
  --buckets "len=0,len=1,len=2-3,len=4-10,len>10" \
  --output_path "$OUT_DIR/A2_sparse_keep1_all.csv"

python3 code/a2_sparse_short_ranker.py blend \
  --data_path data/rec_data \
  --seq_col item_seq_raw \
  --base_a2 "$STABLE_A2" \
  --alt_a2 "$OUT_DIR/A2_sparse.csv" \
  --keep_topn 1 \
  --buckets "len=0,len=1,len=2-3" \
  --output_path "$OUT_DIR/A2_sparse_keep1_short.csv"

python3 code/a2_sparse_short_ranker.py blend \
  --data_path data/rec_data \
  --seq_col item_seq_raw \
  --base_a2 "$STABLE_A2" \
  --alt_a2 "$OUT_DIR/A2_sparse.csv" \
  --keep_topn 2 \
  --buckets "len=0,len=1,len=2-3,len=4-10,len>10" \
  --output_path "$OUT_DIR/A2_sparse_keep2_all.csv"

make_pkg() {
  local name="$1"
  local a2="$2"
  local out="$OUT_DIR/$name"
  mkdir -p "$out"
  cp "$STABLE_A1" "$out/A1.csv"
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

echo
echo "============================================================"
echo "[打包] 候选提交包"
echo "============================================================"
make_pkg "stable_a1_a2_sparse_pure" "$OUT_DIR/A2_sparse.csv"
make_pkg "stable_a1_a2_sparse_keep1_all" "$OUT_DIR/A2_sparse_keep1_all.csv"
make_pkg "stable_a1_a2_sparse_keep1_short" "$OUT_DIR/A2_sparse_keep1_short.csv"
make_pkg "stable_a1_a2_sparse_keep2_all" "$OUT_DIR/A2_sparse_keep2_all.csv"

echo
echo "============================================================"
echo "[差异] 相对稳定A2"
echo "============================================================"
for name in A2_sparse A2_sparse_keep1_all A2_sparse_keep1_short A2_sparse_keep2_all; do
  echo
  echo "---- $name ----"
  python3 code/a2_compare_submissions.py \
    --base_a2 "$STABLE_A2" \
    --new_a2 "$OUT_DIR/${name}.csv" \
    --test_csv data/rec_data/test.csv \
    --seq_col item_seq_raw \
    --topk 10 \
    --topn 10
done

echo
echo "Exp-113候选目录：$OUT_DIR"
