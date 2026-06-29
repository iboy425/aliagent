#!/usr/bin/env bash
set -euo pipefail

# Exp-067：A1更偏全标签模型 + A2扩展到 len2-3 的进攻候选
#
# 这个候选比 Exp-066 更激进：
# - A1：仍使用 0.90 全标签模型融合；
# - A2：从 len=0,len=1 扩展到 len=2-3。
#
# 使用建议：
# - 先看 Exp-065/066 的 A2 线上反馈；
# - 如果 len=1 没有伤分，再考虑提交本候选。

cd "$(dirname "$0")/.."

FULLTRAIN_WEIGHT="${FULLTRAIN_WEIGHT:-0.90}" \
A2_BUCKETS="${A2_BUCKETS:-len=0,len=1,len=2-3}" \
OUT_DIR="${OUT_DIR:-output/exp067_submit_a1_blend90_a2_len0_len1_len2_3}" \
./scripts/build_joint_a1_sign_a2_bucket_candidate.sh
