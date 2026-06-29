#!/usr/bin/env bash
set -euo pipefail

# Exp-066：A1更偏全标签模型 + A2 len0+len1 联合候选
#
# 相比 Exp-065：
# - A1：全标签模型权重从 0.85 提到 0.90，减少分折模型对最终分布的牵制；
# - A2：继续使用 len=0,len=1，承接 Exp-063 中 len=0 已验证有效的方向。

cd "$(dirname "$0")/.."

FULLTRAIN_WEIGHT="${FULLTRAIN_WEIGHT:-0.90}" \
A2_BUCKETS="${A2_BUCKETS:-len=0,len=1}" \
OUT_DIR="${OUT_DIR:-output/exp066_submit_a1_blend90_a2_len0_len1}" \
./scripts/build_joint_a1_sign_a2_bucket_candidate.sh
