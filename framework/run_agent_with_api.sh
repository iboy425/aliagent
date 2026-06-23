#!/usr/bin/env bash
# 使用真实LLM API运行一轮Agent闭环。
# 注意：不要把API Key写入配置文件或提交到版本控制，只通过环境变量传入。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.local"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${LLM_API_KEY:-}" && -n "${DASHSCOPE_API_KEY:-}" ]]; then
  export LLM_API_KEY="${DASHSCOPE_API_KEY}"
fi

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "错误：请先设置 LLM_API_KEY 或 DASHSCOPE_API_KEY 环境变量。"
  echo '示例：export DASHSCOPE_API_KEY="你的API Key"'
  exit 1
fi

if [[ -z "${LLM_BASE_URL:-}" ]]; then
  echo "错误：请先设置 LLM_BASE_URL 环境变量。"
  echo '示例：export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"'
  exit 1
fi

MODEL="${LLM_MODEL:-qwen-plus}"
OUT_DIR="${AGENT_OUTPUT_DIR:-output/repro_agent_api_round1}"
BUDGET="${AGENT_BUDGET:-1}"

echo "LLM配置：model=${MODEL}, base_url=${LLM_BASE_URL}"

python3 main.py \
  --config repro_agent_step6.yaml \
  --task 1 \
  --task 2 \
  --budget "${BUDGET}" \
  --max_iterations "${BUDGET}" \
  --time_limit 1800 \
  --output_dir "${OUT_DIR}" \
  --device cpu \
  --llm_model "${MODEL}"
