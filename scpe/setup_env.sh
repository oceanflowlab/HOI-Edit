#!/usr/bin/env bash
# 在本目录创建独立 venv 并安装依赖
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  echo "✅ venv 已存在: ${ROOT}/.venv"
else
  python3 -m venv "${ROOT}/.venv"
  echo "✅ 已创建 venv: ${ROOT}/.venv"
fi

"${ROOT}/.venv/bin/pip" install -U pip
"${ROOT}/.venv/bin/pip" install -r "${ROOT}/requirements.txt"

echo ""
echo "用法:"
echo "  source ${ROOT}/.venv/bin/activate"
echo "  cp env.example env.local && 编辑 env.local 填入 API Key"
echo "  source env.local && ./run_minimal.sh all"
