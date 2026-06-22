#!/usr/bin/env bash
# 便携版 Dino 环境安装（在新机器上运行一次）
# 依赖: conda, CUDA, GROUNDING_DINO_ROOT 已 clone
set -euo pipefail

_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=workspace.conf
source "${_EVAL_DIR}/workspace.conf"

ENV_PREFIX="${DINO_ENV_PREFIX:-${EVAL_WORKSPACE}/envs/Dino}"
GROUNDING_DINO="${GROUNDING_DINO_ROOT:?set GROUNDING_DINO_ROOT in env/local.conf}"
REQ="${EVAL_WORKSPACE}/env/requirements-dino.txt"

echo ">>> Creating Dino env at ${ENV_PREFIX}"
conda create -y -p "${ENV_PREFIX}" python=3.10 pip

"${ENV_PREFIX}/bin/pip" install -r "${REQ}"
"${ENV_PREFIX}/bin/pip" install -r "${GROUNDING_DINO}/requirements.txt"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
(cd "${GROUNDING_DINO}" && "${ENV_PREFIX}/bin/pip" install -e .) || {
  echo "⚠️  pip install -e failed; use PYTHONPATH=${GROUNDING_DINO}"
}

echo ""
echo "Add to env/local.conf:"
echo "  export DINO_ENV_PY=\"${ENV_PREFIX}/bin/python\""
echo "  export GROUNDING_DINO_ROOT=\"${GROUNDING_DINO}\""
