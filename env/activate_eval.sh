#!/usr/bin/env bash
# 进入评测环境: source env/activate_eval.sh
_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=workspace.conf
source "${_EVAL_DIR}/workspace.conf"

echo "EVAL_WORKSPACE=${EVAL_WORKSPACE}"
echo "DATA_V7_CR=${DATA_V7_CR}"
echo "DINO_ENV_PY=${DINO_ENV_PY}"
echo "SS_ENV_PY=${SS_ENV_PY}"
echo "GOOGLE_ENV_PY=${GOOGLE_ENV_PY}"
if [[ -z "${GROUNDING_DINO_ROOT}" || ! -d "${GROUNDING_DINO_ROOT}" ]]; then
  echo "⚠️  GROUNDING_DINO_ROOT 未配置，请编辑 env/local.conf"
fi
