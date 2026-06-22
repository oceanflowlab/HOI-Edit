#!/usr/bin/env bash
# CR v7 完整流程：QA + HOI + 计分
set -euo pipefail

_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_WS}/env/local.conf" ]] && source "${_WS}/env/local.conf"
source "${_WS}/env/workspace.conf"

SCORES_ONLY="${SCORES_ONLY:-0}"
SKIP_SCORES="${SKIP_SCORES:-0}"

if [[ "${SCORES_ONLY}" != "1" ]]; then
  bash "${_WS}/run_qa_hoi.sh" "$@"
fi

if [[ "${SKIP_SCORES}" != "1" ]]; then
  _score="${SCORE_MODEL:-${MODELS:-}}"
  _score="${_score%%,*}"
  _score="${_score// /}"
  if [[ -z "${_score}" ]]; then
    echo "❌ 计分需要 SCORE_MODEL 或 MODELS（与 QA/HOI 使用的名称相同）"
    exit 1
  fi
  echo "=== Phase 3: Scores (${_score}) ==="
  args=(--workspace "${EVAL_WORKSPACE}" --model "${_score}" --decimals "${DECIMALS:-4}")
  [[ -n "${LEGACY_SCORES_ROOT:-}" ]] && args+=(--legacy-root "${LEGACY_SCORES_ROOT}")
  "${GOOGLE_ENV_PY:-python3}" "${EVAL_DIR}/compute_scoring_final_scores.py" "${args[@]}"
fi
