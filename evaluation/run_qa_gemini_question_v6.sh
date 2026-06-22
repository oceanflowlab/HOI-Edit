#!/usr/bin/env bash
# Gemini 2.5 Pro VQA on question_v6 (scoring_final JSONs)
set -euo pipefail

_WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${_WS_ROOT}/env/workspace.conf"
# shellcheck source=/dev/null
source "${EVAL_DIR}/cr_proxy_defaults.sh"
apply_cr_proxy_if_unset

QUESTION_FIELD="${QUESTION_FIELD:-question_v6}"
GOOGLE_ENV_PY="${GOOGLE_ENV_PY:-python3}"
QA_SCRIPT="${EVAL_DIR}/run_question_answering.py"
EVAL_RUNS_DIR="${EVAL_RUNS_DIR:-${EVAL_WORKSPACE}/eval_runs}"

L1L2_JSON="${L1L2_JSON:-${DATA_V7_CR}/collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json}"
L3_JSON="${L3_JSON:-${DATA_V7_CR}/collected_annotations_bboxes_v7_L3_questions_scoring_final.json}"
ORIG_L1L2="${DATA_V7_CR}/data_v7_L12"
ORIG_L3="${DATA_V7_CR}/data_v7_L3"

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_OVERRIDE}"
elif [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS}"
else
  MODELS=()
fi

SKIP_L1L2="${SKIP_L1L2:-0}"
SKIP_L3="${SKIP_L3:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
INCLUDE_ORIGINAL="${INCLUDE_ORIGINAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

frames_root_for() {
  local name="$1"
  if [[ -n "${FRAMES_DIR:-}" ]]; then
    echo "${FRAMES_DIR}"
  else
    echo "${DATA_V7_CR}/${name}_frames"
  fi
}

mkdir -p "${EVAL_RUNS_DIR}"
LOG_DIR="${EVAL_DIR}/logs/qa_gemini_v6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/qa.log"

echo "=== Gemini 2.5 Pro QA (question_v6) ==="
echo "EVAL_WORKSPACE=${EVAL_WORKSPACE}"
echo "Models: ${MODELS[*]:-(none)}"
echo "Output: ${EVAL_RUNS_DIR}/qa_results_v6_{L1L2,L3}_<name>.json"
echo ""

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "❌ GEMINI_API_KEY 未设置。请 export GEMINI_API_KEY 或在 env/local.conf 中配置"
  exit 1
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "❌ MODELS 未设置"
  exit 1
fi

run_one() {
  local model_name="$1" split="$2" json_path="$3" orig_dir="$4" edit_dir="$5" out_path="$6" log_file="$7"
  [[ -f "${json_path}" ]] || { echo "❌ JSON missing: ${json_path}"; return 1; }
  [[ -d "${edit_dir}" ]] || { echo "⚠️  skip ${model_name}/${split}: no ${edit_dir}"; return 0; }

  local extra=()
  [[ "${INCLUDE_ORIGINAL}" == "1" ]] && extra+=(--include-original)

  echo ">>> QA ${model_name} / ${split}" | tee -a "${MAIN_LOG}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: ${GOOGLE_ENV_PY} ${QA_SCRIPT} --json ${json_path} ..." | tee -a "${MAIN_LOG}"
    return 0
  fi

  set +e
  env GEMINI_API_KEY="${GEMINI_API_KEY}" "${GOOGLE_ENV_PY}" -u "${QA_SCRIPT}" \
    --json "${json_path}" \
    --image-dir "${orig_dir}" \
    --output "${out_path}" \
    --dataset-type "${split}" \
    --models "${model_name}" \
    --model-edited-dir "${model_name}=${edit_dir}" \
    --question-field "${QUESTION_FIELD}" \
    "${extra[@]}" \
    2>&1 | tee -a "${log_file}" | tee -a "${MAIN_LOG}"
  local code=${PIPESTATUS[0]}
  set -e
  [[ ${code} -eq 0 ]] || return "${code}"
}

run_one_bg() { run_one "$@" & }

wait_jobs() {
  local fail=0
  for pid in $(jobs -rp); do
    set +e; wait "${pid}"; code=$?; set -e
    [[ ${code} -ne 0 ]] && fail=1
  done
  return "${fail}"
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do sleep 1; done
}

FAIL=0
for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_NAME="${MODEL_NAME// /}"
  [[ -z "${MODEL_NAME}" ]] && continue
  FRAMES_ROOT="$(frames_root_for "${MODEL_NAME}")"

  if [[ ${SKIP_L1L2} -eq 0 ]]; then
    wait_for_slot
    run_one_bg "${MODEL_NAME}" L1L2 "${L1L2_JSON}" "${ORIG_L1L2}" \
      "${FRAMES_ROOT}/L1L2" \
      "${EVAL_RUNS_DIR}/qa_results_v6_L1L2_${MODEL_NAME}.json" \
      "${LOG_DIR}/qa_${MODEL_NAME}_L1L2.log"
  fi
  if [[ ${SKIP_L3} -eq 0 ]]; then
    wait_for_slot
    run_one_bg "${MODEL_NAME}" L3 "${L3_JSON}" "${ORIG_L3}" \
      "${FRAMES_ROOT}/L3" \
      "${EVAL_RUNS_DIR}/qa_results_v6_L3_${MODEL_NAME}.json" \
      "${LOG_DIR}/qa_${MODEL_NAME}_L3.log"
  fi
done

wait_jobs || FAIL=1
[[ ${FAIL} -eq 0 ]] || exit 1
echo "✅ Gemini QA done → ${EVAL_RUNS_DIR}"
