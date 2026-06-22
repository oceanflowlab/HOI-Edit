#!/usr/bin/env bash
# =============================================================================
# CR v7 统一评测入口：Gemini 2.5 Pro QA + HOI Check
#
# 用法:
#   cp env/local.conf.example env/local.conf   # 配置 Python 路径与 GEMINI_API_KEY
#   export GEMINI_API_KEY='your-key'           # 或在 local.conf 中设置
#   MODELS=your_model_name bash run_qa_hoi.sh
#
# 常用:
#   MODELS=your_model_name bash run_qa_hoi.sh
#   SPLITS=L3 bash run_qa_hoi.sh
#   SKIP_QA=1 bash run_qa_hoi.sh               # 只跑 HOI
#   SKIP_HOI=1 bash run_qa_hoi.sh               # 只跑 QA
#   HOI_CHECK_ONLY=1 SKIP_QA=1 bash run_qa_hoi.sh  # 已有 DINO/SAM2，只补 HOI
#   DRY_RUN=1 bash run_qa_hoi.sh                # 打印命令不执行
# =============================================================================

set -euo pipefail

_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_WS}/env/local.conf" ]] && source "${_WS}/env/local.conf"
source "${_WS}/env/workspace.conf"

# ---------- 可调参数 ----------
SPLITS="${SPLITS:-L1L2,L3}"
MODELS="${MODELS:-}"
SKIP_QA="${SKIP_QA:-0}"
SKIP_HOI="${SKIP_HOI:-0}"
HOI_CHECK_ONLY="${HOI_CHECK_ONLY:-0}"
SKIP_DINO="${SKIP_DINO:-0}"
SKIP_SAM_TRACK="${SKIP_SAM_TRACK:-0}"
SKIP_RESIZE="${SKIP_RESIZE:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-1}"
SKIP_DINO_SETUP="${SKIP_DINO_SETUP:-1}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_QA="${MAX_PARALLEL_QA:-2}"

QA_SCRIPT="${EVAL_DIR}/run_qa_gemini_question_v6.sh"
HOI_SCRIPT="${EVAL_DIR}/run_full_eval_v7_google.sh"
SETUP_DINO="${SETUP_DINO:-${EVAL_WORKSPACE}/env/setup_dino_env.sh}"

resolve_frames_root() {
  local name="$1"
  if [[ -n "${FRAMES_DIR:-}" ]]; then
    echo "${FRAMES_DIR}"
  else
    echo "${DATA_V7_CR}/${name}_frames"
  fi
}

declare -A JSON_PATH=(
  [L1L2]="${DATA_V7_CR}/collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json"
  [L3]="${DATA_V7_CR}/collected_annotations_bboxes_v7_L3_questions_scoring_final.json"
)
declare -A ORIG_DIR=(
  [L1L2]="${DATA_V7_CR}/data_v7_L12"
  [L3]="${DATA_V7_CR}/data_v7_L3"
)

LOG_DIR="${EVAL_DIR}/logs/run_qa_hoi_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/run.log"
mkdir -p "${LOG_DIR}" "${EVAL_RUNS_DIR}"

log() { echo "$@" | tee -a "${MAIN_LOG}"; }

setup_dino_env() {
  [[ "${SKIP_DINO_SETUP}" == "1" || "${SKIP_HOI}" == "1" ]] && return 0
  [[ "${HOI_CHECK_ONLY}" == "1" && "${SKIP_DINO}" == "1" && "${SKIP_SAM_TRACK}" == "1" ]] && return 0
  log ">>> 初始化 DINO 环境"
  [[ -f "${SETUP_DINO}" ]] && bash "${SETUP_DINO}" >> "${MAIN_LOG}" 2>&1 || true
  [[ -f "${EVAL_WORKSPACE}/env/activate_eval.sh" ]] && source "${EVAL_WORKSPACE}/env/activate_eval.sh"
  export PYTHONPATH="${GROUNDING_DINO_ROOT}:${PYTHONPATH:-}"
}

run_hoi_split() {
  local model_name="$1" split="$2"
  local image_dir="$(resolve_frames_root "${model_name}")/${split}"
  local resized_dir="${EVAL_WORKSPACE}/final_eval_data_edited_${model_name}_${split}_resized"
  [[ "${HOI_CHECK_ONLY}" == "1" && -d "${resized_dir}" ]] && image_dir="${resized_dir}"

  export EVAL_V7_SPLIT_TAG="${split}"
  export EVAL_V7_IMAGE_DIR="${image_dir}"
  export EVAL_V7_JSON="${JSON_PATH[$split]}"
  export EVAL_V7_ORIG_DIR="${ORIG_DIR[$split]}"

  local args=(--model "${model_name}" --datasets V7 --gpu-id "${GPU_ID}" --output-dir "${EVAL_RUNS_DIR}")
  [[ "${SKIP_CONVERT}" == "1" ]] && args+=(--skip-convert)
  [[ "${SKIP_RESIZE}" == "1" ]] && args+=(--skip-resize)
  [[ "${SKIP_DINO}" == "1" ]] && args+=(--skip-dino)
  [[ "${SKIP_SAM_TRACK}" == "1" ]] && args+=(--skip-sam-track)
  [[ "${HOI_CHECK_ONLY}" == "1" ]] && args+=(--hoi-check-only --skip-convert --skip-resize --skip-dino --skip-sam-track)

  log ""
  log "=== HOI ${model_name} / ${split} ==="
  log "  JSON:  ${JSON_PATH[$split]}"
  log "  编辑图: ${image_dir}"
  log "  原图:  ${ORIG_DIR[$split]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN: bash ${HOI_SCRIPT} ${args[*]}"
    return 0
  fi

  env GEMINI_API_KEY="${GEMINI_API_KEY:-}" DINO_ENV_PY="${DINO_ENV_PY:-}" SS_ENV_PY="${SS_ENV_PY:-}" \
    GOOGLE_ENV_PY="${GOOGLE_ENV_PY:-}" \
    bash "${HOI_SCRIPT}" "${args[@]}" 2>&1 | tee -a "${LOG_DIR}/hoi_${model_name}_${split}.log" | tee -a "${MAIN_LOG}"
}

log "=========================================="
log "CR v7 QA + HOI (Gemini 2.5 Pro)"
log "  WORKSPACE=${EVAL_WORKSPACE}"
log "  SPLITS=${SPLITS}  MODELS=${MODELS}"
log "  SKIP_QA=${SKIP_QA}  SKIP_HOI=${SKIP_HOI}"
log "  OUTPUT=${EVAL_RUNS_DIR}"
log "=========================================="

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  log "❌ GEMINI_API_KEY 未设置。请 export 或在 env/local.conf 中配置"
  exit 1
fi

if [[ -z "${MODELS// /}" ]]; then
  log "❌ MODELS 未设置。请指定模型名，例如: MODELS=your_model_name bash run_qa_hoi.sh"
  log "   编辑图默认目录: data/<your_model_name>_frames/{L1L2,L3}/"
  log "   或设置 FRAMES_DIR=/path/to/frames 覆盖编辑图根目录"
  exit 1
fi

# ---------- Phase 1: QA ----------
if [[ "${SKIP_QA}" == "0" ]]; then
  log ""
  log "========== Phase 1: Gemini QA (question_v6) =========="
  if [[ "${DRY_RUN}" == "1" ]]; then
    DRY_RUN=1 MODELS="${MODELS}" MODELS_OVERRIDE="${MODELS}" MAX_PARALLEL="${MAX_PARALLEL_QA}" \
      SKIP_L1L2=$([[ "${SPLITS}" == *L1L2* ]] && echo 0 || echo 1) \
      SKIP_L3=$([[ "${SPLITS}" == *L3* ]] && echo 0 || echo 1) \
      bash "${QA_SCRIPT}" | tee -a "${MAIN_LOG}"
  else
    MODELS="${MODELS}" MODELS_OVERRIDE="${MODELS}" MAX_PARALLEL="${MAX_PARALLEL_QA}" \
      SKIP_L1L2=$([[ "${SPLITS}" == *L1L2* ]] && echo 0 || echo 1) \
      SKIP_L3=$([[ "${SPLITS}" == *L3* ]] && echo 0 || echo 1) \
      bash "${QA_SCRIPT}" 2>&1 | tee -a "${MAIN_LOG}"
  fi
else
  log ">>> 跳过 QA (SKIP_QA=1)"
fi

# ---------- Phase 2: HOI ----------
if [[ "${SKIP_HOI}" == "0" ]]; then
  setup_dino_env
  log ""
  log "========== Phase 2: Gemini HOI Check =========="
  IFS=',' read -ra MA <<< "${MODELS}"
  IFS=',' read -ra SA <<< "${SPLITS}"
  fail=0
  for m in "${MA[@]}"; do
    m="${m// /}"
    [[ -d "$(resolve_frames_root "${m}")" ]] || { log "⚠️  未知模型 ${m}，将尝试 $(resolve_frames_root "${m}")"; }
    for s in "${SA[@]}"; do
      s="${s// /}"
      run_hoi_split "${m}" "${s}" || fail=1
    done
  done
  [[ ${fail} -eq 0 ]] || exit 1
else
  log ">>> 跳过 HOI (SKIP_HOI=1)"
fi

log ""
log "✅ 完成。输出目录: ${EVAL_RUNS_DIR}"
log "   QA:  qa_results_v6_{L1L2,L3}_<model>.json"
log "   HOI: <model>_{L1L2,L3}_full/.../results_*_google_full.json"
