#!/bin/bash

set -euo pipefail
# --- cr_eval_workspace portable ---
_WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${_WS_ROOT}/env/workspace.conf"


# ============================================
# 通用评测流程 - V7 L3 数据（Google 官方 Gemini API 版本）
# V7：EVAL_V7_IMAGE_DIR、EVAL_V7_JSON 必填；EVAL_V7_ORIG_DIR 由调用脚本设置
# 支持所有模型：qwen_plus, nanobanana, 等
# 支持数据集类型：L1L2, L3
# HOI Check：google.genai SDK + gemini-2.5-pro（gemini3_*_google_newsim.py）
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# paths from env/workspace.conf (EVAL_WORKSPACE, EVAL_DIR, SAM2_ROOT)
# --------------------------
# 代理配置（Google genai SDK 通常需要代理；未设置时沿用 v6 google 默认）
# --------------------------
# shellcheck source=cr_proxy_defaults.sh
source "${SCRIPT_DIR}/cr_proxy_defaults.sh"
apply_cr_proxy_if_unset
# CUDA_VISIBLE_DEVICES 将在参数解析后设置

# --------------------------
# 环境配置
# --------------------------
# 优先使用外部 export（如 run_cr_wan22_setup_and_eval.sh 配置的新环境）
# DINO_ENV_PY from env/workspace.conf
WAN_ENV_PY="${GOOGLE_ENV_PY}"
SS_ENV_PY="${SS_ENV_PY}"
# HOI Check 需要 google-genai（系统 python3 通常未安装）
GOOGLE_ENV_PY="${GOOGLE_ENV_PY:-${WAN_ENV_PY}}"

# --------------------------
# 脚本路径
# --------------------------
DINO_INFER_SCRIPT="${EVAL_DIR}/inference_on_multi_image_eval_optimized.py"
DINO_INFER_SCRIPT_OPTIMIZED="${EVAL_DIR}/inference_on_multi_image_eval_optimized.py"  # 优化版本：只加载一次模型
USE_OPTIMIZED_DINO=1  # 是否使用优化版本的DINO脚本（默认：1=使用，避免每次加载模型时下载tokenizer）
# SAM2_SCRIPT legacy removed
SAM2_TRACKING_SCRIPT="${SAM2_ROOT}/run_sam2_tracking_for_eval.py"
GOOGLE_SCRIPT_NEW="${EVAL_DIR}/gemini3_final_hoicheck_new_noquestion_track_google_newsim.py"
GOOGLE_SCRIPT_OLD="${EVAL_DIR}/gemini3_final_hoicheck_new_noquestion_track_google.py"
GOOGLE_SCRIPT_CF="${EVAL_DIR}/gemini3_final_hoicheck_new_noquestion_track_google_newsim_2jiehe_cf.py"
if [[ -n "${GOOGLE_HOI_SCRIPT:-}" ]]; then
  GOOGLE_SCRIPT="${GOOGLE_HOI_SCRIPT}"
elif [[ -f "${GOOGLE_SCRIPT_NEW}" ]]; then
  GOOGLE_SCRIPT="${GOOGLE_SCRIPT_NEW}"
else
  GOOGLE_SCRIPT="${GOOGLE_SCRIPT_OLD}"
fi
HOI_RESULT_TAG="${HOI_RESULT_TAG:-}"
CONVERT_SCRIPT="${EVAL_DIR}/convert_images_for_eval.py"
RESIZE_SCRIPT="${EVAL_DIR}/resize_edited_images_to_original.py"

# --------------------------
# 基础路径配置
# --------------------------
DATA_V6_DIR="${SHARED_DIR}/organized_hoi_dataset/data_v6"
DATA_V7_DIR="${SHARED_DIR}/data_v7"
# ORIG_IMAGE_DIR 将根据数据集类型动态设置

# --------------------------
# 参数解析
# --------------------------
MODEL_NAME=""
DATASET_TYPES=""  # 支持多个数据集类型，用逗号分隔，如 "L1L2,L3" 或 "L3,V6"
SKIP_DINO=0
SKIP_SAM_TRACK=0
SKIP_HOI_CHECK=0
SKIP_CONVERT=0
SKIP_RESIZE=0  # 如果为1，则跳过图像缩放步骤
RESIZE_INPLACE=0  # 如果为1，则在原图像路径下直接resize（覆盖原图像）
HOI_CHECK_ONLY=0  # 如果为1，则跳过所有前面的步骤，直接执行HOI Check
GPU_ID=0
RUN_BACKGROUND=0  # 如果为1，则在后台运行（使用nohup）
LOG_FILE=""  # 日志文件路径（后台运行时使用）
OUTPUT_DIR=""  # 指定输出目录（如果不指定，则使用时间戳）
CR_WAN22_QA2_L1L2=0  # data/ wan22 qa2 frames + L1L2 questions
PRE_CONVERT_FRAMES_ROOT=""  # 非空时在主流程前对整棵 frames 目录做 *_edited.png 规范化

usage() {
  echo "用法: $0 --model MODEL_NAME [选项]"
  echo ""
  echo "参数说明:"
  echo "  --model MODEL_NAME      模型名称（必需），如: qwen_plus, nanobanana"
  echo "  --datasets TYPES        数据集类型，用逗号分隔，如: L1L2,L3 (默认: L1L2,L3)"
  echo "                          支持的类型: L1L2, L3, V7"
  echo "  --skip-dino            跳过DINO检测步骤"
  echo "  --skip-sam-track       跳过SAM追踪步骤"
  echo "  --skip-hoi-check      跳过 Google Gemini HOI Check 步骤"
  echo "  --skip-convert         跳过图片格式转换步骤"
  echo "  --skip-resize          跳过图像缩放步骤"
  echo "  --resize-inplace       在原图像路径下直接resize（覆盖原图像）"
  echo "  --hoi-check-only       直接执行HOI Check步骤（跳过所有前面的步骤）"
  echo "  --use-optimized-dino   使用优化版DINO脚本（只加载一次模型，速度更快）"
  echo "  --gpu-id ID           指定CUDA_VISIBLE_DEVICES的GPU ID (默认: 0)"
  echo "  --background          在后台运行（使用nohup，输出到日志文件）"
  echo "  --log-file FILE       指定日志文件路径（默认: output_<MODEL>_v7_google.log）"
  echo "  --output-dir DIR      指定输出目录（默认: 使用时间戳创建新目录）"
  echo "                        例如: --output-dir runs/20251114_005019"
  echo "  --cr-wan22-qa2-l1l2   预设: data/wan22_official3_enhanced_qa2_frames/L1L2"
  echo "                        + collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json"
  echo "                        （先对 frames 根目录做 jpg→*_edited.png，再跑 V7 评测）"
  echo ""
  echo "示例:"
  echo "  $0 --model nanobanana --datasets L1L2,L3"
  echo "  $0 --model qwen_plus --datasets L3"
  echo "  $0 --model nanobanana --datasets L1L2,L3 --skip-dino"
  echo "  $0 --model nanobanana --datasets L1L2,L3 --gpu-id 0"
  echo "  $0 --model qwen_plus --hoi-check-only  # 直接执行HOI Check，跳过前面的所有步骤"
  echo "  $0 --model knotext --datasets L1L2 --hoi-check-only --background  # 后台运行"
  echo ""
  echo "  V7 示例（需先 export 路径）:"
  echo "    export EVAL_V7_IMAGE_DIR=.../data_v7/data_v7_edited_images_12"
  echo "    export EVAL_V7_JSON=.../data_v7/collected_annotations_bboxes_v7_L3_questions_data_v7_12.json"
  echo "    # EVAL_V7_ORIG_DIR 可不设，脚本默认 .../data_v7/new_results_v7_original_flat"
  echo "    # 指定其他原图根：export V7_EVAL_ORIG_OVERRIDE=/path"
  echo "    $0 --model v7_edited_12 --datasets V7"
  echo "    # 旧版 split（JSON key 为 1/xxx.jpg）仍须自行 export 三项路径。"
  echo ""
  echo "  CR wan22 qa2 L1L2 示例:"
  echo "    $0 --cr-wan22-qa2-l1l2 --gpu-id 0"
  echo "    $0 --cr-wan22-qa2-l1l2 --skip-convert --hoi-check-only"
  echo ""
  echo "  CR L1L2/L3 分目录命名（与 run_cr_wan22_setup_and_eval.sh 一致）:"
  echo "    export EVAL_V7_SPLIT_TAG=L1L2   # resize/DINO/SAM2/eval_runs 均带 _L1L2 后缀"
  echo "    export EVAL_V7_IMAGE_DIR=... export EVAL_V7_JSON=..."
  echo ""
  echo "  tool_unmentioned tool 框 SAM2 追踪（可选，默认关闭）:"
  echo "    export TRACK_TOOL_BBOXES=1"
  echo "    # 可选 TRACK_TOOL_UNMENTIONED_ALL=1 追踪所有含 tool_bboxes 样本"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --datasets)
      DATASET_TYPES="$2"
      shift 2
      ;;
    --skip-dino)
      SKIP_DINO=1
      shift
      ;;
    --skip-sam-track)
      SKIP_SAM_TRACK=1
      shift
      ;;
    --skip-hoi-check)
      SKIP_HOI_CHECK=1
      shift
      ;;
    --skip-convert)
      SKIP_CONVERT=1
      shift
      ;;
    --skip-resize)
      SKIP_RESIZE=1
      shift
      ;;
    --resize-inplace)
      RESIZE_INPLACE=1
      shift
      ;;
    --hoi-check-only)
      HOI_CHECK_ONLY=1
      SKIP_CONVERT=1
      SKIP_RESIZE=1
      SKIP_DINO=1
      SKIP_SAM_TRACK=1
      shift
      ;;
    --use-optimized-dino)
      USE_OPTIMIZED_DINO=1
      shift
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --background)
      RUN_BACKGROUND=1
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --cr-wan22-qa2-l1l2)
      CR_WAN22_QA2_L1L2=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "❌ 未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

# CR wan22 qa2 L1L2 预设路径
if [[ ${CR_WAN22_QA2_L1L2} -eq 1 ]]; then
  DATA_V7_CR="${DATA_V7_CR:-${HOI_EDIT_DATA_DIR:-${SHARED_DIR}/data}}"
  PRE_CONVERT_FRAMES_ROOT="${DATA_V7_CR}/wan22_official3_enhanced_qa2_frames"
  export EVAL_V7_IMAGE_DIR="${PRE_CONVERT_FRAMES_ROOT}/L1L2"
  export EVAL_V7_JSON="${DATA_V7_CR}/collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json"
  export EVAL_V7_ORIG_DIR="${DATA_V7_CR}/data_v7_L12"
  DATASET_TYPES="V7"
  if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_NAME="wan22_official3_enhanced_qa2"
  fi
fi

if [[ -z "${MODEL_NAME}" ]]; then
  echo "❌ 必须指定模型名称: --model MODEL_NAME（或使用 --cr-wan22-qa2-l1l2）"
  usage
  exit 1
fi

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 如果指定后台运行，使用nohup包装执行
if [[ ${RUN_BACKGROUND} -eq 1 ]]; then
  if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE="${EVAL_DIR}/output_${MODEL_NAME}_v7_google.log"
  fi
  echo "🔄 后台运行模式"
  echo "📝 日志文件: ${LOG_FILE}"
  echo ""
  
  # 构建完整的命令
  CMD_ARGS=(
    --model "${MODEL_NAME}"
    --datasets "${DATASET_TYPES}"
  )
  [[ ${SKIP_DINO} -eq 1 ]] && CMD_ARGS+=(--skip-dino)
  [[ ${SKIP_SAM_TRACK} -eq 1 ]] && CMD_ARGS+=(--skip-sam-track)
  [[ ${SKIP_HOI_CHECK} -eq 1 ]] && CMD_ARGS+=(--skip-hoi-check)
  [[ ${SKIP_CONVERT} -eq 1 ]] && CMD_ARGS+=(--skip-convert)
  [[ ${SKIP_RESIZE} -eq 1 ]] && CMD_ARGS+=(--skip-resize)
  [[ ${RESIZE_INPLACE} -eq 1 ]] && CMD_ARGS+=(--resize-inplace)
  [[ ${HOI_CHECK_ONLY} -eq 1 ]] && CMD_ARGS+=(--hoi-check-only)
  [[ ${USE_OPTIMIZED_DINO} -eq 1 ]] && CMD_ARGS+=(--use-optimized-dino)
  [[ -n "${OUTPUT_DIR}" ]] && CMD_ARGS+=(--output-dir "${OUTPUT_DIR}")
  CMD_ARGS+=(--gpu-id "${GPU_ID}")
  
  # 使用nohup在后台运行，并将输出重定向到日志文件
  nohup bash "${SCRIPT_DIR}/run_full_eval_v7_google.sh" "${CMD_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
  
  PID=$!
  echo "✅ 进程已在后台启动"
  echo "📌 进程ID: ${PID}"
  echo "📝 日志文件: ${LOG_FILE}"
  echo ""
  echo "查看日志: tail -f ${LOG_FILE}"
  echo "查看进程: ps aux | grep ${PID}"
  echo "停止进程: kill ${PID}"
  exit 0
fi

# 默认数据集类型
if [[ -z "${DATASET_TYPES}" ]]; then
  DATASET_TYPES="L1L2,L3"
fi

# 解析数据集类型列表
IFS=',' read -ra DATASET_ARRAY <<< "${DATASET_TYPES}"

# 验证数据集类型
VALID_TYPES=("L1L2" "L3" "V7")
for dataset_type in "${DATASET_ARRAY[@]}"; do
  valid=0
  for valid_type in "${VALID_TYPES[@]}"; do
    if [[ "${dataset_type}" == "${valid_type}" ]]; then
      valid=1
      break
    fi
  done
  if [[ ${valid} -eq 0 ]]; then
    echo "❌ 无效的数据集类型: ${dataset_type}"
    echo "   可选: L1L2, L3, V7"
    exit 1
  fi
done

# V7：扁平原图根目录（resize 用 join(ORIG, JSON key)）。默认无需再 export EVAL_V7_ORIG_DIR。
#   默认：${SHARED_DIR}/data_v7/new_results_v7_original_flat
# 若 tmux 里仍指向旧路径 .../IPT2V2026/new_results_v7/data_v7，会自动改回默认扁平原图。
# 显式使用其他目录：export V7_EVAL_ORIG_OVERRIDE=/path/to/orig_flat
DATA_V7_DIR_FOR_ORIG="${SHARED_DIR}/data_v7"
V7_DEFAULT_ORIG_FLAT="${DATA_V7_DIR_FOR_ORIG}/new_results_v7_original_flat"
has_v7=0
for _dt in "${DATASET_ARRAY[@]}"; do
  if [[ "${_dt}" == "V7" ]]; then
    has_v7=1
    break
  fi
done
if [[ ${has_v7} -eq 1 ]]; then
  if [[ -n "${V7_EVAL_ORIG_OVERRIDE:-}" ]]; then
    export EVAL_V7_ORIG_DIR="${V7_EVAL_ORIG_OVERRIDE}"
  elif [[ -z "${EVAL_V7_ORIG_DIR:-}" ]]; then
    export EVAL_V7_ORIG_DIR="${V7_DEFAULT_ORIG_FLAT}"
  elif [[ "${EVAL_V7_ORIG_DIR}" == *"/IPT2V2026/new_results_v7/data_v7"* ]] && [[ "${EVAL_V7_ORIG_DIR}" != *"new_results_v7_original_flat"* ]]; then
    export EVAL_V7_ORIG_DIR="${V7_DEFAULT_ORIG_FLAT}"
  fi
fi

# --------------------------
# 函数：获取数据集路径
# --------------------------
get_dataset_paths() {
  local dataset_type="$1"
  local model_name="$2"
  
  local image_dir=""
  local json_file=""
  local orig_image_dir=""
  
  case "${dataset_type}" in
    L1L2)
      image_dir="${SHARED_DIR}/final_eval_data_edited_${model_name}_v6_L1L2"
      json_file="${DATA_V6_DIR}/collected_annotations_bboxes_v6_L1L2_questions.json"
      orig_image_dir="${DATA_V6_DIR}/v6_images_L1L2"
      ;;
    L3)
      image_dir="${SHARED_DIR}/final_eval_data_edited_${model_name}_v6_L3"
      json_file="${DATA_V6_DIR}/collected_annotations_bboxes_v6_L3_questions.json"
      orig_image_dir="${DATA_V6_DIR}/v6_images_L3"
      ;;
    V7)
      if [[ -z "${EVAL_V7_IMAGE_DIR:-}" || -z "${EVAL_V7_JSON:-}" ]]; then
        echo "❌ 数据集 V7 需要环境变量: EVAL_V7_IMAGE_DIR, EVAL_V7_JSON" >&2
        echo "   （EVAL_V7_ORIG_DIR 可选，未设置时默认 data_v7/new_results_v7_original_flat）" >&2
        return 1
      fi
      image_dir="${EVAL_V7_IMAGE_DIR}"
      json_file="${EVAL_V7_JSON}"
      orig_image_dir="${EVAL_V7_ORIG_DIR:-${V7_DEFAULT_ORIG_FLAT}}"
      ;;
    *)
      echo "❌ 未知的数据集类型: ${dataset_type}" >&2
      return 1
      ;;
  esac
  
  echo "${image_dir}|${json_file}|${orig_image_dir}"
}

# 目录/结果命名后缀：CR 评测 export EVAL_V7_SPLIT_TAG=L1L2|L3 时用 L1L2/L3，不用 V7
get_path_suffix() {
  local dataset_type="$1"
  if [[ "${dataset_type}" == "V7" && -n "${EVAL_V7_SPLIT_TAG:-}" ]]; then
    echo "${EVAL_V7_SPLIT_TAG}"
  else
    echo "${dataset_type}"
  fi
}

# resize 输出目录
get_resized_image_dir() {
  local model_name="$1"
  local dataset_type="$2"
  local path_suffix
  path_suffix="$(get_path_suffix "${dataset_type}")"
  if [[ "${dataset_type}" == "V7" && -n "${EVAL_V7_SPLIT_TAG:-}" ]]; then
    echo "${SHARED_DIR}/final_eval_data_edited_${model_name}_${path_suffix}_resized"
  else
    echo "${SHARED_DIR}/final_eval_data_edited_${model_name}_v6_${path_suffix}_resized"
  fi
}

# --------------------------
# 检查必要的文件
# --------------------------
check_path() {
  local path="$1"
  local type_desc="$2"
  if [[ -e "$path" ]]; then
    echo "✅ ${type_desc}: ${path}"
  else
    echo "❌ ${type_desc} 未找到: ${path}"
    return 1
  fi
}

echo "=========================================="
echo "🚀 通用评测流程 - V6/V7 数据集（Google Gemini 官方 API）"
echo "=========================================="
echo "📦 模型名称: ${MODEL_NAME}"
echo "📊 数据集类型: ${DATASET_TYPES}"
echo "🖥️  GPU ID: ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "☁️  API: Google Gemini 官方 SDK (${GOOGLE_SCRIPT##*/})"
if [[ ${HOI_CHECK_ONLY} -eq 1 ]]; then
  echo "⚡ 模式: 仅执行HOI Check（跳过前面的所有步骤）"
fi
echo ""

# 检查脚本文件
if [[ ${SKIP_DINO} -eq 0 ]]; then
  if [[ ! -x "${DINO_ENV_PY}" ]]; then
    echo "❌ DINO 解释器不可执行: ${DINO_ENV_PY}"
    echo "   请: export DINO_ENV_PY=${DINO_ENV_PY}"
    echo "   或: bash ${SETUP_DINO}"
    exit 1
  fi
  if [[ ${USE_OPTIMIZED_DINO} -eq 1 ]]; then
    if [[ ! -f "${DINO_INFER_SCRIPT_OPTIMIZED}" ]]; then
      echo "❌ 未找到优化版DINO脚本: ${DINO_INFER_SCRIPT_OPTIMIZED}"
      exit 1
    fi
  elif [[ ! -f "${DINO_INFER_SCRIPT}" ]]; then
    echo "❌ 未找到DINO推理脚本: ${DINO_INFER_SCRIPT}"
    exit 1
  fi
fi

if [[ ${SKIP_SAM_TRACK} -eq 0 ]]; then
  if [[ ! -f "${SAM2_TRACKING_SCRIPT}" ]]; then
    echo "⚠️  未找到SAM2追踪脚本: ${SAM2_TRACKING_SCRIPT}"
  fi
  if [[ ! -x "${SS_ENV_PY}" ]]; then
    echo "⚠️  SAM2 Python 不可用: ${SS_ENV_PY}（可用 --skip-sam-track 跳过）"
  fi
fi

if [[ ${SKIP_HOI_CHECK} -eq 0 ]]; then
  if [[ ! -f "${GOOGLE_SCRIPT}" ]]; then
    echo "❌ 未找到 Google HOI Check 脚本: ${GOOGLE_SCRIPT}"
    exit 1
  fi
  if [[ ! -x "${GOOGLE_ENV_PY}" ]]; then
    echo "❌ Google HOI Python 不可执行: ${GOOGLE_ENV_PY}"
    echo "   请: export GOOGLE_ENV_PY=/path/to/python  (需已安装 google-genai)"
    exit 1
  fi
  if ! "${GOOGLE_ENV_PY}" -c "from google import genai" 2>/dev/null; then
    echo "❌ ${GOOGLE_ENV_PY} 缺少 google-genai: pip install google-genai"
    exit 1
  fi
  echo "✅ Google HOI Python: ${GOOGLE_ENV_PY}"
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "❌ 未设置 GEMINI_API_KEY，HOI Check 无法运行"
    exit 1
  fi
fi

# 检查数据集路径
for dataset_type in "${DATASET_ARRAY[@]}"; do
  paths=$(get_dataset_paths "${dataset_type}" "${MODEL_NAME}")
  IFS='|' read -ra PATHS <<< "${paths}"
  image_dir="${PATHS[0]}"
  json_file="${PATHS[1]}"
  orig_image_dir="${PATHS[2]}"
  
  if ! check_path "${image_dir}" "${dataset_type}图像目录"; then
    echo "⚠️  警告: ${dataset_type} 数据集的图像目录不存在，将跳过"
  fi
  if ! check_path "${json_file}" "${dataset_type} JSON文件"; then
    echo "⚠️  警告: ${dataset_type} 数据集的JSON文件不存在，将跳过"
  fi
  if ! check_path "${orig_image_dir}" "${dataset_type}原始图像目录"; then
    echo "⚠️  警告: ${dataset_type} 数据集的原始图像目录不存在，将跳过"
  fi
done

# --------------------------
# 运行评测
# --------------------------
# 设置 eval_runs 目录路径
EVAL_RUNS_DIR="${SHARED_DIR}/eval_runs"

if [[ -n "${OUTPUT_DIR}" ]]; then
  # 如果指定了输出目录，使用指定的目录
  RUN_OUTPUT_ROOT="${OUTPUT_DIR}"
  if [[ ! "${RUN_OUTPUT_ROOT}" =~ ^/ ]]; then
    # 如果不是绝对路径，则相对于EVAL_DIR
    RUN_OUTPUT_ROOT="${EVAL_DIR}/${OUTPUT_DIR}"
  fi
  echo "📂 使用指定的输出目录: ${RUN_OUTPUT_ROOT}"
else
  # 默认使用 eval_runs 目录，按照 {model}_{dataset}_full/{model}_{dataset}/ 结构组织
  RUN_OUTPUT_ROOT="${EVAL_RUNS_DIR}"
  echo "📂 使用 eval_runs 目录: ${RUN_OUTPUT_ROOT}"
fi
mkdir -p "${RUN_OUTPUT_ROOT}"

echo ""
echo "📂 输出根目录: ${RUN_OUTPUT_ROOT}"
echo ""

# CR 预设：在主流程前对整个 frames 根目录做指令格式转换（含 L1L2 子目录）
if [[ -n "${PRE_CONVERT_FRAMES_ROOT}" && ${SKIP_CONVERT} -eq 0 ]]; then
  echo "------------------------------------------"
  echo "▶️  CR Prep: 规范化 frames → *_edited.png"
  echo "   目录: ${PRE_CONVERT_FRAMES_ROOT}"
  echo "   JSON: ${EVAL_V7_JSON}"
  echo "------------------------------------------"
  if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "❌ 转换脚本不存在: ${CONVERT_SCRIPT}"
    exit 1
  fi
  python3 "${CONVERT_SCRIPT}" \
    --image_dir "${PRE_CONVERT_FRAMES_ROOT}" \
    --json_path "${EVAL_V7_JSON}"
  echo ""
elif [[ -n "${PRE_CONVERT_FRAMES_ROOT}" ]]; then
  echo ">>> CR Prep: 跳过 frames 根目录转换（--skip-convert）"
  echo ""
fi

# 处理每个数据集
for dataset_type in "${DATASET_ARRAY[@]}"; do
  echo "------------------------------------------"
  
  # 获取数据集路径
  paths=$(get_dataset_paths "${dataset_type}" "${MODEL_NAME}")
  IFS='|' read -ra PATHS <<< "${paths}"
  IMAGE_DIR="${PATHS[0]}"
  INPUT_JSON="${PATHS[1]}"
  ORIG_IMAGE_DIR="${PATHS[2]}"
  PATH_SUFFIX="$(get_path_suffix "${dataset_type}")"
  echo "▶️  开始处理数据集: ${dataset_type}（目录后缀: ${PATH_SUFFIX}）"
  if [[ -n "${EVAL_V7_REQUIRE_SPLIT_TAG:-}" && "${PATH_SUFFIX}" != "${EVAL_V7_REQUIRE_SPLIT_TAG}" ]]; then
    echo "❌ 目录后缀 PATH_SUFFIX=${PATH_SUFFIX}，但要求 EVAL_V7_REQUIRE_SPLIT_TAG=${EVAL_V7_REQUIRE_SPLIT_TAG}"
    echo "   请勿使用 --datasets L1L2/L3；2jiehe 请用: bash evaluation/run_2jiehe_*_eval_google.sh"
    exit 1
  fi
  if [[ "${PATH_SUFFIX}" == "2jiehe" ]]; then
    echo "📌 2jiehe 产物目录: ${MODEL_NAME}_2jiehe_*（与 CR 的 ${MODEL_NAME}_L1L2_* 分离）"
  fi
  echo "------------------------------------------"
  
  # 检查路径是否存在
  if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "⚠️  图像目录不存在，跳过: ${IMAGE_DIR}"
    continue
  fi
  
  if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "⚠️  JSON文件不存在，跳过: ${INPUT_JSON}"
    continue
  fi
  
  # eval_runs / 检测 / 追踪 均用 PATH_SUFFIX（CR 时为 L1L2 或 L3，而非 V7）
  if [[ -n "${OUTPUT_DIR}" ]]; then
    MODEL_OUTPUT_DIR="${RUN_OUTPUT_ROOT}/${MODEL_NAME}_${PATH_SUFFIX}"
  else
    MODEL_OUTPUT_DIR="${RUN_OUTPUT_ROOT}/${MODEL_NAME}_${PATH_SUFFIX}_full/${MODEL_NAME}_${PATH_SUFFIX}"
  fi
  mkdir -p "${MODEL_OUTPUT_DIR}"
  
  PERSON_DET_DIR="${EVAL_DIR}/${MODEL_NAME}_${PATH_SUFFIX}_detection_human"
  OBJECT_DET_DIR="${EVAL_DIR}/${MODEL_NAME}_${PATH_SUFFIX}_detection_object"
  TRACK_DIR="${SHARED_DIR}/sam2/final_output_object_track_${MODEL_NAME}_${PATH_SUFFIX}"
  TEMP_DIR="${MODEL_OUTPUT_DIR}/temp_images"
  
  if [[ -n "${HOI_RESULT_TAG}" ]]; then
    OUTPUT_JSON="${MODEL_OUTPUT_DIR}/results_${MODEL_NAME}_${PATH_SUFFIX}_${HOI_RESULT_TAG}.json"
    echo "📝 HOI 结果（${HOI_RESULT_TAG}）: ${OUTPUT_JSON}"
  else
    OUTPUT_JSON="${MODEL_OUTPUT_DIR}/results_${MODEL_NAME}_${PATH_SUFFIX}_google_full.json"
    if [[ -f "${OUTPUT_JSON}" ]]; then
      echo "📝 将更新 Google 版结果文件: ${OUTPUT_JSON}"
    else
      echo "📝 将创建 Google 版结果文件: ${OUTPUT_JSON}"
    fi
  fi
  
  mkdir -p "${TEMP_DIR}"
  
  echo "📁 图像目录: ${IMAGE_DIR}"
  echo "📄 输入JSON: ${INPUT_JSON}"
  echo ""
  
  # ==========================================
  # Step 0: 图片格式转换（JPG -> PNG，规范化文件名）
  # ==========================================
  if [[ ${SKIP_CONVERT} -eq 0 ]]; then
    echo ">>> Step 0: 图片格式转换和规范化"
    echo "开始时间: $(date)"
    
    if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
      echo "⚠️  转换脚本不存在: ${CONVERT_SCRIPT}"
      echo "   跳过格式转换步骤"
    else
      echo "  - 检查并转换图片格式..."
      python3 "${CONVERT_SCRIPT}" \
        --image_dir "${IMAGE_DIR}" \
        --json_path "${INPUT_JSON}"
      
      if [[ $? -ne 0 ]]; then
        echo "⚠️  图片格式转换失败，但继续执行后续步骤"
      else
        echo "  ✅ 图片格式转换完成"
      fi
    fi
    echo "完成时间: $(date)"
    echo ""
  else
    echo ">>> Step 0: 跳过图片格式转换"
    echo ""
  fi
  
  # ==========================================
  # Step 0.5: 将编辑图像缩放到原始图像尺寸
  # ==========================================
  if [[ ${SKIP_RESIZE} -eq 0 ]]; then
    echo ">>> Step 0.5: 将编辑图像缩放到原始图像尺寸"
    echo "开始时间: $(date)"
    echo "⚠️  注意: 此步骤可以避免尺寸不匹配导致的框错位问题"
    
    if [[ ! -f "${RESIZE_SCRIPT}" ]]; then
      echo "⚠️  缩放脚本不存在: ${RESIZE_SCRIPT}"
      echo "   跳过图像缩放步骤，使用原始编辑图像"
      RESIZED_IMAGE_DIR="${IMAGE_DIR}"  # 使用原始编辑图像目录
    else
      # 缩放后的图像目录
      RESIZED_IMAGE_DIR="$(get_resized_image_dir "${MODEL_NAME}" "${dataset_type}")"
      
      echo "  - 输入JSON: ${INPUT_JSON}"
      echo "  - 原始图像目录: ${ORIG_IMAGE_DIR}"
      echo "  - 编辑图像目录: ${IMAGE_DIR}"
      echo "  - 输出图像目录: ${RESIZED_IMAGE_DIR}"
      echo ""
      
      # 创建输出目录
      mkdir -p "${RESIZED_IMAGE_DIR}"
      
      # 执行缩放脚本
      echo "  - 执行图像缩放..."
      RESIZE_ARGS=(
        --input_json "${INPUT_JSON}"
        --original_image_dir "${ORIG_IMAGE_DIR}"
        --edited_image_dir "${IMAGE_DIR}"
      )
      
      if [[ ${RESIZE_INPLACE} -eq 1 ]]; then
        echo "  ⚠️  模式: 在原图像路径下直接resize（将覆盖原图像）"
        RESIZE_ARGS+=(--inplace)
      else
        echo "  📁 模式: 保存到新目录（不覆盖原图像）"
        RESIZE_ARGS+=(--output_image_dir "${RESIZED_IMAGE_DIR}")
      fi
      
      python3 "${RESIZE_SCRIPT}" "${RESIZE_ARGS[@]}"
      
      if [[ $? -ne 0 ]]; then
        echo "⚠️  图像缩放失败，使用原始编辑图像"
        RESIZED_IMAGE_DIR="${IMAGE_DIR}"  # 使用原始编辑图像目录
      else
        echo "  ✅ 图像缩放完成"
        if [[ ${RESIZE_INPLACE} -eq 1 ]]; then
          echo "  📁 图像已在原路径下缩放（已覆盖原图像）"
          # inplace模式：图像已在原路径下，不需要更新IMAGE_DIR
        else
          echo "  📁 缩放后的图像目录: ${RESIZED_IMAGE_DIR}"
          # 非inplace模式：更新IMAGE_DIR为缩放后的图像目录，后续步骤将使用缩放后的图像
          IMAGE_DIR="${RESIZED_IMAGE_DIR}"
        fi
      fi
    fi
    echo "完成时间: $(date)"
    echo ""
  else
    echo ">>> Step 0.5: 跳过图像缩放（使用原始编辑图像）"
    RESIZED_IMAGE_DIR="${IMAGE_DIR}"  # 使用原始编辑图像目录
    echo ""
  fi
  
  # ==========================================
  # Step 1: DINO 检测（人物和物体）
  # ==========================================
  if [[ ${SKIP_DINO} -eq 0 ]]; then
    echo ">>> Step 1: DINO 检测（人物和物体）"
    echo "开始时间: $(date)"
    
    # 清理之前的检测结果
    # 使用更稳健的删除方式，避免因目录非空或权限问题导致失败
    if [[ -d "${PERSON_DET_DIR}" ]]; then
      rm -rf "${PERSON_DET_DIR}" 2>/dev/null || {
        echo "⚠️  警告: 无法删除人物检测目录，尝试清空内容..."
        find "${PERSON_DET_DIR}" -mindepth 1 -delete 2>/dev/null || true
      }
    fi
    if [[ -d "${OBJECT_DET_DIR}" ]]; then
      rm -rf "${OBJECT_DET_DIR}" 2>/dev/null || {
        echo "⚠️  警告: 无法删除物体检测目录，尝试清空内容..."
        find "${OBJECT_DET_DIR}" -mindepth 1 -delete 2>/dev/null || true
      }
    fi
    mkdir -p "${PERSON_DET_DIR}"
    mkdir -p "${OBJECT_DET_DIR}"
    
    # 检测人物
    echo "  - 检测人物..."
    if [[ ${USE_OPTIMIZED_DINO} -eq 1 ]]; then
      echo "  ⚡ 使用优化版DINO脚本（只加载一次模型，避免代理超时问题）"
      # 临时禁用代理并使用HuggingFace镜像（避免代理超时）
      NO_PROXY_ENV="http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= HF_ENDPOINT=https://hf-mirror.com"
      echo "  ℹ️  临时禁用代理并使用HuggingFace镜像以避免超时"
      env ${NO_PROXY_ENV} "${DINO_ENV_PY}" "${DINO_INFER_SCRIPT_OPTIMIZED}" \
        --image-root "${IMAGE_DIR}" \
        --json-path "${INPUT_JSON}" \
        --output-dir "${PERSON_DET_DIR}" \
        --origin_prompt human \
        --gpu-id "${CUDA_VISIBLE_DEVICES}"
    else
      echo "  ⚠️  使用原版DINO脚本（每张图片都重新加载模型，较慢）"
      # 临时禁用代理并使用HuggingFace镜像（避免代理超时）
      NO_PROXY_ENV="http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= HF_ENDPOINT=https://hf-mirror.com"
      echo "  ℹ️  临时禁用代理并使用HuggingFace镜像以避免超时"
      env ${NO_PROXY_ENV} "${DINO_ENV_PY}" "${DINO_INFER_SCRIPT}" \
        --image-root "${IMAGE_DIR}" \
        --json-path "${INPUT_JSON}" \
        --output-dir "${PERSON_DET_DIR}" \
        --origin_prompt human
    fi
    
    if [[ $? -ne 0 ]]; then
      echo "❌ 人物检测失败！"
      exit 1
    fi
    echo "  ✅ 人物检测完成"
    
    # 检测物体
    echo "  - 检测物体..."
    if [[ ${USE_OPTIMIZED_DINO} -eq 1 ]]; then
      echo "  ⚡ 使用优化版DINO脚本（只加载一次模型，避免代理超时问题）"
      # 临时禁用代理并使用HuggingFace镜像（避免代理超时）
      NO_PROXY_ENV="http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= HF_ENDPOINT=https://hf-mirror.com"
      echo "  ℹ️  临时禁用代理并使用HuggingFace镜像以避免超时"
      env ${NO_PROXY_ENV} "${DINO_ENV_PY}" "${DINO_INFER_SCRIPT_OPTIMIZED}" \
        --image-root "${IMAGE_DIR}" \
        --json-path "${INPUT_JSON}" \
        --output-dir "${OBJECT_DET_DIR}" \
        --origin_prompt object \
        --gpu-id "${CUDA_VISIBLE_DEVICES}"
    else
      echo "  ⚠️  使用原版DINO脚本（每张图片都重新加载模型，较慢）"
      # 临时禁用代理并使用HuggingFace镜像（避免代理超时）
      NO_PROXY_ENV="http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= HF_ENDPOINT=https://hf-mirror.com"
      echo "  ℹ️  临时禁用代理并使用HuggingFace镜像以避免超时"
      env ${NO_PROXY_ENV} "${DINO_ENV_PY}" "${DINO_INFER_SCRIPT}" \
        --image-root "${IMAGE_DIR}" \
        --json-path "${INPUT_JSON}" \
        --output-dir "${OBJECT_DET_DIR}" \
        --origin_prompt object
    fi
    
    if [[ $? -ne 0 ]]; then
      echo "❌ 物体检测失败！"
      exit 1
    fi
    echo "  ✅ 物体检测完成"
    echo "完成时间: $(date)"
    echo ""
  else
    echo ">>> Step 1: 跳过DINO检测（使用已有结果）"
    if [[ ! -d "${PERSON_DET_DIR}" ]]; then
      echo "⚠️  警告: 人物检测目录不存在: ${PERSON_DET_DIR}"
    fi
    if [[ ! -d "${OBJECT_DET_DIR}" ]]; then
      echo "⚠️  警告: 物体检测目录不存在: ${OBJECT_DET_DIR}"
    fi
    echo ""
  fi
  
  # ==========================================
  # Step 2: SAM2 追踪
  # ==========================================
  if [[ ${SKIP_SAM_TRACK} -eq 0 ]]; then
    echo ">>> Step 2: SAM2 追踪"
    echo "开始时间: $(date)"
    echo "⚠️  注意: SAM2追踪需要较长时间，请耐心等待..."
    
    if [[ ${FORCE_SAM_TRACK:-0} -eq 1 && -d "${TRACK_DIR}" ]]; then
      echo "  🔄 FORCE_SAM_TRACK=1: 删除已有追踪目录并重新生成"
      rm -rf "${TRACK_DIR}"
    fi
    if [[ -d "${TRACK_DIR}" ]]; then
      echo "  ✅ 追踪目录已存在: ${TRACK_DIR}"
      echo "  ℹ️  如需重新生成: rm -rf \"${TRACK_DIR}\" 或 FORCE_SAM_TRACK=1"
      echo "  ℹ️  跳过tracking步骤，使用已有结果"
    else
      echo "  📁 追踪目录不存在，开始生成tracking结果..."
      
      if [[ ! -f "${SAM2_TRACKING_SCRIPT}" ]]; then
        echo "  ❌ 未找到SAM2追踪脚本: ${SAM2_TRACKING_SCRIPT}"
        echo "  ⚠️  跳过tracking步骤"
      else
        # 检查必要的输入
        if [[ ! -d "${IMAGE_DIR}" ]]; then
          echo "  ❌ 图像目录不存在: ${IMAGE_DIR}"
          echo "  ⚠️  跳过tracking步骤"
        elif [[ ! -f "${INPUT_JSON}" ]]; then
          echo "  ❌ JSON文件不存在: ${INPUT_JSON}"
          echo "  ⚠️  跳过tracking步骤"
        else
          echo "  - 输入JSON: ${INPUT_JSON}"
          echo "  - 原始图像目录: ${ORIG_IMAGE_DIR}"
          echo "  - 编辑图像目录: ${IMAGE_DIR}"
          echo "  - 输出目录: ${TRACK_DIR}"
          echo ""
          
          # 创建输出目录
          mkdir -p "${TRACK_DIR}"
          
          # 执行tracking脚本
          echo "  - 执行SAM2追踪..."
          START_TIME_TRACK=$(date +%s)
          
          SAM2_EXTRA_ARGS=()
          if [[ "${TRACK_TOOL_BBOXES:-0}" == "1" ]]; then
            echo "  - tool_bboxes 追踪: 开启 (tool_unmentioned)"
            SAM2_EXTRA_ARGS+=(--track-tool-bboxes)
            if [[ "${TRACK_TOOL_UNMENTIONED_ALL:-0}" == "1" ]]; then
              SAM2_EXTRA_ARGS+=(--tool-unmentioned-all)
            fi
          fi

          "${SS_ENV_PY}" "${SAM2_TRACKING_SCRIPT}" \
            --input_json "${INPUT_JSON}" \
            --original_image_dir "${ORIG_IMAGE_DIR}" \
            --edited_image_dir "${IMAGE_DIR}" \
            --output_dir "${TRACK_DIR}" \
            "${SAM2_EXTRA_ARGS[@]}"
          
          EXIT_CODE_TRACK=$?
          END_TIME_TRACK=$(date +%s)
          ELAPSED_TRACK=$((END_TIME_TRACK - START_TIME_TRACK))
          ELAPSED_TRACK_MIN=$((ELAPSED_TRACK / 60))
          ELAPSED_TRACK_SEC=$((ELAPSED_TRACK % 60))
          
          if [[ ${EXIT_CODE_TRACK} -eq 0 ]]; then
            echo ""
            echo "  ✅ SAM2追踪完成！耗时 ${ELAPSED_TRACK_MIN}分${ELAPSED_TRACK_SEC}秒"
            echo "  📁 结果输出: ${TRACK_DIR}"
          else
            echo ""
            echo "  ❌ SAM2追踪失败 (退出码: ${EXIT_CODE_TRACK})"
            echo "  ⚠️  继续执行后续步骤，但tracking结果可能不可用"
          fi
        fi
      fi
    fi
    echo "完成时间: $(date)"
    echo ""
  else
    echo ">>> Step 2: 跳过SAM2追踪（使用已有结果）"
    if [[ ! -d "${TRACK_DIR}" ]]; then
      echo "⚠️  警告: 追踪目录不存在: ${TRACK_DIR}"
    fi
    echo ""
  fi
  
  # ==========================================
  # Step 3: Google Gemini HOI Check
  # ==========================================
  if [[ ${SKIP_HOI_CHECK} -eq 0 ]]; then
    echo ">>> Step 3: Google Gemini HOI Check (gemini-2.5-pro)"
    echo "开始时间: $(date)"
    
    # 检查图像目录中是否有图片（使用-L跟随符号链接）
    # 注意：V7 等为 image_dir/1/*.png 子目录结构，不能用 -maxdepth 1（会误报 0 张并跳过 HOI）
    IMAGE_COUNT=$(find -L "${IMAGE_DIR}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
    echo "📊 找到 ${IMAGE_COUNT} 张图片"
    
    if [[ ${IMAGE_COUNT} -eq 0 ]]; then
      echo "⚠️  警告: 图像目录中没有找到图片，跳过此数据集"
      continue
    fi
    
    PY_ARGS=(
      --input_json_path "${INPUT_JSON}"
      --image_dir_path "${IMAGE_DIR}"
      --original_image_dir_path "${ORIG_IMAGE_DIR}"
      --output_json_path "${OUTPUT_JSON}"
      --temp_dir "${TEMP_DIR}"
    )
    
    # 添加人物检测目录
    if [[ -d "${PERSON_DET_DIR}" ]]; then
      PY_ARGS+=(--person_dir_path "${PERSON_DET_DIR}")
      echo "✅ 使用人物检测目录: ${PERSON_DET_DIR}"
    fi
    
    # 添加物体检测目录
    if [[ -d "${OBJECT_DET_DIR}" ]]; then
      PY_ARGS+=(--object_dir_path "${OBJECT_DET_DIR}")
      echo "✅ 使用物体检测目录: ${OBJECT_DET_DIR}"
    fi
    
    # 添加追踪目录
    if [[ -d "${TRACK_DIR}" ]]; then
      PY_ARGS+=(--object_track_dir "${TRACK_DIR}")
      PY_ARGS+=(--track_frame_key "frame_00001")
      echo "✅ 使用追踪目录: ${TRACK_DIR}"
    else
      echo "⚠️  未找到追踪目录，将跳过追踪相关功能"
    fi
    
    echo ""
    echo "执行命令:"
    echo "${GOOGLE_ENV_PY} ${GOOGLE_SCRIPT} ${PY_ARGS[*]}"
    echo ""
    
    START_TIME=$(date +%s)
    "${GOOGLE_ENV_PY}" "${GOOGLE_SCRIPT}" "${PY_ARGS[@]}"
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    if [[ ${EXIT_CODE} -eq 0 ]]; then
      echo ""
      echo "✅ Google Gemini HOI Check 完成！耗时 ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
      echo "📁 结果输出: ${OUTPUT_JSON}"
    else
      echo ""
      echo "❌ Google Gemini HOI Check 失败 (退出码: ${EXIT_CODE})"
      exit ${EXIT_CODE}
    fi
    echo "完成时间: $(date)"
    echo ""
  else
    echo ">>> Step 3: 跳过 Google Gemini HOI Check"
    echo ""
  fi
  
  echo "✅ 数据集 ${dataset_type} 处理完成！"
  echo ""
done

echo "=========================================="
echo "🎉 全部数据集处理完成！"
echo "=========================================="
echo "📂 结果目录: ${RUN_OUTPUT_ROOT}"
echo ""
echo "结果文件位置："
for dataset_type in "${DATASET_ARRAY[@]}"; do
  path_suffix="$(get_path_suffix "${dataset_type}")"
  if [[ -n "${OUTPUT_DIR}" ]]; then
    MODEL_OUTPUT_DIR="${RUN_OUTPUT_ROOT}/${MODEL_NAME}_${path_suffix}"
  else
    MODEL_OUTPUT_DIR="${RUN_OUTPUT_ROOT}/${MODEL_NAME}_${path_suffix}_full/${MODEL_NAME}_${path_suffix}"
  fi
  if [[ -n "${HOI_RESULT_TAG}" ]]; then
    OUTPUT_JSON="${MODEL_OUTPUT_DIR}/results_${MODEL_NAME}_${path_suffix}_${HOI_RESULT_TAG}.json"
  else
    OUTPUT_JSON="${MODEL_OUTPUT_DIR}/results_${MODEL_NAME}_${path_suffix}_google_full.json"
  fi
  echo "  - ${path_suffix} 结果: ${OUTPUT_JSON}"
done
echo ""
