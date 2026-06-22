#!/usr/bin/env bash
# 检查发布包脚本与数据是否就绪
set -euo pipefail
_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_EVAL_DIR}/workspace.conf"

ok=0
fail=0

check() {
  local name="$1" cond="$2"
  if eval "${cond}"; then
    echo "✅ ${name}"
    ok=$((ok + 1))
  else
    echo "❌ ${name}"
    fail=$((fail + 1))
  fi
}

echo "=== CR Eval Release Check ==="
echo "EVAL_WORKSPACE=${EVAL_WORKSPACE}"
echo ""

# 入口
check "run_qa_hoi.sh" "[[ -x '${EVAL_WORKSPACE}/run_qa_hoi.sh' ]]"
check "run_eval.sh" "[[ -x '${EVAL_WORKSPACE}/run_eval.sh' ]]"

# 数据
check "L1L2 JSON (499)" "[[ -f '${DATA_V7_CR}/collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json' ]]"
check "L3 JSON (136)" "[[ -f '${DATA_V7_CR}/collected_annotations_bboxes_v7_L3_questions_scoring_final.json' ]]"
check "原图 L12" "[[ -d '${DATA_V7_CR}/data_v7_L12' ]]"
check "原图 L3" "[[ -d '${DATA_V7_CR}/data_v7_L3' ]]"

# 评测脚本
check "QA shell" "[[ -f '${EVAL_DIR}/run_qa_gemini_question_v6.sh' ]]"
check "QA python" "[[ -f '${EVAL_DIR}/run_question_answering.py' ]]"
check "HOI shell" "[[ -f '${EVAL_DIR}/run_full_eval_v7_google.sh' ]]"
check "HOI python" "[[ -f '${EVAL_DIR}/gemini3_final_hoicheck_new_noquestion_track_google_newsim.py' ]]"
check "计分" "[[ -f '${EVAL_DIR}/compute_scoring_final_scores.py' ]]"
check "DINO infer" "[[ -f '${EVAL_DIR}/inference_on_multi_image_eval_optimized.py' ]]"
check "SAM2 tracking" "[[ -f '${SAM2_ROOT}/run_sam2_tracking_for_eval.py' ]]"

# 模型权重
check "GroundingDINO weights" "[[ -f '${GROUNDING_DINO_CHECKPOINT}' ]]"
check "SAM2 checkpoint" "[[ -f '${SAM2_CHECKPOINT}' ]]"

# 依赖库
check "GroundingDINO package" "[[ -d '${GROUNDING_DINO_ROOT}/groundingdino' ]]"
check "SAM2 package" "[[ -d '${SAM2_ROOT}/sam2' ]]"

# 环境配置模板
check "local.conf.example" "[[ -f '${EVAL_WORKSPACE}/env/local.conf.example' ]]"
check "requirements-dino" "[[ -f '${EVAL_WORKSPACE}/env/requirements-dino.txt' ]]"
check "requirements-sam2" "[[ -f '${EVAL_WORKSPACE}/env/requirements-sam2.txt' ]]"
check "requirements-gemini" "[[ -f '${EVAL_WORKSPACE}/env/requirements-hoi-google.txt' ]]"

# 安全：不应含硬编码 API key
if grep -rqE 'AIza[0-9A-Za-z_-]{20,}' \
  "${EVAL_DIR}"/*.py "${EVAL_DIR}"/*.sh \
  "${EVAL_WORKSPACE}/env"/*.conf "${EVAL_WORKSPACE}/env"/*.example \
  "${EVAL_WORKSPACE}/run"*.sh 2>/dev/null; then
  echo "❌ 安全检查: 发现硬编码 API key"
  fail=$((fail + 1))
else
  echo "✅ 安全检查: 无硬编码 API key"
  ok=$((ok + 1))
fi

# Python 语法
for py in "${EVAL_DIR}"/*.py; do
  if python3 -m py_compile "${py}" 2>/dev/null; then
    echo "✅ syntax: $(basename "${py}")"
    ok=$((ok + 1))
  else
    echo "❌ syntax: $(basename "${py}")"
    fail=$((fail + 1))
  fi
done

echo ""
echo "Pass: ${ok}  Fail: ${fail}"
[[ "${fail}" -eq 0 ]]
