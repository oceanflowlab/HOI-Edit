#!/usr/bin/env bash
# 最基本 ACE I2V 流程（3 步，可分段跑）
#
#   learn     Round1: 从参考视频学 Playbook (epoch0)
#   enhance   Round2: 用 Playbook 生成 enhanced_prompt
#   wan22     Wan2.2 图生视频
#   qa2       对生成视频抽帧评估（可选）
#   all       依次 learn → enhance → wan22 → qa2
#
# 准备:
#   ./setup_env.sh
#   cp env.example env.local && 编辑 API Key
#   source env.local
#
# 调试（每 split 只跑 2 条）:
#   LIMIT=2 ./run_minimal.sh all
#
# 跳过已完成的轮次:
#   SKIP_LEARN=1 ./run_minimal.sh enhance
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="${ROOT}/scripts"
OUT="${ROOT}/output"
LOG="${ROOT}/logs"
mkdir -p "$OUT" "$LOG"

# 加载密钥与 DATA_ROOT
if [[ -f "${ROOT}/env.local" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/env.local"
fi
DATA_ROOT="${DATA_ROOT:-$(cd "${ROOT}/../data" && pwd)}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  elif [[ -x "${ROOT}/../.venv/bin/python" ]]; then
    PYTHON="${ROOT}/../.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

PLAYBOOK="${PLAYBOOK:-${OUT}/playbook.json}"
PLAYBOOK_SEED="${PLAYBOOK_SEED:-${ROOT}/data/playbook_seed_${ACE_LANG:-cn}.json}"
PROMPTS_JSON="${PROMPTS_JSON:-${OUT}/enhanced_prompts.json}"
WAN22_OUT="${WAN22_OUT:-${OUT}/wan22_videos}"
QA2_OUT="${QA2_OUT:-${OUT}/qa2_frames}"

JSON_L1L2="${JSON_L1L2:-${DATA_ROOT}/collected_annotations_bboxes_v7_L1L2_questions.json}"
JSON_L3="${JSON_L3:-${DATA_ROOT}/collected_annotations_bboxes_v7_L3_questions.json}"
IMG_L1L2="${IMG_L1L2:-${DATA_ROOT}/data_v7_L12}"
IMG_L3="${IMG_L3:-${DATA_ROOT}/data_v7_L3}"
VID_EPOCH0_L1L2="${VID_EPOCH0_L1L2:-${DATA_ROOT}/epoch_0_L1L2}"
VID_EPOCH0_L3="${VID_EPOCH0_L3:-${DATA_ROOT}/epoch_0_L3}"

LIMIT_FLAG=()
[[ -n "${LIMIT:-}" ]] && LIMIT_FLAG=(--limit "$LIMIT")

run_py() {
  env -u all_proxy -u ALL_PROXY \
    HTTP_PROXY="${HTTP_PROXY:-}" \
    HTTPS_PROXY="${HTTPS_PROXY:-}" \
    http_proxy="${http_proxy:-}" \
    https_proxy="${https_proxy:-}" \
    GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
    DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}" \
    ACE_LANG="${ACE_LANG:-cn}" \
    "$PYTHON" "$@"
}

preflight_gemini() {
  [[ -n "${GEMINI_API_KEY:-}" ]] || { echo "❗ 需要 GEMINI_API_KEY（env.local）"; exit 1; }
}

preflight_dashscope() {
  [[ -n "${DASHSCOPE_API_KEY:-}" ]] || { echo "❗ 需要 DASHSCOPE_API_KEY（env.local）"; exit 1; }
}

init_playbook() {
  if [[ "${SKIP_PLAYBOOK_INIT:-0}" == "1" && -f "$PLAYBOOK" ]]; then
    echo "📘 续跑，保留 Playbook: $PLAYBOOK"
    return
  fi
  if [[ -f "$PLAYBOOK_SEED" ]]; then
    cp "$PLAYBOOK_SEED" "$PLAYBOOK"
    echo "📘 从 seed 初始化: $PLAYBOOK_SEED -> $PLAYBOOK"
  else
    echo "📘 无 seed，由 ace_i2v_official3 创建默认 3 条 Playbook"
  fi
}

run_learn_split() {
  local split="$1" json img vid
  case "$split" in
    L1L2) json="$JSON_L1L2" img="$IMG_L1L2" vid="$VID_EPOCH0_L1L2" ;;
    L3)   json="$JSON_L3"   img="$IMG_L3"   vid="$VID_EPOCH0_L3" ;;
    *) echo "未知 split: $split"; exit 1 ;;
  esac
  [[ -f "$json" ]] || { echo "❗ 缺少标注: $json"; exit 1; }
  [[ -d "$img" ]] || { echo "❗ 缺少原图: $img"; exit 1; }
  [[ -d "$vid" ]] || { echo "❗ 缺少参考视频: $vid"; exit 1; }

  echo "=== LEARN [$split] epoch0 ==="
  run_py "${SCRIPTS}/ace_i2v_official3.py" --mode epoch0_only \
    --json-path "$json" \
    --image-dir "$img" \
    --original-video-dir "$vid" \
    --playbook-file "$PLAYBOOK" \
    --ace-lang "${ACE_LANG:-cn}" \
    --trace-dir "${LOG}/traces_${split}" \
    --save-every "${SAVE_EVERY:-10}" \
    ${LIMIT_FLAG+"${LIMIT_FLAG[@]}"} \
    2>&1 | tee -a "${LOG}/learn_${split}.log"
}

run_enhance_split() {
  local split="$1" json img out_part
  case "$split" in
    L1L2) json="$JSON_L1L2" img="$IMG_L1L2" out_part="${LOG}/enhanced_L1L2.json" ;;
    L3)   json="$JSON_L3"   img="$IMG_L3"   out_part="${LOG}/enhanced_L3.json" ;;
    *) echo "未知 split: $split"; exit 1 ;;
  esac
  [[ -f "$PLAYBOOK" ]] || { echo "❗ 先跑 learn 或指定 PLAYBOOK"; exit 1; }

  echo "=== ENHANCE [$split] ==="
  run_py "${SCRIPTS}/ace_i2v_official3.py" --mode enhance_prompts_only \
    --json-path "$json" \
    --image-dir "$img" \
    --playbook-file "$PLAYBOOK" \
    --ace-lang "${ACE_LANG:-cn}" \
    --output-prompts-json "$out_part" \
    ${LIMIT_FLAG+"${LIMIT_FLAG[@]}"} \
    2>&1 | tee -a "${LOG}/enhance_${split}.log"
}

merge_prompts() {
  echo "=== 合并 enhanced prompts -> $PROMPTS_JSON ==="
  run_py - <<PY
import json
from pathlib import Path
parts = {
    "L1L2": Path("${LOG}/enhanced_L1L2.json"),
    "L3": Path("${LOG}/enhanced_L3.json"),
}
by_split = {}
for split, path in parts.items():
    if path.exists():
        by_split[split] = json.loads(path.read_text(encoding="utf-8"))
out = {
    "_meta": {"format": "by_split", "source": "ace_i2v_basic/run_minimal.sh"},
    **by_split,
}
Path("${PROMPTS_JSON}").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"L1L2={len(by_split.get('L1L2',{}))} L3={len(by_split.get('L3',{}))} -> ${PROMPTS_JSON}")
PY
}

run_wan22() {
  [[ -f "$PROMPTS_JSON" ]] || { echo "❗ 先跑 enhance"; exit 1; }
  if [[ "${WAN22_BACKEND:-dashscope}" == "local" ]]; then
    echo "=== WAN2.2 LOCAL A14B -> ${WAN22_OUT} ==="
    ENHANCED_JSON="$PROMPTS_JSON" OUT_L1L2="${WAN22_OUT}/L1L2" OUT_L3="${WAN22_OUT}/L3" \
      "${ROOT}/run_wan22_local_a14b.sh" 2>&1 | tee -a "${LOG}/wan22_local.log"
    return
  fi
  preflight_dashscope
  echo "=== WAN2.2 DashScope API -> $WAN22_OUT ==="
  run_py "${SCRIPTS}/wan22_generate_from_enhanced_prompts.py" \
    --enhanced-prompts-json "$PROMPTS_JSON" \
    --image-dir-l1l2 "$IMG_L1L2" \
    --image-dir-l3 "$IMG_L3" \
    --output-dir-l1l2 "${WAN22_OUT}/L1L2" \
    --output-dir-l3 "${WAN22_OUT}/L3" \
    --split "${SPLIT:-all}" \
    ${LIMIT_FLAG+"${LIMIT_FLAG[@]}"} \
    ${FORCE_REGEN:+--no-skip-existing} \
    2>&1 | tee -a "${LOG}/wan22.log"
}

run_qa2_split() {
  local split="$1" json vid out
  case "$split" in
    L1L2) json="$JSON_L1L2" vid="${WAN22_OUT}/L1L2" out="${QA2_OUT}/L1L2" ;;
    L3)   json="$JSON_L3"   vid="${WAN22_OUT}/L3"   out="${QA2_OUT}/L3" ;;
    *) echo "未知 split: $split"; exit 1 ;;
  esac
  mkdir -p "$out"
  echo "=== QA2 [$split] -> $out ==="
  run_py "${SCRIPTS}/ace_v2f_qa2.py" \
    --json-path "$json" \
    --video-dir "$vid" \
    --fallback-video-dir "$vid" \
    --output-dir "$out" \
    --num-frames "${NUM_FRAMES:-15}" \
    --ace-lang "${ACE_LANG:-cn}" \
    2>&1 | tee -a "${LOG}/qa2_${split}.log"
}

phase_learn() {
  preflight_gemini
  init_playbook
  run_learn_split L1L2
  run_learn_split L3
}

phase_enhance() {
  local pb_norm="${OUT}/playbook_normalized.json"
  if [[ -f "$pb_norm" ]]; then
    echo "📘 enhance 使用 normalized Playbook: $pb_norm"
    PLAYBOOK="$pb_norm"
  fi
  preflight_gemini
  run_enhance_split L1L2
  run_enhance_split L3
  merge_prompts
}

phase_wan22() { run_wan22; }

phase_qa2() {
  preflight_gemini
  run_qa2_split L1L2
  run_qa2_split L3
}

CMD="${1:-all}"
case "$CMD" in
  learn)
    [[ "${SKIP_LEARN:-0}" == "1" ]] && echo "⏩ SKIP_LEARN=1" && exit 0
    phase_learn ;;
  enhance)
    phase_enhance ;;
  wan22)
    phase_wan22 ;;
  qa2)
    phase_qa2 ;;
  all)
    [[ "${SKIP_LEARN:-0}" != "1" ]] && phase_learn
    phase_enhance
    phase_wan22
    [[ "${SKIP_QA2:-0}" != "1" ]] && phase_qa2 ;;
  *)
    echo "用法: $0 {learn|enhance|wan22|qa2|all}"
    exit 1 ;;
esac

echo ""
echo "✅ 完成 ($CMD)"
echo "   DATA_ROOT:  $DATA_ROOT"
echo "   Playbook:   $PLAYBOOK"
echo "   Prompts:    $PROMPTS_JSON"
echo "   Videos:     $WAN22_OUT"
echo "   QA2:        $QA2_OUT"
echo "   Logs:       $LOG"
