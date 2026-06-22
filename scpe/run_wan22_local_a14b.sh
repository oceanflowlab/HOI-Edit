#!/usr/bin/env bash
# 本地 Wan2.2 I2V A14B 批量生成（原版仓库 + 本地权重）
#
# 环境变量（见 env.example）:
#   WAN22_REPO, WAN22_CKPT_DIR, WAN22_DEVICE, WAN22_RUN_MODE, DATA_ROOT
#
# 用法:
#   source env.local
#   ./run_wan22_local_a14b.sh
#   LIMIT=2 SPLIT=L1L2 ./run_wan22_local_a14b.sh
#   WAN22_RUN_MODE=subprocess WAN22_LAUNCHER=torchrun WAN22_NPROC=8 \
#     WAN22_SUBPROCESS_EXTRA="--dit_fsdp --t5_fsdp --cfg_size 2 --ulysses_size 4 --vae_parallel" \
#     ./run_wan22_local_a14b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/env.local" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/env.local"
fi

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

ENHANCED_JSON="${ENHANCED_JSON:-${ROOT}/output/enhanced_prompts.json}"
OUT_L1L2="${OUT_L1L2:-${ROOT}/output/wan22_local_a14b/L1L2}"
OUT_L3="${OUT_L3:-${ROOT}/output/wan22_local_a14b/L3}"
SPLIT="${SPLIT:-all}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${WAN22_SIZE:-}" ]] && EXTRA+=(--size "$WAN22_SIZE")
[[ "${FORCE_REGEN:-0}" == "1" ]] && EXTRA+=(--no-skip-existing)
[[ -n "${WAN22_RUN_MODE:-}" ]] && EXTRA+=(--mode "$WAN22_RUN_MODE")
[[ -n "${WAN22_REPO:-}" ]] && EXTRA+=(--wan-repo "$WAN22_REPO")
[[ -n "${WAN22_CKPT_DIR:-}" ]] && EXTRA+=(--ckpt-dir "$WAN22_CKPT_DIR")

exec "$PYTHON" "${ROOT}/scripts/wan22_local_i2v_a14b_generate.py" \
  --enhanced-prompts-json "$ENHANCED_JSON" \
  --output-dir-l1l2 "$OUT_L1L2" \
  --output-dir-l3 "$OUT_L3" \
  --split "$SPLIT" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
