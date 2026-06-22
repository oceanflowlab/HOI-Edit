#!/usr/bin/env bash
# 检查工作区依赖是否就绪（简化版，完整检查见 verify_scripts.sh）
set -euo pipefail
_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${_EVAL_DIR}/verify_scripts.sh"
