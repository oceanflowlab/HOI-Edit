#!/usr/bin/env bash
# CR / V7 评测默认代理（Gemini Google API）
# 用法: source evaluation/cr_proxy_defaults.sh && apply_cr_proxy_if_unset

_WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${_WS_ROOT}/env/workspace.conf"

# 可在 env/local.conf 中设置 CR_DEFAULT_PROXY_URL；留空则不自动设代理
CR_DEFAULT_PROXY_URL="${CR_DEFAULT_PROXY_URL:-}"

_apply_cr_proxy_exports() {
  [[ -n "${CR_DEFAULT_PROXY_URL}" ]] || return 0
  export http_proxy="${CR_DEFAULT_PROXY_URL}"
  export https_proxy="${CR_DEFAULT_PROXY_URL}"
  export ftp_proxy="${CR_DEFAULT_PROXY_URL}"
  export all_proxy="${CR_DEFAULT_PROXY_URL}"
  export HTTP_PROXY="${http_proxy}"
  export HTTPS_PROXY="${https_proxy}"
  export FTP_PROXY="${ftp_proxy}"
  export ALL_PROXY="${all_proxy}"
}

apply_cr_proxy_if_unset() {
  if [[ -n "${http_proxy:-}" || -n "${HTTP_PROXY:-}" ]]; then
    return 0
  fi
  _apply_cr_proxy_exports
}

apply_cr_proxy_force() {
  _apply_cr_proxy_exports
}

unset_cr_proxy_env() {
  unset http_proxy https_proxy ftp_proxy all_proxy \
    HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY \
    no_proxy NO_PROXY \
    socks_proxy SOCKS_PROXY socks5_proxy SOCKS5_PROXY \
    GIT_HTTP_PROXY GIT_HTTPS_PROXY 2>/dev/null || true
}
