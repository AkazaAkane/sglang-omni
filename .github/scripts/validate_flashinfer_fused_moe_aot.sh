#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

VENV_NAME="$1"
PYTHON="${VENV_NAME}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(mktemp -d)"
TRACE="${WORKSPACE}/fused-moe-aot.exec"

trap 'rm -rf "${WORKSPACE}"' EXIT

strace -f \
  -e trace=execve \
  -o "${TRACE}" \
  env \
    FLASHINFER_WORKSPACE_BASE="${WORKSPACE}" \
    FLASHINFER_DISABLE_JIT=1 \
  "${PYTHON}" "${SCRIPT_DIR}/validate_flashinfer_fused_moe_aot.py"

if grep -E 'execve\(".*(/|^)ninja"' "${TRACE}"; then
  echo "Unexpected Ninja invocation" >&2
  exit 1
fi

if grep -E 'execve\(".*(nvcc|ptxas)"' "${TRACE}" | grep -v -- '--version'; then
  echo "Unexpected CUDA compilation or assembly invocation" >&2
  exit 1
fi

if grep -E \
  'execve\(".*(cicc|cc1plus|collect2|clang\+\+|g\+\+|/c\+\+|/ld)"' \
  "${TRACE}"; then
  echo "Unexpected compiler or linker invocation" >&2
  exit 1
fi

MODULE_WORKSPACE="${WORKSPACE}/.cache/flashinfer/0.6.14/90a/cached_ops/fused_moe_90"
test ! -e "${MODULE_WORKSPACE}/build.ninja"
test ! -e "${MODULE_WORKSPACE}/fused_moe_90.so"
