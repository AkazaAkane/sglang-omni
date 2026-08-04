#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

VENV_NAME="$1"
WHEEL="/opt/flashinfer-wheels/flashinfer_jit_cache-0.6.14+cu130.omni1-py3-none-any.whl"

if [ ! -f "${WHEEL}" ]; then
  echo "FlashInfer JIT cache wheel not found: ${WHEEL}" >&2
  exit 1
fi

uv pip install --python "${VENV_NAME}/bin/python" --no-deps "${WHEEL}"
