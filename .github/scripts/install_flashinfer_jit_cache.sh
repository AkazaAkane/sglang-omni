#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

VENV_NAME="$1"
PYTHON="${VENV_NAME}/bin/python"
EXPECTED_VERSION="0.6.14+cu130.omni1"
WHEEL="/opt/flashinfer-wheels/flashinfer_jit_cache-${EXPECTED_VERSION}-py3-none-linux_x86_64.whl"

if INSTALLED_VERSION="$(
  "${PYTHON}" -c "from importlib.metadata import version; print(version('flashinfer-jit-cache'))" \
    2>/dev/null
)"; then
  if [ "${INSTALLED_VERSION}" = "${EXPECTED_VERSION}" ]; then
    echo "FlashInfer JIT cache already installed: ${INSTALLED_VERSION}"
    exit 0
  fi
  echo "Refusing to replace flashinfer-jit-cache ${INSTALLED_VERSION}" >&2
  exit 1
fi

if [ ! -f "${WHEEL}" ]; then
  echo "FlashInfer JIT cache wheel not found: ${WHEEL}" >&2
  exit 1
fi

uv pip install --python "${PYTHON}" --no-deps "${WHEEL}"
