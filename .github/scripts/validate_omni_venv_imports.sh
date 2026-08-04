#!/usr/bin/env bash
# Import probe for the Omni CI venv (matches packages exercised in real CI jobs).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

if [ -z "${OMNI_CI_HOME:-}" ]; then
  echo "OMNI_CI_HOME is not set" >&2
  exit 1
fi

VENV_NAME="$1"
PYTHON="${OMNI_CI_HOME}/${VENV_NAME}/bin/python"

if [ ! -x "${PYTHON}" ]; then
  echo "python not found: ${PYTHON}" >&2
  exit 1
fi

if ! "${PYTHON}" -c "
import hashlib

import av
import flashinfer_jit_cache
import torch
import transformers
import sglang
import zhon.hanzi
from whisper.normalizers import EnglishTextNormalizer

assert flashinfer_jit_cache.__version__ == '0.6.14+cu130.omni1'
artifact = (
    flashinfer_jit_cache.get_jit_cache_dir()
    + '/fused_moe_90/fused_moe_90.so'
)
with open(artifact, 'rb') as handle:
    assert hashlib.file_digest(handle, 'sha256').hexdigest() == (
        'ee5a6a87a13a271c0418e07ea76fd4cba'
        '5cbee1af2d7dab53890caacb9254e14'
    )
" 2>/dev/null; then
  echo "::error::${VENV_NAME} import probe failed at ${OMNI_CI_HOME}/${VENV_NAME}" >&2
  exit 1
fi

echo "Import probe ok: ${OMNI_CI_HOME}/${VENV_NAME}"
