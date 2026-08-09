# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS upstream compatibility shims."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import masking_utils
from transformers.models.llama.configuration_llama import LlamaConfig

# Note (Akazaakane): loaded by path so the module under test never imports sglang.
_COMPAT_PATH = (
    Path(__file__).resolve().parents[3] / "sglang_omni/models/qwen3_tts/compat.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen3_tts_compat", _COMPAT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)

# Note (Akazaakane): must be captured before any test applies the patch.
_PRISTINE = {name: getattr(masking_utils, name) for name in compat._MASK_FACTORY_NAMES}
# Note (Akazaakane): the factories qwen-tts splats mask_kwargs into, per the
# Qwen3TTSTokenizerV2 decoder forward.
_SHIMMED_FACTORIES = (
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "create_chunked_causal_mask",
)


@pytest.fixture
def pristine_masking_utils(monkeypatch: pytest.MonkeyPatch):
    for name, original in _PRISTINE.items():
        monkeypatch.setattr(masking_utils, name, original)
    return masking_utils


def _mask_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=8, num_attention_heads=2, num_hidden_layers=1, vocab_size=16
    )
    config._attn_implementation = "eager"
    config.sliding_window = 2
    config.attention_chunk_size = 2
    return config


def _qwen_tts_kwargs(config: LlamaConfig) -> dict:
    return {
        "config": config,
        "input_embeds": torch.zeros(1, 4, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "cache_position": torch.arange(4),
        "past_key_values": None,
    }


def _supported_kwargs(config: LlamaConfig) -> dict:
    return {
        "config": config,
        "inputs_embeds": torch.zeros(1, 4, 8),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "past_key_values": None,
    }


@pytest.mark.parametrize("name", _SHIMMED_FACTORIES)
def test_unpatched_transformers_rejects_the_qwen_tts_call_shape(
    pristine_masking_utils, name: str
) -> None:
    with pytest.raises(TypeError, match="input_embeds"):
        getattr(pristine_masking_utils, name)(**_qwen_tts_kwargs(_mask_config()))


@pytest.mark.parametrize("name", _SHIMMED_FACTORIES)
def test_shim_accepts_input_embeds_and_absorbs_cache_position(
    pristine_masking_utils, name: str
) -> None:
    config = _mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    shimmed = getattr(pristine_masking_utils, name)(**_qwen_tts_kwargs(config))
    expected = _PRISTINE[name](**_supported_kwargs(config))

    assert shimmed is not None
    torch.testing.assert_close(shimmed, expected)


@pytest.mark.parametrize("name", _SHIMMED_FACTORIES)
def test_shim_passes_the_supported_call_shape_through(
    pristine_masking_utils, name: str
) -> None:
    config = _mask_config()
    compat.apply_qwen_tts_transformers_compatibility_patches()

    torch.testing.assert_close(
        getattr(pristine_masking_utils, name)(**_supported_kwargs(config)),
        _PRISTINE[name](**_supported_kwargs(config)),
    )


def test_shim_is_idempotent(pristine_masking_utils) -> None:
    compat.apply_qwen_tts_transformers_compatibility_patches()
    patched = {
        name: getattr(pristine_masking_utils, name) for name in _SHIMMED_FACTORIES
    }
    for name, fn in patched.items():
        assert fn is not _PRISTINE[name]

    compat.apply_qwen_tts_transformers_compatibility_patches()

    for name, fn in patched.items():
        assert getattr(pristine_masking_utils, name) is fn


def test_shim_is_a_no_op_when_transformers_already_accepts_input_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def already_compatible(config, input_embeds, cache_position=None):  # noqa: ARG001
        return None

    monkeypatch.setattr(masking_utils, "create_causal_mask", already_compatible)

    compat.apply_qwen_tts_transformers_compatibility_patches()

    assert masking_utils.create_causal_mask is already_compatible
