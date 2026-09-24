# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.utils.snake_beta as vocoder_kernels


class _StubSnakeBeta(torch.nn.Module):
    """Stand-in with the qwen-tts SnakeBeta attribute layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_features = channels
        self.alpha = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )


_StubSnakeBeta.__name__ = "SnakeBeta"


def test_fuse_vocoder_decoder_keeps_originals_on_prewarm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _StubSnakeBeta(4)
    second = _StubSnakeBeta(4)
    decoder = torch.nn.Sequential(first, torch.nn.Sequential(second))

    monkeypatch.setattr(vocoder_kernels, "HAS_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_prewarm(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prewarm failed")

    monkeypatch.setattr(vocoder_kernels, "prewarm_replacements", fail_prewarm)

    assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 0
    assert decoder[0] is first
    assert decoder[1][0] is second


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused SnakeBeta parity needs CUDA"
)
@pytest.mark.parametrize(
    ("batch", "channels", "frames"),
    [
        (1, 96, 33),
        (2, 192, 96),
        (1, 384, 192),
        (1, 768, 257),
        (1, 96, 65536),
        (1, 96, 122880),
        (4, 1536, 1),
        (8, 384, 64),
        (16, 96, 37845),
        (32, 96, 64),
        (2, 4, 33),
    ],
)
def test_fused_snake_beta_cuda_parity_uses_kernel(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    channels: int,
    frames: int,
) -> None:
    # note (db-ol): on the accelerator runner a missing Triton must fail
    # loudly, a skip here would hide the kernel from CI again.
    assert vocoder_kernels.HAS_TRITON, "Triton is required on accelerator CI"

    torch.manual_seed(0)
    device = torch.device("cuda")
    original = _StubSnakeBeta(channels).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(
        (batch, channels, frames),
        device=device,
        dtype=torch.bfloat16,
    )
    expected = original(x)
    launches: list[tuple[int, int, int]] = []
    original_launch = vocoder_kernels.launch

    def record_launch(
        hidden_states: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        launches.append(tuple(hidden_states.shape))
        return original_launch(hidden_states, alpha, beta, eps)

    monkeypatch.setattr(vocoder_kernels, "launch", record_launch)

    actual = vocoder_kernels.fused_snake_beta(
        x, original.alpha, original.beta, original.no_div_by_zero
    )

    assert actual is not None
    assert launches == [(batch, channels, frames)]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shared_snake_preserves_parameters_and_cpu_fallback(dtype: torch.dtype) -> None:
    original = _StubSnakeBeta(96).to(dtype=dtype).eval()
    decoder = torch.nn.Sequential(original)
    x = torch.randn(2, 96, 17).to(dtype=dtype)
    expected = original(x)
    state = {name: value.clone() for name, value in decoder.state_dict().items()}

    assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 1
    assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 0
    assert decoder[0].alpha is original.alpha
    assert decoder[0].beta is original.beta
    assert not decoder[0].training
    assert decoder.state_dict().keys() == state.keys()
    assert all(
        torch.equal(value, state[name]) for name, value in decoder.state_dict().items()
    )
    assert torch.equal(decoder(x), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_uses_the_module_epsilon() -> None:
    original = _StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    original.no_div_by_zero = 1e-3
    decoder = torch.nn.Sequential(original)
    x = torch.randn(1, 96, 257, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(x)
        assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 1
        assert torch.equal(decoder(x), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["dtype", "layout", "empty", "length"])
def test_shared_snake_cuda_falls_back_outside_envelope(kind: str) -> None:
    batch, channels, frames = 2, 96, 17
    if kind == "empty":
        frames = 0
    elif kind == "length":
        batch, channels, frames = 1, 1, vocoder_kernels.MAX_T + 1
    dtype = torch.float32 if kind == "dtype" else torch.bfloat16
    original = _StubSnakeBeta(channels).to(device="cuda", dtype=dtype).eval()
    x = torch.randn(batch, channels, frames, device="cuda", dtype=dtype)
    if kind == "layout":
        x = x[..., ::2]
    with torch.inference_mode():
        assert (
            vocoder_kernels.fused_snake_beta(
                x, original.alpha, original.beta, original.no_div_by_zero
            )
            is None
        )
        assert torch.equal(vocoder_kernels.FusedSnakeBeta(original)(x), original(x))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_bf16_encodings_and_denormals() -> None:
    original = _StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    encodings = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = encodings.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = values.repeat(96).reshape(1, 96, -1)
    with torch.inference_mode():
        original.alpha.copy_(
            torch.tensor([-90.0, -80.0, 0.0], device="cuda").repeat(32)
        )
        original.beta.copy_(torch.tensor([-90.0, 0.0, 80.0], device="cuda").repeat(32))
        actual = vocoder_kernels.fused_snake_beta(
            x, original.alpha, original.beta, original.no_div_by_zero
        )
        assert actual is not None
        assert torch.equal(actual, original(x))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_prewarm_covers_new_shapes_and_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    vocoder_kernels.prewarm(device)
    compile_kernel = Mock(side_effect=AssertionError("runtime Triton compilation"))
    monkeypatch.setattr(vocoder_kernels.snake_beta_kernel, "compile", compile_kernel)
    with torch.inference_mode():
        for index, frames in enumerate(
            (1, 16, 33, 64, 65, 96, 128, 129, 192, 256, 257, 1024, 122880)
        ):
            channels = (96, 192, 384, 768, 1536)[index % 5]
            original = (
                _StubSnakeBeta(channels).to(device=device, dtype=torch.bfloat16).eval()
            )
            batch = 16 if frames <= 1024 else 1
            x = torch.randn(
                batch, channels, frames, device=device, dtype=torch.bfloat16
            )
            expected = original(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = vocoder_kernels.fused_snake_beta(
                    x, original.alpha, original.beta, original.no_div_by_zero
                )
            graph.replay()
            assert actual is not None
            assert torch.equal(actual, expected)
    compile_kernel.assert_not_called()


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused SnakeBeta compile needs CUDA"
)
def test_fused_snake_beta_survives_a_fullgraph_compile() -> None:
    """A fullgraph compile of a fused decoder raised Unsupported on the launch."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    original = _StubSnakeBeta(96).to(device=device, dtype=torch.bfloat16)
    decoder = torch.nn.Sequential(original)
    x = torch.randn((2, 96, 320), device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(x)

        assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 1
        compiled = torch.compile(decoder, dynamic=False, fullgraph=True)

        assert torch.equal(compiled(x), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_graph_reads_current_inputs_and_parameters() -> None:
    original = _StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    decoder = torch.nn.Sequential(original)
    x = torch.zeros(1, 96, 33, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = decoder(x)
        for value in (0.1, -3.0):
            x.fill_(value)
            original.alpha.fill_(value)
            original.beta.fill_(-value)
            graph.replay()
            assert torch.equal(actual, original(x))


@pytest.mark.benchmark
@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_tts_decoder_and_incremental_pcm_equal() -> None:
    checkpoint = os.environ.get("QWEN3_TTS_TOKENIZER_PATH")
    if checkpoint is None:
        pytest.skip("Set QWEN3_TTS_TOKENIZER_PATH to run the real checkpoint gate")
    from sglang_omni.models.qwen3_tts.compat import (
        apply_qwen_tts_transformers_compatibility_patches,
    )
    from sglang_omni.models.qwen3_tts.incremental_codec import (
        Qwen3TTSIncrementalCodecState,
        Qwen3TTSIncrementalDecoder,
    )

    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        checkpoint,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    codes = [
        torch.randint(
            decoder.config.codebook_size,
            (batch, decoder.config.num_quantizers, frames),
            device="cuda:0",
            generator=generator,
        )
        for batch, frames in ((1, 2), (1, 24), (1, 35), (8, 24))
    ]
    with torch.inference_mode():
        expected = [decoder(value).clone() for value in codes]
        incremental = Qwen3TTSIncrementalDecoder(decoder)
        state = Qwen3TTSIncrementalCodecState()
        parts = codes[1].split((2, 6, 8, 8), dim=-1)
        incremental_expected = [
            incremental.decode(part, state).clone() for part in parts
        ]

        assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 29
        for value, pcm in zip(codes, expected):
            assert torch.equal(decoder(value), pcm), tuple(value.shape)
        incremental = Qwen3TTSIncrementalDecoder(decoder)
        state = Qwen3TTSIncrementalCodecState()
        for part, pcm in zip(parts, incremental_expected):
            assert torch.equal(incremental.decode(part, state), pcm)
