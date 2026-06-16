# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeProcessor:
    """Records every encode_audios_from_path call so tests can inspect batch sizes."""

    def __init__(self, tokens_per_path: dict[str, list[int]] | None = None) -> None:
        self.call_sizes: list[int] = []
        self._lock = threading.Lock()
        self._tokens = tokens_per_path or {}

    def encode_audios_from_path(self, paths: list[str]) -> list[torch.Tensor]:
        with self._lock:
            self.call_sizes.append(len(paths))
        return [
            torch.tensor(self._tokens.get(p, [0]), dtype=torch.long) for p in paths
        ]


# ---------------------------------------------------------------------------
# _BatchedReferenceEncoder
# ---------------------------------------------------------------------------


def test_batched_encoder_always_uses_b1_per_path() -> None:
    """Worker must call encode_audios_from_path with exactly one path at a time.

    Before the fix the worker would call encode_audios_from_path(unique_paths)
    with B>1 when multiple paths coalesced in one drain. That violates the
    content-addressed cache invariant because BF16 linear ops are batch-shape-
    sensitive.
    """
    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    processor = _FakeProcessor()
    encoder = _BatchedReferenceEncoder(
        processor, max_batch_size=8, max_batch_wait_ms=20
    )

    paths = [f"audio_{i}.wav" for i in range(5)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(encoder.encode, p) for p in paths]
        [f.result(timeout=10) for f in futs]

    assert processor.call_sizes, "no encode calls recorded"
    assert all(n == 1 for n in processor.call_sizes), (
        f"Expected all B=1 encodes; got sizes: {processor.call_sizes}"
    )


def test_batched_encoder_deduplicates_same_path_in_one_drain() -> None:
    """Duplicate paths in a single drain batch must produce only one encode call."""
    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    processor = _FakeProcessor(tokens_per_path={"dup.wav": [42, 43]})
    encoder = _BatchedReferenceEncoder(
        processor, max_batch_size=8, max_batch_wait_ms=20
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(encoder.encode, "dup.wav") for _ in range(4)]
        results = [f.result(timeout=10) for f in futs]

    assert len(processor.call_sizes) == 1, (
        f"Expected 1 encode for 4 identical paths; got {len(processor.call_sizes)}"
    )
    assert all(r.tolist() == [42, 43] for r in results)


def test_batched_encoder_isolates_per_path_failures() -> None:
    """A failing path must not propagate its exception to other paths in the batch."""
    from sglang_omni.models.moss_tts_local.stages import _BatchedReferenceEncoder

    class _FailingProcessor:
        def encode_audios_from_path(self, paths: list[str]) -> list[torch.Tensor]:
            assert len(paths) == 1
            if "bad" in paths[0]:
                raise RuntimeError("codec failure")
            return [torch.tensor([99], dtype=torch.long)]

    encoder = _BatchedReferenceEncoder(
        _FailingProcessor(), max_batch_size=8, max_batch_wait_ms=20
    )

    good_future: concurrent.futures.Future = concurrent.futures.Future()
    bad_future: concurrent.futures.Future = concurrent.futures.Future()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {
            "good": pool.submit(encoder.encode, "good.wav"),
            "bad": pool.submit(encoder.encode, "bad.wav"),
        }
        good_result = futs["good"].result(timeout=10)
        with pytest.raises(RuntimeError, match="codec failure"):
            futs["bad"].result(timeout=10)

    assert good_result.tolist() == [99]


# ---------------------------------------------------------------------------
# CachedReferenceEncoder
# ---------------------------------------------------------------------------


def _make_cached_encoder(
    tokens_per_path: dict[str, list[int]] | None = None,
    *,
    max_items: int = 16,
    max_bytes: int = 1024 * 1024,
):
    from sglang_omni.models.moss_tts_local.stages import (
        CachedReferenceEncoder,
        _BatchedReferenceEncoder,
    )

    processor = _FakeProcessor(tokens_per_path)
    inner = _BatchedReferenceEncoder(processor, max_batch_size=8, max_batch_wait_ms=20)
    cached = CachedReferenceEncoder(inner, max_items=max_items, max_bytes=max_bytes)
    return cached, processor


def test_cache_hit_returns_same_tokens_as_miss() -> None:
    """Cache hit must return the same token values as the original miss fill."""
    cached, _ = _make_cached_encoder({"ref.wav": [10, 20, 30]})

    with patch.object(
        type(cached._encoder._processor).__mro__[0],
        "encode_audios_from_path",
        side_effect=lambda paths: cached._encoder._processor.encode_audios_from_path(
            paths
        ),
    ):
        pass  # just use the real fake

    miss_result = cached.encode("ref.wav")
    hit_result = cached.encode("ref.wav")

    assert miss_result.tolist() == [10, 20, 30]
    assert hit_result.tolist() == [10, 20, 30]

    stats = cached.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1


def test_cache_hit_returns_independent_tensor_copy() -> None:
    """Each call must return a fresh tensor; mutations must not affect the cache."""
    cached, _ = _make_cached_encoder({"ref.wav": [1, 2, 3]})

    result1 = cached.encode("ref.wav")
    result1[0] = 999  # mutate the returned tensor

    result2 = cached.encode("ref.wav")
    assert result2.tolist() == [1, 2, 3], "cache returned shared/mutated tensor"


def test_cache_single_flight_dedup_merges_concurrent_misses() -> None:
    """Concurrent requests for the same uncached path must share one encode call."""
    barrier = threading.Barrier(4)
    processor = _FakeProcessor(tokens_per_path={"shared.wav": [7, 8]})

    original_encode = processor.encode_audios_from_path

    def slow_encode(paths: list[str]) -> list[torch.Tensor]:
        barrier.wait(timeout=5)
        return original_encode(paths)

    processor.encode_audios_from_path = slow_encode  # type: ignore[method-assign]

    from sglang_omni.models.moss_tts_local.stages import (
        CachedReferenceEncoder,
        _BatchedReferenceEncoder,
    )

    inner = _BatchedReferenceEncoder(processor, max_batch_size=8, max_batch_wait_ms=20)
    cached = CachedReferenceEncoder(inner, max_items=16, max_bytes=1024 * 1024)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(cached.encode, "shared.wav") for _ in range(4)]
        results = [f.result(timeout=10) for f in futs]

    assert all(r.tolist() == [7, 8] for r in results)
    stats = cached.stats()
    # 1 miss (leader) + 3 merged (followers) = 4 total
    assert stats["misses"] == 1
    assert stats["merged"] == 3


def test_cache_dtype_is_long_on_all_paths() -> None:
    """Both miss and hit returns must be dtype=torch.long (not int32)."""
    cached, _ = _make_cached_encoder({"ref.wav": [5, 6]})

    miss = cached.encode("ref.wav")
    hit = cached.encode("ref.wav")

    assert miss.dtype == torch.long
    assert hit.dtype == torch.long
