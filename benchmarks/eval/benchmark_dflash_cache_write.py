# SPDX-License-Identifier: Apache-2.0
"""Measure compiled DFlash cache writes with shared and separate K/V storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch._functorch.config as functorch_config
from sglang.kernels.ops.kvcache.kvcache import store_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-tokens", type=int, default=65536)
    parser.add_argument("--write-tokens", type=int, default=8)
    parser.add_argument("--row-elements", type=int, default=1024)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("compiled_shared", "compiled_separate", "boundary_shared"),
        default="compiled_shared",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not 0 < arguments.write_tokens < arguments.cache_tokens:
        parser.error("write-tokens must leave room for the reserved zero cache slot")
    elif arguments.row_elements <= 0 or arguments.replays <= 0:
        parser.error("row-elements and replays must be positive")
    else:
        pass
    torch.manual_seed(0)
    functorch_config.enable_autograd_cache = False
    key_states = torch.randn(
        arguments.write_tokens,
        arguments.row_elements,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_states = torch.randn_like(key_states)
    token_indices = torch.arange(1, arguments.write_tokens + 1, device="cuda")
    results: list[dict[str, str | int | float]] = []
    mode = arguments.mode
    if mode == "compiled_separate":
        key_cache = torch.ones(
            arguments.cache_tokens,
            arguments.row_elements,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value_cache = torch.ones_like(key_cache)
    else:
        cache_storage = torch.ones(
            2,
            arguments.cache_tokens,
            arguments.row_elements,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_cache, value_cache = cache_storage.unbind()
    cache_store = (
        torch.compiler.disable(store_cache)
        if mode == "boundary_shared"
        else store_cache
    )

    def write_cache(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        row_bytes = key_states.shape[-1] * key_states.element_size()
        cache_store(
            key_states,
            value_states,
            key_cache,
            value_cache,
            token_indices,
            row_bytes=row_bytes,
            v_row_bytes=row_bytes,
            size_limit=key_cache.shape[0],
        )
        return key_states + value_states

    cache_writer = torch.compile(
        write_cache, dynamic=False, options={"fx_graph_cache": False}
    )
    for _ in range(3):
        cache_writer(key_states, value_states, key_cache, value_cache, token_indices)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = cache_writer(
            key_states, value_states, key_cache, value_cache, token_indices
        )
    graph.replay()
    torch.testing.assert_close(output, key_states + value_states)
    expected_key_cache = torch.ones_like(key_cache)
    expected_value_cache = torch.ones_like(value_cache)
    expected_key_cache.index_copy_(0, token_indices, key_states)
    expected_value_cache.index_copy_(0, token_indices, value_states)
    torch.testing.assert_close(key_cache, expected_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value_cache, rtol=0, atol=0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(arguments.replays):
        graph.replay()
    end.record()
    end.synchronize()
    results.append(
        {
            "mode": mode,
            "cache_tokens": arguments.cache_tokens,
            "write_tokens": arguments.write_tokens,
            "row_elements": arguments.row_elements,
            "replay_microseconds": start.elapsed_time(end) * 1000 / arguments.replays,
        }
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
