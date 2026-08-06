# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace


class FakeExecutionBridge:
    """SGLangExecutionBridge double for scheduler-owned ModelRunner tests."""

    def __init__(self) -> None:
        self.published: list[tuple[object, object]] = []
        self.isolate_sampling_calls: list[bool] = []

    @contextlib.contextmanager
    def forward_context(self, batch: object, *, isolate_sampling: bool = False):
        del batch
        self.isolate_sampling_calls.append(isolate_sampling)
        yield

    def publish_next_tokens(self, batch: object, next_token_ids: object) -> None:
        self.published.append((batch, next_token_ids))

    def record_completion(self):
        import torch

        return torch.cuda.Event()


class FakeServerArgs(SimpleNamespace):
    """ServerArgs double exposing the 0.5.16 override() mutation entry point."""

    def __init__(self, **fields: object) -> None:
        cuda_graph_config = fields.get("cuda_graph_config")
        if cuda_graph_config is None:
            cuda_graph_config = SimpleNamespace()
            fields["cuda_graph_config"] = cuda_graph_config
        prefill = getattr(cuda_graph_config, "prefill", None)
        if prefill is None:
            prefill = SimpleNamespace()
            cuda_graph_config.prefill = prefill
        defaults = {
            "backend": fields.get("cuda_graph_backend_prefill", "disabled"),
            "max_bs": fields.get("cuda_graph_max_bs_prefill", 0),
            "bs": fields.get("cuda_graph_bs_prefill", []),
            "tc_compiler": fields.get("cuda_graph_tc_compiler", "eager"),
        }
        for name, value in defaults.items():
            if not hasattr(prefill, name):
                setattr(prefill, name, value)
        super().__init__(**fields)

    def override(self, source: str, **fields: object) -> None:
        del source
        for name, value in fields.items():
            setattr(self, name, value)
