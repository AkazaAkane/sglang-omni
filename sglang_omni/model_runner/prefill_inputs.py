# SPDX-License-Identifier: Apache-2.0
"""Batch-carried prefill inputs for breakable-CUDA-graph (BCG) prefill.

SGLang's prefill graph runner rejects batches that carry
``forward_batch.input_embeds`` and its replay-time static batch drops dynamic
attributes and ``rids``. It does, however, copy ``mm_inputs`` verbatim into
the static batch, and nothing on the omni-active runner path dereferences
that field. Runner-composed prefill conditioning therefore crosses the graph
boundary as an :class:`OmniPrefillInputs` payload stored in
``forward_batch.mm_inputs``: the model's eager outer forward reads it, passes
the embeddings into its decoder call, and SGLang's replay hook copies them
into the captured ``input_embeds`` slot. The same read path serves the eager
fallback, so graph and eager prefills consume identical inputs.

On a real prefill batch ``mm_inputs`` is never ``None``:
``ScheduleBatch.prepare_for_extend`` appends ``req.multimodal_inputs`` for
every request, so a text-only TTS batch arrives with a per-request list of
``None`` entries. Attaching replaces that placeholder list on the
ForwardBatch only; ``ScheduleBatch.multimodal_inputs`` is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OmniPrefillInputs:
    """Per-forward prefill conditioning composed by an omni model runner.

    ``input_embeds`` covers exactly the extend-window tokens of the batch, in
    model dtype. ``rids`` carries request identity for the eager tail; the
    graph runner's static batch does not preserve ``ForwardBatch.rids``.
    """

    input_embeds: torch.Tensor
    rids: tuple[str, ...]


def attach_omni_prefill_inputs(
    forward_batch: Any, prefill_inputs: OmniPrefillInputs
) -> None:
    """Stow ``prefill_inputs`` on ``forward_batch.mm_inputs``.

    ``forward_batch.input_embeds`` must stay ``None``: the SGLang prefill
    graph gate rejects embeds-carrying batches, and the model forward is the
    single consumer of the payload on both the graph and eager paths.
    """
    if forward_batch.input_embeds is not None:
        raise RuntimeError(
            "OmniPrefillInputs requires forward_batch.input_embeds to stay "
            "None; the model forward consumes the payload instead"
        )
    mm_inputs = forward_batch.mm_inputs
    if isinstance(mm_inputs, OmniPrefillInputs):
        raise RuntimeError(
            "forward_batch.mm_inputs already carries OmniPrefillInputs; "
            "refusing to attach twice to the same batch"
        )
    # prepare_for_extend fills mm_inputs with one entry per request (None for
    # text-only requests), so the field is a list of Nones on every real
    # prefill batch. Replace that placeholder; refuse to displace genuine
    # SGLang multimodal inputs.
    if mm_inputs is not None and any(item is not None for item in mm_inputs):
        raise RuntimeError(
            "forward_batch.mm_inputs carries SGLang multimodal inputs; "
            "refusing to overwrite them with OmniPrefillInputs"
        )
    num_tokens = len(forward_batch.input_ids)
    if prefill_inputs.input_embeds.shape[0] != num_tokens:
        raise RuntimeError(
            "OmniPrefillInputs embeddings must cover the extend-window tokens: "
            f"embeds rows={prefill_inputs.input_embeds.shape[0]}, "
            f"batch tokens={num_tokens}"
        )
    if len(prefill_inputs.rids) != forward_batch.batch_size:
        raise RuntimeError(
            "OmniPrefillInputs rids must cover the batch: "
            f"rids={len(prefill_inputs.rids)}, "
            f"batch_size={forward_batch.batch_size}"
        )
    forward_batch.mm_inputs = prefill_inputs


def get_omni_prefill_inputs(forward_batch: Any) -> OmniPrefillInputs | None:
    """Return the payload attached by :func:`attach_omni_prefill_inputs`.

    Returns ``None`` for batches without a payload, including batches whose
    ``mm_inputs`` carries genuine SGLang multimodal inputs.
    """
    payload = forward_batch.mm_inputs
    if isinstance(payload, OmniPrefillInputs):
        return payload
    return None


__all__ = [
    "OmniPrefillInputs",
    "attach_omni_prefill_inputs",
    "get_omni_prefill_inputs",
]
