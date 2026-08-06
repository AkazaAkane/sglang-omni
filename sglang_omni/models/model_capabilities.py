# SPDX-License-Identifier: Apache-2.0
"""Capability declarations for model architectures."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

PrefillCudaGraphBackend = Literal[
    "breakable",
    "full",
    "tc_piecewise",
]

PrefillCudaGraphIntegration = Literal[
    "direct",
    "adapter",
    "incompatible",
]

PrefillCudaGraphStatus = Literal[
    "validated",
    "experimental",
    "performance_unproven",
    "blocked",
    "unvalidated",
]


@dataclass(frozen=True)
class PrefillCudaGraphBackendCapability:
    backend: PrefillCudaGraphBackend
    status: PrefillCudaGraphStatus
    default_buckets: tuple[int, ...] = ()
    requirements: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PrefillCudaGraphCapability:
    integration: PrefillCudaGraphIntegration
    backends: tuple[PrefillCudaGraphBackendCapability, ...] = ()
    preferred_backend: PrefillCudaGraphBackend | None = None
    incompatible_reason: str | None = None


@dataclass(frozen=True)
class ModelCapabilities:
    """Static capability declaration for a model architecture.

    These flags describe what a model architecture can support. Concrete
    checkpoint and deployment policy stays with ``PipelineConfig`` methods. For
    example, a Qwen3-TTS CustomVoice checkpoint can reject uploaded reference
    audio even though the architecture declares reference-audio support.

    Fields:
    - supports_reference_audio: the architecture can condition on reference
      audio when the selected checkpoint/deployment allows it.
    - supports_batch_vocoder: the architecture can produce batched waveform
      output.
    - supports_streaming_vocoder: the architecture can stream vocoder output
      before the full generation payload is complete.
    - supports_cuda_graph: the architecture has a CUDA graph path.
    - supports_torch_compile: the architecture has an owned ``torch.compile``
      path, including codec, codebook, or frame-sampler compiles. This is not
      limited to the generic SGLang ``enable_torch_compile`` server arg.
    - prefill_cuda_graph: backend-specific Prefill CUDA Graph compatibility and
      deployment policy metadata.
    """

    supports_reference_audio: bool
    supports_batch_vocoder: bool
    supports_streaming_vocoder: bool
    supports_cuda_graph: bool
    supports_torch_compile: bool
    prefill_cuda_graph: PrefillCudaGraphCapability | None = None

    def __post_init__(self) -> None:
        if self.prefill_cuda_graph is not None:
            _validate_prefill_cuda_graph_capability(self.prefill_cuda_graph)


def get_model_capabilities(architecture: str) -> ModelCapabilities | None:
    """Look up capabilities for a registered model architecture."""
    module = _model_package_for_architecture(architecture)
    if module is None:
        return None
    return _module_model_capabilities(module)


def _model_package_for_architecture(architecture: str) -> ModuleType | None:
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    config_cls = PIPELINE_CONFIG_REGISTRY.configs.get(architecture)
    if config_cls is None:
        return None
    package = config_cls.__module__.rsplit(".", 1)[0]
    return importlib.import_module(package)


def _module_model_capabilities(module: ModuleType) -> ModelCapabilities | None:
    capabilities = getattr(module, "CAPABILITIES", None)
    if capabilities is None:
        return None
    return _ensure_model_capabilities(capabilities, f"{module.__name__}.CAPABILITIES")


def _ensure_model_capabilities(capabilities: object, source: str) -> ModelCapabilities:
    if not isinstance(capabilities, ModelCapabilities):
        raise TypeError(f"{source} must be a ModelCapabilities instance")
    return capabilities


def _validate_prefill_cuda_graph_capability(
    capability: PrefillCudaGraphCapability,
) -> None:
    if capability.integration not in ("direct", "adapter", "incompatible"):
        raise ValueError(
            f"invalid Prefill CUDA Graph integration: {capability.integration!r}"
        )

    if capability.integration == "incompatible":
        if not capability.incompatible_reason:
            raise ValueError(
                "incompatible Prefill CUDA Graph capability requires "
                "incompatible_reason"
            )
        if capability.backends:
            raise ValueError(
                "incompatible Prefill CUDA Graph capability cannot declare backends"
            )

    declared_backends: set[str] = set()
    for backend in capability.backends:
        if not isinstance(backend, PrefillCudaGraphBackendCapability):
            raise TypeError(
                "Prefill CUDA Graph backends must be "
                "PrefillCudaGraphBackendCapability instances"
            )
        if backend.backend not in ("breakable", "full", "tc_piecewise"):
            raise ValueError(f"invalid Prefill CUDA Graph backend: {backend.backend!r}")
        if backend.backend in declared_backends:
            raise ValueError(
                f"duplicate Prefill CUDA Graph backend declaration: {backend.backend}"
            )
        declared_backends.add(backend.backend)

        if backend.status not in (
            "validated",
            "experimental",
            "performance_unproven",
            "blocked",
            "unvalidated",
        ):
            raise ValueError(f"invalid Prefill CUDA Graph status: {backend.status!r}")
        if backend.status == "blocked" and not backend.reason:
            raise ValueError(
                f"blocked Prefill CUDA Graph backend {backend.backend} requires a reason"
            )
        _validate_default_buckets(backend.backend, backend.default_buckets)

    if (
        capability.preferred_backend is not None
        and capability.preferred_backend not in declared_backends
    ):
        raise ValueError(
            "preferred Prefill CUDA Graph backend must be present in backends: "
            f"{capability.preferred_backend}"
        )


def _validate_default_buckets(backend: str, buckets: tuple[int, ...]) -> None:
    if not isinstance(buckets, tuple):
        raise ValueError(
            f"default buckets for Prefill CUDA Graph backend {backend} must be a tuple"
        )
    if not buckets:
        raise ValueError(
            f"default buckets for Prefill CUDA Graph backend {backend} "
            "must not be empty"
        )
    if any(
        not isinstance(bucket, int) or isinstance(bucket, bool) for bucket in buckets
    ):
        raise ValueError(
            f"default buckets for Prefill CUDA Graph backend {backend} "
            "must contain integers"
        )
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(
            f"default buckets for Prefill CUDA Graph backend {backend} must be positive"
        )
    if len(set(buckets)) != len(buckets):
        raise ValueError(
            f"default buckets for Prefill CUDA Graph backend {backend} must be unique"
        )
    if any(left >= right for left, right in zip(buckets, buckets[1:])):
        raise ValueError(
            f"default buckets for Prefill CUDA Graph backend {backend} "
            "must be strictly increasing"
        )


__all__ = [
    "ModelCapabilities",
    "PrefillCudaGraphBackend",
    "PrefillCudaGraphBackendCapability",
    "PrefillCudaGraphCapability",
    "PrefillCudaGraphIntegration",
    "PrefillCudaGraphStatus",
    "get_model_capabilities",
]
