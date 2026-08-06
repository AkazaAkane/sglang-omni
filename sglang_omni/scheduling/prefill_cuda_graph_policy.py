# SPDX-License-Identifier: Apache-2.0
"""Shared policy resolution for SGLang Prefill CUDA Graphs.

Invalid explicit startup configuration fails here. Runtime replay eligibility,
including eager fallback beyond captured token buckets, remains owned by SGLang.
Capability metadata never selects a backend automatically; models remain eager
unless a builder or operator explicitly requests a backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.models.model_capabilities import (
    PrefillCudaGraphBackend,
    PrefillCudaGraphCapability,
    PrefillCudaGraphIntegration,
    PrefillCudaGraphStatus,
)


@dataclass(frozen=True)
class ResolvedPrefillCudaGraphPolicy:
    enabled: bool
    requested_backend: str
    resolved_backend: PrefillCudaGraphBackend | None
    integration: PrefillCudaGraphIntegration | None
    status: PrefillCudaGraphStatus | None
    token_buckets: tuple[int, ...]
    requirements: tuple[str, ...]
    reason: str | None
    compiler: str | None = None


def resolve_prefill_cuda_graph_policy(
    *,
    model_name: str,
    capability: PrefillCudaGraphCapability | None,
    requested_backend: str,
    requested_token_buckets: list[int] | tuple[int, ...] | None,
    requested_compiler: str | None = None,
    allow_experimental: bool = False,
    allow_performance_unproven: bool = False,
    runtime_requirements: dict[str, bool] | None = None,
) -> ResolvedPrefillCudaGraphPolicy:
    if requested_backend == "disabled":
        return ResolvedPrefillCudaGraphPolicy(
            enabled=False,
            requested_backend=requested_backend,
            resolved_backend=None,
            integration=None,
            status=None,
            token_buckets=(),
            requirements=(),
            reason=None,
            compiler=None,
        )

    if capability is None:
        raise ValueError(
            f"{model_name} has not declared Prefill CUDA Graph compatibility"
        )
    if capability.integration == "incompatible":
        raise ValueError(
            f"{model_name} is incompatible with Prefill CUDA Graph: "
            f"{capability.incompatible_reason}"
        )

    backend_capability = next(
        (
            backend
            for backend in capability.backends
            if backend.backend == requested_backend
        ),
        None,
    )
    if backend_capability is None:
        declared = ", ".join(backend.backend for backend in capability.backends)
        if not declared:
            declared = "none"
        raise ValueError(
            f"{model_name} has not declared Prefill CUDA Graph backend "
            f"{requested_backend!r}; declared backends: {declared}"
        )
    if capability.integration == "adapter":
        raise ValueError(
            f"{model_name} Prefill CUDA Graph integration requires a concrete "
            "adapter implementation"
        )

    status = backend_capability.status
    if status == "experimental" and not allow_experimental:
        raise ValueError(
            f"{model_name} Prefill CUDA Graph backend {requested_backend!r} is "
            "experimental; set allow_experimental=True to enable it"
        )
    if status == "performance_unproven" and not allow_performance_unproven:
        raise ValueError(
            f"{model_name} Prefill CUDA Graph backend {requested_backend!r} has "
            "unproven performance; set allow_performance_unproven=True to enable it"
        )
    if status == "blocked":
        raise ValueError(
            f"{model_name} Prefill CUDA Graph backend {requested_backend!r} is "
            f"blocked: {backend_capability.reason}"
        )
    if status == "unvalidated":
        raise ValueError(
            f"{model_name} Prefill CUDA Graph backend {requested_backend!r} is "
            "unvalidated"
        )

    requirements = backend_capability.requirements
    available_requirements = runtime_requirements or {}
    unmet_requirements = [
        requirement
        for requirement in requirements
        if available_requirements.get(requirement) is not True
    ]
    if unmet_requirements:
        raise ValueError(
            f"{model_name} Prefill CUDA Graph backend {requested_backend!r} has "
            f"unmet runtime requirements: {', '.join(unmet_requirements)}"
        )

    token_buckets = _validate_token_buckets(
        requested_token_buckets
        if requested_token_buckets is not None
        else backend_capability.default_token_buckets
    )
    if not token_buckets:
        raise ValueError(
            f"{model_name} Prefill CUDA Graph backend {requested_backend!r} "
            "requires at least one token bucket"
        )

    compiler = _resolve_compiler(requested_backend, requested_compiler)

    return ResolvedPrefillCudaGraphPolicy(
        enabled=True,
        requested_backend=requested_backend,
        resolved_backend=backend_capability.backend,
        integration=capability.integration,
        status=status,
        token_buckets=token_buckets,
        requirements=requirements,
        reason=backend_capability.reason,
        compiler=compiler,
    )


def prefill_cuda_graph_server_args(
    policy: ResolvedPrefillCudaGraphPolicy,
) -> dict[str, object]:
    if not policy.enabled:
        return {"cuda_graph_backend_prefill": "disabled"}

    server_args: dict[str, object] = {
        "cuda_graph_backend_prefill": policy.resolved_backend,
        "cuda_graph_bs_prefill": list(policy.token_buckets),
        "cuda_graph_max_bs_prefill": max(policy.token_buckets),
    }
    if policy.compiler is not None:
        server_args["cuda_graph_tc_compiler"] = policy.compiler
    return server_args


def _resolve_compiler(backend: str, requested_compiler: str | None) -> str | None:
    if backend != "tc_piecewise":
        if requested_compiler is not None:
            raise ValueError(
                "Prefill CUDA Graph compiler can only be set for tc_piecewise"
            )
        return None
    compiler = requested_compiler or "eager"
    if compiler not in ("eager", "inductor"):
        raise ValueError("Prefill CUDA Graph compiler must be 'eager' or 'inductor'")
    return compiler


def _validate_token_buckets(
    token_buckets: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(token_buckets, (list, tuple)):
        raise ValueError("Prefill CUDA Graph token buckets must be a list or tuple")
    if any(
        not isinstance(bucket, int) or isinstance(bucket, bool)
        for bucket in token_buckets
    ):
        raise ValueError("Prefill CUDA Graph token buckets must contain integers")

    normalized = tuple(token_buckets)
    if any(bucket <= 0 for bucket in normalized):
        raise ValueError("Prefill CUDA Graph token buckets must be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Prefill CUDA Graph token buckets must be unique")
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError("Prefill CUDA Graph token buckets must be strictly increasing")
    return normalized


__all__ = [
    "ResolvedPrefillCudaGraphPolicy",
    "prefill_cuda_graph_server_args",
    "resolve_prefill_cuda_graph_policy",
]
