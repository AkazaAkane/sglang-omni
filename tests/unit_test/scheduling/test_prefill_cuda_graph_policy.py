# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.model_capabilities import (
    PrefillCudaGraphBackendCapability,
    PrefillCudaGraphCapability,
)
from sglang_omni.scheduling.prefill_cuda_graph_policy import (
    prefill_cuda_graph_server_args,
    resolve_prefill_cuda_graph_policy,
)


def _capability(
    *,
    status: str = "validated",
    default_buckets: tuple[int, ...] = (32, 64),
    requirements: tuple[str, ...] = (),
    reason: str | None = None,
) -> PrefillCudaGraphCapability:
    return PrefillCudaGraphCapability(
        integration="direct",
        backends=(
            PrefillCudaGraphBackendCapability(
                backend="breakable",
                status=status,
                default_buckets=default_buckets,
                requirements=requirements,
                reason=reason,
            ),
        ),
    )


def _resolve(
    capability: PrefillCudaGraphCapability | None,
    **kwargs: object,
):
    return resolve_prefill_cuda_graph_policy(
        model_name="TestModel",
        capability=capability,
        requested_backend="breakable",
        requested_buckets=None,
        **kwargs,
    )


def test_disabled_request_needs_no_declaration_or_valid_buckets() -> None:
    policy = resolve_prefill_cuda_graph_policy(
        model_name="TestModel",
        capability=None,
        requested_backend="disabled",
        requested_buckets=[0, 0],
        runtime_requirements={"missing": False},
    )

    assert policy.enabled is False
    assert policy.requested_backend == "disabled"
    assert policy.resolved_backend is None
    assert policy.buckets == ()


def test_explicit_request_requires_declaration() -> None:
    with pytest.raises(
        ValueError,
        match="TestModel has not declared Prefill CUDA Graph compatibility",
    ):
        _resolve(None)


def test_incompatible_model_is_rejected_with_reason() -> None:
    capability = PrefillCudaGraphCapability(
        integration="incompatible",
        incompatible_reason="custom forward cannot be captured",
    )

    with pytest.raises(ValueError, match="custom forward cannot be captured"):
        _resolve(capability)


def test_undeclared_backend_lists_declared_backends() -> None:
    capability = PrefillCudaGraphCapability(
        integration="adapter",
        backends=(
            PrefillCudaGraphBackendCapability(
                backend="full",
                status="validated",
                default_buckets=(16,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="declared backends: full"):
        _resolve(capability)


def test_adapter_integration_requires_concrete_implementation() -> None:
    capability = PrefillCudaGraphCapability(
        integration="adapter",
        backends=(
            PrefillCudaGraphBackendCapability(
                backend="breakable",
                status="validated",
                default_buckets=(16,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="concrete adapter implementation"):
        _resolve(capability)


def test_validated_backend_uses_default_buckets() -> None:
    policy = _resolve(_capability())

    assert policy.enabled is True
    assert policy.resolved_backend == "breakable"
    assert policy.integration == "direct"
    assert policy.status == "validated"
    assert policy.buckets == (32, 64)


def test_experimental_backend_requires_explicit_allowance() -> None:
    capability = _capability(status="experimental")

    with pytest.raises(ValueError, match="experimental"):
        _resolve(capability)

    assert _resolve(capability, allow_experimental=True).enabled is True


def test_performance_unproven_backend_requires_explicit_allowance() -> None:
    capability = _capability(status="performance_unproven")

    with pytest.raises(ValueError, match="unproven performance"):
        _resolve(capability)

    assert _resolve(capability, allow_performance_unproven=True).enabled is True


def test_blocked_backend_is_rejected_with_reason() -> None:
    capability = _capability(status="blocked", reason="upstream issue 123")

    with pytest.raises(ValueError, match="upstream issue 123"):
        _resolve(capability)


def test_unvalidated_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unvalidated"):
        _resolve(_capability(status="unvalidated"))


def test_explicit_buckets_override_defaults() -> None:
    policy = resolve_prefill_cuda_graph_policy(
        model_name="TestModel",
        capability=_capability(),
        requested_backend="breakable",
        requested_buckets=[8, 16],
    )

    assert policy.buckets == (8, 16)


def test_enabled_backend_requires_buckets() -> None:
    with pytest.raises(ValueError, match="at least one bucket"):
        _resolve(_capability(default_buckets=()))


@pytest.mark.parametrize(
    ("buckets", "error"),
    [
        ([16, 8], "strictly increasing"),
        ([16, 16], "unique"),
        ([0, 16], "positive"),
        ([True, 16], "integers"),
    ],
)
def test_invalid_explicit_buckets_are_rejected(
    buckets: list[int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        resolve_prefill_cuda_graph_policy(
            model_name="TestModel",
            capability=_capability(),
            requested_backend="breakable",
            requested_buckets=buckets,
        )


def test_satisfied_runtime_requirements_are_preserved() -> None:
    policy = _resolve(
        _capability(requirements=("static_inputs", "single_rank")),
        runtime_requirements={"static_inputs": True, "single_rank": True},
    )

    assert policy.requirements == ("static_inputs", "single_rank")


def test_unmet_runtime_requirements_are_listed() -> None:
    with pytest.raises(
        ValueError,
        match="unmet runtime requirements: static_inputs, single_rank",
    ):
        _resolve(
            _capability(requirements=("static_inputs", "single_rank")),
            runtime_requirements={"static_inputs": False},
        )


def test_server_args_mapping() -> None:
    disabled = resolve_prefill_cuda_graph_policy(
        model_name="TestModel",
        capability=None,
        requested_backend="disabled",
        requested_buckets=None,
    )
    enabled = _resolve(_capability(default_buckets=(32, 64, 128)))

    assert prefill_cuda_graph_server_args(disabled) == {
        "cuda_graph_backend_prefill": "disabled"
    }
    assert prefill_cuda_graph_server_args(enabled) == {
        "cuda_graph_backend_prefill": "breakable",
        "cuda_graph_bs_prefill": [32, 64, 128],
        "cuda_graph_max_bs_prefill": 128,
    }


@pytest.mark.parametrize("compiler", [None, "eager", "inductor"])
def test_tc_piecewise_compiler_is_resolved_and_mapped(
    compiler: str | None,
) -> None:
    capability = PrefillCudaGraphCapability(
        integration="direct",
        backends=(
            PrefillCudaGraphBackendCapability(
                backend="tc_piecewise",
                status="validated",
                default_buckets=(16, 32),
            ),
        ),
    )

    policy = resolve_prefill_cuda_graph_policy(
        model_name="TestModel",
        capability=capability,
        requested_backend="tc_piecewise",
        requested_buckets=None,
        requested_compiler=compiler,
    )

    expected_compiler = compiler or "eager"
    assert policy.compiler == expected_compiler
    assert prefill_cuda_graph_server_args(policy) == {
        "cuda_graph_backend_prefill": "tc_piecewise",
        "cuda_graph_bs_prefill": [16, 32],
        "cuda_graph_max_bs_prefill": 32,
        "cuda_graph_tc_compiler": expected_compiler,
    }


def test_compiler_requires_tc_piecewise_backend() -> None:
    with pytest.raises(ValueError, match="only be set for tc_piecewise"):
        resolve_prefill_cuda_graph_policy(
            model_name="TestModel",
            capability=_capability(),
            requested_backend="breakable",
            requested_buckets=None,
            requested_compiler="inductor",
        )


def test_tc_piecewise_rejects_unknown_compiler() -> None:
    capability = PrefillCudaGraphCapability(
        integration="direct",
        backends=(
            PrefillCudaGraphBackendCapability(
                backend="tc_piecewise",
                status="validated",
                default_buckets=(16,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="must be 'eager' or 'inductor'"):
        resolve_prefill_cuda_graph_policy(
            model_name="TestModel",
            capability=capability,
            requested_backend="tc_piecewise",
            requested_buckets=None,
            requested_compiler="unknown",
        )
