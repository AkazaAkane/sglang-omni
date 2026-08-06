# SPDX-License-Identifier: Apache-2.0
"""Builders for SGLang-backed autoregressive engine stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from sglang_omni.models.model_capabilities import (
    PrefillCudaGraphCapability,
    get_model_capabilities,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    get_prefill_cuda_graph_backend,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.prefill_cuda_graph_policy import (
    ResolvedPrefillCudaGraphPolicy,
    prefill_cuda_graph_server_args,
    resolve_prefill_cuda_graph_policy,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)

_MISSING = object()
_CANONICAL_PREFILL_KEYS = {
    "backend",
    "max_bs",
    "bs",
    "tc_compiler",
    "full_prefill_max_req",
}
_PREFILL_BACKEND_KEYS = (
    "cuda_graph_backend_prefill",
    "prefill_cuda_graph_backend",
)
_PREFILL_TOKEN_BUCKET_KEYS = (
    "cuda_graph_bs_prefill",
    "prefill_cuda_graph_token_buckets",
)
_PREFILL_COMPILER_KEYS = (
    "cuda_graph_tc_compiler",
    "prefill_cuda_graph_compiler",
)
_LEGACY_PREFILL_BACKEND_ALIASES = {
    "enable_breakable_cuda_graph": "breakable",
    "disable_piecewise_cuda_graph": "disabled",
    "enforce_piecewise_cuda_graph": "tc_piecewise",
}


class SGLangGenerationEngineBuilder(ABC):
    """Build the model-neutral parts of a SGLang AR engine stage.

    Model-specific builders provide checkpoint preprocessing, model setup,
    request/result adapters, validation policy, and any stage-owned resources.
    Family-specific builders such as :class:`AsrEngineBuilder` and
    :class:`TtsEngineBuilder` define the lifecycle policy for each modality.

    Prefill controls accept upstream ``cuda_graph_*_prefill`` keys or the
    stage-facing ``prefill_cuda_graph_*`` aliases. Prefill ``bs`` values are
    token buckets, not request batch-size buckets.
    """

    model_name: str
    context_length: int
    model_arch_override: str | None = None
    prefill_cuda_graph_capability_architecture: str | None = None
    prefill_cuda_graph_backend: str | None = None
    prefill_cuda_graph_token_buckets: list[int] | tuple[int, ...] | None = None
    prefill_cuda_graph_compiler: str | None = None
    allow_experimental_prefill_cuda_graph: bool = False
    allow_performance_unproven_prefill_cuda_graph: bool = False

    def build(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        gpu_id: int | None = None,
        dtype: str = "bfloat16",
        server_args_overrides: dict[str, Any] | None = None,
    ) -> Any:
        from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
        from sglang_omni.scheduling import sglang_backend

        checkpoint_dir = self.resolve_checkpoint(model_path)
        if gpu_id is not None:
            device = f"cuda:{gpu_id}"
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype

        self.pre_infra_setup(checkpoint_dir)

        stage_defaults = self.generation_defaults(dtype=dtype)
        explicit_overrides = dict(server_args_overrides or {})
        _normalize_prefill_override_precedence(
            lower_defaults=stage_defaults,
            explicit_overrides=explicit_overrides,
        )
        overrides = build_generation_batch_overrides(
            server_args_overrides=explicit_overrides,
            **stage_defaults,
        )
        self.adjust_overrides(overrides)
        prefill_policy = self._resolve_prefill_cuda_graph_policy(
            overrides=overrides,
        )

        server_args = sglang_backend.build_sglang_server_args(
            checkpoint_dir,
            context_length=self.context_length,
            **overrides,
        )
        self.customize_server_args(server_args)
        self._validate_and_log_prefill_cuda_graph_policy(
            prefill_policy,
            server_args,
        )
        self.validate_before_infrastructure(server_args)

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)
        prefill_graph_backend = get_prefill_cuda_graph_backend(server_args)
        if prefill_graph_backend == "breakable":
            # SGLang registers the prefill input_embeds slot only for
            # multimodal model configs; the payload channel needs it.
            infra_kwargs.setdefault("enable_prefill_input_embeds", True)
        want_cuda_graph, (
            model_worker,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            prefill_mgr,
            decode_mgr,
            model_config,
        ) = scheduling_bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            gpu_id,
            **infra_kwargs,
        )
        model = model_worker.model_runner.model

        self.setup_model(
            model_worker=model_worker,
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_id=gpu_id,
            server_args=server_args,
        )

        self.validate_after_model_setup(model, server_args)

        self.compile_model(model, server_args)

        if want_cuda_graph:
            model_worker.model_runner.init_cuda_graphs()
            self.post_cuda_graph_setup(model, server_args)
            if prefill_graph_backend != "disabled":
                from sglang_omni.utils import cuda_graph_batch_validator

                cuda_graph_batch_validator.attest_prefill_cuda_graphs(
                    model_worker.model_runner, server_args
                )

        try:
            # Model-local encoder graphs and caches must be initialized after
            # SGLang's generation graphs to preserve the established order.
            self.setup_model_resources(
                model,
                server_args,
                generation_cuda_graph_enabled=want_cuda_graph,
            )

            output_proc = sglang_backend.SGLangOutputProcessor(
                capture_hidden=False,
                capture_hidden_layers=None,
                model=model,
            )
            self.setup_runtime_resources(model, server_args)
            scheduler, model_runner = self._build_runtime(
                model_worker=model_worker,
                model=model,
                output_proc=output_proc,
                tree_cache=tree_cache,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                server_args=server_args,
                model_config=model_config,
                prefill_manager=prefill_mgr,
                decode_manager=decode_mgr,
            )
            self.post_scheduler_setup(scheduler, model_runner)
            return scheduler
        except Exception:
            self.cleanup_build_failure()
            raise

    def resolve_checkpoint(self, model_path: str) -> str:
        # The shared builder treats checkpoint resolution as a family policy.
        # Subclasses override this when they need a resolved local snapshot.
        return model_path

    def prefill_cuda_graph_capability(self) -> PrefillCudaGraphCapability | None:
        architecture = self.prefill_cuda_graph_capability_architecture
        if architecture is None:
            return None
        capabilities = get_model_capabilities(architecture)
        if capabilities is None:
            return None
        return capabilities.prefill_cuda_graph

    def prefill_cuda_graph_runtime_requirements(self) -> dict[str, bool]:
        """Report model-specific prerequisites before explicit enablement."""
        return {}

    def _resolve_prefill_cuda_graph_policy(
        self,
        *,
        overrides: dict[str, Any],
    ) -> ResolvedPrefillCudaGraphPolicy:
        _normalize_legacy_prefill_aliases(overrides)
        canonical_config, canonical_prefill = _extract_canonical_prefill_config(
            overrides
        )
        uses_canonical_prefill = bool(canonical_prefill)
        merged_backend = _pop_prefill_value(
            overrides,
            "prefill backend",
            *_PREFILL_BACKEND_KEYS,
        )
        merged_token_buckets = _pop_prefill_value(
            overrides,
            "prefill token buckets",
            *_PREFILL_TOKEN_BUCKET_KEYS,
        )
        merged_max_bs = overrides.pop("cuda_graph_max_bs_prefill", _MISSING)
        merged_compiler = _pop_prefill_value(
            overrides,
            "prefill compiler",
            *_PREFILL_COMPILER_KEYS,
        )
        disable_prefill = bool(overrides.pop("disable_prefill_cuda_graph", False))
        disable_all_graphs = bool(overrides.get("disable_cuda_graph"))
        has_convenience_prefill = (
            any(
                value is not _MISSING
                for value in (
                    merged_backend,
                    merged_token_buckets,
                    merged_max_bs,
                    merged_compiler,
                )
            )
            or disable_prefill
        )
        if canonical_prefill and has_convenience_prefill:
            raise ValueError(
                "cuda_graph_config.prefill cannot be combined with Prefill CUDA "
                "Graph convenience settings"
            )

        if (
            disable_prefill
            and merged_backend is not _MISSING
            and merged_backend != "disabled"
        ):
            raise ValueError(
                "Conflicting Prefill CUDA Graph settings: a disable flag cannot "
                "be combined with an enabled prefill backend at the same tier"
            )

        canonical_backend = canonical_prefill.pop("backend", _MISSING)
        clear_builder_prefill_settings = False
        if canonical_backend is not _MISSING:
            requested_backend = canonical_backend
            clear_builder_prefill_settings = requested_backend == "disabled"
        elif merged_backend is not _MISSING:
            requested_backend = merged_backend
            clear_builder_prefill_settings = requested_backend == "disabled"
        else:
            if disable_prefill or disable_all_graphs:
                requested_backend = "disabled"
                clear_builder_prefill_settings = True
            else:
                requested_backend = self.prefill_cuda_graph_backend or "disabled"

        requested_token_buckets = canonical_prefill.pop("bs", _MISSING)
        if requested_token_buckets is _MISSING:
            requested_token_buckets = merged_token_buckets
        if requested_token_buckets is _MISSING:
            requested_token_buckets = (
                None
                if clear_builder_prefill_settings
                else self.prefill_cuda_graph_token_buckets
            )

        requested_max_bs = canonical_prefill.pop("max_bs", _MISSING)
        if requested_max_bs is _MISSING:
            requested_max_bs = merged_max_bs
        if requested_max_bs is None:
            requested_max_bs = _MISSING

        requested_compiler = canonical_prefill.pop("tc_compiler", _MISSING)
        if requested_compiler is _MISSING:
            requested_compiler = merged_compiler
        if requested_compiler is _MISSING:
            requested_compiler = (
                None
                if clear_builder_prefill_settings
                else self.prefill_cuda_graph_compiler
            )

        allow_experimental = _pop_policy_flag(
            overrides,
            "allow_experimental_prefill_cuda_graph",
            self.allow_experimental_prefill_cuda_graph,
        )
        allow_performance_unproven = _pop_policy_flag(
            overrides,
            "allow_performance_unproven_prefill_cuda_graph",
            self.allow_performance_unproven_prefill_cuda_graph,
        )

        if requested_backend == "disabled" and requested_token_buckets is not None:
            raise ValueError(
                "Prefill CUDA Graph token buckets cannot be set when the backend "
                "is disabled"
            )
        if requested_backend == "disabled" and requested_max_bs is not _MISSING:
            raise ValueError(
                "cuda_graph_max_bs_prefill cannot be set when the Prefill CUDA "
                "Graph backend is disabled"
            )
        if requested_backend == "disabled" and requested_compiler is not None:
            raise ValueError(
                "Prefill CUDA Graph compiler cannot be set when the backend is disabled"
            )
        if (
            requested_backend != "full"
            and canonical_prefill.get("full_prefill_max_req") is not None
        ):
            raise ValueError(
                "full_prefill_max_req requires Prefill CUDA Graph backend 'full'"
            )

        capability = (
            None
            if requested_backend == "disabled"
            else self.prefill_cuda_graph_capability()
        )
        policy = resolve_prefill_cuda_graph_policy(
            model_name=self.model_name,
            capability=capability,
            requested_backend=requested_backend,
            requested_token_buckets=requested_token_buckets,
            requested_compiler=requested_compiler,
            allow_experimental=allow_experimental,
            allow_performance_unproven=allow_performance_unproven,
            runtime_requirements=(
                None
                if requested_backend == "disabled"
                else self.prefill_cuda_graph_runtime_requirements()
            ),
        )
        if requested_max_bs is not _MISSING and requested_max_bs != max(
            policy.token_buckets
        ):
            raise ValueError(
                "Conflicting Prefill CUDA Graph maximum token bucket: "
                f"cuda_graph_max_bs_prefill={requested_max_bs!r}, "
                f"resolved maximum={max(policy.token_buckets)!r}"
            )

        if uses_canonical_prefill:
            resolved_prefill = dict(canonical_prefill)
            resolved_prefill["backend"] = (
                policy.resolved_backend if policy.enabled else "disabled"
            )
            if policy.enabled:
                resolved_prefill["bs"] = list(policy.token_buckets)
                resolved_prefill["max_bs"] = max(policy.token_buckets)
                if policy.compiler is not None:
                    resolved_prefill["tc_compiler"] = policy.compiler
            canonical_config["prefill"] = resolved_prefill
            overrides["cuda_graph_config"] = canonical_config
        else:
            overrides.update(prefill_cuda_graph_server_args(policy))
        return policy

    def _validate_and_log_prefill_cuda_graph_policy(
        self,
        policy: ResolvedPrefillCudaGraphPolicy,
        server_args: Any,
    ) -> None:
        actual = server_args.cuda_graph_config.prefill
        actual_backend = actual.backend
        actual_token_buckets = tuple(actual.bs or ())
        actual_max_bs = actual.max_bs
        actual_compiler = actual.tc_compiler
        expected_backend = policy.resolved_backend if policy.enabled else "disabled"
        changed = actual_backend != expected_backend
        downgraded = policy.enabled and actual_backend == "disabled"
        logger.info(
            "Prefill CUDA Graph policy: model=%s requested=%s actual=%s "
            "enabled=%s integration=%s status=%s token_buckets=%s max_bs=%s "
            "compiler=%s changed=%s downgraded=%s",
            self.model_name,
            policy.requested_backend,
            actual_backend,
            str(actual_backend != "disabled").lower(),
            policy.integration,
            policy.status,
            list(actual_token_buckets),
            actual_max_bs,
            actual_compiler,
            str(changed).lower(),
            str(downgraded).lower(),
        )
        errors: list[str] = []
        if changed:
            errors.append(
                f"backend resolved to {actual_backend!r}, expected {expected_backend!r}"
            )
        if policy.enabled and actual_token_buckets != policy.token_buckets:
            errors.append(
                "token buckets resolved to "
                f"{actual_token_buckets!r}, expected {policy.token_buckets!r}"
            )
        if policy.enabled and actual_max_bs != max(policy.token_buckets):
            errors.append(
                f"max_bs resolved to {actual_max_bs!r}, "
                f"expected {max(policy.token_buckets)!r}"
            )
        if policy.compiler is not None and actual_compiler != policy.compiler:
            errors.append(
                f"compiler resolved to {actual_compiler!r}, "
                f"expected {policy.compiler!r}"
            )
        if errors:
            raise ValueError(
                f"{self.model_name} Prefill CUDA Graph configuration changed by "
                "SGLang: " + "; ".join(errors)
            )

    @abstractmethod
    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir

    def validate_before_infrastructure(self, server_args: Any) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        del overrides

    def customize_server_args(self, server_args: Any) -> None:
        del server_args

    def infra_kwargs(self) -> dict[str, Any]:
        return {}

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def get_model_buffer_bs(self, model: Any) -> int | None:
        del model
        return None

    def compile_model(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        del model, server_args

    def setup_model_resources(
        self,
        model: Any,
        server_args: Any,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        del model, server_args, generation_cuda_graph_enabled

    def setup_runtime_resources(self, model: Any, server_args: Any) -> None:
        del model, server_args

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def _build_runtime(
        self,
        *,
        model_worker: Any,
        model: Any,
        output_proc: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
    ) -> tuple[Any, Any]:
        request_builder, result_adapter = self.make_adapters(model)
        scheduler_kwargs = self.extra_scheduler_kwargs()
        model_runner = self.make_model_runner(model_worker, output_proc)
        scheduler = self._make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            extra_scheduler_kwargs=scheduler_kwargs,
        )
        return scheduler, model_runner

    def make_abort_callback(self) -> Any | None:
        return None

    def make_request_finished_callback(self) -> Any | None:
        return None

    def extra_scheduler_callbacks(self) -> dict[str, Any]:
        return {}

    def cleanup_build_failure(self) -> None:
        pass

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
        extra_scheduler_kwargs: dict[str, Any],
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        scheduler_kwargs = {
            "tp_worker": model_worker,
            "tree_cache": tree_cache,
            "req_to_token_pool": req_to_token_pool,
            "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
            "server_args": server_args,
            "model_config": model_config,
            "prefill_manager": prefill_manager,
            "decode_manager": decode_manager,
            "model_runner": model_runner,
            "request_builder": request_builder,
            "result_adapter": result_adapter,
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
        }
        scheduler_kwargs.update(self.extra_scheduler_callbacks())
        scheduler_kwargs.update(extra_scheduler_kwargs)
        return omni_scheduler.OmniScheduler(**scheduler_kwargs)

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        del scheduler, model_runner


def _get_prefill_value(
    values: dict[str, Any],
    label: str,
    *keys: str,
) -> Any:
    present = [
        (key, values[key]) for key in keys if key in values and values[key] is not None
    ]
    if not present:
        return _MISSING
    first_key, first_value = present[0]
    for key, value in present[1:]:
        if not _prefill_values_equal(first_value, value):
            raise ValueError(
                f"Conflicting {label} settings: "
                f"{first_key}={first_value!r}, {key}={value!r}"
            )
    return first_value


def _normalize_prefill_override_precedence(
    *,
    lower_defaults: dict[str, Any],
    explicit_overrides: dict[str, Any],
) -> None:
    _normalize_legacy_prefill_aliases(explicit_overrides)
    explicit_canonical_prefill = _has_canonical_prefill(explicit_overrides)
    explicit_setting_keys = _active_prefill_setting_keys(explicit_overrides)

    if explicit_canonical_prefill:
        if explicit_setting_keys:
            raise ValueError(
                "cuda_graph_config.prefill cannot be combined with Prefill CUDA "
                "Graph convenience settings at the same override tier"
            )
        _remove_lower_prefill_settings(lower_defaults)
        return

    if explicit_setting_keys:
        _remove_canonical_prefill(lower_defaults)
        _remove_shadowed_lower_prefill_fields(
            lower_defaults,
            explicit_setting_keys,
        )

    explicit_backend = _get_prefill_value(
        explicit_overrides,
        "prefill backend",
        *_PREFILL_BACKEND_KEYS,
    )
    explicit_disable = (
        explicit_backend == "disabled"
        or bool(explicit_overrides.get("disable_prefill_cuda_graph"))
        or bool(explicit_overrides.get("disable_cuda_graph"))
    )
    if not explicit_disable:
        return

    if explicit_backend is not _MISSING and explicit_backend != "disabled":
        raise ValueError(
            "Conflicting Prefill CUDA Graph settings: a disable flag cannot be "
            "combined with an enabled prefill backend at the same override tier"
        )
    conflicting_keys = explicit_setting_keys.intersection(
        {
            *_PREFILL_TOKEN_BUCKET_KEYS,
            "cuda_graph_max_bs_prefill",
            *_PREFILL_COMPILER_KEYS,
        }
    )
    if conflicting_keys:
        raise ValueError(
            "Prefill CUDA Graph disable settings cannot be combined with token "
            "buckets, maximum, or compiler settings at the same override tier: "
            + ", ".join(sorted(conflicting_keys))
        )
    _remove_lower_prefill_settings(lower_defaults)


def _normalize_legacy_prefill_aliases(values: dict[str, Any]) -> None:
    for alias, backend in _LEGACY_PREFILL_BACKEND_ALIASES.items():
        enabled = values.pop(alias, False)
        if not enabled:
            continue
        current = _get_prefill_value(
            values,
            "prefill backend",
            *_PREFILL_BACKEND_KEYS,
        )
        if current is not _MISSING and current != backend:
            raise ValueError(
                "Conflicting Prefill CUDA Graph backend settings: "
                f"{alias}=True selects {backend!r}, existing backend={current!r}"
            )
        values["cuda_graph_backend_prefill"] = backend


def _active_prefill_setting_keys(values: dict[str, Any]) -> set[str]:
    keys = {
        key
        for key in (
            *_PREFILL_BACKEND_KEYS,
            *_PREFILL_TOKEN_BUCKET_KEYS,
            "cuda_graph_max_bs_prefill",
            *_PREFILL_COMPILER_KEYS,
        )
        if values.get(key) is not None
    }
    for key in ("disable_prefill_cuda_graph", "disable_cuda_graph"):
        if values.get(key):
            keys.add(key)
    return keys


def _has_canonical_prefill(values: dict[str, Any]) -> bool:
    config = values.get("cuda_graph_config")
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    return isinstance(config, dict) and bool(config.get("prefill"))


def _remove_canonical_prefill(values: dict[str, Any]) -> None:
    config = values.get("cuda_graph_config")
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict) or "prefill" not in config:
        return
    remaining = dict(config)
    remaining.pop("prefill")
    if remaining:
        values["cuda_graph_config"] = remaining
    else:
        values.pop("cuda_graph_config", None)


def _remove_shadowed_lower_prefill_fields(
    lower_defaults: dict[str, Any],
    explicit_setting_keys: set[str],
) -> None:
    groups = (
        _PREFILL_BACKEND_KEYS,
        _PREFILL_TOKEN_BUCKET_KEYS,
        ("cuda_graph_max_bs_prefill",),
        _PREFILL_COMPILER_KEYS,
    )
    for group in groups:
        if explicit_setting_keys.intersection(group):
            for key in group:
                lower_defaults.pop(key, None)
    if explicit_setting_keys.intersection(_PREFILL_BACKEND_KEYS):
        lower_defaults.pop("disable_prefill_cuda_graph", None)


def _remove_lower_prefill_settings(values: dict[str, Any]) -> None:
    _remove_canonical_prefill(values)
    for key in (
        *_PREFILL_BACKEND_KEYS,
        *_PREFILL_TOKEN_BUCKET_KEYS,
        "cuda_graph_max_bs_prefill",
        *_PREFILL_COMPILER_KEYS,
        "disable_prefill_cuda_graph",
        *_LEGACY_PREFILL_BACKEND_ALIASES,
    ):
        values.pop(key, None)


def _extract_canonical_prefill_config(
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_config = overrides.pop("cuda_graph_config", None)
    if raw_config is None:
        return {}, {}
    if hasattr(raw_config, "to_dict"):
        raw_config = raw_config.to_dict()
    if not isinstance(raw_config, dict):
        raise ValueError("cuda_graph_config must be a dictionary")

    canonical_config = dict(raw_config)
    raw_prefill = canonical_config.pop("prefill", {})
    if not isinstance(raw_prefill, dict):
        raise ValueError("cuda_graph_config.prefill must be a dictionary")
    unknown_keys = set(raw_prefill) - _CANONICAL_PREFILL_KEYS
    if unknown_keys:
        raise ValueError(
            "cuda_graph_config.prefill contains unsupported settings: "
            + ", ".join(sorted(unknown_keys))
        )
    if canonical_config:
        overrides["cuda_graph_config"] = canonical_config
    return canonical_config, dict(raw_prefill)


def _pop_prefill_value(
    values: dict[str, Any],
    label: str,
    *keys: str,
) -> Any:
    value = _get_prefill_value(values, label, *keys)
    for key in keys:
        values.pop(key, None)
    return value


def _prefill_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    return left == right


def _pop_policy_flag(
    overrides: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = overrides.pop(key, _MISSING)
    if value is not _MISSING:
        return bool(value)
    return bool(default)


class AsrEngineBuilder(SGLangGenerationEngineBuilder):
    """Shared lifecycle policy for SGLang-backed ASR stages."""

    def resolve_checkpoint(self, model_path: str) -> str:
        # ASR model loaders accept either a repo id or a local path and should
        # preserve the operator-provided value through server-args creation.
        return model_path

    def validate_before_infrastructure(self, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
        )

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.model_runner.base import ModelRunner

        return ModelRunner(model_worker, output_proc)


class TtsEngineBuilder(SGLangGenerationEngineBuilder):
    """Compatibility builder preserving the historical TTS contract."""

    @abstractmethod
    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        raise NotImplementedError

    def resolve_checkpoint(self, model_path: str) -> str:
        return _resolve_checkpoint(model_path)

    def validate_before_infrastructure(self, server_args: Any) -> None:
        del server_args

    def validate_after_model_setup(self, model: Any, server_args: Any) -> None:
        validate_generation_batch_policy(
            model_name=self.model_name,
            server_args=server_args,
            model_buffer_bs=self.get_model_buffer_bs(model),
        )

    def make_scheduler(
        self,
        *,
        model_worker: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
        model_runner: Any,
        request_builder: Any,
        result_adapter: Any,
    ) -> Any:
        from sglang_omni.scheduling import omni_scheduler

        return omni_scheduler.OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **self.extra_scheduler_kwargs(),
        )

    def _build_runtime(
        self,
        *,
        model_worker: Any,
        model: Any,
        output_proc: Any,
        tree_cache: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        server_args: Any,
        model_config: Any,
        prefill_manager: Any,
        decode_manager: Any,
    ) -> tuple[Any, Any]:
        model_runner = self.make_model_runner(model_worker, output_proc)
        request_builder, result_adapter = self.make_adapters(model)
        scheduler = self.make_scheduler(
            model_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
            model_runner=model_runner,
            request_builder=request_builder,
            result_adapter=result_adapter,
        )
        return scheduler, model_runner
