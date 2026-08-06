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


class SGLangGenerationEngineBuilder(ABC):
    """Build the model-neutral parts of a SGLang AR engine stage.

    Model-specific builders provide checkpoint preprocessing, model setup,
    request/result adapters, validation policy, and any stage-owned resources.
    Family-specific builders such as :class:`AsrEngineBuilder` and
    :class:`TtsEngineBuilder` define the lifecycle policy for each modality.
    """

    model_name: str
    context_length: int
    model_arch_override: str | None = None
    prefill_cuda_graph_backend: str | None = None
    prefill_cuda_graph_buckets: list[int] | tuple[int, ...] | None = None
    allow_experimental_prefill_cuda_graph = False
    allow_performance_unproven_prefill_cuda_graph = False

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

        overrides = build_generation_batch_overrides(
            server_args_overrides=server_args_overrides,
            **self.generation_defaults(dtype=dtype),
        )
        self.adjust_overrides(overrides)
        prefill_policy = self._resolve_prefill_cuda_graph_policy(
            overrides=overrides,
            server_args_overrides=server_args_overrides,
        )
        overrides.update(prefill_cuda_graph_server_args(prefill_policy))
        self._log_prefill_cuda_graph_policy(prefill_policy)

        server_args = sglang_backend.build_sglang_server_args(
            checkpoint_dir,
            context_length=self.context_length,
            **overrides,
        )
        self.customize_server_args(server_args)
        self.validate_before_infrastructure(server_args)

        infra_kwargs = dict(self.infra_kwargs())
        if self.model_arch_override is not None:
            infra_kwargs.setdefault("model_arch_override", self.model_arch_override)
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
        from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

        candidates = (
            self.model_arch_override,
            getattr(self, "architecture", None),
        )
        for architecture in candidates:
            if not isinstance(architecture, str) or not architecture:
                continue
            capabilities = get_model_capabilities(architecture)
            if capabilities is not None:
                return capabilities.prefill_cuda_graph

        model_package = ".".join(self.__class__.__module__.split(".")[:3])
        package_architectures = {
            config_cls.architecture
            for config_cls in set(PIPELINE_CONFIG_REGISTRY.configs.values())
            if ".".join(config_cls.__module__.split(".")[:3]) == model_package
        }
        if len(package_architectures) != 1:
            return None
        architecture = package_architectures.pop()
        capabilities = get_model_capabilities(architecture)
        if capabilities is None:
            return None
        return capabilities.prefill_cuda_graph

    def prefill_cuda_graph_runtime_requirements(self) -> dict[str, bool]:
        return {}

    def _resolve_prefill_cuda_graph_policy(
        self,
        *,
        overrides: dict[str, Any],
        server_args_overrides: dict[str, Any] | None,
    ) -> ResolvedPrefillCudaGraphPolicy:
        incoming = dict(server_args_overrides or {})
        merged_backend = _pop_prefill_value(
            overrides,
            "prefill backend",
            "cuda_graph_backend_prefill",
            "prefill_cuda_graph_backend",
        )
        override_backend = _get_prefill_value(
            incoming,
            "prefill backend",
            "cuda_graph_backend_prefill",
            "prefill_cuda_graph_backend",
        )
        stage_backend = self.prefill_cuda_graph_backend
        if override_backend is not _MISSING:
            if stage_backend is not None and stage_backend != override_backend:
                raise ValueError(
                    "Conflicting Prefill CUDA Graph backend settings: "
                    f"builder={stage_backend!r}, server_args_overrides="
                    f"{override_backend!r}"
                )
            requested_backend = override_backend
        elif stage_backend is not None:
            if merged_backend is not _MISSING and merged_backend != stage_backend:
                raise ValueError(
                    "Conflicting Prefill CUDA Graph backend settings: "
                    f"builder={stage_backend!r}, stage={merged_backend!r}"
                )
            requested_backend = stage_backend
        elif merged_backend is not _MISSING:
            requested_backend = merged_backend
        else:
            requested_backend = "disabled"

        merged_buckets = _pop_prefill_value(
            overrides,
            "prefill buckets",
            "cuda_graph_bs_prefill",
            "prefill_cuda_graph_buckets",
        )
        override_buckets = _get_prefill_value(
            incoming,
            "prefill buckets",
            "cuda_graph_bs_prefill",
            "prefill_cuda_graph_buckets",
        )
        stage_buckets = self.prefill_cuda_graph_buckets
        if override_buckets is not _MISSING:
            if stage_buckets is not None and tuple(stage_buckets) != tuple(
                override_buckets
            ):
                raise ValueError(
                    "Conflicting Prefill CUDA Graph bucket settings: "
                    f"builder={stage_buckets!r}, server_args_overrides="
                    f"{override_buckets!r}"
                )
            requested_buckets = override_buckets
        elif stage_buckets is not None:
            if merged_buckets is not _MISSING and tuple(merged_buckets) != tuple(
                stage_buckets
            ):
                raise ValueError(
                    "Conflicting Prefill CUDA Graph bucket settings: "
                    f"builder={stage_buckets!r}, stage={merged_buckets!r}"
                )
            requested_buckets = stage_buckets
        elif merged_buckets is not _MISSING:
            requested_buckets = merged_buckets
        else:
            requested_buckets = None

        allow_experimental = _pop_policy_flag(
            overrides,
            incoming,
            "allow_experimental_prefill_cuda_graph",
            self.allow_experimental_prefill_cuda_graph,
        )
        allow_performance_unproven = _pop_policy_flag(
            overrides,
            incoming,
            "allow_performance_unproven_prefill_cuda_graph",
            self.allow_performance_unproven_prefill_cuda_graph,
        )
        requested_max_bs = overrides.pop("cuda_graph_max_bs_prefill", _MISSING)

        if requested_backend == "disabled" and requested_buckets is not None:
            raise ValueError(
                "Prefill CUDA Graph buckets cannot be set when the backend is disabled"
            )
        if requested_backend == "disabled" and requested_max_bs is not _MISSING:
            raise ValueError(
                "cuda_graph_max_bs_prefill cannot be set when the Prefill CUDA "
                "Graph backend is disabled"
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
            requested_buckets=requested_buckets,
            allow_experimental=allow_experimental,
            allow_performance_unproven=allow_performance_unproven,
            runtime_requirements=(
                None
                if requested_backend == "disabled"
                else self.prefill_cuda_graph_runtime_requirements()
            ),
        )
        if requested_max_bs is not _MISSING and requested_max_bs != max(policy.buckets):
            raise ValueError(
                "Conflicting Prefill CUDA Graph maximum bucket: "
                f"cuda_graph_max_bs_prefill={requested_max_bs!r}, "
                f"resolved maximum={max(policy.buckets)!r}"
            )
        return policy

    def _log_prefill_cuda_graph_policy(
        self,
        policy: ResolvedPrefillCudaGraphPolicy,
    ) -> None:
        if not policy.enabled:
            logger.info(
                "Prefill CUDA Graph policy: model=%s requested=%s enabled=false",
                self.model_name,
                policy.requested_backend,
            )
            return
        buckets = ",".join(str(bucket) for bucket in policy.buckets)
        logger.info(
            "Prefill CUDA Graph policy: model=%s requested=%s resolved=%s "
            "integration=%s status=%s buckets=[%s]",
            self.model_name,
            policy.requested_backend,
            policy.resolved_backend,
            policy.integration,
            policy.status,
            buckets,
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
    present = [(key, values[key]) for key in keys if key in values]
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
    incoming: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = overrides.pop(key, _MISSING)
    if key in incoming:
        return bool(incoming[key])
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
