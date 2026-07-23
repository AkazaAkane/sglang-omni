# SPDX-License-Identifier: Apache-2.0
"""Explicit process-placement overrides for pipeline stages."""

from __future__ import annotations

from sglang_omni.config.schema import PipelineConfig


def apply_stage_process_overrides(
    pipeline_config: PipelineConfig,
    *,
    isolate_stages: list[str] | None = None,
) -> PipelineConfig:
    """Return a config with selected non-TP stages or roles in their own process."""
    if not isolate_stages:
        return pipeline_config

    config = pipeline_config.model_copy(deep=True)
    stages = {stage.name: stage for stage in config.stages}
    role_map = type(config).isolation_role_to_stage()

    for requested_name in isolate_stages:
        stage = stages.get(requested_name)
        if stage is None:
            resolved_name = role_map.get(requested_name)
            stage = stages.get(resolved_name) if resolved_name is not None else None
        if stage is None:
            raise ValueError(f"Unknown stage or isolation role: {requested_name}")
        if stage.tp_size > 1:
            raise ValueError(
                f"Stage {stage.name!r} already uses one process per TP rank"
            )
        stage.process = stage.name

    return config
