# SPDX-License-Identifier: Apache-2.0
"""Explicit process-placement overrides for pipeline stages."""

from __future__ import annotations

from sglang_omni.config.schema import PipelineConfig


def apply_stage_process_overrides(
    pipeline_config: PipelineConfig,
    *,
    isolate_stages: list[str] | None = None,
) -> PipelineConfig:
    """Return a config with selected non-TP stages placed in their own process."""
    if not isolate_stages:
        return pipeline_config

    config = pipeline_config.model_copy(deep=True)
    stages = {stage.name: stage for stage in config.stages}

    for stage_name in isolate_stages:
        stage = stages.get(stage_name)
        if stage is None:
            raise ValueError(f"Unknown stage: {stage_name}")
        if stage.tp_size > 1:
            raise ValueError(
                f"Stage {stage_name!r} already uses one process per TP rank"
            )
        stage.process = stage.name

    return config
