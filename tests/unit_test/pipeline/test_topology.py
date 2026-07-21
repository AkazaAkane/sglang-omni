# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    apply_stage_process_overrides,
    build_process_topology_plan,
    build_stage_placement_plan,
)
from sglang_omni.config.manager import ConfigManager

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(
    name: str,
    *,
    gpu: int | list[int] | None = None,
    fraction: float | None = None,
    process: str | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        factory=_FACTORY,
        gpu=gpu,
        process=process,
        tp_size=tp_size,
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=fraction)
        ),
        next=next_stage,
        terminal=terminal,
    )


def _topology(config: PipelineConfig):
    gpu_placement = build_stage_placement_plan(config)
    return build_process_topology_plan(config, gpu_placement)


def test_stage_process_parses_from_schema_and_dotted_overrides() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="old0", next_stage="b"),
            _stage("b", process="old1", terminal=True),
        ],
    )

    merged = ConfigManager(config).merge_config(
        {"stages.0.process": "p0", "stages.1.process": "p1"}
    )

    assert [stage.process for stage in merged.stages] == ["p0", "p1"]


def test_process_override_none_preserves_declared_topology() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(config, isolate_stages=None)

    assert overridden is config
    assert [stage.process for stage in overridden.stages] == [
        "pipeline",
        "pipeline",
    ]


def test_process_override_isolates_named_stage_without_mutating_source() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(config, isolate_stages=["b"])

    assert overridden is not config
    assert [stage.process for stage in config.stages] == ["pipeline", "pipeline"]
    assert [stage.process for stage in overridden.stages] == ["pipeline", "b"]


def test_process_override_isolates_multiple_stages_separately() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )

    overridden = apply_stage_process_overrides(
        config,
        isolate_stages=["b", "c"],
    )

    assert [stage.process for stage in overridden.stages] == ["pipeline", "b", "c"]


def test_process_override_rejects_unknown_stage() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("a", process="pipeline", terminal=True)],
    )

    with pytest.raises(ValueError, match="Unknown stage: missing"):
        apply_stage_process_overrides(config, isolate_stages=["missing"])


def test_process_override_rejects_tp_stage() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "thinker",
                gpu=[0, 1],
                tp_size=2,
                terminal=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="already uses one process per TP rank"):
        apply_stage_process_overrides(config, isolate_stages=["thinker"])


def test_process_override_same_gpu_requires_memory_fractions() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="pipeline", next_stage="b"),
            _stage("b", gpu=0, process="pipeline", terminal=True),
        ],
    )
    overridden = apply_stage_process_overrides(config, isolate_stages=["b"])
    gpu_placement = build_stage_placement_plan(overridden)

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_process_topology_plan(overridden, gpu_placement)


def test_process_override_same_gpu_accepts_valid_memory_fractions() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "a",
                gpu=0,
                fraction=0.40,
                process="pipeline",
                next_stage="b",
            ),
            _stage(
                "b",
                gpu=0,
                fraction=0.40,
                process="pipeline",
                terminal=True,
            ),
        ],
    )
    overridden = apply_stage_process_overrides(config, isolate_stages=["b"])

    topology = _topology(overridden)

    assert [(group.name, group.stage_names) for group in topology.groups] == [
        ("pipeline", ("a",)),
        ("b", ("b",)),
    ]


def test_serve_cli_accepts_repeatable_isolate_stage(monkeypatch) -> None:
    import importlib

    from typer.testing import CliRunner

    from sglang_omni.cli import app

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="pipeline", next_stage="b"),
            _stage("b", process="pipeline", next_stage="c"),
            _stage("c", process="pipeline", terminal=True),
        ],
    )
    launched: list[PipelineConfig] = []
    serve_module = importlib.import_module("sglang_omni.cli.serve")
    monkeypatch.setattr(
        ConfigManager,
        "from_file",
        staticmethod(lambda _path: ConfigManager(config)),
    )
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **_kwargs: launched.append(pipeline_config),
    )

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            "ignored.yaml",
            "--isolate-stage",
            "b",
            "--isolate-stage",
            "c",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(launched) == 1
    assert [stage.process for stage in launched[0].stages] == ["pipeline", "b", "c"]


def test_non_tp_stages_must_declare_process() -> None:
    with pytest.raises(ValueError, match="Non-TP stages must declare process"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", process="p0", next_stage="b"),
                _stage("b", terminal=True),
            ],
        )


def test_missing_non_tp_process_declaration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Non-TP stages must declare process"):
        PipelineConfig(
            model_path="dummy",
            stages=[_stage("a", next_stage="b"), _stage("b", terminal=True)],
        )


def test_tp_process_names_are_derived_when_process_is_missing() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("thinker", gpu=[0, 1], tp_size=2, terminal=True)],
    )

    topology = _topology(config)

    assert topology.groups == ()
    assert topology.tp_stage_to_processes == {"thinker": ("thinker_tp0", "thinker_tp1")}


def test_tp_process_field_is_used_as_rank_process_prefix() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "thinker",
                gpu=[0, 1],
                tp_size=2,
                process="model",
                terminal=True,
            )
        ],
    )

    topology = _topology(config)

    assert topology.tp_stage_to_processes == {"thinker": ("model_tp0", "model_tp1")}


def test_same_process_same_gpu_does_not_require_memory_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p0", terminal=True),
        ],
    )

    topology = _topology(config)

    assert [
        (group.name, group.stage_names, group.gpu_id) for group in topology.groups
    ] == [("p0", ("a", "b"), 0)]


def test_same_gpu_multiple_processes_accepts_explicit_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.20, process="p0", next_stage="b"),
            _stage("b", gpu=0, fraction=0.30, process="p0", next_stage="c"),
            _stage("c", gpu=0, fraction=0.40, process="p1", terminal=True),
        ],
    )

    topology = _topology(config)

    assert [
        (group.name, group.stage_names, group.gpu_id) for group in topology.groups
    ] == [
        ("p0", ("a", "b"), 0),
        ("p1", ("c",), 0),
    ]


def test_same_gpu_multiple_processes_rejects_missing_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.20, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p1", terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_process_topology_plan(config, gpu_placement)


def test_same_gpu_multiple_processes_rejects_over_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.70, process="p0", next_stage="b"),
            _stage("b", gpu=0, fraction=0.40, process="p1", terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(config)


def test_one_process_group_cannot_span_multiple_gpus() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=1, process="p0", terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="spans multiple GPUs"):
        build_process_topology_plan(config, gpu_placement)


def test_tp_process_names_must_not_collide_with_non_tp_process_group() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="thinker_tp0", next_stage="thinker"),
            _stage("thinker", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="collide"):
        build_process_topology_plan(config, gpu_placement)


def test_tp_process_names_must_be_unique_across_tp_stages() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=[0, 1], tp_size=2, process="model", next_stage="b"),
            _stage("b", gpu=[2, 3], tp_size=2, process="model", terminal=True),
        ],
    )
    gpu_placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="Duplicate TP process names"):
        build_process_topology_plan(config, gpu_placement)
