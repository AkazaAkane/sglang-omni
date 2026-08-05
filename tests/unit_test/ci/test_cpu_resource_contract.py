# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.utils.ci_resource_contract import (
    collect_cpu_resource_contract,
    format_cpu_list,
    parse_cpu_list,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cpu_list_round_trip() -> None:
    cpus = {0, 1, 2, 7, 64, 65}
    assert parse_cpu_list(format_cpu_list(cpus)) == cpus
    assert format_cpu_list(cpus) == "0-2,7,64-65"


def test_partition_contract_records_normalized_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tests.utils.ci_resource_contract._read_cgroup_cpuset", lambda: "0-2,7"
    )
    evidence = collect_cpu_resource_contract(
        environ={
            "OMNI_CI_CPUSET": "0-2,7",
            "OMNI_CI_CPUSET_TOPOLOGY_VERSION": "h100-v1",
            "NVIDIA_VISIBLE_DEVICES": "0,1",
        },
        effective_cpus={0, 1, 2, 7},
        require_partition=True,
    )
    assert evidence["valid"] is True
    assert evidence["resource_contract"]["mode"] == "partitioned"
    assert evidence["resource_contract"]["requested_cpuset"] == "0-2,7"
    assert evidence["resource_contract"]["effective_cpuset"] == "0-2,7"
    assert evidence["resource_contract"]["cgroup_cpuset"] == "0-2,7"


def test_partition_contract_rejects_unenforced_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tests.utils.ci_resource_contract._read_cgroup_cpuset", lambda: "0-1"
    )
    evidence = collect_cpu_resource_contract(
        environ={
            "OMNI_CI_CPUSET": "0-1",
            "OMNI_CI_CPUSET_TOPOLOGY_VERSION": "h100-v1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        effective_cpus={0, 1, 2, 3},
        require_partition=True,
    )
    assert evidence["valid"] is False
    assert "effective affinity does not match" in evidence["errors"][0]


@pytest.mark.parametrize("cgroup_cpuset", [None, "0-3"])
def test_partition_contract_rejects_missing_or_incompatible_cgroup_boundary(
    monkeypatch: pytest.MonkeyPatch,
    cgroup_cpuset: str | None,
) -> None:
    monkeypatch.setattr(
        "tests.utils.ci_resource_contract._read_cgroup_cpuset",
        lambda: cgroup_cpuset,
    )
    evidence = collect_cpu_resource_contract(
        environ={
            "OMNI_CI_CPUSET": "0-1",
            "OMNI_CI_CPUSET_TOPOLOGY_VERSION": "h100-v1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        effective_cpus={0, 1},
        require_partition=True,
    )
    assert evidence["valid"] is False
    assert any("cgroup cpuset" in error for error in evidence["errors"])


def test_partition_contract_requires_runner_metadata() -> None:
    evidence = collect_cpu_resource_contract(
        environ={},
        effective_cpus={0, 1},
        require_partition=True,
    )
    assert evidence["valid"] is False
    assert evidence["errors"] == [
        "OMNI_CI_CPUSET is required for partitioned CI",
        "NVIDIA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is required",
    ]


def test_h100_containers_receive_the_runner_partition() -> None:
    h100_jobs: list[str] = []
    for workflow_path in sorted((REPO_ROOT / ".github/workflows").glob("*.yaml")):
        workflow = yaml.load(
            workflow_path.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        for job_name, job in workflow["jobs"].items():
            runners = job.get("runs-on", [])
            if isinstance(runners, str):
                runners = [runners]
            if "h100" not in runners:
                continue
            location = f"{workflow_path.name}:{job_name}"
            h100_jobs.append(location)
            options = job["container"]["options"]
            assert "-e OMNI_CI_CPUSET" in options, location
            assert "-e OMNI_CI_CPUSET_TOPOLOGY_VERSION" in options, location
            steps = job["steps"]
            assert steps[0].get("uses") == "actions/checkout@v4", location
            assert steps[1].get("uses") == (
                "./.github/actions/validate-omni-ci-cpu-partition"
            ), location
    assert h100_jobs


def test_validator_runs_with_isolated_system_python(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(REPO_ROOT / ".github/scripts/validate_omni_ci_cpu_partition.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_asr_stage_2_validates_and_uploads_partition_evidence() -> None:
    validator = (
        REPO_ROOT / ".github/actions/validate-omni-ci-cpu-partition/action.yaml"
    ).read_text(encoding="utf-8")
    asr = (REPO_ROOT / ".github/workflows/test-asr-ci.yaml").read_text(encoding="utf-8")
    assert "validate_omni_ci_cpu_partition.py" in validator
    assert "/tmp/omni-ci-resource-contract.json" in asr
