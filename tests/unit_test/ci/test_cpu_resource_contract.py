# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

from tests.utils.ci_resource_contract import (
    collect_cpu_resource_contract,
    format_cpu_list,
    parse_cpu_list,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
H100_WORKFLOWS = (
    "cleanup-pr-ci-home-on-close.yaml",
    "omni-ci.yaml",
    "test.yaml",
    "test-asr-ci.yaml",
    "test-tts-ci.yaml",
    "test-qwen3-omni-ci.yaml",
)


def test_cpu_list_round_trip() -> None:
    cpus = {0, 1, 2, 7, 64, 65}
    assert parse_cpu_list(format_cpu_list(cpus)) == cpus
    assert format_cpu_list(cpus) == "0-2,7,64-65"


def test_partition_contract_records_normalized_affinity() -> None:
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


def test_partition_contract_rejects_unenforced_affinity() -> None:
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
    for name in H100_WORKFLOWS:
        workflow = yaml.load(
            (REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        for job_name, job in workflow["jobs"].items():
            if "h100" not in job.get("runs-on", []):
                continue
            options = job["container"]["options"]
            assert "-e OMNI_CI_CPUSET" in options, f"{name}:{job_name}"
            assert "-e OMNI_CI_CPUSET_TOPOLOGY_VERSION" in options, f"{name}:{job_name}"


def test_asr_stage_2_validates_and_uploads_partition_evidence() -> None:
    setup = (REPO_ROOT / ".github/actions/omni-setup/action.yaml").read_text(
        encoding="utf-8"
    )
    asr = (REPO_ROOT / ".github/workflows/test-asr-ci.yaml").read_text(encoding="utf-8")
    assert "validate_omni_ci_cpu_partition.py" in setup
    assert "/tmp/omni-ci-resource-contract.json" in asr
