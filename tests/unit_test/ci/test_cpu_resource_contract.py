# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.utils.ci_resource_contract import (
    _read_cgroup_cpuset,
    collect_cpu_resource_contract,
    format_cpu_list,
    parse_cpu_list,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cpu_list_round_trip() -> None:
    cpus = {0, 1, 2, 7, 64, 65}
    assert parse_cpu_list(format_cpu_list(cpus)) == cpus
    assert format_cpu_list(cpus) == "0-2,7,64-65"


@pytest.mark.parametrize(
    "value",
    ["", "0,,1", ",0", "0,", "-3", "3-", "1--2", "-1", "0-a", "0-1000000"],
)
def test_cpu_list_rejects_invalid_components(value: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_list(value)


def _write_proc_files(
    tmp_path: Path,
    *,
    cgroup: str,
    mountinfo: str,
) -> tuple[Path, Path]:
    proc_cgroup = tmp_path / "cgroup"
    proc_mountinfo = tmp_path / "mountinfo"
    proc_cgroup.write_text(cgroup, encoding="utf-8")
    proc_mountinfo.write_text(mountinfo, encoding="utf-8")
    return proc_cgroup, proc_mountinfo


def test_read_cgroup_v2_private_namespace(tmp_path: Path) -> None:
    mount_point = tmp_path / "cgroup2"
    mount_point.mkdir()
    source = mount_point / "cpuset.cpus.effective"
    source.write_text("0-3\n", encoding="utf-8")
    proc_cgroup, proc_mountinfo = _write_proc_files(
        tmp_path,
        cgroup="0::/\n",
        mountinfo=f"29 23 0:26 / {mount_point} rw - cgroup2 cgroup rw\n",
    )

    assert _read_cgroup_cpuset(proc_cgroup, proc_mountinfo) == ("0-3", str(source))


def test_read_cgroup_v2_host_namespace_scope(tmp_path: Path) -> None:
    mount_point = tmp_path / "cgroup2"
    scope = mount_point / "system.slice/docker.scope"
    scope.mkdir(parents=True)
    (mount_point / "cpuset.cpus.effective").write_text("0-7\n", encoding="utf-8")
    source = scope / "cpuset.cpus.effective"
    source.write_text("2-3\n", encoding="utf-8")
    proc_cgroup, proc_mountinfo = _write_proc_files(
        tmp_path,
        cgroup="0::/system.slice/docker.scope\n",
        mountinfo=f"29 23 0:26 / {mount_point} rw - cgroup2 cgroup rw\n",
    )

    assert _read_cgroup_cpuset(proc_cgroup, proc_mountinfo) == ("2-3", str(source))


def test_read_cgroup_v1_cpuset_hierarchy(tmp_path: Path) -> None:
    mount_point = tmp_path / "cpuset"
    cgroup = mount_point / "docker/id"
    cgroup.mkdir(parents=True)
    source = cgroup / "cpuset.effective_cpus"
    source.write_text("4,5\n", encoding="utf-8")
    proc_cgroup, proc_mountinfo = _write_proc_files(
        tmp_path,
        cgroup="5:cpuset:/docker/id\n",
        mountinfo=f"31 23 0:28 / {mount_point} rw - cgroup cgroup rw,cpuset\n",
    )

    assert _read_cgroup_cpuset(proc_cgroup, proc_mountinfo) == ("4-5", str(source))


def test_read_cgroup_non_root_mount(tmp_path: Path) -> None:
    mount_point = tmp_path / "bound-cgroup"
    mount_point.mkdir()
    source = mount_point / "cpuset.cpus.effective"
    source.write_text("6-7\n", encoding="utf-8")
    proc_cgroup, proc_mountinfo = _write_proc_files(
        tmp_path,
        cgroup="0::/docker/id\n",
        mountinfo=(f"29 23 0:26 /docker/id {mount_point} rw - cgroup2 cgroup rw\n"),
    )

    assert _read_cgroup_cpuset(proc_cgroup, proc_mountinfo) == ("6-7", str(source))


@pytest.mark.parametrize(
    ("cgroup", "mountinfo"),
    [
        ("malformed\n", "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"),
        ("0::/\n", "malformed\n"),
        ("0::/missing\n", "29 23 0:26 / /missing rw - cgroup2 cgroup rw\n"),
        (
            "5:cpu:/docker/id\n",
            "31 23 0:28 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n",
        ),
    ],
)
def test_read_cgroup_returns_unavailable_for_invalid_or_missing_layout(
    tmp_path: Path,
    cgroup: str,
    mountinfo: str,
) -> None:
    proc_cgroup, proc_mountinfo = _write_proc_files(
        tmp_path,
        cgroup=cgroup,
        mountinfo=mountinfo,
    )
    assert _read_cgroup_cpuset(proc_cgroup, proc_mountinfo) == (None, None)


def test_read_cgroup_returns_unavailable_for_unreadable_proc_file(
    tmp_path: Path,
) -> None:
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.mkdir()
    assert _read_cgroup_cpuset(proc_cgroup, tmp_path / "mountinfo") == (None, None)


def test_read_cgroup_returns_unavailable_for_unreadable_mountinfo(
    tmp_path: Path,
) -> None:
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/\n", encoding="utf-8")
    proc_mountinfo = tmp_path / "mountinfo"
    proc_mountinfo.mkdir()
    assert _read_cgroup_cpuset(proc_cgroup, proc_mountinfo) == (None, None)


def test_partition_contract_records_normalized_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tests.utils.ci_resource_contract._read_cgroup_cpuset",
        lambda: ("0-2,7", "/sys/fs/cgroup/job/cpuset.cpus.effective"),
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
    assert evidence["resource_contract"]["cgroup_cpuset_source"] == (
        "/sys/fs/cgroup/job/cpuset.cpus.effective"
    )


def test_partition_contract_rejects_unenforced_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tests.utils.ci_resource_contract._read_cgroup_cpuset",
        lambda: ("0-1", "/sys/fs/cgroup/job/cpuset.cpus.effective"),
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
        lambda: (
            cgroup_cpuset,
            "/sys/fs/cgroup/job/cpuset.cpus.effective" if cgroup_cpuset else None,
        ),
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
    workflow_root = REPO_ROOT / ".github/workflows"
    workflow_paths = sorted(
        (*workflow_root.glob("*.yaml"), *workflow_root.glob("*.yml"))
    )
    for workflow_path in workflow_paths:
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
            assert steps[0].get("uses", "").startswith("actions/checkout@"), location
            assert steps[1].get("uses") == (
                "./.github/actions/validate-omni-ci-cpu-partition"
            ), location
    assert h100_jobs


def test_close_cleanup_runs_before_partition_failure_is_propagated() -> None:
    workflow = yaml.load(
        (REPO_ROOT / ".github/workflows/cleanup-pr-ci-home-on-close.yaml").read_text(
            encoding="utf-8"
        ),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["cleanup-pr-ci-home"]["steps"]
    assert steps[1]["continue-on-error"] == "true"
    assert steps[2]["if"] == "always()"
    assert "steps.validate-cpu-partition.outcome == 'failure'" in steps[3]["if"]


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
    assert "OMNI_CI_RESOURCE_CONTRACT_REQUIRED" in asr
    assert "/tmp/omni-ci-resource-contract.json" in asr
