# SPDX-License-Identifier: Apache-2.0
"""CPU resource-contract evidence for host-sensitive CI benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        start, separator, end = item.strip().partition("-")
        if not start:
            continue
        first = int(start)
        last = int(end) if separator else first
        if last < first:
            raise ValueError(f"invalid CPU range {item!r}")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError("CPU list selects no CPUs")
    return cpus


def format_cpu_list(cpus: set[int]) -> str:
    ordered = sorted(cpus)
    if not ordered:
        raise ValueError("CPU list selects no CPUs")
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _read_cgroup_cpuset(
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> str | None:
    if not proc_cgroup.is_file():
        return None
    for line in proc_cgroup.read_text(encoding="utf-8").splitlines():
        _, controllers, raw_path = line.split(":", 2)
        relative = raw_path.lstrip("/")
        if not controllers:
            candidates = [cgroup_root / relative / "cpuset.cpus.effective"]
        elif "cpuset" in controllers.split(","):
            base = cgroup_root / "cpuset" / relative
            candidates = [base / "cpuset.effective_cpus", base / "cpuset.cpus"]
        else:
            continue
        for candidate in candidates:
            if candidate.is_file() and (value := candidate.read_text().strip()):
                return format_cpu_list(parse_cpu_list(value))
    return None


def collect_cpu_resource_contract(
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
    effective_cpus: set[int] | None = None,
    require_partition: bool = False,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    requested_spec = env.get("OMNI_CI_CPUSET", "").strip()
    topology_version = env.get("OMNI_CI_CPUSET_TOPOLOGY_VERSION", "").strip()
    visible_devices = (
        env.get("NVIDIA_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES") or ""
    ).strip()
    effective = (
        set(os.sched_getaffinity(0)) if effective_cpus is None else effective_cpus
    )
    errors: list[str] = []
    requested: set[int] | None = None
    if requested_spec:
        try:
            requested = parse_cpu_list(requested_spec)
        except ValueError as exc:
            errors.append(f"OMNI_CI_CPUSET is invalid: {exc}")
    elif require_partition:
        errors.append("OMNI_CI_CPUSET is required for partitioned CI")
    if requested is not None and requested != effective:
        errors.append(
            "effective affinity does not match OMNI_CI_CPUSET: "
            f"requested={format_cpu_list(requested)} "
            f"effective={format_cpu_list(effective)}"
        )
    if requested_spec and not topology_version:
        errors.append("OMNI_CI_CPUSET_TOPOLOGY_VERSION is required with OMNI_CI_CPUSET")
    if require_partition and not visible_devices:
        errors.append("NVIDIA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is required")

    return {
        "resource_contract": {
            "mode": "partitioned" if requested_spec else "none",
            "requested_cpuset": (
                format_cpu_list(requested) if requested is not None else requested_spec
            ),
            "effective_cpuset": format_cpu_list(effective),
            "cgroup_cpuset": _read_cgroup_cpuset(),
            "visible_devices": visible_devices or None,
            "topology_version": topology_version or None,
        },
        "valid": not errors,
        "errors": errors,
    }
