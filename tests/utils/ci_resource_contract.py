# SPDX-License-Identifier: Apache-2.0
"""CPU resource-contract evidence for host-sensitive CI benchmarks."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

# Note (Akazaakane): Cpulists come from runner-controlled text; cap expansion
# well above Linux CPU limits so malformed ranges cannot allocate unbounded sets.
_MAX_CPU_LIST_EXPANSION = 1_000_000


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    selected_count = 0
    for item in value.split(","):
        item = item.strip()
        if not item or item.count("-") > 1:
            raise ValueError(f"invalid CPU range {item!r}")
        start, separator, end = item.partition("-")
        if not start.isascii() or not start.isdecimal():
            raise ValueError(f"invalid CPU range {item!r}")
        if separator and (not end.isascii() or not end.isdecimal()):
            raise ValueError(f"invalid CPU range {item!r}")
        first = int(start)
        last = int(end) if separator else first
        if last < first:
            raise ValueError(f"invalid CPU range {item!r}")
        selected_count += last - first + 1
        if selected_count > _MAX_CPU_LIST_EXPANSION:
            raise ValueError("CPU list selects too many CPUs")
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


def _decode_mountinfo_path(value: str) -> str:
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _cgroup_cpuset_candidates(
    *,
    mountinfo: str,
    controllers: str,
    raw_path: str,
) -> list[Path]:
    candidates: list[Path] = []
    cgroup_path = PurePosixPath(raw_path)
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) <= separator + 3:
            continue
        filesystem = fields[separator + 1]
        super_options = fields[separator + 3].split(",")
        if controllers:
            if filesystem != "cgroup" or "cpuset" not in super_options:
                continue
            filenames = ("cpuset.effective_cpus", "cpuset.cpus")
        else:
            if filesystem != "cgroup2":
                continue
            filenames = ("cpuset.cpus.effective",)
        mount_root = PurePosixPath(_decode_mountinfo_path(fields[3]))
        try:
            relative = cgroup_path.relative_to(mount_root)
        except ValueError:
            continue
        mount_point = Path(_decode_mountinfo_path(fields[4]))
        # Note (Akazaakane): mountinfo's root maps the hierarchy path from
        # /proc/self/cgroup into the process-visible mount point.
        base = mount_point.joinpath(*relative.parts)
        candidates.extend(base / filename for filename in filenames)
    return candidates


def _read_cgroup_cpuset(
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    proc_mountinfo: Path = Path("/proc/self/mountinfo"),
) -> tuple[str | None, str | None]:
    try:
        cgroup_lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
        mountinfo = proc_mountinfo.read_text(encoding="utf-8")
    except OSError:
        return None, None
    for line in cgroup_lines:
        try:
            _, controllers, raw_path = line.split(":", 2)
        except ValueError:
            continue
        if controllers and "cpuset" not in controllers.split(","):
            continue
        candidates = _cgroup_cpuset_candidates(
            mountinfo=mountinfo,
            controllers=controllers,
            raw_path=raw_path,
        )
        for candidate in candidates:
            try:
                value = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not value:
                continue
            try:
                return format_cpu_list(parse_cpu_list(value)), str(candidate)
            except ValueError:
                continue
    return None, None


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
    cgroup_cpuset, cgroup_cpuset_source = _read_cgroup_cpuset()
    if requested is not None:
        if cgroup_cpuset is None:
            errors.append("effective cgroup cpuset is unavailable")
        elif parse_cpu_list(cgroup_cpuset) != requested:
            errors.append(
                "effective cgroup cpuset does not match OMNI_CI_CPUSET: "
                f"requested={format_cpu_list(requested)} "
                f"cgroup={cgroup_cpuset}"
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
            "cgroup_cpuset": cgroup_cpuset,
            "cgroup_cpuset_source": cgroup_cpuset_source,
            "visible_devices": visible_devices or None,
            "topology_version": topology_version or None,
        },
        "valid": not errors,
        "errors": errors,
    }
