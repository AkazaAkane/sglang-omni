#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_SUFFIXES = (".so", ".o")
COMPILER_NAMES = ("cc1plus", "cicc", "ninja", "nvcc", "ptxas")
COMPILER_DRIVERS = ("ninja", "nvcc")
COMPILER_BUCKETS = ("flashinfer", "torchinductor", "other")
MONITOR_INTERVAL_SECONDS = 1.0


def _state_dir() -> Path:
    return Path(os.environ["RUNNER_TEMP"]) / "flashinfer-cache-stats"


def _run_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _package_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _torch_environment() -> tuple[str, str]:
    try:
        torch = importlib.import_module("torch")
        return (
            str(torch.__version__),
            str(torch.version.cuda or "unknown"),
        )
    except ImportError:
        return "unknown", "unknown"


def _gpu_arches() -> list[str]:
    output = _run_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    )
    if output == "unknown":
        return ["unknown"]
    arches = set()
    for line in output.splitlines():
        capability = line.strip()
        major, separator, minor = capability.partition(".")
        if separator and major.isdigit() and minor.isdigit():
            arches.add(f"sm{major}{minor}")
    return sorted(arches) or ["unknown"]


def _deps_hash() -> str:
    path = Path("pyproject.toml")
    if not path.is_file():
        return "unknown"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> tuple[str, dict[str, Any], list[str]]:
    torch_version, torch_cuda_version = _torch_environment()
    nvcc_version = _run_output(["nvcc", "--version"])
    components = {
        "dependency_hash": _deps_hash(),
        "flashinfer_version": _package_version(
            "flashinfer-python", "flashinfer-python-cu12"
        ),
        "gpu_arches": _gpu_arches(),
        "nvcc_version": nvcc_version,
        "torch_cuda_version": torch_cuda_version,
        "torch_version": torch_version,
    }
    unknown_fields = sorted(
        key
        for key, value in components.items()
        if value == "unknown"
        or (isinstance(value, list) and (not value or "unknown" in value))
    )
    encoded = json.dumps(components, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:20], components, unknown_fields


def _cache_root() -> tuple[Path, str]:
    override = os.environ.get("FLASHINFER_CACHE_DIR")
    if override:
        return Path(override).expanduser(), "FLASHINFER_CACHE_DIR"
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if workspace:
        return Path(workspace).expanduser() / ".cache" / "flashinfer", "workspace"
    return Path.home() / ".cache" / "flashinfer", "default"


def _manifest(cache_root: Path) -> tuple[dict[str, dict[str, int]], int]:
    artifacts: dict[str, dict[str, int]] = {}
    cache_bytes = 0
    if not cache_root.is_dir():
        return artifacts, cache_bytes
    for path in cache_root.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        cache_bytes += stat.st_size
        if path.name.endswith(ARTIFACT_SUFFIXES):
            artifacts[path.relative_to(cache_root).as_posix()] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
    return dict(sorted(artifacts.items())), cache_bytes


def _write_manifest(path: Path, artifacts: dict[str, dict[str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("path_json\tsize\tmtime_ns\n")
        for artifact_path, metadata in artifacts.items():
            handle.write(
                f"{json.dumps(artifact_path)}\t{metadata['size']}\t"
                f"{metadata['mtime_ns']}\n"
            )


def _read_manifest(path: Path) -> dict[str, dict[str, int]]:
    artifacts: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return artifacts
    with path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            try:
                artifact_path_json, size, mtime_ns = line.rstrip("\n").split("\t")
                artifact_path = json.loads(artifact_path_json)
                if not isinstance(artifact_path, str):
                    continue
                artifacts[artifact_path] = {
                    "mtime_ns": int(mtime_ns),
                    "size": int(size),
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return artifacts


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pr_number(ci_home: str) -> int | None:
    name = Path(ci_home).name
    if not name.startswith("pr-"):
        return None
    try:
        return int(name.removeprefix("pr-"))
    except ValueError:
        return None


def _utc_timestamp(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, timezone.utc).isoformat()


def _path_matches_root(path: str, root: Path) -> bool:
    if not path:
        return False
    try:
        return os.path.commonpath((path, str(root))) == str(root)
    except ValueError:
        return False


def _compiler_bucket(
    cwd: str, cmdline: str, cache_root: Path, inductor_root: Path
) -> tuple[str, str]:
    if _path_matches_root(cwd, cache_root):
        return "flashinfer", "cwd"
    if _path_matches_root(cwd, inductor_root):
        return "torchinductor", "cwd"
    if str(cache_root) in cmdline or "flashinfer" in cmdline.lower():
        return "flashinfer", "cmdline"
    if str(inductor_root) in cmdline or "torchinductor" in cmdline.lower():
        return "torchinductor", "cmdline"
    return "other", "unattributed"


def _compiler_processes(cache_root: Path, inductor_root: Path) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            name = (proc_dir / "comm").read_text(encoding="utf-8").strip()
            if name not in COMPILER_NAMES:
                continue
            stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
            _, separator, stat_tail = stat_text.rpartition(") ")
            stat_fields = stat_tail.split()
            if not separator or len(stat_fields) <= 19:
                continue
            cpu_ticks = int(stat_fields[11]) + int(stat_fields[12])
            start_ticks = int(stat_fields[19])
            try:
                cwd = os.readlink(proc_dir / "cwd")
            except OSError:
                cwd = ""
            cmdline = (
                (proc_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        bucket, match_source = _compiler_bucket(cwd, cmdline, cache_root, inductor_root)
        processes.append(
            {
                "bucket": bucket,
                "cpu_ticks": cpu_ticks,
                "cwd": cwd,
                "match_source": match_source,
                "name": name,
                "pid": int(proc_dir.name),
                "start_ticks": start_ticks,
            }
        )
    return sorted(processes, key=lambda process: process["pid"])


def monitor(state_dir: Path, cache_root: Path, inductor_root: Path) -> None:
    samples_path = state_dir / "compiler_samples.jsonl"
    stop_path = state_dir / "monitor.stop"
    clock_ticks_per_second = os.sysconf("SC_CLK_TCK")
    monitor_started_boottime_ticks = int(
        time.clock_gettime(time.CLOCK_BOOTTIME) * clock_ticks_per_second
    )
    observed_processes: set[tuple[int, int]] = set()
    previously_active = False
    with samples_path.open("a", encoding="utf-8") as handle:
        while True:
            stopping = stop_path.exists()
            processes = _compiler_processes(cache_root, inductor_root)
            for process in processes:
                process_key = (process["pid"], process["start_ticks"])
                if process_key in observed_processes:
                    process.pop("cwd")
                else:
                    observed_processes.add(process_key)
            sample = {
                "clock_ticks_per_second": clock_ticks_per_second,
                "monitor_started_boottime_ticks": monitor_started_boottime_ticks,
                "monotonic_ns": time.monotonic_ns(),
                "processes": processes,
                "timestamp_ns": time.time_ns(),
            }
            if processes or previously_active or stopping:
                handle.write(json.dumps(sample, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
                handle.flush()
            previously_active = bool(processes)
            if stopping:
                return
            time.sleep(MONITOR_INTERVAL_SECONDS)


def _start_monitor(state_dir: Path, cache_root: Path, inductor_root: Path) -> None:
    environment = os.environ.copy()
    environment.pop("RUNNER_TRACKING_ID", None)
    with Path(os.devnull).open("wb") as devnull:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "monitor",
                "--state-dir",
                str(state_dir),
                "--cache-root",
                str(cache_root),
                "--inductor-root",
                str(inductor_root),
            ],
            env=environment,
            start_new_session=True,
            stderr=devnull,
            stdout=devnull,
        )
    (state_dir / "monitor.pid").write_text(str(process.pid), encoding="utf-8")


def _stop_monitor(state_dir: Path) -> None:
    (state_dir / "monitor.stop").touch()
    pid_path = state_dir / "monitor.pid"
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8"))
    except ValueError:
        return
    for _ in range(30):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)


def _compiler_summary(state_dir: Path) -> dict[str, Any]:
    samples_path = state_dir / "compiler_samples.jsonl"
    samples: list[dict[str, Any]] = []
    if samples_path.is_file():
        with samples_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(sample, dict):
                    samples.append(sample)
    samples.sort(
        key=lambda sample: (
            sample.get("monotonic_ns")
            if isinstance(sample.get("monotonic_ns"), int)
            else 0
        )
    )
    clock_ticks_per_second = next(
        (
            sample.get("clock_ticks_per_second")
            for sample in samples
            if isinstance(sample.get("clock_ticks_per_second"), int)
        ),
        os.sysconf("SC_CLK_TCK"),
    )
    monitor_started_ticks = next(
        (
            sample.get("monitor_started_boottime_ticks")
            for sample in samples
            if isinstance(sample.get("monitor_started_boottime_ticks"), int)
        ),
        None,
    )
    bucket_data: dict[str, dict[str, Any]] = {
        bucket: {
            "active_sample_count": 0,
            "active_wall_ns": 0,
            "attribution_observations": Counter(),
            "cpu_ticks": 0,
            "first_active_ns": None,
            "last_active_end_ns": None,
            "observed_pids": set(),
            "peak_matched_processes": 0,
            "peak_worker_processes": 0,
            "sampled_process_ns_by_name": Counter(),
            "worker_process_ns": 0,
        }
        for bucket in COMPILER_BUCKETS
    }
    last_cpu_ticks: dict[tuple[int, int], int] = {}
    active_sample_count = 0
    for index, sample in enumerate(samples):
        monotonic_ns = sample.get("monotonic_ns")
        if not isinstance(monotonic_ns, int):
            continue
        next_monotonic_ns = monotonic_ns
        if index + 1 < len(samples):
            candidate = samples[index + 1].get("monotonic_ns")
            if isinstance(candidate, int):
                next_monotonic_ns = max(candidate, monotonic_ns)
        interval_ns = next_monotonic_ns - monotonic_ns
        raw_processes = sample.get("processes", [])
        processes = raw_processes if isinstance(raw_processes, list) else []
        valid_processes = [
            process for process in processes if isinstance(process, dict)
        ]
        if valid_processes:
            active_sample_count += 1
        for bucket in COMPILER_BUCKETS:
            attributed = [
                process
                for process in valid_processes
                if process.get("bucket") == bucket
            ]
            if not attributed:
                continue
            data = bucket_data[bucket]
            workers = [
                process
                for process in attributed
                if process.get("name") not in COMPILER_DRIVERS
            ]
            data["active_sample_count"] += 1
            data["active_wall_ns"] += interval_ns
            data["worker_process_ns"] += len(workers) * interval_ns
            data["peak_matched_processes"] = max(
                data["peak_matched_processes"], len(attributed)
            )
            data["peak_worker_processes"] = max(
                data["peak_worker_processes"], len(workers)
            )
            if data["first_active_ns"] is None:
                data["first_active_ns"] = monotonic_ns
            data["last_active_end_ns"] = next_monotonic_ns
            for process in attributed:
                name = process.get("name")
                pid = process.get("pid")
                start_ticks = process.get("start_ticks")
                cpu_ticks = process.get("cpu_ticks")
                if isinstance(name, str):
                    data["sampled_process_ns_by_name"][name] += interval_ns
                match_source = process.get("match_source")
                if isinstance(match_source, str):
                    data["attribution_observations"][match_source] += 1
                if not all(
                    isinstance(value, int) for value in (pid, start_ticks, cpu_ticks)
                ):
                    continue
                process_key = (pid, start_ticks)
                previous_ticks = last_cpu_ticks.get(process_key)
                if previous_ticks is None:
                    delta_ticks = (
                        cpu_ticks
                        if monitor_started_ticks is not None
                        and start_ticks >= monitor_started_ticks
                        else 0
                    )
                else:
                    delta_ticks = max(cpu_ticks - previous_ticks, 0)
                last_cpu_ticks[process_key] = cpu_ticks
                data["cpu_ticks"] += delta_ticks
                data["observed_pids"].add(process_key)

    summaries = {}
    for bucket, data in bucket_data.items():
        active_wall_ns = data["active_wall_ns"]
        first_active_ns = data["first_active_ns"]
        last_active_end_ns = data["last_active_end_ns"]
        cpu_seconds = data["cpu_ticks"] / clock_ticks_per_second
        summaries[bucket] = {
            "active_sample_count": data["active_sample_count"],
            "active_wall_ms": int(active_wall_ns / 1e6),
            "attribution_observations": dict(
                sorted(data["attribution_observations"].items())
            ),
            "average_worker_processes_when_active": (
                data["worker_process_ns"] / active_wall_ns if active_wall_ns else 0
            ),
            "compiler_window_ms": (
                int((last_active_end_ns - first_active_ns) / 1e6)
                if first_active_ns is not None and last_active_end_ns is not None
                else 0
            ),
            "cpu_core_hours": cpu_seconds / 3600,
            "cpu_seconds": cpu_seconds,
            "observed_pid_count": len(data["observed_pids"]),
            "peak_matched_processes": data["peak_matched_processes"],
            "peak_worker_processes": data["peak_worker_processes"],
            "sampled_process_seconds_by_name": {
                name: duration_ns / 1e9
                for name, duration_ns in sorted(
                    data["sampled_process_ns_by_name"].items()
                )
            },
        }
    return {
        "active_sample_count": active_sample_count,
        "buckets": summaries,
        "clock_ticks_per_second": clock_ticks_per_second,
        "configured_sample_interval_seconds": MONITOR_INTERVAL_SECONDS,
        "sample_count": len(samples),
    }


def start() -> None:
    state_dir = _state_dir()
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)
    cache_root, cache_root_source = _cache_root()
    cache_root = cache_root.resolve()
    inductor_root = Path(
        os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path(os.environ["OMNI_CI_HOME"]) / ".torchinductor"),
        )
    ).resolve()
    artifacts, cache_bytes = _manifest(cache_root)
    env_key, environment, unknown_fields = _environment()
    started_ns = time.time_ns()
    context = {
        "cache_artifacts_before": len(artifacts),
        "cache_bytes_before": cache_bytes,
        "cache_root": str(cache_root),
        "cache_root_exists_before": cache_root.is_dir(),
        "cache_root_source": cache_root_source,
        "cache_seeded_from_image": os.environ.get(
            "FLASHINFER_CACHE_SEEDED_FROM_IMAGE", "false"
        ).lower()
        == "true",
        "environment": environment,
        "environment_key": env_key,
        "environment_key_valid": not unknown_fields,
        "environment_unknown_fields": unknown_fields,
        "github_job": os.environ.get("GITHUB_JOB"),
        "omni_ci_home": os.environ["OMNI_CI_HOME"],
        "pr_number": _pr_number(os.environ["OMNI_CI_HOME"]),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "schema_version": 2,
        "sha": os.environ.get("GITHUB_SHA"),
        "shared_snapshot_enabled": False,
        "started_at": _utc_timestamp(started_ns),
        "started_ns": started_ns,
        "torchinductor_cache_root": str(inductor_root),
    }
    _write_manifest(state_dir / "before.tsv", artifacts)
    (state_dir / "context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _start_monitor(state_dir, cache_root, inductor_root)


def _changed_artifacts(
    cache_root: Path,
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for artifact_path, metadata in after.items():
        previous = before.get(artifact_path)
        if previous == metadata:
            continue
        full_path = cache_root / artifact_path
        try:
            sha256 = _file_hash(full_path)
        except OSError:
            sha256 = "unavailable"
        changed.append(
            {
                "mtime_ns": metadata["mtime_ns"],
                "path": artifact_path,
                "sha256": sha256,
                "size": metadata["size"],
                "status": "created" if previous is None else "modified",
            }
        )
    return changed


def _write_changed(path: Path, changed: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("path_json\tsize\tmtime_ns\tsha256\tstatus\n")
        for artifact in changed:
            handle.write(
                f"{json.dumps(artifact['path'])}\t{artifact['size']}\t"
                f"{artifact['mtime_ns']}\t"
                f"{artifact['sha256']}\t{artifact['status']}\n"
            )


def finish(stage_label: str) -> None:
    state_dir = _state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    context_path = state_dir / "context.json"
    if not context_path.is_file():
        summary = {
            "github_job": os.environ.get("GITHUB_JOB"),
            "job_status": os.environ.get("JOB_STATUS"),
            "result": "not_started",
            "schema_version": 2,
            "stage_label": stage_label,
        }
        (state_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("FLASHINFER_CACHE " + json.dumps(summary, separators=(",", ":")))
        return

    _stop_monitor(state_dir)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    cache_root = Path(context["cache_root"])
    before = _read_manifest(state_dir / "before.tsv")
    after, cache_bytes_after = _manifest(cache_root)
    cache_root_exists_after = cache_root.is_dir()
    changed = _changed_artifacts(cache_root, before, after)
    _write_manifest(state_dir / "after.tsv", after)
    _write_changed(state_dir / "created.tsv", changed)

    finished_ns = time.time_ns()
    created = sum(artifact["status"] == "created" for artifact in changed)
    modified = len(changed) - created
    if changed:
        result = "partial_miss" if before else "cold_miss"
    elif before:
        result = "no_new_artifacts"
    else:
        result = "no_cache_activity"
    artifact_write_span_ms = 0
    if changed:
        mtimes = [artifact["mtime_ns"] for artifact in changed]
        artifact_write_span_ms = int((max(mtimes) - min(mtimes)) / 1e6)
    compiler_summary = _compiler_summary(state_dir)
    flashinfer_compiler = compiler_summary["buckets"]["flashinfer"]
    summary = {
        **context,
        "aggregation_eligible": bool(
            context["environment_key_valid"]
            and (context["cache_root_exists_before"] or cache_root_exists_after)
        ),
        "artifact_write_span_ms": artifact_write_span_ms,
        "cache_artifacts_after": len(after),
        "cache_artifacts_created": created,
        "cache_artifacts_modified": modified,
        "cache_bytes_after": cache_bytes_after,
        "cache_root_exists_after": cache_root_exists_after,
        "compiler_monitor": compiler_summary,
        "duration_ms": int((finished_ns - context["started_ns"]) / 1e6),
        "flashinfer_compile_observed": bool(
            changed and flashinfer_compiler["active_sample_count"]
        ),
        "finished_at": _utc_timestamp(finished_ns),
        "finished_ns": finished_ns,
        "job_status": os.environ.get("JOB_STATUS"),
        "result": result,
        "snapshot_found": False,
        "source": (
            "image_seed" if context["cache_seeded_from_image"] else "per_pr_cache"
        ),
        "stage_label": stage_label,
    }
    (state_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_summary = {
        "artifacts_after": summary["cache_artifacts_after"],
        "artifacts_before": summary["cache_artifacts_before"],
        "artifacts_created": summary["cache_artifacts_created"],
        "artifacts_modified": summary["cache_artifacts_modified"],
        "aggregation_eligible": summary["aggregation_eligible"],
        "environment_key": summary["environment_key"],
        "environment_unknown_fields": summary["environment_unknown_fields"],
        "flashinfer_cpu_core_hours": flashinfer_compiler["cpu_core_hours"],
        "flashinfer_compile_observed": summary["flashinfer_compile_observed"],
        "flashinfer_compiler_window_ms": flashinfer_compiler["compiler_window_ms"],
        "result": result,
        "source": summary["source"],
        "stage_label": stage_label,
    }
    print("FLASHINFER_CACHE " + json.dumps(log_summary, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start")
    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--stage-label", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--state-dir", type=Path, required=True)
    monitor_parser.add_argument("--cache-root", type=Path, required=True)
    monitor_parser.add_argument("--inductor-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "start":
        start()
    elif args.command == "finish":
        finish(args.stage_label)
    else:
        monitor(args.state_dir, args.cache_root, args.inductor_root)


if __name__ == "__main__":
    main()
