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

ARTIFACT_SUFFIXES = (".so", ".o", ".cuda.o")
COMPILER_NAMES = ("cc1plus", "cicc", "ninja", "nvcc", "ptxas")
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


def _torch_environment() -> tuple[str, str, list[str]]:
    try:
        torch = importlib.import_module("torch")
        arches = {
            f"sm{major}{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(index)
                for index in range(torch.cuda.device_count())
            )
        }
        return (
            str(torch.__version__),
            str(torch.version.cuda or "unknown"),
            sorted(arches) or ["unknown"],
        )
    except (ImportError, RuntimeError):
        return "unknown", "unknown", ["unknown"]


def _image_id() -> str:
    hostname = os.environ.get("HOSTNAME")
    if not hostname:
        return "unknown"
    return _run_output(["docker", "inspect", "--format={{.Image}}", hostname])


def _deps_hash() -> str:
    path = Path("pyproject.toml")
    if not path.is_file():
        return "unknown"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> tuple[str, dict[str, Any]]:
    torch_version, torch_cuda_version, gpu_arches = _torch_environment()
    nvcc_version = _run_output(["nvcc", "--version"])
    components = {
        "dependency_hash": _deps_hash(),
        "flashinfer_version": _package_version(
            "flashinfer-python", "flashinfer-python-cu12"
        ),
        "gpu_arches": gpu_arches,
        "image_id": _image_id(),
        "nvcc_version": nvcc_version,
        "nvcc_version_sha256": hashlib.sha256(nvcc_version.encode()).hexdigest(),
        "torch_cuda_version": torch_cuda_version,
        "torch_version": torch_version,
    }
    encoded = json.dumps(components, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:20], components


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
        handle.write("path\tsize\tmtime_ns\n")
        for artifact_path, metadata in artifacts.items():
            escaped_path = artifact_path.replace("\t", "\\t").replace("\n", "\\n")
            handle.write(
                f"{escaped_path}\t{metadata['size']}\t{metadata['mtime_ns']}\n"
            )


def _read_manifest(path: Path) -> dict[str, dict[str, int]]:
    artifacts: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return artifacts
    with path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            artifact_path, size, mtime_ns = line.rstrip("\n").split("\t")
            artifacts[artifact_path] = {
                "mtime_ns": int(mtime_ns),
                "size": int(size),
            }
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


def _compiler_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for comm_path in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = comm_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if name in COMPILER_NAMES:
            counts[name] += 1
    return counts


def monitor(state_dir: Path) -> None:
    samples_path = state_dir / "compiler_samples.jsonl"
    stop_path = state_dir / "monitor.stop"
    with samples_path.open("a", encoding="utf-8") as handle:
        while not stop_path.exists():
            counts = _compiler_counts()
            sample = {
                "counts": dict(sorted(counts.items())),
                "timestamp_ns": time.time_ns(),
                "total": sum(counts.values()),
            }
            handle.write(json.dumps(sample, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
            handle.flush()
            time.sleep(MONITOR_INTERVAL_SECONDS)


def _start_monitor(state_dir: Path) -> None:
    environment = os.environ.copy()
    environment.pop("RUNNER_TRACKING_ID", None)
    with Path(os.devnull).open("wb") as devnull:
        process = subprocess.Popen(
            [sys.executable, __file__, "monitor", "--state-dir", str(state_dir)],
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
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    active = [sample for sample in samples if sample["total"] > 0]
    process_seconds_by_name: Counter[str] = Counter()
    for sample in samples:
        for name, count in sample["counts"].items():
            process_seconds_by_name[name] += count * MONITOR_INTERVAL_SECONDS
    process_seconds = (
        sum(sample["total"] for sample in samples) * MONITOR_INTERVAL_SECONDS
    )
    compiler_window_ms = 0
    if active:
        compiler_window_ms = int(
            (active[-1]["timestamp_ns"] - active[0]["timestamp_ns"]) / 1e6
            + MONITOR_INTERVAL_SECONDS * 1000
        )
    return {
        "active_sample_count": len(active),
        "active_wall_ms": int(len(active) * MONITOR_INTERVAL_SECONDS * 1000),
        "average_processes_when_active": (
            sum(sample["total"] for sample in active) / len(active) if active else 0
        ),
        "compiler_window_ms": compiler_window_ms,
        "peak_processes": max((sample["total"] for sample in samples), default=0),
        "process_hours": process_seconds / 3600,
        "process_seconds_by_name": dict(sorted(process_seconds_by_name.items())),
        "sample_count": len(samples),
        "sample_interval_seconds": MONITOR_INTERVAL_SECONDS,
    }


def start() -> None:
    state_dir = _state_dir()
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)
    cache_root = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]) / ".cache" / "flashinfer"
    artifacts, cache_bytes = _manifest(cache_root)
    env_key, environment = _environment()
    started_ns = time.time_ns()
    context = {
        "cache_artifacts_before": len(artifacts),
        "cache_bytes_before": cache_bytes,
        "cache_root": str(cache_root),
        "cache_root_exists_before": cache_root.is_dir(),
        "cache_seeded_from_image": os.environ.get(
            "FLASHINFER_CACHE_SEEDED_FROM_IMAGE", "false"
        ).lower()
        == "true",
        "environment": environment,
        "environment_key": env_key,
        "github_job": os.environ.get("GITHUB_JOB"),
        "omni_ci_home": os.environ["OMNI_CI_HOME"],
        "pr_number": _pr_number(os.environ["OMNI_CI_HOME"]),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "schema_version": 1,
        "sha": os.environ.get("GITHUB_SHA"),
        "shared_snapshot_enabled": False,
        "started_at": _utc_timestamp(started_ns),
        "started_ns": started_ns,
    }
    _write_manifest(state_dir / "before.tsv", artifacts)
    (state_dir / "context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _start_monitor(state_dir)


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
        handle.write("path\tsize\tmtime_ns\tsha256\tstatus\n")
        for artifact in changed:
            escaped_path = artifact["path"].replace("\t", "\\t").replace("\n", "\\n")
            handle.write(
                f"{escaped_path}\t{artifact['size']}\t{artifact['mtime_ns']}\t"
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
            "schema_version": 1,
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
    summary = {
        **context,
        "artifact_write_span_ms": artifact_write_span_ms,
        "cache_artifacts_after": len(after),
        "cache_artifacts_created": created,
        "cache_artifacts_modified": modified,
        "cache_bytes_after": cache_bytes_after,
        "compiler_monitor": compiler_summary,
        "duration_ms": int((finished_ns - context["started_ns"]) / 1e6),
        "flashinfer_compile_observed": bool(
            changed and compiler_summary["active_sample_count"]
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
        "compiler_process_hours": summary["compiler_monitor"]["process_hours"],
        "compiler_window_ms": summary["compiler_monitor"]["compiler_window_ms"],
        "environment_key": summary["environment_key"],
        "flashinfer_compile_observed": summary["flashinfer_compile_observed"],
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
    args = parser.parse_args()
    if args.command == "start":
        start()
    elif args.command == "finish":
        finish(args.stage_label)
    else:
        monitor(args.state_dir)


if __name__ == "__main__":
    main()
