# SPDX-License-Identifier: Apache-2.0
"""Track immutable inputs, file identities and resumable scoring progress."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import types
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REFERENCE_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
REFERENCE_FILES = {
    "asr": (
        "v1_v1.5/get_transcript/asr.py",
        "aedaee0d50f2bc47947caf6f3899939461e290225a98c0ddc434c360189596cf",
    ),
    "timing": (
        "v1_v1.5/evaluation/get_timing.py",
        "4f551da4194ab4d9584db964f4b27223914eecf98cbc312388ecf45eaf6f8a17",
    ),
    "behavior": (
        "v1_v1.5/evaluation/eval_behavior.py",
        "0ff8179a437503581d65787da3a43924b45310c31a98d1bcbadce5c8605ca6f2",
    ),
    "instruction": (
        "v1_v1.5/evaluation/instruction/behavior.txt",
        "19e5477dac9a9a1e11de126783a0b820b3ecb70db5e91181824fa944e1947977",
    ),
}
ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
JUDGE_MODEL = "gpt-4o-2024-08-06"
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
JUDGE_MAX_ATTEMPTS = 3
ASR_END_TOLERANCE_S = 0.08
TIMING_END_TOLERANCE_S = 1e-3
EOF_TOLERANCE_S = 0.05
VARIANTS = {
    "overlap": ("input.wav", "output.wav"),
    "clean": ("clean_input.wav", "clean_output.wav"),
}
AUDIO_FILES = tuple(name for pair in VARIANTS.values() for name in pair)
EVENT_SELECTION_RULE = (
    "campaign_event_selected_v1 (NOT official): stop = official stop intervals "
    "intersecting [event_start, event_end]; response = first official response "
    "interval whose start >= event_start"
)
PACKAGES = (
    "torch",
    "torchaudio",
    "numpy",
    "soundfile",
    "silero-vad",
    "nemo_toolkit",
    "openai",
    "tqdm",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    with open(path, "rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fp:
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def finite(*values: Any) -> bool:
    return all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in values
    )


class HashCache:
    """sha256 by (resolved path, size, mtime_ns); trees are immutable after export."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path) if path.exists() else {}
        self.dirty = False

    def get(self, path: Path) -> str:
        real = path.resolve()
        st = real.stat()
        key = f"{real}|{st.st_size}|{st.st_mtime_ns}"
        if key not in self.data:
            self.data[key] = sha256_file(real)
            self.dirty = True
        return self.data[key]

    def save(self) -> None:
        if self.dirty:
            atomic_write_json(self.path, self.data)
            self.dirty = False


class Progress:
    def __init__(self, out: Path, phase: str, total: int) -> None:
        self.path = out / "progress" / f"{phase}.json"
        self.state = {
            "phase": phase,
            "pid": os.getpid(),
            "started_at": utc_now(),
            "total_units": total,
            "counts": Counter(),
            "finished": False,
        }
        self.write()

    def add(self, status: str) -> None:
        self.state["counts"][status] += 1
        self.write()

    def write(self, finished: bool = False) -> None:
        self.state["updated_at"] = utc_now()
        self.state["finished"] = finished
        atomic_write_json(
            self.path, {**self.state, "counts": dict(self.state["counts"])}
        )


def load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest(path: Path, projection: Path | None) -> dict:
    """Load reference-manifest.json into the canonical form, optionally via project(doc)."""
    doc = read_json(path)
    if projection is not None:
        doc = load_module(projection, "fdb_manifest_projection").project(doc)
    samples = doc.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}: 'samples' must be a non-empty list")
    seen = set()
    for sample in samples:
        sid = sample.get("sample_id")
        if (
            not isinstance(sid, str)
            or sid.count("/") != 1
            or any(part in ("", ".", "..") for part in sid.split("/"))
            or sid in seen
            or ".." in sid
        ):
            raise ValueError(f"{path}: invalid or duplicate sample_id {sid!r}")
        seen.add(sid)
        category = sid.split("/")[0]
        if sample.setdefault("category", category) != category:
            raise ValueError(f"{sid}: category disagrees with sample_id")
        variants = sample.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise ValueError(f"{sid}: variants must be exactly {sorted(VARIANTS)}")
        for name, variant in variants.items():
            if not isinstance(variant.get("eligible"), bool):
                raise ValueError(f"{sid}/{name}: 'eligible' must be a bool")
            if not variant["eligible"] and not variant.get("reasons"):
                raise ValueError(f"{sid}/{name}: ineligible variant needs reasons")
        span = sample.get("event_span_s")
        if span is not None and not (
            len(span) == 2 and finite(*span) and span[0] < span[1]
        ):
            raise ValueError(f"{sid}: invalid event_span_s {span}")
    return doc


class Engine:
    def __init__(
        self,
        out: Path,
        name: str,
        tree: Path,
        manifest_path: Path,
        projection: Path | None,
    ) -> None:
        if not name.replace("-", "").replace("_", "").isalnum():
            raise SystemExit(f"invalid engine name {name!r}")
        self.name, self.tree = name, tree.resolve()
        self.root = out / "engines" / name
        source_bytes = manifest_path.read_bytes()
        manifest_sha = hashlib.sha256(source_bytes).hexdigest()
        receipt_path = self.root / "manifest-receipt.json"
        frozen = {
            "tree": str(self.tree),
            "source_manifest_sha256": manifest_sha,
            "projection": str(projection.resolve()) if projection else None,
            "projection_sha256": sha256_file(projection) if projection else None,
        }
        receipt = read_json(receipt_path) if receipt_path.exists() else None

        def check(keys: list[str]) -> None:
            changed = [k for k in keys if receipt.get(k) != frozen[k]]
            if changed:
                raise SystemExit(
                    f"{name}: {', '.join(changed)} changed since first phase; use a new --out"
                )

        if receipt is not None:
            check(list(frozen))
        self.manifest = load_manifest(manifest_path, projection)
        frozen["projected_manifest_sha256"] = canonical_hash(self.manifest)
        if receipt is not None:
            check(["projected_manifest_sha256"])
        else:
            atomic_write_bytes(self.root / "source-manifest.json", source_bytes)
            atomic_write_json(self.root / "projected-manifest.json", self.manifest)
            atomic_write_json(
                receipt_path,
                {
                    "engine": name,
                    "source_manifest": str(manifest_path.resolve()),
                    **frozen,
                    "samples": len(self.manifest["samples"]),
                    "created_at": utc_now(),
                },
            )
        self.samples = {s["sample_id"]: s for s in self.manifest["samples"]}

    def sample_dir(self, sid: str) -> Path:
        return self.root / "samples" / sid

    def source_audio(self, sid: str, fname: str) -> Path:
        return self.tree / sid / fname

    def link(self, dst: Path, src: Path) -> None:
        """Read-only view of source audio inside the output tree."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            if dst.resolve() != src.resolve():
                raise SystemExit(f"{dst} points to {dst.resolve()}, expected {src}")
            return
        os.symlink(src.resolve(), dst)

    def eligible(self, sid: str, variant: str) -> bool:
        return self.samples[sid]["variants"][variant]["eligible"]


def selected(engine: Engine, only: list[str]) -> list[str]:
    sids = sorted(engine.samples)
    if only:
        unknown = set(only) - set(sids)
        if unknown:
            raise SystemExit(f"{engine.name}: unknown --only {sorted(unknown)}")
        sids = [s for s in sids if s in only]
    return sids


def record_identity(out: Path, phase: str, extra: dict) -> dict:
    identity = {
        "phase": phase,
        "recorded_at": utc_now(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "wrapper_sha256": {
            path.name: sha256_file(path)
            for path in sorted(Path(__file__).parent.glob("reference_*.py"))
        },
        "reference_revision": REFERENCE_REVISION,
        "reference_files": {
            k: {"path": v[0], "sha256": v[1]} for k, v in REFERENCE_FILES.items()
        },
        **extra,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(out / "identity" / f"{phase}-{stamp}.json", identity)
    return identity


@contextlib.contextmanager
def phase_log(out: Path, phase: str) -> Iterator[None]:
    """Official scripts print per file; keep that output with the run, not on the console."""
    path = out / "logs" / f"{phase}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp, contextlib.redirect_stdout(fp):
        print(f"=== {phase} {utc_now()} pid={os.getpid()}")
        yield


def audio_duration(path: Path) -> float:
    import soundfile

    info = soundfile.info(str(path))
    return info.frames / info.samplerate
