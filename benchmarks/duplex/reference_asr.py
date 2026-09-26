# SPDX-License-Identifier: Apache-2.0
"""Transcribe reference audio with a local Parakeet checkpoint and resumable receipts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import time
import types
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.duplex.reference_core import (
    ASR_END_TOLERANCE_S,
    ASR_MODEL_ID,
    VARIANTS,
    Engine,
    HashCache,
    Progress,
    atomic_write_bytes,
    atomic_write_json,
    audio_duration,
    canonical_hash,
    finite,
    load_module,
    package_versions,
    phase_log,
    read_json,
    record_identity,
    selected,
    sha256_file,
    utc_now,
)


def asr_config(asr_path: Path, nemo_sha: str, device: str) -> dict:
    return {
        "reference_asr_sha256": sha256_file(asr_path),
        "task": "default",
        "timestamps": True,
        "model_id": ASR_MODEL_ID,
        "checkpoint_sha256": nemo_sha,
        "device_type": device,
        "entry": "get_time_aligned_transcription(root, 'default', 'audio.wav')",
        "model_bridge": "nemo_asr.models.ASRModel.from_pretrained -> preloaded restore_from(local .nemo)",
        "nemo_toolkit": package_versions()["nemo_toolkit"],
        "torch": package_versions()["torch"],
    }


def local_model_hub(model: Any, device: str) -> types.SimpleNamespace:
    """Stand-in for the official module's nemo_asr: returns the one preloaded local model."""

    class Placement:
        def cuda(self) -> Any:
            # Note (wenyao): official hardcodes .cuda(); --device cpu keeps the CPU model.
            return model.cuda() if device == "cuda" else model

    def from_pretrained(model_name: str) -> Placement:
        if model_name != ASR_MODEL_ID:
            raise RuntimeError(f"official ASR requested {model_name!r}")
        return Placement()

    return types.SimpleNamespace(
        models=types.SimpleNamespace(
            ASRModel=types.SimpleNamespace(from_pretrained=from_pretrained)
        )
    )


def load_nemo_model(nemo_path: Path, device: str) -> Any:
    import torch

    nemo_asr = importlib.import_module("nemo.collections.asr")
    if device == "cuda":
        count = torch.cuda.device_count()
        if count != 1:
            raise SystemExit(
                f"--device cuda needs exactly one visible GPU, found {count}"
            )
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_path), map_location=torch.device(device)
    )
    model.eval()
    return model


def validate_transcript(
    doc: Any, duration_s: float, tolerance_s: float = ASR_END_TOLERANCE_S
) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != {"text", "chunks"}:
        return ["keys must be exactly text, chunks"]
    errors = []
    if not isinstance(doc["text"], str) or not isinstance(doc["chunks"], list):
        return ["text must be str and chunks list"]
    words = []
    for i, chunk in enumerate(doc["chunks"]):
        ts = chunk.get("timestamp") if isinstance(chunk, dict) else None
        if (
            not isinstance(chunk, dict)
            or set(chunk) != {"text", "timestamp"}
            or not isinstance(chunk["text"], str)
        ):
            errors.append(f"chunk {i} malformed")
            continue
        words.append(chunk["text"])
        if not (isinstance(ts, list) and len(ts) == 2 and finite(*ts)):
            errors.append(f"chunk {i} timestamp not two finite numbers")
        elif not 0 <= ts[0] <= ts[1] <= duration_s + tolerance_s:
            errors.append(f"chunk {i} timestamp {ts} outside [0, {duration_s}]")
    if doc["text"] != " ".join(words).strip():
        errors.append("text differs from joined chunk words")
    return errors


def transcribe_one(official: types.ModuleType, audio: Path, stage_root: Path) -> bytes:
    """Run the unmodified official function on a single-file staging root.

    Returns the official JSON bytes verbatim; re-serializing would reorder keys
    and change the judge payload built from json.load of these files.
    """
    if stage_root.exists():
        shutil.rmtree(stage_root)
    item = stage_root / "item"
    item.mkdir(parents=True)
    os.symlink(audio.resolve(), item / "audio.wav")
    if str(item / "audio.wav").count("audio.wav") != 1:
        raise RuntimeError("staging path would break official path.replace")
    try:
        official.get_time_aligned_transcription(str(stage_root), "default", "audio.wav")
        return (item / "audio.json").read_bytes()
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def run_asr(
    args: Namespace,
    engines: list[Engine],
    paths: dict,
    hashes: HashCache,
    official: types.ModuleType | None = None,
    model: Any = None,
) -> Counter:
    nemo_path = Path(args.nemo)
    nemo_sha = hashes.get(nemo_path)
    if args.nemo_sha256 and nemo_sha != args.nemo_sha256:
        raise SystemExit(f"{nemo_path} sha256 {nemo_sha} != --nemo-sha256")
    config = asr_config(paths["asr"], nemo_sha, args.device)
    config_hash = canonical_hash(config)
    cache_root = args.out / "asr-cache" / config_hash[:16]
    atomic_write_json(cache_root / "config.json", config)

    uses: list[tuple[Engine, str, str, Path, str]] = []
    for engine in engines:
        for sid in selected(engine, args.only):
            for variant, files in VARIANTS.items():
                if not engine.eligible(sid, variant):
                    continue
                for fname in files:
                    src = engine.source_audio(sid, fname)
                    uses.append(
                        (
                            engine,
                            sid,
                            fname,
                            src,
                            hashes.get(src) if src.exists() else "",
                        )
                    )
    hashes.save()
    pending = sorted(
        {
            (sha, src)
            for *_, src, sha in uses
            if sha and not cache_ok(cache_root / sha, args.retry_failed)
        },
        key=lambda x: x[0],
    )
    pending = list({sha: src for sha, src in pending}.items())[: args.limit]
    counts: Counter = Counter()
    pending_shas = {sha for sha, _ in pending}
    failed_cache_shas = set()
    progress = Progress(args.out, "asr", len(pending))

    if pending:
        if official is None:
            official = load_module(paths["asr"], "fdb_v15_asr_3e799c4")
        if model is None:
            model = load_nemo_model(nemo_path, args.device)
        official.nemo_asr = local_model_hub(model, args.device)
        record_identity(
            args.out,
            "asr",
            {
                "asr_config": config,
                "asr_config_hash": config_hash,
                "nemo_path": str(nemo_path.resolve()),
            },
        )
    with phase_log(args.out, "asr"):
        for sha, src in pending:
            unit = cache_root / sha
            started = time.monotonic()
            receipt = {
                "audio_sha256": sha,
                "first_source": str(src),
                "config_hash": config_hash,
                "duration_s": audio_duration(src),
                "started_at": utc_now(),
            }
            try:
                raw = transcribe_one(
                    official, src, args.out / "asr-stage" / str(os.getpid())
                )
                atomic_write_bytes(unit / "audio.json", raw)
                doc = json.loads(raw)
                receipt.update(
                    status="ok",
                    words=len(doc.get("chunks", [])),
                    output_sha256=sha256_file(unit / "audio.json"),
                )
            except Exception as exc:
                receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            receipt["elapsed_s"] = round(time.monotonic() - started, 3)
            atomic_write_json(unit / "receipt.json", receipt)
            counts[receipt["status"]] += 1
            progress.add(receipt["status"])

    for engine, sid, fname, src, sha in uses:
        sample = engine.sample_dir(sid)
        stem = fname.rsplit(".", 1)[0]
        mat_path = sample / "receipts" / f"asr-{stem}.json"
        if (
            mat_path.exists()
            and read_json(mat_path).get("config_hash", config_hash) != config_hash
        ):
            raise SystemExit(f"{mat_path}: ASR config changed; use a new --out")
        if not sha:
            atomic_write_json(mat_path, {"status": "missing_audio", "source": str(src)})
            counts["missing_audio"] += 1
            continue
        engine.link(sample / fname, src)
        unit = cache_root / sha
        cached = (
            read_json(unit / "receipt.json")
            if (unit / "receipt.json").exists()
            else None
        )
        if cached is None or cached["status"] != "ok":
            if cached and sha not in pending_shas and sha not in failed_cache_shas:
                counts["reused_failed"] += 1
                failed_cache_shas.add(sha)
            atomic_write_json(
                mat_path,
                {
                    "status": "failed" if cached else "not_run",
                    "audio_sha256": sha,
                    "config_hash": config_hash,
                },
            )
            continue
        doc = read_json(unit / "audio.json")
        errors = validate_transcript(doc, cached["duration_s"])
        if errors:
            status = "invalid_transcript"
            counts[status] += 1
        else:
            status = "ok"
            raw = (unit / "audio.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != cached["output_sha256"]:
                raise SystemExit(f"{unit}/audio.json changed after transcription")
            atomic_write_bytes(sample / f"{stem}.json", raw)
        atomic_write_json(
            mat_path,
            {
                "status": status,
                "errors": errors,
                "audio_sha256": sha,
                "config_hash": config_hash,
                "cache": str(unit),
                "transcript_sha256": cached["output_sha256"],
                "words": (
                    len(doc["chunks"])
                    if isinstance(doc, dict) and isinstance(doc.get("chunks"), list)
                    else None
                ),
            },
        )
    progress.write(finished=True)
    return counts


def cache_ok(unit: Path, retry_failed: bool) -> bool:
    receipt = unit / "receipt.json"
    if not receipt.exists():
        return False
    return read_json(receipt)["status"] == "ok" or not retry_failed
