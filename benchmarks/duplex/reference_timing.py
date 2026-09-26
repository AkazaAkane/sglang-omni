# SPDX-License-Identifier: Apache-2.0
"""Run pinned timing formulas and retain validated intervals and VAD evidence."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from benchmarks.duplex.reference_core import (
    EOF_TOLERANCE_S,
    TIMING_END_TOLERANCE_S,
    VARIANTS,
    Engine,
    HashCache,
    Progress,
    atomic_write_json,
    audio_duration,
    canonical_hash,
    finite,
    package_versions,
    phase_log,
    read_json,
    record_identity,
    selected,
    sha256_file,
    utc_now,
)
from benchmarks.duplex.reference_source import load_official_timing, soundfile_load_wav


def timing_config(paths: dict, bridge: dict, loader: str) -> dict:
    versions = package_versions()
    return {
        "reference_timing_sha256": sha256_file(paths["timing"]),
        "entry": "process_folder(folder)",
        "vad_branch": bridge["vad_branch"],
        "silero_vad": versions["silero-vad"],
        "silero_jit_sha256": bridge["silero_jit_sha256"],
        "audio_loader": loader,
        "torch": versions["torch"],
        "torchaudio": versions["torchaudio"],
        "constants": bridge["constants"],
    }


def select_loader(module: ModuleType, probe: Path, choice: str) -> str:
    if choice == "official":
        return "official torchaudio.load"
    if choice == "auto":
        try:
            module.load_wav(probe)
            return "official torchaudio.load"
        except (ImportError, RuntimeError):
            pass
    module.load_wav = soundfile_load_wav(module.SR)
    return "soundfile.read(float32)+torchaudio.functional.resample bridge"


def validate_intervals(doc: Any, input_s: float, output_s: float) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != {
        "latency_stop_list",
        "latency_resp_list",
    }:
        return ["unexpected latency_intervals keys"]
    errors = []
    limit = max(input_s, output_s) + TIMING_END_TOLERANCE_S
    for key, intervals in doc.items():
        for iv in intervals:
            if not (len(iv) == 2 and finite(*iv) and 0 <= iv[0] <= iv[1] <= limit):
                errors.append(f"{key} interval {iv} invalid")
    return errors


def run_timing(
    args: Namespace,
    engines: list[Engine],
    paths: dict,
    hashes: HashCache,
    silero_loader: Callable[[], Any] | None = None,
) -> Counter:
    units = []
    for engine in engines:
        for sid in selected(engine, args.only):
            for variant in VARIANTS:
                if engine.eligible(sid, variant):
                    units.append((engine, sid, variant))
    counts: Counter = Counter()
    if not units:
        return counts
    module, bridge = load_official_timing(paths["timing"], silero_loader)
    probe_engine, probe_sid, probe_variant = units[0]
    probe = probe_engine.source_audio(probe_sid, VARIANTS[probe_variant][0])
    loader = select_loader(module, probe, args.audio_loader)
    config = timing_config(paths, bridge, loader)
    config_hash = canonical_hash(config)
    record_identity(
        args.out,
        "timing",
        {"timing_config": config, "timing_config_hash": config_hash, "bridge": bridge},
    )

    captured: list = []
    seg_sec, vad_ts = module.seg_sec, module._vad_ts
    module._vad_ts = (
        lambda wav: captured.append(("raw", vad_ts(wav))) or captured[-1][1]
    )
    module.seg_sec = (
        lambda wav, gap: captured.append(("merged", gap, seg_sec(wav, gap)))
        or captured[-1][2]
    )

    todo = []
    for engine, sid, variant in units:
        receipt_path = engine.sample_dir(sid) / "receipts" / f"timing-{variant}.json"
        if receipt_path.exists():
            old = read_json(receipt_path)
            if old.get("status") == "ok":
                in_name, out_name = VARIANTS[variant]
                for name, key in (
                    (in_name, "input_sha256"),
                    (out_name, "output_sha256"),
                ):
                    if hashes.get(engine.source_audio(sid, name)) != old[key]:
                        raise ValueError(f"Audio changed after timing: {sid}/{name}")
                folder = engine.sample_dir(sid)
                if variant == "clean":
                    folder = folder / "clean"
                if sha256_file(folder / module.OUT_FILENAME) != old["intervals_sha256"]:
                    raise ValueError(f"Timing intervals changed: {sid}/{variant}")
            if old.get("status") != "ok" and args.retry_failed:
                todo.append((engine, sid, variant, receipt_path))
            elif old.get("config_hash") != config_hash:
                raise SystemExit(
                    f"{receipt_path}: timing config changed; use a new --out"
                )
            else:
                counts["reused"] += 1
                if old.get("status") != "ok":
                    counts[f"reused_{old['status']}"] += 1
            continue
        todo.append((engine, sid, variant, receipt_path))
    todo = todo[: args.limit]
    progress = Progress(args.out, "timing", len(todo))
    with phase_log(args.out, "timing"):
        for engine, sid, variant, receipt_path in todo:
            in_name, out_name = VARIANTS[variant]
            sample = engine.sample_dir(sid)
            folder = sample if variant == "overlap" else sample / "clean"
            src_in, src_out = engine.source_audio(sid, in_name), engine.source_audio(
                sid, out_name
            )
            receipt = {
                "variant": variant,
                "config_hash": config_hash,
                "folder": str(folder),
                "official_inputs": {
                    "input.wav": str(src_in),
                    "output.wav": str(src_out),
                },
            }
            if not (src_in.exists() and src_out.exists()):
                receipt["status"] = "missing_audio"
            else:
                try:
                    engine.link(sample / in_name, src_in)
                    engine.link(sample / out_name, src_out)
                    if variant == "clean":
                        engine.link(folder / "input.wav", src_in)
                        engine.link(folder / "output.wav", src_out)
                    receipt.update(
                        input_sha256=hashes.get(src_in),
                        output_sha256=hashes.get(src_out),
                        input_s=audio_duration(src_in),
                        output_s=audio_duration(src_out),
                    )
                    captured.clear()
                    module.process_folder(folder)
                    doc = read_json(folder / module.OUT_FILENAME)
                    merged = [c for c in captured if c[0] == "merged"]
                    raw = [c[1] for c in captured if c[0] == "raw"]
                    user, model = [list(map(list, m[2])) for m in merged]
                    errors = validate_intervals(
                        doc, receipt["input_s"], receipt["output_s"]
                    )
                    receipt.update(
                        status="invalid_intervals" if errors else "ok",
                        errors=errors,
                        intervals_sha256=sha256_file(folder / module.OUT_FILENAME),
                        raw_vad_samples={"user": raw[0], "model": raw[1]},
                        user_segments=user,
                        model_segments=model,
                        labels=eof_labels(
                            user, model, receipt["input_s"], receipt["output_s"]
                        ),
                        stop_n=len(doc["latency_stop_list"]),
                        resp_n=len(doc["latency_resp_list"]),
                    )
                except Exception as exc:
                    receipt.update(
                        status="failed", error=f"{type(exc).__name__}: {exc}"
                    )
            receipt["finished_at"] = utc_now()
            atomic_write_json(receipt_path, receipt)
            counts[receipt["status"]] += 1
            progress.add(receipt["status"])
    hashes.save()
    progress.write(finished=True)
    return counts


def eof_labels(user: list, model: list, input_s: float, output_s: float) -> dict:
    """Diagnostic censoring labels from the official merged segments; no interval is altered."""
    starts = [s for s, _ in model]
    return {
        "model_speech_at_output_end": bool(model)
        and model[-1][1] >= output_s - EOF_TOLERANCE_S,
        "user_speech_at_input_end": bool(user)
        and user[-1][1] >= input_s - EOF_TOLERANCE_S,
        "user_ends_without_later_model_start": [
            e for _, e in user if not any(s > e for s in starts)
        ],
    }
