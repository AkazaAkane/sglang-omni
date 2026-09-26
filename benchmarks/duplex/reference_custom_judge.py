# SPDX-License-Identifier: Apache-2.0
"""Run a separately identified custom judge over immutable reference transcripts."""

from __future__ import annotations

import copy
import fcntl
import math
import time
from argparse import Namespace
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.duplex.reference_behavior import (
    behavior_units,
    build_request,
    openai_transport,
    run_judgment,
)
from benchmarks.duplex.reference_core import (
    AUDIO_FILES,
    C_LABELS,
    REFERENCE_REVISION,
    VARIANTS,
    Engine,
    Progress,
    atomic_write_json,
    canonical_hash,
    read_json,
    record_identity,
    selected,
    sha256_file,
    utc_now,
)
from benchmarks.duplex.reference_source import load_official_behavior


class CustomDecoding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(ge=-1)
    min_p: float = Field(ge=0, le=1)
    repetition_penalty: float = Field(gt=0)
    max_tokens: int = Field(gt=0)


class CustomJudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    served_model: str = Field(min_length=1)
    precision: Literal["bf16"]
    enable_thinking: Literal[False]
    decoding: CustomDecoding
    seeds: list[int] = Field(min_length=1, max_length=3)
    server_launch_receipt: str = Field(min_length=1)
    server_launch_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_custom_request(
    official: SimpleNamespace,
    engine: Engine,
    sid: str,
    config: CustomJudgeConfig,
    experiment_hash: str,
) -> dict:
    request = build_request(official, engine.sample_dir(sid))
    body = request["body"]
    body.update(
        model=config.served_model,
        temperature=config.decoding.temperature,
        top_p=config.decoding.top_p,
        max_tokens=config.decoding.max_tokens,
        extra_body={
            "top_k": config.decoding.top_k,
            "min_p": config.decoding.min_p,
            "repetition_penalty": config.decoding.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        },
    )
    identity = {
        "experiment_hash": experiment_hash,
        "engine": engine.name,
        "sample_id": sid,
        "body": body,
        "seeds": config.seeds,
        "transcript_sha256": request["transcript_sha256"],
        "audio_sha256": {
            name: sha256_file(engine.source_audio(sid, name)) for name in AUDIO_FILES
        },
        "asr_receipt_sha256": {
            name: sha256_file(engine.sample_dir(sid) / "receipts" / f"asr-{name}.json")
            for name in ("input", "clean_input", "output", "clean_output")
        },
    }
    return {**identity, "request_hash": canonical_hash(identity)}


def summarize_custom(
    args: Namespace,
    engines: list[Engine],
    states: dict[str, dict[str, dict]],
    experiment: dict,
) -> dict:
    summary = {
        "generated_at": utc_now(),
        "scope": "custom_behavior_judge_non_official",
        "experiment_hash": canonical_hash(experiment),
        "experiment": experiment,
        "proportion_denominator": "valid custom labels only",
        "confidence_interval": {
            "method": "Wilson score",
            "confidence": 0.95,
            "note": "dataset sampling uncertainty; not human agreement or repeated-generation variance",
        },
        "engines": {},
    }
    for engine in engines:
        sids = selected(engine, args.only)
        groups = {"all": sids}
        for sid in sids:
            groups.setdefault(engine.samples[sid]["category"], []).append(sid)
        summary["engines"][engine.name] = {}
        for group, members in sorted(groups.items()):
            rows = [states[engine.name][sid] for sid in members]
            statuses = Counter(row["status"] for row in rows)
            labels = Counter(row["label"] for row in rows if row["status"] == "valid")
            valid_n = sum(labels.values())
            proportions = {}
            z_squared = NormalDist().inv_cdf(0.975) ** 2
            for label in C_LABELS:
                if valid_n:
                    proportion = labels[label] / valid_n
                    denominator = 1 + z_squared / valid_n
                    center = (proportion + z_squared / (2 * valid_n)) / denominator
                    margin = (
                        math.sqrt(
                            z_squared * proportion * (1 - proportion) / valid_n
                            + z_squared**2 / (4 * valid_n**2)
                        )
                        / denominator
                    )
                    ci95 = [max(0.0, center - margin), min(1.0, center + margin)]
                else:
                    proportion, ci95 = None, None
                proportions[label] = {
                    "count": labels[label],
                    "proportion": proportion,
                    "ci95": ci95,
                }
            reasons = Counter(
                reason
                for sid in members
                for variant in engine.samples[sid]["variants"].values()
                if not variant["eligible"]
                for reason in variant["reasons"]
            )
            summary["engines"][engine.name][group] = {
                "selected_pairs": len(members),
                "selected_sessions": len(members) * len(VARIANTS),
                "eligible_sessions": sum(
                    engine.eligible(sid, variant)
                    for sid in members
                    for variant in VARIANTS
                ),
                "eligible_pairs": sum(row["eligible"] for row in rows),
                "asr_ready_pairs": sum(row["asr_ready"] for row in rows),
                "attempted_pairs": sum(row["attempts"] > 0 for row in rows),
                "attempts": sum(row["attempts"] for row in rows),
                "status": dict(sorted(statuses.items())),
                "ineligible_variant_reasons": dict(sorted(reasons.items())),
                "valid_n": valid_n,
                "valid_label_proportions": proportions,
            }
    atomic_write_json(args.out / "summary.json", summary)
    atomic_write_json(args.out / "sample-status.json", states)
    return summary


def run_custom(
    args: Namespace,
    engines: list[Engine],
    paths: dict[str, Path],
    transport: Callable[[dict, int], dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Counter:
    for source in (args.source_scores, *(engine.tree for engine in engines)):
        if args.out.resolve().is_relative_to(
            source.resolve()
        ) or source.resolve().is_relative_to(args.out.resolve()):
            raise SystemExit(
                "custom --out must be independent of source scores and audio"
            )
    config = CustomJudgeConfig.model_validate(read_json(args.judge_config))
    launch_path = (args.judge_config.parent / config.server_launch_receipt).resolve()
    if sha256_file(launch_path) != config.server_launch_receipt_sha256:
        raise SystemExit("server launch receipt differs from the pinned SHA-256")
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    custom = copy.copy(official)
    custom.model = config.served_model
    experiment = {
        "scope": "custom_behavior_judge_non_official",
        "reference_revision": REFERENCE_REVISION,
        "reference_files": {name: sha256_file(path) for name, path in paths.items()},
        "custom_judge_sha256": sha256_file(Path(__file__)),
        "source_scores": str(args.source_scores.resolve()),
        "source_receipts": {
            engine.name: read_json(engine.root / "manifest-receipt.json")
            for engine in engines
        },
        "selected_samples": {
            engine.name: selected(engine, args.only) for engine in engines
        },
        "config": config.model_dump(),
        "server_launch_receipt": read_json(launch_path),
    }
    experiment_hash = canonical_hash(experiment)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "another custom judge phase owns this output directory"
            ) from None
        identity_path = args.out / "experiment.json"
        if identity_path.exists():
            if read_json(identity_path) != experiment:
                raise SystemExit("custom judge identity changed; use a new --out")
        elif any(path.name != ".lock" for path in args.out.iterdir()):
            raise SystemExit(
                "custom judge requires a new --out or matching experiment.json"
            )
        else:
            atomic_write_json(identity_path, experiment)
        states, todo = {}, []
        for engine in engines:
            ready, blocked = behavior_units(engine, args.only)
            states[engine.name] = {}
            for sid in selected(engine, args.only):
                folder = args.out / "engines" / engine.name / "samples" / sid
                result_path = folder / "result.json"
                previous = read_json(result_path) if result_path.exists() else None
                row = {
                    "eligible": all(engine.eligible(sid, v) for v in VARIANTS),
                    "asr_ready": sid in ready,
                    "attempts": len(previous["attempts"]) if previous else 0,
                    "label": None,
                    "status": blocked.get(sid, "not_judged"),
                }
                states[engine.name][sid] = row
                if sid in blocked:
                    continue
                request = build_custom_request(
                    official, engine, sid, config, experiment_hash
                )
                request_path = folder / "request.json"
                if request_path.exists():
                    if read_json(request_path) != request:
                        row["status"] = "stale_request"
                        continue
                else:
                    atomic_write_json(request_path, request)
                if previous is not None:
                    if previous["request_hash"] != request["request_hash"]:
                        row["status"] = "result_request_mismatch"
                        continue
                    row.update(status=previous["status"], label=previous["label"])
                    if previous["status"] != "failed" or not args.retry_failed:
                        continue
                todo.append((result_path, request, previous, row))
        if args.phase == "custom-judge":
            todo = todo[: args.limit]
            progress = Progress(args.out, "custom-judge", len(todo))
            record_identity(
                args.out,
                "custom-judge",
                {
                    "experiment_hash": experiment_hash,
                    "base_url": args.base_url,
                    "timeout_s": args.timeout_s,
                    "retry_sleep_s": args.retry_sleep_s,
                    "sdk_max_retries": 0,
                },
            )
            if todo:
                transport = transport or openai_transport(
                    args.api_key_env, args.timeout_s, args.base_url
                )
            for result_path, request, previous, row in todo:
                result = run_judgment(
                    custom,
                    request["body"],
                    request["seeds"],
                    transport,
                    args.retry_sleep_s,
                    sleep,
                )
                if result["status"] in ("valid", "invalid_label"):
                    choice = result["attempts"][-1]["response"]["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        result.update(status="invalid_finish", label=None)
                result.update(
                    request_hash=request["request_hash"], finished_at=utc_now()
                )
                if previous:
                    result["attempts"] = previous["attempts"] + result["attempts"]
                atomic_write_json(result_path, result)
                row.update(
                    status=result["status"],
                    label=result["label"],
                    attempts=len(result["attempts"]),
                )
                progress.add(result["status"])
            progress.write(finished=True)
        summary = summarize_custom(args, engines, states, experiment)
        counts: Counter = Counter()
        for engine in summary["engines"].values():
            counts.update(engine["all"]["status"])
        return counts
