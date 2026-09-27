# SPDX-License-Identifier: Apache-2.0
"""Prepare exact reference judge requests and retain bounded API outcomes."""

from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from benchmarks.duplex.reference_core import (
    C_LABELS,
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MODEL,
    REFERENCE_FILES,
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


def build_request(official: SimpleNamespace, sample: Path) -> dict:
    """Exact official payload from the four official transcript JSON files."""
    docs = {}
    for name in ("input_clean", "input_noisy", "output_clean", "output_noisy"):
        fname = {
            "input_clean": "clean_input.json",
            "input_noisy": "input.json",
            "output_clean": "clean_output.json",
            "output_noisy": "output.json",
        }[name]
        with open(sample / fname, "r") as fp:
            docs[name] = json.load(fp)
    user_msg = official.template(
        *(
            official.json_dict_to_compact_text(docs[n])
            for n in ("input_clean", "input_noisy", "output_clean", "output_noisy")
        )
    )
    body = {
        "model": official.model,
        "messages": [
            {"role": "system", "content": official.instruction},
            {"role": "user", "content": user_msg},
        ],
    }
    seeds = [official.initial_seed + i for i in range(JUDGE_MAX_ATTEMPTS)]
    return {
        "body": body,
        "seeds": seeds,
        "transcript_sha256": {
            f: sha256_file(sample / f)
            for f in (
                "input.json",
                "clean_input.json",
                "output.json",
                "clean_output.json",
            )
        },
        "request_hash": canonical_hash(
            {
                "body": body,
                "seeds": seeds,
                "instruction_sha256": REFERENCE_FILES["instruction"][1],
                "eval_behavior_sha256": REFERENCE_FILES["behavior"][1],
            }
        ),
    }


def behavior_units(engine: Engine, only: list[str]) -> tuple[list[str], dict[str, str]]:
    ready, blocked = [], {}
    for sid in selected(engine, only):
        if not all(engine.eligible(sid, v) for v in VARIANTS):
            blocked[sid] = "variant_ineligible"
            continue
        receipt_directory = engine.sample_dir(sid) / "receipts"
        receipts = {
            stem: (
                read_json(receipt_directory / f"asr-{stem}.json")
                if (receipt_directory / f"asr-{stem}.json").exists()
                else {"status": "not_run"}
            )
            for stem in ("input", "clean_input", "output", "clean_output")
        }
        states = [receipt["status"] for receipt in receipts.values()]
        if all(s == "ok" for s in states):
            for stem, receipt in receipts.items():
                transcript = engine.sample_dir(sid) / f"{stem}.json"
                if sha256_file(transcript) != receipt["transcript_sha256"]:
                    blocked[sid] = "stale_request"
                if (
                    sha256_file(engine.source_audio(sid, f"{stem}.wav"))
                    != receipt["audio_sha256"]
                ):
                    blocked[sid] = "asr_audio_changed"
            if sid not in blocked:
                ready.append(sid)
        else:
            blocked[sid] = "asr_" + next(s for s in states if s != "ok")
    return ready, blocked


def run_prepare_judge(args: Namespace, engines: list[Engine], paths: dict) -> Counter:
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    counts: Counter = Counter()
    for engine in engines:
        ready, blocked = behavior_units(engine, args.only)
        counts.update(f"blocked_{r}" for r in blocked.values())
        for sid in ready:
            path = engine.sample_dir(sid) / "judge" / "request.json"
            request = build_request(official, engine.sample_dir(sid))
            if path.exists():
                old = read_json(path)
                if old["request_hash"] != request["request_hash"]:
                    raise SystemExit(
                        f"{path}: prepared request differs from current transcripts"
                    )
                counts["reused"] += 1
                continue
            atomic_write_json(path, {**request, "prepared_at": utc_now()})
            counts["prepared"] += 1
    record_identity(
        args.out, "prepare-judge", {"judge_model": JUDGE_MODEL, "counts": dict(counts)}
    )
    return counts


def run_judgment(
    official: SimpleNamespace,
    body: dict,
    seeds: list[int],
    transport: Callable[[dict, int], dict],
    sleep_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Official eval_behavior loop (seed += 1 per exception), bounded to len(seeds) attempts."""
    attempts = []
    for i, seed in enumerate(seeds):
        attempt = {"seed": seed, "started_at": utc_now()}
        attempts.append(attempt)
        try:
            response = transport(body, seed)
            attempt["response"] = response
            prediction = response["choices"][0]["message"]["content"]
            result = official.parse_eval(prediction)
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            if i + 1 < len(seeds):
                sleep(sleep_s)
            continue
        served = response.get("model")
        labels = result.get("behaviour") if isinstance(result, dict) else None
        if served != official.model:
            status = "model_mismatch"
        elif isinstance(labels, list) and len(labels) == 1 and labels[0] in C_LABELS:
            status = "valid"
        else:
            status = "invalid_label"
        return {
            "status": status,
            "parsed": result,
            "label": labels[0] if status == "valid" else None,
            "served_model": served,
            "system_fingerprint": response.get("system_fingerprint"),
            "attempts": attempts,
        }
    return {"status": "failed", "parsed": None, "label": None, "attempts": attempts}


def openai_transport(
    api_key_env: str, timeout_s: float, base_url: str | None = None
) -> Callable[[dict, int], dict]:
    from openai import OpenAI

    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(f"judge needs credentials in ${api_key_env}")
    client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout_s)
    keep = (
        "id",
        "object",
        "created",
        "model",
        "system_fingerprint",
        "service_tier",
        "choices",
        "usage",
    )

    def transport(body: dict, seed: int) -> dict:
        try:
            response = client.chat.completions.create(**body, seed=seed).model_dump(
                mode="json"
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            raise RuntimeError(
                f"{type(exc).__name__} status={status}: {str(exc).replace(key, '[REDACTED]')}"
            ) from None
        return {k: response.get(k) for k in keep}

    return transport


def run_judge(
    args: Namespace,
    engines: list[Engine],
    paths: dict,
    transport: Callable[[dict, int], dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Counter:
    if args.judge != JUDGE_MODEL:
        raise SystemExit(
            f"--judge must be exactly {JUDGE_MODEL}; other judges are a separate experiment"
        )
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    todo, counts = [], Counter()
    for engine in engines:
        _, blocked = behavior_units(engine, args.only)
        for sid in selected(engine, args.only):
            sample = engine.sample_dir(sid)
            req_path, res_path = (
                sample / "judge" / "request.json",
                sample / "judge" / "result.json",
            )
            if not req_path.exists():
                continue
            if sid in blocked:
                counts[blocked[sid]] += 1
                continue
            prepared = read_json(req_path)
            if (
                build_request(official, sample)["request_hash"]
                != prepared["request_hash"]
            ):
                counts["stale_request"] += 1
                atomic_write_json(
                    sample / "judge" / "stale.json",
                    {"at": utc_now(), "request": str(req_path)},
                )
                continue
            if res_path.exists():
                old = read_json(res_path)
                if old["request_hash"] != prepared["request_hash"]:
                    raise SystemExit(f"{res_path}: result belongs to another request")
                if old["status"] != "failed" or not args.retry_failed:
                    counts["reused"] += 1
                    if old["status"] != "valid":
                        counts[f"reused_{old['status']}"] += 1
                    continue
            todo.append((sample, prepared))
    todo = todo[: args.max_requests if args.max_requests is not None else args.limit]
    progress = Progress(args.out, "judge", len(todo))
    if todo:
        record_identity(
            args.out,
            "judge",
            {
                "judge_model": JUDGE_MODEL,
                "max_attempts": JUDGE_MAX_ATTEMPTS,
                "retry_sleep_s": args.retry_sleep_s,
                "timeout_s": args.timeout_s,
                "sdk_max_retries": 0,
                "requests": len(todo),
            },
        )
        transport = transport or openai_transport(
            args.api_key_env, args.timeout_s, args.base_url
        )
    for sample, prepared in todo:
        result = run_judgment(
            official,
            prepared["body"],
            prepared["seeds"],
            transport,
            args.retry_sleep_s,
            sleep,
        )
        result.update(request_hash=prepared["request_hash"], finished_at=utc_now())
        atomic_write_json(sample / "judge" / "result.json", result)
        if result["parsed"] is not None:
            atomic_write_json(sample / "content_tag.json", result["parsed"])
        counts[result["status"]] += 1
        progress.add(result["status"])
    progress.write(finished=True)
    return counts
