# SPDX-License-Identifier: Apache-2.0
"""Summarize reference intervals and behavior with explicit coverage counts."""

from __future__ import annotations

import math
import random
import statistics
from argparse import Namespace
from collections import Counter
from types import SimpleNamespace

from benchmarks.duplex.reference_behavior import behavior_units, build_request
from benchmarks.duplex.reference_core import (
    C_LABELS,
    EVENT_SELECTION_RULE,
    REFERENCE_REVISION,
    VARIANTS,
    Engine,
    atomic_write_json,
    read_json,
    selected,
    sha256_file,
    utc_now,
)
from benchmarks.duplex.reference_source import load_official_behavior


def quantile(sorted_values: list[float], q: float) -> float:
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cluster_bootstrap(clusters: list[list[float]], replicates: int, seed: str) -> dict:
    """Percentile CI of the pooled mean, resampling whole samples (clusters) with replacement."""
    sums = [sum(c) for c in clusters]
    lens = [len(c) for c in clusters]
    n = len(clusters)
    if n == 0 or sum(lens) == 0:
        return {"unit": "sample", "unit_n": n, "replicates": 0, "ci95": None}
    rng = random.Random(seed)
    stats, empty = [], 0
    for _ in range(replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        count = sum(lens[i] for i in idx)
        if count == 0:
            empty += 1
            continue
        stats.append(sum(sums[i] for i in idx) / count)
    stats.sort()
    return {
        "unit": "sample",
        "unit_n": n,
        "replicates": replicates,
        "empty_replicates": empty,
        "seed": seed,
        "method": "percentile",
        "ci95": [quantile(stats, 0.025), quantile(stats, 0.975)] if stats else None,
    }


def describe(clusters: list[list[float]], replicates: int, seed: str) -> dict:
    values = sorted(v for c in clusters for v in c)
    return {
        "interval_n": len(values),
        "samples": len(clusters),
        "zero_interval_samples": sum(1 for c in clusters if not c),
        "mean_s": statistics.fmean(values) if values else None,
        "median_s": statistics.median(values) if values else None,
        "bootstrap_pooled_mean": cluster_bootstrap(clusters, replicates, seed),
    }


def summarize_timing(
    engine: Engine, sids: list[str], variant: str, args: Namespace, group: str
) -> dict:
    ledger = Counter()
    reasons = Counter()
    flags = Counter()
    stop, resp, ev_stop, ev_resp = [], [], [], []
    for sid in sids:
        meta = engine.samples[sid]["variants"][variant]
        flags.update(meta.get("flags", []))
        if not meta["eligible"]:
            ledger["ineligible"] += 1
            reasons.update(meta["reasons"])
            continue
        path = engine.sample_dir(sid) / "receipts" / f"timing-{variant}.json"
        receipt = read_json(path) if path.exists() else {"status": "not_run"}
        ledger[receipt["status"]] += 1
        if receipt["status"] != "ok":
            continue
        folder = (
            engine.sample_dir(sid)
            if variant == "overlap"
            else engine.sample_dir(sid) / "clean"
        )
        intervals = folder / "latency_intervals.json"
        if sha256_file(intervals) != receipt["intervals_sha256"]:
            raise ValueError(f"Timing intervals changed after scoring: {intervals}")
        doc = read_json(intervals)
        for key, value in receipt["labels"].items():
            if value:
                ledger[f"label_{key}"] += 1
        s = [e - b for b, e in doc["latency_stop_list"]]
        r = [e - b for b, e in doc["latency_resp_list"]]
        stop.append(s)
        resp.append(r)
        span = engine.samples[sid].get("event_span_s")
        if span is not None:
            ev_stop.append(
                [
                    e - b
                    for b, e in doc["latency_stop_list"]
                    if b < span[1] and e > span[0]
                ]
            )
            first = next(
                ([b, e] for b, e in doc["latency_resp_list"] if b >= span[0]), None
            )
            ev_resp.append([first[1] - first[0]] if first else [])
    out = {
        "population": len(sids),
        "status": dict(ledger),
        "ineligible_reasons": dict(reasons),
        "manifest_flags": dict(flags),
        "official_all_intervals": {
            "reduction": "campaign reduction of official per-sample intervals (end - start); "
            "official source defines no scalar aggregate",
            "stop": describe(stop, args.bootstrap, f"{args.seed}:{group}:stop"),
            "response": describe(resp, args.bootstrap, f"{args.seed}:{group}:resp"),
        },
    }
    if ev_stop:
        out["event_selected_non_official"] = {
            "rule": EVENT_SELECTION_RULE,
            "stop": describe(ev_stop, args.bootstrap, f"{args.seed}:{group}:ev_stop"),
            "response": describe(
                ev_resp, args.bootstrap, f"{args.seed}:{group}:ev_resp"
            ),
        }
    return out


def summarize_asr(engine: Engine, sids: list[str]) -> dict:
    ledger = Counter()
    for sid in sids:
        for variant, files in VARIANTS.items():
            for fname in files:
                if not engine.eligible(sid, variant):
                    ledger[f"{fname}:ineligible"] += 1
                    continue
                path = (
                    engine.sample_dir(sid)
                    / "receipts"
                    / f"asr-{fname.rsplit('.', 1)[0]}.json"
                )
                receipt = (
                    read_json(path)
                    if path.exists()
                    else {"status": "not_run", "words": None}
                )
                ledger[f"{fname}:{receipt['status']}"] += 1
                if receipt["status"] == "ok" and receipt["words"] == 0:
                    ledger[f"{fname}:ok_empty_transcript"] += 1
    return dict(sorted(ledger.items()))


def summarize_behavior(
    engine: Engine,
    sids: list[str],
    official: SimpleNamespace,
    args: Namespace,
    group: str,
) -> dict:
    ledger, labels, parsed = Counter(), Counter(), []
    valid_labels = []
    ready, blocked = behavior_units(engine, sids)
    for sid in sids:
        if sid in blocked:
            ledger[blocked[sid]] += 1
            continue
        judge = engine.sample_dir(sid) / "judge"
        if not (judge / "request.json").exists():
            ledger["not_prepared"] += 1
            continue
        prepared = read_json(judge / "request.json")
        if (
            build_request(official, engine.sample_dir(sid))["request_hash"]
            != prepared["request_hash"]
        ):
            ledger["stale_request"] += 1
            continue
        if not (judge / "result.json").exists():
            ledger["not_judged"] += 1
            continue
        result = read_json(judge / "result.json")
        if result["request_hash"] != prepared["request_hash"]:
            ledger["result_request_mismatch"] += 1
            continue
        ledger[result["status"]] += 1
        if result["parsed"] is not None:
            parsed.append(result["parsed"])
        if result["status"] == "valid":
            labels[result["label"]] += 1
            valid_labels.append(result["label"])
    try:
        _, totals, ratios = official.stats_by_axis(parsed)
        official_fmt = {
            ax: {k: round(v, 2) for k, v in sorted(ratios[ax].items())} for ax in ["C"]
        }
        official_fmt["C_total_tags"] = totals["C"]
    except Exception as exc:
        official_fmt = {"error": f"{type(exc).__name__}: {exc}"}
    proportions = {
        label: {
            "count": labels[label],
            "proportion": labels[label] / len(valid_labels) if valid_labels else None,
            "bootstrap": cluster_bootstrap(
                [[1.0 if x == label else 0.0] for x in valid_labels],
                args.bootstrap,
                f"{args.seed}:{group}:{label}",
            ),
        }
        for label in C_LABELS
    }
    return {
        "population": len(sids),
        "status": dict(ledger),
        "valid_n": len(valid_labels),
        "valid_label_proportions": proportions,
        "official_format_ratios": official_fmt,
        "official_format_note": "official stats_by_axis over every parsed response, rounded to 2 dp",
    }


def run_summarize(args: Namespace, engines: list[Engine], paths: dict) -> dict:
    official = load_official_behavior(paths["behavior"], paths["instruction"])
    summary = {
        "generated_at": utc_now(),
        "reference_revision": REFERENCE_REVISION,
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "unit": "sample (one generation each)",
            "note": "sampling uncertainty over dataset samples; no repeated generations were run",
        },
        "engines": {},
    }
    for engine in engines:
        sids = selected(engine, args.only)
        groups = {"all": sids}
        for sid in sids:
            groups.setdefault(engine.samples[sid]["category"], []).append(sid)
        summary["engines"][engine.name] = {
            group: {
                "timing_official_overlap": summarize_timing(
                    engine, members, "overlap", args, f"{engine.name}:{group}:overlap"
                ),
                "timing_supplementary_clean": summarize_timing(
                    engine, members, "clean", args, f"{engine.name}:{group}:clean"
                ),
                "asr_files": summarize_asr(engine, members),
                "behavior": summarize_behavior(
                    engine, members, official, args, f"{engine.name}:{group}"
                ),
            }
            for group, members in sorted(groups.items())
        }
    atomic_write_json(args.out / "summary.json", summary)
    return summary
