# SPDX-License-Identifier: Apache-2.0
"""Compute client-observable serving metrics from native duplex recordings."""

from __future__ import annotations

import base64
import binascii
import json
import math
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.client import PACKET_BYTES, SAMPLE_RATE, SEND_RECEIPTS_FILE
from benchmarks.duplex.profiles import PROFILES, ProfileName

PLAYBACK_EPSILON_S = 1e-9
LATE_SEND_THRESHOLD_S = 0.02


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "n": 0,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "n": len(ordered),
        "p50": percentile(0.5),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def playback_underruns(
    packets: list[tuple[float, float]], end_s: float, reserve_s: float
) -> tuple[int, float, float]:
    assert packets
    start_s = packets[0][0] + reserve_s
    buffer_s = 0.0
    previous_s = start_s
    count = 0
    total_s = 0.0
    worst_s = 0.0
    for arrival_s, duration_s in [*packets, (end_s, 0.0)]:
        if arrival_s > end_s:
            break
        if arrival_s > previous_s:
            deficit_s = max(0.0, arrival_s - previous_s - buffer_s)
            if deficit_s > PLAYBACK_EPSILON_S:
                count += 1
                total_s += deficit_s
                worst_s = max(worst_s, deficit_s)
            buffer_s = max(0.0, buffer_s - (arrival_s - previous_s))
            previous_s = arrival_s
        buffer_s += duration_s
    return count, total_s, worst_s


def session_metrics(
    trace_path: Path,
    *,
    session_id: str,
    input_duration_s: float,
    profile: ProfileName,
    reserve_s: float,
) -> dict[str, JsonValue]:
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    receipts = json.loads(
        trace_path.with_name(SEND_RECEIPTS_FILE).read_text(encoding="utf-8")
    )
    start_s = receipts["session_start_s"]
    appends = receipts["appends"]
    audio_packets: list[tuple[float, float]] = []
    event_types: list[str] = []
    errors: list[str] = []
    output_samples = 0
    for record in records:
        direction = record["direction"]
        event = record["event"]
        event_type = event.get("type")
        if direction == "error":
            errors.append(str(event.get("message", "client error")))
        elif direction == "receive":
            event_types.append(event_type)
            if event_type == "error":
                errors.append(str(event.get("error", "server error")))
            elif event_type == "response.done":
                response = event.get("response")
                if (
                    not isinstance(response, dict)
                    or response.get("status") != "completed"
                ):
                    errors.append(f"response did not complete: {response}")
            elif event_type == "session.closed" and event.get("reason") not in (
                None,
                "client_closed",
            ):
                errors.append(f"unexpected session close: {event.get('reason')}")
            elif event_type == "response.output_audio.delta":
                try:
                    pcm = base64.b64decode(event["delta"], validate=True)
                    if not pcm or len(pcm) % 2:
                        raise ValueError("output audio is empty or not PCM16")
                    duration_s = len(pcm) / (2 * PROFILES[profile].output_sample_rate)
                    output_samples += len(pcm) // 2
                    audio_packets.append((record["time_s"], duration_s))
                except (KeyError, ValueError, binascii.Error) as exc:
                    errors.append(f"invalid output audio: {exc}")
    expected_appends = math.ceil(input_duration_s * SAMPLE_RATE * 2 / PACKET_BYTES)
    if len(appends) != expected_appends:
        errors.append(f"sent {len(appends)}/{expected_appends} input frames")
    if "sglang.input_audio.drained" not in event_types:
        errors.append("input did not drain")
    if "session.closed" not in event_types:
        errors.append("session did not close")
    if audio_packets and "response.done" not in event_types:
        errors.append("output response did not complete")
    if not audio_packets:
        errors.append("no output audio")
    if start_s is None:
        errors.append("session did not start")
    lateness = [max(0.0, r["start_s"] - r["scheduled_s"]) for r in appends]
    gaps = [
        current[0] - previous[0]
        for previous, current in zip(audio_packets, audio_packets[1:])
    ]
    gap_excess = [
        max(0.0, current[0] - previous[0] - previous[1])
        for previous, current in zip(audio_packets, audio_packets[1:])
    ]
    output_drift: list[float] = []
    if audio_packets:
        ideal_arrival_s = audio_packets[0][0]
        for arrival_s, duration_s in audio_packets:
            output_drift.append(arrival_s - ideal_arrival_s)
            ideal_arrival_s += duration_s
    else:
        pass
    late_send_count = sum(value > LATE_SEND_THRESHOLD_S for value in lateness)
    underrun_count, underrun_total_s, underrun_worst_s = (
        playback_underruns(
            audio_packets, start_s + input_duration_s + reserve_s, reserve_s
        )
        if audio_packets
        and start_s is not None
        and audio_packets[0][0] < start_s + input_duration_s
        else (1, input_duration_s, input_duration_s)
    )
    return {
        "session_id": session_id,
        "input_duration_s": input_duration_s,
        "trace_file": str(trace_path),
        "receipts_file": str(trace_path.with_name(SEND_RECEIPTS_FILE)),
        "success": not errors,
        "errors": errors,
        "ttfa_s": (
            audio_packets[0][0] - start_s
            if audio_packets and start_s is not None
            else None
        ),
        "send_lateness_s": distribution(lateness),
        "late_send_count": late_send_count,
        "late_send_rate": late_send_count / len(lateness) if lateness else None,
        "output_gap_s": distribution(gaps),
        "output_gap_excess_s": distribution(gap_excess),
        "output_drift_s": distribution(output_drift),
        "final_output_drift_s": output_drift[-1] if output_drift else None,
        "send_lateness_values_s": lateness,
        "output_gap_values_s": gaps,
        "output_gap_excess_values_s": gap_excess,
        "output_drift_values_s": output_drift,
        "output_samples": output_samples,
        "output_duration_s": output_samples / PROFILES[profile].output_sample_rate,
        "output_coverage": output_samples
        / PROFILES[profile].output_sample_rate
        / input_duration_s,
        "underrun_count": underrun_count,
        "underrun_total_s": underrun_total_s,
        "underrun_worst_s": underrun_worst_s,
        "underrun_ratio": underrun_total_s / input_duration_s,
    }


def aggregate_sessions(sessions: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    send_count = sum(len(s["send_lateness_values_s"]) for s in sessions)
    late_send_count = sum(s["late_send_count"] for s in sessions)
    underrun_total_s = sum(s["underrun_total_s"] for s in sessions)
    return {
        "attempted_sessions": len(sessions),
        "successful_sessions": sum(bool(s["success"]) for s in sessions),
        "ttfa_s": distribution(
            [s["ttfa_s"] for s in sessions if s["ttfa_s"] is not None]
        ),
        "send_lateness_s": distribution(
            [value for s in sessions for value in s["send_lateness_values_s"]]
        ),
        "late_send_threshold_s": LATE_SEND_THRESHOLD_S,
        "late_send_count": late_send_count,
        "late_send_rate": late_send_count / send_count if send_count else None,
        "output_gap_s": distribution(
            [value for s in sessions for value in s["output_gap_values_s"]]
        ),
        "output_gap_excess_s": distribution(
            [value for s in sessions for value in s["output_gap_excess_values_s"]]
        ),
        "output_drift_s": distribution(
            [value for s in sessions for value in s["output_drift_values_s"]]
        ),
        "final_output_drift_s": distribution(
            [
                s["final_output_drift_s"]
                for s in sessions
                if s["final_output_drift_s"] is not None
            ]
        ),
        "output_coverage": distribution([s["output_coverage"] for s in sessions]),
        "underrun_sessions": sum(bool(s["underrun_count"]) for s in sessions),
        "underrun_count": sum(s["underrun_count"] for s in sessions),
        "underrun_total_s": underrun_total_s,
        "underrun_worst_s": max((s["underrun_worst_s"] for s in sessions), default=0.0),
        "underrun_ratio": underrun_total_s
        / sum(s["input_duration_s"] for s in sessions),
    }
