# SPDX-License-Identifier: Apache-2.0
"""CPU-only serving timing and coordinated session tests."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex.client import (
    PACKET_BYTES,
    PACKET_MS,
    SEND_RECEIPTS_FILE,
    run_session,
    scheduled_send_s,
)
from benchmarks.duplex.profiles import DEFAULT_PROFILE
from benchmarks.duplex.serving import run_concurrency
from benchmarks.duplex.serving_metrics import aggregate_sessions, session_metrics
from tests.unit_test.benchmarks.test_duplex_client import DuplexPeer

INPUT_DURATION_S = 0.32
OUTPUT_PACKET = base64.b64encode(b"\x00\x00" * 1764).decode("ascii")


def recorded_session(
    trace_path: Path,
    output_times_s: list[float],
    *,
    send_delay_s: float = 0.0,
    failed: bool = False,
) -> dict[str, JsonValue]:
    trace_path.parent.mkdir(parents=True)
    receipts = [
        {
            "event_id": f"append-{index}",
            "seq": index,
            "scheduled_s": 10 + index * 0.08,
            "start_s": 10 + index * 0.08 + send_delay_s,
            "completed_s": 10 + index * 0.08 + send_delay_s + 0.001,
        }
        for index in range(4)
    ]
    records = [
        {
            "direction": "receive",
            "time_s": timestamp_s,
            "event": {"type": "response.output_audio.delta", "delta": OUTPUT_PACKET},
        }
        for timestamp_s in output_times_s
    ]
    records.extend(
        [
            {
                "direction": "receive",
                "time_s": 10.34,
                "event": {"type": "response.done", "response": {"status": "completed"}},
            },
            {
                "direction": "receive",
                "time_s": 10.35,
                "event": {"type": "sglang.input_audio.drained"},
            },
            {
                "direction": "receive",
                "time_s": 10.36,
                "event": {"type": "session.closed"},
            },
        ]
    )
    if failed:
        records.append(
            {
                "direction": "error",
                "time_s": 10.37,
                "event": {"message": "server disconnected"},
            }
        )
    trace_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
        json.dumps({"session_start_s": 10.0, "appends": receipts})
    )
    return session_metrics(
        trace_path,
        session_id=trace_path.parent.name,
        input_duration_s=INPUT_DURATION_S,
        profile=DEFAULT_PROFILE,
        reserve_s=0.08,
    )


def test_synthetic_serving_metrics(tmp_path: Path) -> None:
    perfect = recorded_session(
        tmp_path / "perfect" / "trace.jsonl", [10, 10.08, 10.16, 10.24]
    )
    assert perfect["success"] is True
    assert perfect["ttfa_s"] == pytest.approx(0)
    assert perfect["output_gap_s"]["p99"] == pytest.approx(0.08)
    assert perfect["output_coverage"] == pytest.approx(1)
    assert perfect["underrun_count"] == 0

    stalled = recorded_session(
        tmp_path / "stalled" / "trace.jsonl", [10, 10.08, 10.28, 10.36]
    )
    assert stalled["output_gap_s"]["max"] == pytest.approx(0.2)
    assert stalled["underrun_count"] == 1
    assert stalled["underrun_total_s"] == pytest.approx(0.04)

    sparse = recorded_session(tmp_path / "sparse" / "trace.jsonl", [10, 10.08])
    assert sparse["output_coverage"] == pytest.approx(0.5)
    assert sparse["underrun_total_s"] == pytest.approx(0.16)

    late = recorded_session(
        tmp_path / "late" / "trace.jsonl",
        [10, 10.08, 10.16, 10.24],
        send_delay_s=0.05,
    )
    assert late["send_lateness_s"]["p99"] == pytest.approx(0.05)

    silent = recorded_session(tmp_path / "silent" / "trace.jsonl", [])
    assert silent["success"] is False
    assert silent["ttfa_s"] is None
    assert silent["output_coverage"] == 0
    assert silent["underrun_total_s"] == INPUT_DURATION_S

    failed = recorded_session(
        tmp_path / "failed" / "trace.jsonl", [10, 10.08], failed=True
    )
    aggregate = aggregate_sessions([perfect, failed, silent])
    assert aggregate["attempted_sessions"] == 3
    assert aggregate["successful_sessions"] == 1
    assert aggregate["output_coverage"]["p50"] == pytest.approx(0.5)
    assert aggregate["ttfa_s"]["n"] == 2


def test_scheduled_deadlines_do_not_follow_late_sends() -> None:
    assert [scheduled_send_s(10, index) for index in range(4)] == pytest.approx(
        [10, 10.08, 10.16, 10.24]
    )
    assert scheduled_send_s(10, 3) == pytest.approx(10.24)


def test_late_start_does_not_burst_input(tmp_path: Path) -> None:
    async def run() -> list[dict[str, JsonValue]]:
        peer = DuplexPeer()
        async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            gate = asyncio.get_running_loop().create_future()
            gate.set_result(time.perf_counter() - 0.25)
            trace_path = tmp_path / "trace.jsonl"
            await run_session(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (3 * PACKET_BYTES // 2),
                scenario="continuous",
                trace_path=trace_path,
                start_gate=gate,
            )
            return json.loads(trace_path.with_name(SEND_RECEIPTS_FILE).read_text())[
                "appends"
            ]

    appends = asyncio.run(run())
    assert len(appends) == 3
    assert appends[0]["start_s"] - appends[0]["scheduled_s"] >= 0.25
    assert [r["scheduled_s"] - appends[0]["scheduled_s"] for r in appends] == (
        pytest.approx([0, 0.08, 0.16])
    )
    assert all(
        current["start_s"] - previous["start_s"] >= PACKET_MS / 1000 - 0.005
        for previous, current in zip(appends, appends[1:])
    )


def test_coordinator_keeps_failed_session(tmp_path: Path) -> None:
    async def run() -> dict[str, JsonValue]:
        peers: list[DuplexPeer] = []

        async def handler(websocket: ServerConnection) -> None:
            peer = DuplexPeer("close_without_update" if len(peers) == 0 else "healthy")
            peers.append(peer)
            await peer.handler(websocket)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            result = await run_concurrency(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (4 * PACKET_BYTES // 2),
                concurrency=3,
                profile=DEFAULT_PROFILE,
                output_dir=tmp_path / "c3",
                timeout_s=5,
                reserve_s=0.08,
            )
        assert len(peers) == 3
        return result

    summary = asyncio.run(run())
    assert summary["configured_sessions"] == 2
    assert summary["aggregate"]["attempted_sessions"] == 3
    assert summary["aggregate"]["successful_sessions"] == 2
    assert [session["success"] for session in summary["sessions"]] == [
        False,
        True,
        True,
    ]
    for session in summary["sessions"]:
        assert Path(session["trace_file"]).exists()
        receipts = json.loads(Path(session["receipts_file"]).read_text())
        if session["success"]:
            assert receipts["session_start_s"] == summary["common_start_s"]
        else:
            assert receipts["session_start_s"] is None


def test_serving_cli_sweep_against_fake_server(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\x00\x00" * (4 * PACKET_BYTES // 2))

    async def run() -> tuple[int, str]:
        async def handler(websocket: ServerConnection) -> None:
            await DuplexPeer().handler(websocket)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "benchmarks.duplex.serving",
                "--url",
                f"ws://127.0.0.1:{port}/v1/realtime",
                "--audio",
                str(audio_path),
                "--profile",
                "nemotron",
                "--concurrencies",
                "1,2",
                "--output-dir",
                str(tmp_path / "results"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, stderr.decode()
            return process.returncode, stdout.decode()

    _, output = asyncio.run(run())
    assert "1/1" in output
    assert "2/2" in output
    summaries = json.loads((tmp_path / "results" / "summary.json").read_text())["runs"]
    assert [run["aggregate"]["attempted_sessions"] for run in summaries] == [1, 2]
    assert [run["aggregate"]["successful_sessions"] for run in summaries] == [1, 2]
