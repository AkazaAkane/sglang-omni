# SPDX-License-Identifier: Apache-2.0
"""Verify immutable audio windows, lifecycle diagnostics and input integrity."""
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from benchmarks.duplex import reference_audio as ra
from benchmarks.duplex.reference_export import export_runs
from benchmarks.eval.benchmark_duplex_reference import main

RATE = 16000
T0 = 1000.0


def pcm_input(samples):
    rng = np.random.default_rng(samples)
    return rng.integers(-3000, 3000, samples, dtype="<i2").tobytes()


def b64(data):
    return base64.b64encode(data).decode()


def tone(seconds, rate, value=1000):
    return np.full(round(seconds * rate), value, "<i2").tobytes()


def trace(
    pcm,
    engine,
    *,
    deltas=(),
    out_rate=RATE,
    errors=(),
    client_errors=(),
    closed_at=None,
    pace=None,
    drop_last=False,
    corrupt=None,
    mark=True,
    sent=True,
    created=("r1",),
    done=(),
    sent_delay=None,
    mark_at=None,
):
    """Records in time order; deltas are (elapsed_s, bytes or raw base64 string)."""
    samples = np.frombuffer(pcm, "<i2")
    window = len(samples) / RATE
    rows = []
    if engine == "sglang":
        rows.append(
            (
                T0 - 0.01,
                "receive",
                {
                    "type": "session.updated",
                    "session": {
                        "audio": {
                            "output": {
                                "format": {"type": "audio/pcm", "rate": out_rate}
                            }
                        }
                    },
                },
            )
        )
    else:
        rows.append((T0 - 0.01, "receive", {"type": "session.created", "session": {}}))
    frames = range(-(-len(samples) // 1280))
    for index in frames:
        if drop_last and index == frames[-1]:
            continue
        chunk = samples[index * 1280 : (index + 1) * 1280]
        when = T0 + index * 0.08 + (pace(index) if pace else 0)
        event_id = f"a{index}"
        if engine == "sglang":
            data = chunk.tobytes() if corrupt != index else (chunk + 1).tobytes()
            event = {
                "type": "input_audio_buffer.append",
                "event_id": event_id,
                "audio": b64(data),
                "sglang": {"seq": index, "t_start_ms": index * 1280 / RATE * 1000},
            }
            rows.append((when, "send", event))
        else:
            frame = np.zeros(1280, "<f4")
            frame[: len(chunk)] = chunk.astype(np.float32) / 32768
            if corrupt == index:
                frame[-1] = 0.5
            event = {
                "type": "input_audio_buffer.append",
                "event_id": event_id,
                "audio": b64(frame.tobytes()),
                "format": "pcm_f32le",
                "sample_rate_hz": RATE,
            }
            rows.append(
                (
                    when,
                    "send",
                    event,
                    {
                        "client_source": {
                            "index": index,
                            "start_s": index * 1280 / RATE,
                            "valid_samples": len(chunk),
                            "padded_samples": 1280 - len(chunk),
                        }
                    },
                )
            )
            if sent:
                delay = sent_delay(index) if sent_delay else 1e-4
                rows.append(
                    (
                        when + delay,
                        "sent",
                        {"event_id": event_id, "type": event["type"]},
                    )
                )
    for rid in created:
        rows.append(
            (
                T0 + 0.01,
                "receive",
                {"type": "response.created", "response": {"id": rid}},
            )
        )
    for elapsed, data in deltas:
        event = {
            "type": "response.output_audio.delta",
            "response_id": "r1",
            "delta": data if isinstance(data, str) else b64(data),
        }
        if engine == "vllm":
            event.update(format="pcm16", sample_rate_hz=out_rate)
        rows.append((T0 + elapsed, "receive", event))
    for rid in done:
        rows.append(
            (
                T0 + window + 0.3,
                "receive",
                {
                    "type": "response.done",
                    "response": {"id": rid, "status": "completed"},
                },
            )
        )
    for elapsed in errors:
        rows.append(
            (T0 + elapsed, "receive", {"type": "error", "error": {"message": "boom"}})
        )
    for elapsed in client_errors:
        rows.append((T0 + elapsed, "error", {"message": "client failure"}))
    if mark and engine == "vllm":
        rows.append(
            (
                T0 + (window if mark_at is None else mark_at),
                "mark",
                {"type": "observation_window.end", "receiver_alive": True},
            )
        )
    if closed_at is None:
        closed_at = window + 0.6
    if closed_at is not False:
        rows.append((T0 + closed_at, "receive", {"type": "session.closed"}))
    rows.sort(key=lambda row: row[0])
    return "".join(
        json.dumps(
            {
                "direction": r[1],
                "time_s": r[0],
                "event": r[2],
                **(r[3] if len(r) > 3 else {}),
            }
        )
        + "\n"
        for r in rows
    )


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_run(self, variants: dict[str, tuple[bytes, str]]) -> Path:
        """Write a paired SGLang capture with the supplied PCM and trace text."""
        run = self.root / "run"
        sample_id = "user_interruption/1"
        states = {}
        for variant, (pcm, trace_text) in variants.items():
            directory = Path("samples") / sample_id / variant
            (run / directory).mkdir(parents=True)
            (run / directory / "input.pcm").write_bytes(pcm)
            (run / directory / "continuous.jsonl").write_text(trace_text)
            states[variant] = {
                "directory": str(directory),
                "status": "captured",
                "input": {"sha256": hashlib.sha256(pcm).hexdigest()},
                "source": {"file": f"{sample_id}/input.wav", "sha256": None},
            }
        (run / "manifest.json").write_text(json.dumps({"profile": "sglang"}))
        (run / "run.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "samples": [{"id": sample_id, "variants": states}],
                }
            )
        )
        return run

    def analyze(self, engine, pcm, text, expected=None):
        directory = self.root / f"v{len(list(self.root.iterdir()))}"
        directory.mkdir()
        (directory / "input.pcm").write_bytes(pcm)
        (directory / "continuous.jsonl").write_text(text)
        expected = expected or hashlib.sha256(pcm).hexdigest()
        return ra.analyze_variant(directory, engine, expected)


class WindowEligibility(Fixture):
    pcm = pcm_input(8000)

    def test_missing_terminal_is_lifecycle_only(self):
        record, pcm, out = self.analyze(
            "vllm",
            self.pcm,
            trace(
                self.pcm,
                "vllm",
                deltas=[(0.1, tone(0.08, 22050))],
                out_rate=22050,
                created=("r1", "r2"),
                done=("r1",),
            ),
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(record["lifecycle"]["responses_missing_terminal"], ["r2"])
        self.assertIsNone(record["lifecycle"]["natural_completion"])
        self.assertEqual(len(out), 8000)
        self.assertEqual(pcm, self.pcm)

    def test_server_error_or_disconnect_before_window_end_invalidates(self):
        for kwargs, expect in (
            ({"errors": [0.2]}, "native server error"),
            ({"client_errors": [0.3]}, "client error"),
            ({"closed_at": 0.3, "mark": False}, "session.closed"),
        ):
            record, _, out = self.analyze(
                "vllm", self.pcm, trace(self.pcm, "vllm", **kwargs)
            )
            self.assertFalse(record["window"]["valid"])
            self.assertIn(expect, " ".join(record["window"]["reasons"]))
            self.assertIsNone(out)

    def test_liveness_after_window_must_be_proven(self):
        record, _, _ = self.analyze(
            "sglang", self.pcm, trace(self.pcm, "sglang", closed_at=False)
        )
        self.assertIn(
            "receiver liveness after window end unproven", record["window"]["reasons"]
        )

    def test_post_window_error_keeps_window_audio(self):
        record, _, out = self.analyze(
            "vllm",
            self.pcm,
            trace(
                self.pcm,
                "vllm",
                deltas=[(0.1, tone(0.1, RATE)), (0.9, "!!!")],
                errors=[0.8],
                client_errors=[0.85],
            ),
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(len(record["lifecycle"]["post_window_errors"]), 3)
        self.assertEqual(int(np.count_nonzero(out)), 1600)

    def test_healthy_silence_is_valid(self):
        for engine in ("vllm", "sglang"):
            record, _, out = self.analyze(engine, self.pcm, trace(self.pcm, engine))
            self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
            self.assertTrue(record["output"]["silent"])
            self.assertEqual(out.tolist(), [0] * 8000)

    def test_audio_after_window_is_excluded_not_backdated(self):
        record, _, out = self.analyze(
            "vllm",
            self.pcm,
            trace(self.pcm, "vllm", deltas=[(0.5001, tone(0.2, RATE))]),
        )
        self.assertTrue(record["window"]["valid"])
        self.assertEqual(record["output"]["packets_after_window_excluded"], 1)
        self.assertFalse(np.any(out))

    def test_fifo_backlog_is_cropped_at_window_end(self):
        record, _, out = self.analyze(
            "vllm",
            self.pcm,
            trace(
                self.pcm,
                "vllm",
                deltas=[(0.05, tone(0.3, RATE, 1)), (0.06, tone(0.3, RATE, 2))],
            ),
        )
        expect = np.zeros(8000, "<i2")
        expect[800:5600] = 1
        expect[5600:8000] = 2
        np.testing.assert_array_equal(out, expect)
        self.assertTrue(record["boundary"]["playout_active_at_T"])
        self.assertAlmostEqual(record["boundary"]["queued_audio_cropped_s"], 0.15)

    def test_native_rate_resampled_to_exact_input_count(self):
        pcm = pcm_input(7999)
        record, _, out = self.analyze(
            "sglang",
            pcm,
            trace(pcm, "sglang", deltas=[(0.0, tone(0.6, 22050))], out_rate=22050),
        )
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(len(out), 7999)
        self.assertEqual(
            record["output"]["resample"],
            {"method": "scipy.signal.resample_poly", "up": 320, "down": 441},
        )

    def test_malformed_output_in_window_invalidates(self):
        for bad in ("!!!", b64(b"\x01"), ""):
            record, _, out = self.analyze(
                "vllm", self.pcm, trace(self.pcm, "vllm", deltas=[(0.1, bad)])
            )
            self.assertFalse(record["window"]["valid"])
            self.assertIsNone(out)
        record, _, _ = self.analyze(
            "vllm",
            self.pcm,
            trace(self.pcm, "vllm", deltas=[(0.1, tone(0.08, RATE))], out_rate=24000),
        )
        self.assertTrue(record["window"]["valid"])
        self.assertEqual(record["output"]["native_rate"], 24000)

    def test_malformed_input_hash_and_frames_invalidate(self):
        cases = [
            ("vllm", {"corrupt": 6}, None, "differs from input.pcm/zero padding"),
            ("vllm", {"corrupt": 1}, None, "differs from input.pcm/zero padding"),
            ("sglang", {"corrupt": 2}, None, "serialized PCM16 differs"),
            ("vllm", {}, "0" * 64, "sha256 differs from run.json"),
            (
                "vllm",
                {"pace": lambda i: 0.09 if i >= 3 else 0},
                None,
                "pacing deviation",
            ),
            ("sglang", {"drop_last": True}, None, "incomplete append population"),
            ("vllm", {"sent": False}, None, "send completion not recorded"),
        ]
        for engine, kwargs, expected, message in cases:
            record, _, _ = self.analyze(
                engine, self.pcm, trace(self.pcm, engine, **kwargs), expected
            )
            self.assertFalse(record["window"]["valid"], (engine, kwargs))
            self.assertIn(message, " ".join(record["window"]["reasons"]))


class TraceClock(Fixture):
    pcm = pcm_input(8000)

    def test_append_send_completion_after_window_invalidates(self):
        record, _, _ = self.analyze(
            "vllm",
            self.pcm,
            trace(self.pcm, "vllm", sent_delay=lambda i: 0.3 if i == 6 else 1e-4),
        )
        self.assertFalse(record["window"]["valid"])
        self.assertIn(
            "completed after v2 deadline", " ".join(record["window"]["reasons"])
        )

    def test_early_mark_is_not_liveness(self):
        record, _, _ = self.analyze(
            "vllm", self.pcm, trace(self.pcm, "vllm", mark_at=0.01, closed_at=False)
        )
        self.assertFalse(record["window"]["valid"])
        self.assertIn(
            "window mark before window end", " ".join(record["window"]["reasons"])
        )

    def test_new_format_requires_mark(self):
        record, _, _ = self.analyze(
            "vllm", self.pcm, trace(self.pcm, "vllm", mark=False)
        )
        self.assertIn("window mark missing", " ".join(record["window"]["reasons"]))

    def test_malformed_or_nonmonotonic_clock_invalidates(self):
        lines = trace(self.pcm, "vllm").splitlines(keepends=True)
        for bad in ("NaN", "Infinity", '"1001.0"', "true"):
            text = lines[:]
            row = json.loads(text[5])
            text[5] = json.dumps(row).replace(
                f'"time_s": {row["time_s"]}', f'"time_s": {bad}'
            )
            text[5] += "\n"
            record, _, _ = self.analyze("vllm", self.pcm, "".join(text))
            self.assertFalse(record["window"]["valid"], bad)
            self.assertIn("clock", " ".join(record["window"]["reasons"]))
        text = lines[:]
        text[4], text[5] = text[5], text[4]
        record, _, _ = self.analyze("vllm", self.pcm, "".join(text))
        self.assertIn(
            "trace clock not monotonic", " ".join(record["window"]["reasons"])
        )

    def test_legacy_trace_inference_is_tagged(self):
        for engine, kwargs in (
            ("vllm", {"sent": False, "mark": False}),
            ("sglang", {}),
        ):
            record, _, _ = self.analyze(
                engine, self.pcm, trace(self.pcm, engine, **kwargs)
            )
            self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
            self.assertTrue(record["input_check"]["legacy_inference"])
            self.assertIn("legacy", record["input_check"]["send_completion_evidence"])
        record, _, _ = self.analyze("vllm", self.pcm, trace(self.pcm, "vllm"))
        self.assertFalse(record["input_check"]["legacy_inference"])


class ReferenceExport(Fixture):
    def test_declared_send_receipts_cannot_fall_back_to_legacy(self):
        pcm = pcm_input(8000)
        text = trace(pcm, "sglang")
        run = self.make_run({v: (pcm, text) for v in ("overlap", "clean")})
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"] = {"transport": {"input_send_receipts": ra.SEND_RECEIPTS}}
        path.write_text(json.dumps(manifest))
        result = export_runs([run], self.root / "export", "sglang")
        self.assertEqual(result["counts"]["eligible_variants"], 0)
        for state in result["samples"][0]["variants"].values():
            self.assertIn("missing", " ".join(state["reasons"]))
            self.assertFalse(state["input_check"]["legacy_inference"])

    def test_public_export_preserves_population_and_source_bytes(self):
        pcm = pcm_input(8000)
        text = trace(pcm, "sglang", out_rate=24000, deltas=[(0.1, tone(0.2, 24000))])
        run = self.make_run({v: (pcm, text) for v in ("overlap", "clean")})
        before = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob("*")
            if p.is_file()
        }
        out = self.root / "export"
        result = export_runs(
            [run], out, "sglang", ["user_interruption/1", "background_speech/2"]
        )
        self.assertEqual(
            result["counts"],
            {
                "selected_pairs": 2,
                "selected_variants": 4,
                "eligible_variants": 2,
                "eligible_pairs": 1,
            },
        )
        for variant in result["samples"][1]["variants"].values():
            self.assertFalse(variant["eligible"])
            self.assertTrue(variant["reasons"])
        audio, rate = soundfile.read(out / "user_interruption/1/output.wav")
        self.assertEqual((len(audio), rate), (8000, 16000))
        self.assertGreater(np.count_nonzero(audio), 0)
        after = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)
        with self.assertRaises(FileExistsError):
            export_runs([run], out, "sglang")
        with self.assertRaises(ValueError):
            export_runs([run], run / "export", "sglang")

    def test_export_cli_accepts_valid_silence(self):
        pcm = pcm_input(8000)
        run = self.make_run(
            {v: (pcm, trace(pcm, "sglang")) for v in ("overlap", "clean")},
        )
        out = self.root / "export"
        self.assertEqual(
            main(
                ["export", "--engine", "sglang", "--run", str(run), "--out", str(out)]
            ),
            0,
        )
        audio, _ = soundfile.read(out / "user_interruption/1/output.wav")
        self.assertEqual(np.count_nonzero(audio), 0)

    def test_completion_grace_accepts_tiny_tail_but_rejects_stall(self):
        pcm = pcm_input(7681)
        text = trace(pcm, "vllm", pace=lambda i: 0.001 if i else 0)
        record, _, audio = self.analyze("vllm", pcm, text)
        self.assertTrue(record["window"]["valid"], record["window"]["reasons"])
        self.assertEqual(len(audio), 7681)
        self.assertEqual(record["input_check"]["append_completions_after_T"], 1)
        stalled = trace(pcm, "vllm", sent_delay=lambda i: 0.3 if i == 6 else 1e-4)
        record, _, audio = self.analyze("vllm", pcm, stalled)
        self.assertFalse(record["window"]["valid"])
        self.assertIsNone(audio)
