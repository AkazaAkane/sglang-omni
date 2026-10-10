# SPDX-License-Identifier: Apache-2.0
"""API tests for the native model servers beside qwen3_asr_server, run against
the real models.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_model_servers.py
"""

from __future__ import annotations

import http.client
import json
import struct
import subprocess
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server, clip, long_wav, sse_events
from websockets.exceptions import InvalidHandshake
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
WHISPER_LARGE_V3_REPO = "mlx-community/whisper-large-v3-mlx"
COHERE_REPO = "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
VAD_REPO = "mlx-community/silero-vad-v6"
# Voxt's settings for cutting long audio at speech.
VAD_FIELDS = {
    "vad_threshold": "0.5",
    "vad_min_speech_ms": "220",
    "vad_min_silence_ms": "420",
    "vad_speech_pad_ms": "180",
    "vad_merge_gap_seconds": "1.0",
    "vad_max_chunk_seconds": "24.0",
}


class ModelServer(Server):
    """Another model's server binary, started and spoken to as qwen3_asr_server is."""

    def __init__(self, binary: str, model_kind: str, repo: str) -> None:
        super().__init__(
            binary, model_kind, Path(DATA_ROOT) / "models" / repo.replace("/", "_")
        )


@pytest.fixture(scope="module")
def whisper_server() -> Iterator[ModelServer]:
    running = ModelServer("whisper_server", "whisper", WHISPER_REPO)
    yield running
    running.stop()


def test_whisper_ready_event_names_a_loopback_endpoint(
    whisper_server: ModelServer,
) -> None:
    assert whisper_server.ready["event"] == "ready"
    assert whisper_server.ready["host"] == "127.0.0.1"
    assert whisper_server.ready["server_pid"] == whisper_server.process.pid
    assert whisper_server.ready["model_name"].startswith("voxt-whisper-")
    status, body = whisper_server.request("GET", "/health")
    assert (status, json.loads(body)) == (
        200,
        {"status": "healthy", "running": True, "request_states": {}},
    )


def test_whisper_final_request_streams_text_and_generation_metadata(
    whisper_server: ModelServer,
) -> None:
    status, body = whisper_server.post_form(
        {
            "stream": "true",
            "language": "en",
            "max_new_tokens": "1024",
            "temperature": "0.0",
            "include_generation_metadata": "true",
        },
        clip("0006_en_short"),
    )
    assert status == 200
    assert sse_events(body) == [
        {
            "type": "transcript.text.done",
            "text": "Surely you are not thinking of going off there.",
            "generation_metadata": {
                "generated_token_count": 10,
                "language": "en",
                "finish_reason": "stop",
            },
        },
        "[DONE]",
    ]


def test_whisper_plain_request_returns_json_text(whisper_server: ModelServer) -> None:
    status, body = whisper_server.post_form({"language": "zh"}, clip("0152_zh_short"))
    assert (status, json.loads(body)) == (
        200,
        {"text": "互联网结合了大众传播和人际传播的要素"},
    )


def test_whisper_large_v3_server_transcribes() -> None:
    server = ModelServer("whisper_server", "whisper", WHISPER_LARGE_V3_REPO)
    try:
        status, body = server.post_form(
            {"language": "en", "max_new_tokens": "128"}, clip("0006_en_short")
        )
        assert status == 200
        assert json.loads(body)["text"].strip()
    finally:
        server.stop()


def corrupt_npz(path: Path, field: str) -> None:
    """A stored one-member archive whose directory lies about a size."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("w.npy", b"\0" * 64)
    data = bytearray(path.read_bytes())
    end = len(data) - 22
    directory = struct.unpack_from("<I", data, end + 16)[0]
    if field == "directory_offset":
        struct.pack_into("<I", data, end + 16, len(data))
    else:
        struct.pack_into("<H", data, directory + 28, 0xFFFF)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("field", ["directory_offset", "name_length"])
def test_whisper_corrupt_npz_fails_to_start(tmp_path: Path, field: str) -> None:
    source = Path(DATA_ROOT) / "models" / WHISPER_LARGE_V3_REPO.replace("/", "_")
    for item in source.iterdir():
        if item.name != "weights.npz":
            (tmp_path / item.name).symlink_to(item)
    corrupt_npz(tmp_path / "weights.npz", field)
    server = Server("whisper_server", "whisper", tmp_path)
    assert server.ready["event"] == "failed"
    assert "corrupt" in server.ready["reason"]
    assert server.process.wait(timeout=30) != 0


@pytest.mark.parametrize(
    ("fields", "wav"),
    [
        ({}, None),
        ({}, b"not audio"),
        ({"include_generation_metadata": "true"}, "wav"),
        ({"max_new_tokens": "many"}, "wav"),
        ({"temperature": "warm"}, "wav"),
        ({"temperature": "nan"}, "wav"),
        ({"temperature": "inf"}, "wav"),
        ({"temperature": "-1"}, "wav"),
        ({"max_new_tokens": "-1"}, "wav"),
    ],
)
def test_whisper_invalid_requests_are_rejected(
    whisper_server: ModelServer, fields: dict[str, str], wav: bytes | str | None
) -> None:
    status, body = whisper_server.post_form(
        fields, clip("0006_en_short") if wav == "wav" else wav
    )
    assert status == 400
    assert "detail" in json.loads(body)


def test_whisper_bad_number_error_names_its_field(whisper_server: ModelServer) -> None:
    status, body = whisper_server.post_form(
        {"max_new_tokens": "many"}, clip("0006_en_short")
    )
    assert status == 400
    assert "max_new_tokens" in json.loads(body)["detail"]


def test_whisper_language_with_a_truncated_utf8_sequence_is_unknown(
    whisper_server: ModelServer,
) -> None:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n'.encode()
        + b"en\xf0\r\n"
        + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'.encode()
        + clip("0006_en_short")
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, response = whisper_server.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    # Note (khazic): an unknown language sends no language token, as Swift does; the
    # truncated sequence must not be read past.
    assert status == 200
    assert "text" in json.loads(response)
    assert whisper_server.request("GET", "/health")[0] == 200


def test_whisper_disconnected_stream_stops_its_decode(
    whisper_server: ModelServer,
) -> None:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        ).encode()
        + long_wav(600)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection(
        "127.0.0.1", whisper_server.port, timeout=30
    )
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    time.sleep(2.0)
    assert json.loads(whisper_server.request("GET", "/health")[1])[
        "request_states"
    ] == {"running": 1}
    closed_at = time.monotonic()
    # Note (khazic): http.client keeps the socket open while the response object is.
    response.close()
    connection.close()
    while (
        json.loads(whisper_server.request("GET", "/health")[1])["request_states"] != {}
    ):
        assert (
            time.monotonic() - closed_at < 2.0
        ), "the decode kept running after its client left"
        time.sleep(0.05)


def test_whisper_serves_no_realtime_api(whisper_server: ModelServer) -> None:
    with pytest.raises(InvalidHandshake):
        connect(f"ws://127.0.0.1:{whisper_server.port}/v1/realtime")


def test_whisper_shutdown_reports_stopped() -> None:
    running = ModelServer("whisper_server", "whisper", WHISPER_REPO)
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0


@pytest.mark.parametrize(
    ("binary", "other_kind"),
    [("whisper_server", "qwen3_asr"), ("cohere_transcribe_server", "whisper")],
)
def test_server_serves_only_its_model_kind(binary: str, other_kind: str) -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / binary),
            "--supervised",
            "--model-kind",
            other_kind,
            "--model-directory",
            "x",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.fixture(scope="module")
def cohere_server() -> Iterator[ModelServer]:
    running = ModelServer("cohere_transcribe_server", "cohere_transcribe", COHERE_REPO)
    yield running
    running.stop()


def test_cohere_final_request_streams_text_and_generation_metadata(
    cohere_server: ModelServer,
) -> None:
    assert cohere_server.ready["model_name"].startswith("voxt-cohere_transcribe-")
    status, body = cohere_server.post_form(
        {
            "stream": "true",
            "language": "zh",
            "use_punctuation": "true",
            "max_new_tokens": "1024",
            "temperature": "0.0",
            "chunk_duration": "1200",
            "min_chunk_duration": "1",
            "include_generation_metadata": "true",
        },
        clip("0152_zh_short"),
    )
    assert status == 200
    assert sse_events(body) == [
        {
            "type": "transcript.text.done",
            "text": "互联网结合了大众传播和人际传播的要素。",
            "generation_metadata": {
                "generated_token_count": 20,
                "language": "zh",
                "finish_reason": "stop",
            },
        },
        "[DONE]",
    ]


def test_cohere_cuts_long_audio_at_speech(cohere_server: ModelServer) -> None:
    vad_directory = str(Path(DATA_ROOT) / "models" / VAD_REPO.replace("/", "_"))
    status, body = cohere_server.post_form(
        {"language": "en", "vad_model_directory": vad_directory, **VAD_FIELDS},
        clip("0344_en_long"),
    )
    assert status == 200
    text = json.loads(body)["text"]
    assert text.startswith("In every way they sought to undermine the authority")
    assert len(text.split("\n")) > 1


def test_cohere_voice_activity_needs_every_setting(cohere_server: ModelServer) -> None:
    fields = {"vad_model_directory": "x", **VAD_FIELDS}
    fields.pop("vad_threshold")
    status, body = cohere_server.post_form(fields, clip("0006_en_short"))
    assert status == 400
    assert "vad_threshold" in json.loads(body)["detail"]


def test_cohere_reports_the_language_it_decoded(cohere_server: ModelServer) -> None:
    status, body = cohere_server.post_form(
        {"language": "fr-FR", "stream": "true", "include_generation_metadata": "true"},
        clip("0006_en_short"),
    )
    assert status == 200
    assert sse_events(body)[0]["generation_metadata"]["language"] == "en"


@pytest.mark.parametrize("temperature", ["nan", "-1"])
def test_cohere_rejects_a_bad_temperature(
    cohere_server: ModelServer, temperature: str
) -> None:
    status, body = cohere_server.post_form(
        {"temperature": temperature}, clip("0006_en_short")
    )
    assert status == 400
    assert "temperature" in json.loads(body)["detail"]
