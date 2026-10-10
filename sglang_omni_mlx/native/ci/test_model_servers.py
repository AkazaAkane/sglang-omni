# SPDX-License-Identifier: Apache-2.0
"""API tests for the native model servers beside qwen3_asr_server, run against
the real models.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_model_servers.py
"""

from __future__ import annotations

import http.client
import json
import subprocess
import time
import uuid
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


def test_whisper_server_serves_only_whisper() -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "whisper_server"),
            "--supervised",
            "--model-kind",
            "qwen3_asr",
            "--model-directory",
            "x",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
