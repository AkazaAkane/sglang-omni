# SPDX-License-Identifier: Apache-2.0
"""Clean-environment smoke test for Qwen3-TTS.

Builds a venv from only ``pyproject.toml`` plus the documented ``qwen-tts``
install line, boots Qwen3-TTS Base in it, and serves one request end to end.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from sglang_omni.utils import find_available_port
from tests.utils import start_server_from_cmd, stop_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISSING_DEPS_SCRIPT = PROJECT_ROOT / ".github/scripts/omni_missing_dependencies.py"
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CONFIG_PATH = PROJECT_ROOT / "examples/configs/qwen3_tts_0_6b.yaml"
REFERENCE_AUDIO_ROOT = PROJECT_ROOT / "docs/_static/audio"
REFERENCE_AUDIO = REFERENCE_AUDIO_ROOT / "female-voice.wav"
# Note (Akazaakane): Qwen3-TTS Base rejects an empty ref_text; this is the
# transcript published for the clip in docs/cookbook/dots_tts.md.
REFERENCE_TEXT = (
    "By repeating what students say, teachers can demonstrate that they are "
    "listening. By extending what students say."
)
# Note (Akazaakane): keep in step with the install block in
# docs/basic_usage/tts.md and docs/cookbook/qwen3_tts.md.
QWEN_TTS_EXTRA_DEPS = ("sox", "einops")
QWEN_TTS_PACKAGE_SPEC = "qwen-tts==0.1.1"
CLEAN_VENV_PYTHON = "3.12"
INSTALL_TIMEOUT_S = 1800
# Note (Akazaakane): three stages capture CUDA graphs and the talker compiles,
# so cold startup runs well past the 300s the higgs preset allows.
STARTUP_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 300

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None, reason="uv is required to build the clean install venv"
)


def _run(command: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=INSTALL_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


@pytest.fixture(scope="module")
def clean_install_python(tmp_path_factory) -> Path:
    """Build the documented environment and return its interpreter.

    Note (Akazaakane): layered on the image with --system-site-packages rather
    than resolved from scratch. pyproject.toml pins torch, sglang and
    flash-attn, which the CI image already provides.
    """
    venv_dir = tmp_path_factory.mktemp("qwen3-tts-clean-install") / "venv"
    _run(
        ["uv", "venv", "--system-site-packages", str(venv_dir), "-p", CLEAN_VENV_PYTHON]
    )
    python = venv_dir / "bin" / "python"

    missing = _run([str(python), str(MISSING_DEPS_SCRIPT), "pyproject.toml"])
    for requirement in missing.stdout.split():
        _run([str(python), "-m", "pip", "install", requirement])

    overrides = _run(
        [str(python), str(MISSING_DEPS_SCRIPT), "--overrides", "pyproject.toml"]
    ).stdout.split()
    if overrides:
        _run([str(python), "-m", "pip", "install", "--no-deps", *overrides])

    _run(["uv", "pip", "install", "--python", str(python), "--no-deps", "-e", "."])
    # Note (Akazaakane): --no-deps, and no onnxruntime (already a pyproject dep)
    # — an unconstrained resolve lifts numpy past the numba==0.65.1 ceiling.
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            *QWEN_TTS_EXTRA_DEPS,
        ]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            QWEN_TTS_PACKAGE_SPEC,
        ]
    )
    return python


@pytest.fixture(scope="module")
def qwen3_tts_server(clean_install_python: Path, tmp_path_factory) -> Iterator[str]:
    run_dir = tmp_path_factory.mktemp("qwen3-tts-clean-install-server")
    port = find_available_port()
    command = [
        str(clean_install_python),
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_PATH,
        "--config",
        str(CONFIG_PATH),
        "--allowed-local-media-path",
        str(REFERENCE_AUDIO_ROOT),
        "--port",
        str(port),
    ]
    process = start_server_from_cmd(
        command,
        run_dir / "server.log",
        port,
        timeout=STARTUP_TIMEOUT_S,
        env={"PYTHONPATH": str(PROJECT_ROOT)},
        tee=True,
        strip_proxy=True,
    )
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        stop_server(process)


@pytest.mark.benchmark
def test_clean_install_serves_one_speech_request(qwen3_tts_server: str) -> None:
    assert REFERENCE_AUDIO.exists(), f"missing reference clip: {REFERENCE_AUDIO}"

    payload = {
        "model": MODEL_PATH,
        "voice": "default",
        "input": "SGLang-Omni is a great project!",
        "references": [
            {
                "audio_path": str(REFERENCE_AUDIO),
                "text": REFERENCE_TEXT,
            }
        ],
    }
    request = Request(
        f"{qwen3_tts_server}/v1/audio/speech",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=REQUEST_TIMEOUT_S) as response:
        assert response.status == 200, f"speech request failed: {response.status}"
        audio = response.read()

    assert audio, "server returned an empty audio body"
    assert audio[:4] == b"RIFF", f"expected a WAV payload, got {audio[:16]!r}"
