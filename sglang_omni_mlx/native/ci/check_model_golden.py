# SPDX-License-Identifier: Apache-2.0
"""Checks a native model's parity tool against its golden outputs on the frozen corpus.

    python check_model_golden.py --runtime-bin DIR --data-root DIR --golden FILE
        [--write] [--outputs DIR]
    python check_model_golden.py --golden FILE --import OUTPUTS

Like check_golden.py, with the same per-chip golden outputs and tolerance, for a
model served by its own binary. The golden file also names the parity tool
(whisper_transcribe, ...), its request flags, and the language each clip
language is sent with (a user with that main language); every corpus clip is
transcribed, one tool run per language sent.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from check_golden import COMPARED_FIELDS, model_directory, run


def transcribe(
    command: list[str], clips: list[Path], language: str | None
) -> dict[str, dict]:
    language_flags = ["--language", language] if language is not None else []
    completed = subprocess.run(
        command + language_flags + [str(clip) for clip in clips],
        check=True,
        capture_output=True,
        text=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        row = json.loads(line)
        results[Path(row["file"]).stem] = {
            field: row[field] for field in COMPARED_FIELDS
        }
    return results


def transcribe_with_tool(
    runtime_bin: Path, data_root: Path, golden: dict, manifest: dict[str, dict]
) -> dict[str, dict]:
    command = [
        str(runtime_bin / golden["tool"]),
        "--model-path",
        str(model_directory(data_root, golden["model"])),
    ]
    for flag, value in golden["request"].items():
        command += [f"--{flag.replace('_', '-')}", str(value)]
    clips_by_request_language: dict[str | None, list[Path]] = {}
    for clip_id, clip in manifest.items():
        clips_by_request_language.setdefault(
            golden["language_by_clip_language"][clip["lang"]], []
        ).append(data_root / "corpus" / "v1" / "clips" / f"{clip_id}.wav")
    results: dict[str, dict] = {}
    for language, clips in clips_by_request_language.items():
        results.update(transcribe(command, clips, language))
    return {clip_id: results[clip_id] for clip_id in manifest}


if __name__ == "__main__":
    run(transcribe_with_tool)
