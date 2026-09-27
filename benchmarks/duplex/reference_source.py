# SPDX-License-Identifier: Apache-2.0
"""Load pinned external scoring code without implicit model downloads."""

from __future__ import annotations

import ast
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from benchmarks.duplex.reference_core import (
    JUDGE_MODEL,
    REFERENCE_FILES,
    REFERENCE_REVISION,
    load_module,
    sha256_file,
)


def verify_reference(source: Path) -> dict[str, Path]:
    """Refuse any checkout whose used files differ from the pinned revision."""
    paths = {}
    for key, (rel, expected) in REFERENCE_FILES.items():
        path = source / rel
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"{rel} sha256 {actual} != pinned {expected} ({REFERENCE_REVISION})"
            )
        paths[key] = path
    return paths


def load_official_timing(
    path: Path, silero_loader: Callable[[], Any] | None = None
) -> tuple[types.ModuleType, dict]:
    """Import get_timing.py with its unpinned torch.hub.load bound to packaged Silero.

    Returns (module, bridge_record). Formulas and constants are the file's own.
    """
    import torch

    if silero_loader is None:

        def silero_loader() -> Any:
            from silero_vad import load_silero_vad

            return load_silero_vad(onnx=False)

    calls = []

    def hub_load(
        repo_or_dir: str, model: str, *args: Any, **kwargs: Any
    ) -> tuple[Any, None]:
        call = {
            "repo_or_dir": repo_or_dir,
            "model": model,
            "args": list(args),
            "kwargs": kwargs,
        }
        if call != {
            "repo_or_dir": "snakers4/silero-vad",
            "model": "silero_vad",
            "args": [],
            "kwargs": {"trust_repo": True, "onnx": False},
        }:
            raise RuntimeError(f"unexpected torch.hub.load call {call}")
        calls.append(call)
        return silero_loader(), None

    original = torch.hub.load
    torch.hub.load = hub_load
    try:
        module = load_module(path, "fdb_v15_get_timing_3e799c4")
    finally:
        torch.hub.load = original
    silero = sys.modules.get("silero_vad")
    record = {
        "vad_branch": (
            "VoiceActivityDetector"
            if hasattr(module, "_VAD")
            else "get_speech_timestamps"
        ),
        "torch_hub_load_calls": calls,
        "silero_bridge": "torch.hub.load('snakers4/silero-vad', ...) -> silero_vad.load_silero_vad(onnx=False)",
        "silero_module_file": getattr(silero, "__file__", None),
        "silero_jit_sha256": silero_jit_hash(silero),
        "constants": {
            k: getattr(module, k)
            for k in ("SR", "USER_MERGE_GAP", "MODEL_MERGE_GAP", "OUT_FILENAME")
        },
    }
    return module, record


def silero_jit_hash(silero: types.ModuleType | None) -> str | None:
    if silero is None or not getattr(silero, "__file__", None):
        return None
    jit = Path(silero.__file__).parent / "data" / "silero_vad.jit"
    return sha256_file(jit) if jit.exists() else None


def soundfile_load_wav(sr_target: int) -> Callable[[Path], Any]:
    """Bridge for torchaudio.load without torchcodec: same float32 [C,T] then official resample/squeeze."""
    import soundfile
    import torch
    import torchaudio

    def load_wav(p: Path) -> Any:
        data, sr = soundfile.read(str(p), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())
        if sr != sr_target:
            wav = torchaudio.functional.resample(wav, sr, sr_target)
        return wav.squeeze(0)

    return load_wav


def load_official_behavior(path: Path, instruction_path: Path) -> types.SimpleNamespace:
    # Note (wenyao): Importing the reference module would initialize an unused OpenAI client.
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    names = (
        "json_dict_to_compact_text",
        "extract_json",
        "parse_eval",
        "stats_by_axis",
    )
    namespace = {
        "json": json,
        "Counter": Counter,
        "Dict": Dict,
        "Any": Any,
        "Union": Union,
        "List": List,
    }
    exec(
        compile(
            ast.Module(body=[funcs[n] for n in names], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )

    final_input = [
        n
        for n in ast.walk(funcs["eval_behavior_all"])
        if isinstance(n, ast.Assign)
        and [getattr(t, "id", None) for t in n.targets] == ["final_input"]
    ]
    if len(final_input) != 1 or not isinstance(final_input[0].value, ast.JoinedStr):
        raise RuntimeError("eval_behavior_all final_input f-string not found")
    fields = (
        "input_clean_text",
        "input_noisy_text",
        "output_clean_text",
        "output_noisy_text",
    )
    lam = ast.Expression(
        ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=f) for f in fields],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=final_input[0].value,
        )
    )
    ast.fix_missing_locations(lam)
    template = eval(compile(lam, str(path), "eval"), {})

    with open(instruction_path, "r", encoding="utf-8") as fp:
        instruction = fp.read()
    return types.SimpleNamespace(
        template=template,
        instruction=instruction,
        model=JUDGE_MODEL,
        initial_seed=1,
        **{n: namespace[n] for n in names},
    )
