# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from sglang_omni.models.qwen3_omni.prompt_frontend import parse_ordered_chat
from sglang_omni.preprocessing.text import ensure_chat_template


def test_ensure_chat_template_uses_remote_fallback(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_load_chat_template(
        model_path: str, *, local_files_only: bool = True
    ) -> str | None:
        calls.append((model_path, local_files_only))
        if model_path == "base-model":
            return "template"
        return None

    monkeypatch.setattr(
        "sglang_omni.preprocessing.text.load_chat_template",
        fake_load_chat_template,
    )
    tokenizer = SimpleNamespace(chat_template=None)

    ensure_chat_template(
        tokenizer,
        model_path="fp8-model",
        fallback_model_paths=("base-model",),
    )

    assert tokenizer.chat_template == "template"
    assert calls == [("fp8-model", True), ("base-model", False)]


def test_ensure_chat_template_does_not_fetch_when_template_exists(monkeypatch) -> None:
    def fail_load_chat_template(
        model_path: str, *, local_files_only: bool = True
    ) -> str | None:
        raise AssertionError("chat template should not be loaded")

    monkeypatch.setattr(
        "sglang_omni.preprocessing.text.load_chat_template",
        fail_load_chat_template,
    )
    tokenizer = SimpleNamespace(chat_template="existing")

    ensure_chat_template(
        tokenizer,
        model_path="fp8-model",
        fallback_model_paths=("base-model",),
    )

    assert tokenizer.chat_template == "existing"


def test_ordered_content_keeps_media_in_original_turns() -> None:
    image = Image.new("RGB", (2, 2), "red")
    waveform = np.zeros(1600, dtype=np.float32)
    ordered = parse_ordered_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "image", "image": image},
                    {"type": "text", "text": "After"},
                ],
            },
            {"role": "assistant", "content": "Remembered."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": waveform},
                    {"type": "text", "text": "Compare."},
                ],
            },
        ]
    )
    assert ordered.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Before"},
                {"type": "image"},
                {"type": "text", "text": "After"},
            ],
        },
        {"role": "assistant", "content": "Remembered."},
        {
            "role": "user",
            "content": [{"type": "audio"}, {"type": "text", "text": "Compare."}],
        },
    ]
    assert ordered.images == [image]
    assert ordered.audios[0] is waveform


def test_inline_urls_and_base64_audio_are_collected_in_order() -> None:
    ordered = parse_ordered_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "first.png"}},
                    {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "AAAA", "format": "wav"},
                    },
                    {"type": "image_url", "image_url": {"url": "second.png"}},
                ],
            }
        ]
    )
    assert ordered.images == ["first.png", "second.png"]
    assert ordered.videos == ["clip.mp4"]
    assert ordered.audios == ["data:audio/wav;base64,AAAA"]
    assert [part["type"] for part in ordered.messages[0]["content"]] == [
        "image",
        "video",
        "audio",
        "image",
    ]


@pytest.mark.parametrize(
    "part", [{"type": "unknown"}, {"type": "image_url"}, {"type": "text", "text": None}]
)
def test_invalid_content_is_rejected(part) -> None:
    with pytest.raises(ValueError):
        parse_ordered_chat([{"role": "user", "content": [part]}])
