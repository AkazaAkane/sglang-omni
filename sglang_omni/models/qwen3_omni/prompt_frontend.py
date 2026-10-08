# SPDX-License-Identifier: Apache-2.0
"""Preserve structured chat parts and collect their ordered media sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
from PIL import Image


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class MediaPart(TypedDict):
    type: Literal["image", "audio", "video"]


class ChatMessage(TypedDict):
    role: str
    content: str | list[TextPart | MediaPart]


@dataclass(kw_only=True)
class OrderedChat:
    messages: list[ChatMessage] = field(default_factory=list)
    images: list[str | Image.Image] = field(default_factory=list)
    audios: list[str | npt.NDArray[np.float32]] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)


def parse_ordered_chat(messages: object) -> OrderedChat:
    if not isinstance(messages, list):
        raise ValueError("Chat messages must be a list")
    else:
        pass
    ordered = OrderedChat()
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("role", "user"), str
        ):
            raise ValueError("Each chat message requires a string role")
        else:
            role = message.get("role", "user")
            content = message.get("content", "")
        if isinstance(content, str):
            ordered.messages.append({"role": role, "content": content})
            continue
        elif not isinstance(content, list):
            raise ValueError("Chat content must be a string or ordered parts")
        else:
            pass
        parts: list[TextPart | MediaPart] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(
                    "Qwen3-Omni content parts must be structured dictionaries"
                )
            else:
                part_type = part.get("type")
            if part_type == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("Text content parts require string text")
                else:
                    parts.append({"type": "text", "text": part["text"]})
            elif part_type in (
                "image",
                "image_url",
                "audio",
                "audio_url",
                "input_audio",
                "video",
                "video_url",
            ):
                source = part.get(str(part_type))
                if part_type == "input_audio":
                    if (
                        not isinstance(source, dict)
                        or not isinstance(source.get("data"), str)
                        or not isinstance(source.get("format"), str)
                    ):
                        raise ValueError("input_audio requires base64 data and format")
                    else:
                        source = (
                            f"data:audio/{source['format']};base64,{source['data']}"
                        )
                elif isinstance(source, dict):
                    source = source.get("url")
                else:
                    pass
                if part_type in ("image", "image_url") and isinstance(
                    source, (str, Image.Image)
                ):
                    ordered.images.append(source)
                    parts.append({"type": "image"})
                elif part_type in ("audio", "audio_url", "input_audio") and isinstance(
                    source, (str, np.ndarray)
                ):
                    ordered.audios.append(source)
                    parts.append({"type": "audio"})
                elif part_type in ("video", "video_url") and isinstance(source, str):
                    ordered.videos.append(source)
                    parts.append({"type": "video"})
                else:
                    raise ValueError(f"Invalid source for {part_type} content")
            else:
                raise ValueError(f"Unsupported Qwen3-Omni content type: {part_type}")
        ordered.messages.append({"role": role, "content": parts})
    return ordered
