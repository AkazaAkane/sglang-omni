from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang_omni.models.minicpm_o.components import preprocessor as preprocessor_mod
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.prompt_frontend import render_ordered_chat
from sglang_omni.models.minicpm_o.video_frontend import MiniCPMVideoIO, TimedVideo
from sglang_omni.proto import OmniRequest, StagePayload


def make_payload(inputs: dict) -> StagePayload:
    return StagePayload(
        request_id="video-test",
        request=OmniRequest(inputs=inputs),
        data=None,
    )


class FakeProcessor:
    def __init__(self) -> None:
        self.images = None
        self.audios = None
        self.options = None

    def __call__(self, prompt_text, *, images, audios, return_tensors, **options):
        self.images = images
        self.audios = audios
        self.options = options
        image_count = len(images[0]) if images else 0
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "image_bound": [[torch.tensor([0, 1])] * image_count],
            "pixel_values": [[torch.zeros(1, 2) for _ in range(image_count)]],
            "tgt_sizes": [[torch.tensor([1, 1]) for _ in range(image_count)]],
            "audio_bounds": [[]],
            "audio_feature_lens": [[]],
            "audio_features": [],
        }


async def empty_images(images):
    return images if images and isinstance(images[0], Image.Image) else []


async def explicit_audios(_audios, *, target_sr):
    if _audios and isinstance(_audios[0], np.ndarray):
        return _audios
    else:
        return [np.array([0.25, 0.5], dtype=np.float32)] if _audios else []


@pytest.mark.parametrize("use_audio_in_video", [None, False, True])
@pytest.mark.parametrize("explicit_audio", [False, True])
def test_minicpm_preprocessor_uses_only_requested_video_audio(
    monkeypatch,
    use_audio_in_video,
    explicit_audio,
) -> None:
    fake_processor = FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        fake_processor  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    preprocessor.tokenizer = SimpleNamespace()
    monkeypatch.setattr(
        preprocessor,
        "render_chat_template",
        lambda messages, **_: str(messages),
    )

    video = torch.stack(
        [
            torch.zeros((3, 2, 2)),
            torch.ones((3, 2, 2)) * 0.5,
        ]
    )
    captured_video_kwargs = {}

    async def videos(videos, **kwargs):
        captured_video_kwargs.update(kwargs)
        audio = (
            [np.array([1.0, 2.0], dtype=np.float32)]
            if kwargs["extract_audio"]
            else None
        )
        return [video], [2.0], audio

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", empty_images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)

    async def timed_video(source, **options):
        captured_video_kwargs.update(
            fps=options["fps"],
            max_frames=options["max_frames"],
            min_pixels=options["min_pixels"],
            max_pixels=options["max_pixels"],
            total_pixels=options["total_pixels"],
            extract_audio=options["use_audio"],
            audio_target_sr=16000,
        )
        return TimedVideo(
            frames=[Image.new("RGB", (2, 2)) for _ in range(2)],
            audio_segments=[np.array([1.0, 2.0], dtype=np.float32)] * 2,
            timestamps_seconds=[0, 1],
            duration_seconds=2,
        )

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.prompt_frontend.load_timed_video", timed_video
    )
    monkeypatch.setattr(
        preprocessor_mod,
        "compute_video_cache_key",
        lambda *args, **_kwargs: "video-cache",
    )

    payload = make_payload(
        {
            "messages": [{"role": "user", "content": "What happens?"}],
            "videos": ["clip.mp4"],
            **({"audios": ["question.wav"]} if explicit_audio else {}),
            **(
                {"use_audio_in_video": use_audio_in_video}
                if use_audio_in_video is not None
                else {}
            ),
            "video_fps": 2,
            "video_max_frames": 8,
            "video_min_pixels": 128,
            "video_max_pixels": 4096,
            "video_total_pixels": 8192,
        }
    )

    result = asyncio.run(preprocessor(payload))

    assert captured_video_kwargs == {
        "fps": 2,
        "max_frames": 8,
        "min_pixels": 128,
        "max_pixels": 4096,
        "total_pixels": 8192,
        "extract_audio": bool(use_audio_in_video),
        "audio_target_sr": 16000,
    }
    assert len(fake_processor.images[0]) == 2
    expected_options = {"max_slice_nums": 1, "use_image_id": False}
    expected_audio_count = int(explicit_audio) + 2 * int(bool(use_audio_in_video))
    if use_audio_in_video:
        expected_options["audio_parts"] = [[0] * expected_audio_count]
    assert fake_processor.options == expected_options
    if expected_audio_count:
        assert len(fake_processor.audios[0]) == expected_audio_count
        if explicit_audio:
            np.testing.assert_array_equal(
                fake_processor.audios[0][0],
                np.array([0.25, 0.5], dtype=np.float32),
            )
        if use_audio_in_video:
            np.testing.assert_array_equal(
                fake_processor.audios[0][-1],
                np.array([1.0, 2.0], dtype=np.float32),
            )
    else:
        assert fake_processor.audios is None
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == 2
    assert prompt_text.count("<audio>./</audio>") == expected_audio_count
    assert payload.request.inputs is None


@pytest.mark.parametrize(
    ("with_image", "with_audio", "with_video"),
    [(True, False, False), (False, True, False), (True, True, True)],
)
def test_minicpm_video_options_preserve_other_media(
    monkeypatch, with_image, with_audio, with_video
) -> None:
    fake_processor = FakeProcessor()
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        fake_processor  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    image = Image.new("RGB", (2, 2), color="red")
    frame = Image.new("RGB", (2, 2), color="blue")

    async def images(raw_images):
        return [image] if raw_images else []

    async def videos(raw_videos, **kwargs):
        return [[frame]], [1.0], None

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_audio_list_async", explicit_audios)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)
    result = asyncio.run(
        preprocessor(
            make_payload(
                {
                    "messages": [{"role": "user", "content": "Describe this."}],
                    "images": [image] if with_image else None,
                    "audios": ["question.wav"] if with_audio else None,
                    "videos": ["clip.mp4"] if with_video else None,
                }
            )
        )
    )

    # The processor has one policy for the whole image list, including mixed inputs.
    assert fake_processor.options == (
        {"max_slice_nums": 1, "use_image_id": False} if with_video else {}
    )
    expected_images = ([image] if with_image else []) + ([frame] if with_video else [])
    if expected_images:
        assert len(fake_processor.images[0]) == len(expected_images)
        for actual, expected in zip(fake_processor.images[0], expected_images):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        assert fake_processor.images is None
    if with_audio:
        np.testing.assert_array_equal(
            fake_processor.audios[0][0], np.array([0.25, 0.5], dtype=np.float32)
        )
    else:
        assert fake_processor.audios is None
    prompt_text = result.data["prompt"]["prompt_text"]
    assert prompt_text.count("<image>./</image>") == len(expected_images)
    assert prompt_text.count("<audio>./</audio>") == int(with_audio)


@pytest.mark.parametrize("changed", ["image", "video"])
def test_minicpm_visual_cache_key_tracks_decoded_content(monkeypatch, changed) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor._processor = (
        FakeProcessor()  # noqa: leading-underscore  # production name
    )
    preprocessor.speech_enabled = False
    monkeypatch.setattr(
        preprocessor, "render_chat_template", lambda messages, **_: str(messages)
    )
    media = {"image": Image.new("RGB", (2, 2), "red"), "video": torch.zeros(2, 3, 2, 2)}

    async def images(raw_images):
        return [media["image"]]

    async def videos(raw_videos, **kwargs):
        return [media["video"]], [1.0], None

    monkeypatch.setattr(preprocessor_mod, "ensure_image_list_async", images)
    monkeypatch.setattr(preprocessor_mod, "ensure_video_list_async", videos)

    def cache_key(name: str = "same") -> str:
        inputs = {
            "messages": [{"role": "user", "content": "Describe this."}],
            "images": [f"https://media.invalid/{name}.png"],
            "videos": [f"https://media.invalid/{name}.mp4"],
        }
        data = asyncio.run(preprocessor(make_payload(inputs))).data
        key = data["encoder_inputs"]["image_encoder"]["cache_key"]
        assert data["mm_inputs"]["image"]["cache_key"] == key
        return key

    before = cache_key()
    assert cache_key() == before
    # New content behind the same URL must not reuse the previous entry.
    media[changed] = (
        Image.new("RGB", (2, 2), "blue")
        if changed == "image"
        else torch.ones(2, 3, 2, 2)
    )
    after = cache_key()
    assert after != before
    # Identical content at another address shares the entry.
    assert cache_key("other") == after


def test_inline_video_interleaves_audio_and_keeps_surrounding_text(monkeypatch) -> None:
    frames = [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")]
    segments = [np.zeros(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]

    async def load_video(source, **options):
        assert source == "clip.mp4"
        return TimedVideo(
            frames=frames,
            audio_segments=segments,
            timestamps_seconds=[0, 1],
            duration_seconds=2,
        )

    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.prompt_frontend.load_timed_video", load_video
    )
    rendered = asyncio.run(
        render_ordered_chat(
            [
                {
                    "role": "user",
                    "content": [
                        "Before\n",
                        {
                            "type": "video_url",
                            "video_url": {"url": "clip.mp4", "use_audio": True},
                        },
                        "After",
                    ],
                },
            ]
        )
    )
    assert (
        rendered.messages[0]["content"]
        == "Before\n<image>./</image><audio>./</audio><image>./</image><audio>./</audio>After"
    )
    assert rendered.audio_turn_indices == [0, 0]
    assert rendered.omni_mode
    assert rendered.has_video
    assert rendered.images == frames
    for actual, expected in zip(rendered.audios, segments):
        np.testing.assert_array_equal(actual, expected)


def test_video_decoder_preserves_audio_timeline(tmp_path, monkeypatch) -> None:
    from fractions import Fraction

    import av

    video_path = tmp_path / "timeline.mp4"
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(23):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), index, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    waveform = np.arange(36800, dtype=np.float32)
    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.video_frontend.librosa.load",
        lambda path, sr, mono: (waveform, sr),
    )
    video = MiniCPMVideoIO(use_audio=True).load_file(video_path)
    assert video.timestamps_seconds == [0, 1, 2]
    assert [len(segment) for segment in video.audio_segments] == [16000, 16000, 4800]
    for index, segment in enumerate(video.audio_segments):
        np.testing.assert_array_equal(
            segment, waveform[index * 16000 : (index + 1) * 16000]
        )
