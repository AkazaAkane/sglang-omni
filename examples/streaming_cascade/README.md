# Image Voice Cascade Demo

Upload an image, ask spoken questions with a continuous microphone, and receive streamed image-grounded text and speech. An 800 ms pause ends each question; speaking over an answer interrupts it. The profile uses Silero VAD, Qwen3-ASR-1.7B, Qwen3-VL-4B-Instruct through SGLang Chat Completions, and Qwen3-TTS-12Hz-1.7B-Base with the existing Vivian reference recording.

## Docker Environment

Run these commands from the repository root. The tested profile uses GPU 0 for the VLM and GPU 1 for ASR and TTS. The validation machine has two NVIDIA H200 GPUs with 143771 MiB each; this is a verified layout, not a minimum hardware requirement.

```bash
docker run -d --name sglang-omni-d1 --gpus all --network host --ipc host \
  -v "$PWD:/workspace/sglang-omni" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/sglang-omni -e SGLANG_OMNI_AUTO_CLONE=0 \
  -e SGLANG_OMNI_STRICT_PORT=1 -e OMP_NUM_THREADS=4 --entrypoint bash \
  docker.io/hongccc/sglang-omni@sha256:06ed306707bcf3d9e9beb5ba7ce66f09a666cac11c23455ece50c34199c0e985 \
  -c 'exec sleep infinity'
docker exec sglang-omni-d1 python -m pip install --no-deps --no-build-isolation -e .
docker exec sglang-omni-d1 python -m pip install pytest pytest-asyncio pre-commit
```

The image contains SGLang 0.5.22, PyTorch 2.14.1+cu130, Transformers 5.19.0, Pillow, HTTPX and the model dependencies. For a Podman installation exposing Docker-compatible commands, replace `--gpus all` with `--device nvidia.com/gpu=all --security-opt label=disable`.

| Model | Validated Hugging Face Revision |
| --- | --- |
| Qwen/Qwen3-ASR-1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| Qwen/Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | `fd4b254389122332181a7c3db7f27e918eec64e3` |

The VLM launch command pins its revision. To reproduce ASR and TTS weights exactly, download the revisions above with `huggingface_hub.snapshot_download` and replace each service command's `--model-path` with the returned container path; retain the adapter's public `model` identifier. GPU placement, ports and service commands are defined in `profiles/qwen3-asr-qwen3-vl-tts-base/launch.yaml`.

## Script

Place `picture.png`, `question.wav` and optionally `followup.wav` in the mounted repository. Input WAV files must be mono, 16-bit PCM at 16000 Hz. The script starts missing services, reuses existing services on the configured ports, and waits for each answer before sending the next file.

```bash
docker exec sglang-omni-d1 python examples/streaming_cascade/run.py \
  question.wav followup.wav --image picture.png --commit \
  --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-vl-tts-base/launch.yaml \
  --output cascade-response.wav
```

The console prints transcripts and text deltas; `cascade-response.wav` contains the complete 24 kHz PCM16 answers. `--image` applies only to the first question; follow-ups retain its image in conversation history. Omit `followup.wav` for one question. `--commit` disables automatic turn detection for this scripted session and waits for each answer before the next file. The live browser uses VAD; this profile does not use Smart Turn.

To isolate the VLM interface, start its service with the command in the profile, then run:

```bash
docker exec sglang-omni-d1 python examples/streaming_cascade/probe_vlm.py \
  picture.png 'What color is the cup in the picture?' \
  --base-url http://127.0.0.1:18010/v1 --request-output vlm-request.json
docker exec sglang-omni-d1 curl -N http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @vlm-request.json
```

The saved request uses JPEG or PNG data URLs and `stream: true`. The adapter validates SSE completion and its finish reason.

## Supported Semantics

The image is fixed at the utterance's speech onset in live mode, or when manual recording starts. ASR reads audio and utterance controls only; it waits for the first VAD boundary rather than accumulating startup silence, and crops the overlapping audio packet to that boundary. The image profile uses zero VAD prefix padding so an image uploaded just before speech is selected at speech onset rather than preceding silence. Dialog stores that image in the corresponding user message, including when the transcript arrives after another upload or the reply waits behind an earlier answer. Image bytes stay structured through the Session Graph and become `text` and `image_url` content parts at the HTTP request boundary.

The browser acknowledges the initial image before opening its microphone, then keeps capturing while text and speech stream back. A new speech boundary interrupts the current reply, clears the player's queued audio and reports the played position through `conversation.item.truncate`. Later uploads are sent on the microphone timeline and apply to the next utterance. Removing the preview does not retract acknowledged uploads or visual history.

Only one static JPEG or PNG per question is supported. Uploads are limited to 512 KiB and 4194304 pixels; malformed, animated and oversized images are rejected. Realtime rejection is nonfatal and does not consume the image slot. No image is valid input; the VLM receives an explicit missing-image instruction when no visual history is retained.

Image history has independent `max_history_images` and `max_history_image_bytes` budgets, defaulting to eight images and 4 MiB. Older attachments are removed when either budget is exceeded; their user and assistant text remains. Pending uploads are also bounded by these limits. Image bytes count toward session resource accounting. Start a new session when text history reaches its own limit. Continuous camera capture, multiple images per question, video, and synchronized audio/video are outside this demo.

An explicit `input:image_frame` edge to Dialog requires an `openai_chat` consumer with `supports_images: true`. Wiring images to a text-only generator is rejected at load and during component updates. Unqualified `input` edges keep their audio-only meaning.

## Validation

Record "What color is the cup in the picture?" as `question.wav` and "Does the cup have a handle?" as `followup.wav`. Real-model tests create diagnostic red and blue mug images and exercise all three services without mocking them. In another terminal, start the realtime server with `docker exec sglang-omni-d1 sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-vl-tts-base/launch.yaml --enable-realtime --host 0.0.0.0 --port 18020` before running these checks.

```bash
docker exec \
  -e SGLANG_OMNI_TEST_REALTIME_PORT=18020 \
  -e SGLANG_OMNI_CASCADE_AUDIO=/workspace/sglang-omni/question.wav \
  -e SGLANG_OMNI_CASCADE_FOLLOWUP_AUDIO=/workspace/sglang-omni/followup.wav \
  -e SGLANG_OMNI_IMAGE_RESULTS=/workspace/sglang-omni/image-results \
  sglang-omni-d1 python -m pytest tests/test_model/test_streaming_cascade_images.py -q
docker exec sglang-omni-d1 python -m pytest \
  tests/unit_test/models/test_streaming_cascade_images.py \
  tests/unit_test/models/test_streaming_cascade.py \
  tests/unit_test/pipeline/test_streaming_cascade.py \
  tests/unit_test/serve/test_realtime_image_input.py \
  tests/unit_test/serve/test_session_bridge.py -q
```

On 2026-10-27, all seven real-model cases passed in 89.65 seconds through the playground's WebSocket proxy in this Docker environment. Identical audio yielded "red" or "blue" according to the image; submitting B immediately after A's question preserved A's answer; a follow-up retained the handle context; two concurrent sessions returned their own colors; missing, corrupt and over-limit images allowed a subsequent valid question. The live cases additionally verified startup silence, image replacement during speech, automatic turn completion without client commits, and speech-triggered cancellation followed by an answer about the new image. `SGLANG_OMNI_IMAGE_RESULTS` saves complete protocol events and nonempty answer WAV files.

The delayed-ASR unit test forces the interleaving with events around the HTTP boundary. Other unit tests cover structured conversion, image budgets, replacement, session isolation, text-only rejection, PCM cropping for either VAD/audio arrival order, preserving image timestamps within audio units and published transcript timestamps on the shared clock, and switching from an idle VAD microphone to manual turns. Accuracy outside these diagnostic images and performance on other hardware are not established by these checks.
