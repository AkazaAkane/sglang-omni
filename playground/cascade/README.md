# Cascade Image Playground

The cascade playground supports single-image upload with a continuous microphone, automatic turn detection and speech-triggered interruption. See the [image cascade guide](../../examples/streaming_cascade/README.md) for Docker setup, model revisions, GPU layout, the script example, limits and real-model acceptance tests.

## Start

From the repository root, start these commands in separate terminals after creating the Docker environment:

```bash
docker exec sglang-omni-d1 sgl-omni serve \
  --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-vl-tts-base/launch.yaml \
  --enable-realtime --host 0.0.0.0 --port 18020
docker exec sglang-omni-d1 python playground/cascade/app.py \
  --api-base http://127.0.0.1:18020 --host 0.0.0.0 --port 18080
```

Open `http://localhost:18080`, upload an image and start the session. The microphone starts after the server acknowledges the image and stays open during answers. Pause for 800 ms to submit a question; speaking over the answer interrupts it and clears buffered playback. The microphone button mutes or unmutes input, and retries capture after a permission failure. Use localhost or HTTPS for browser microphone access from another machine.

## Upload Lifecycle

Uploading validates and previews the selected image locally. Before the session starts the status shows Ready; during a live session uploads are sent immediately and show Sending, then Received after acknowledgment. Rejected uploads leave the microphone and session available for another image or a question using retained visual history.

The image is selected at the utterance's speech onset. Replacing it during speech or an answer applies to the next utterance, including when the earlier transcript is delayed. An unchanged preview is not resent on follow-up questions. Removing the preview does not retract an already acknowledged upload or erase earlier images from conversation history; start a new session for fresh visual context.

Image uploads are shown whenever the deployment grants image input. A deployment without automatic turn detection retains the manual record/submit controls when it grants client commit. Audio-only deployments retain their continuous microphone workflow. The bundled icons are generated from Lucide 1.61.0 and include its ISC license notice.

## Verification

On 2026-10-27, Playwright exercised the actual Docker cascade with Qwen3-ASR-1.7B, Qwen3-VL-4B-Instruct, and Qwen3-TTS-12Hz-1.7B-Base. Chromium captured controlled WebAudio speech through the page's continuous microphone worklet and played nonempty generated audio. Red and blue images produced their respective answers; replacing red with blue during speech preserved the current red answer; speaking over the next blue reply cancelled it and a handle follow-up completed. No client commit or explicit response cancellation was sent. The player reported the interrupted playback position through truncation. Mute, preview removal, session cleanup, and desktop/mobile screenshots also passed, with no page errors, horizontal overflow or caption overlap.

All 19 browser unit tests pass. They cover acknowledgment ordering, continuous capture during replies, playback clearing on speech, recoverable rejection, microphone permission failure and image resubmission on retry, disconnect cleanup, pending audio flushing before image upload or manual commit, and existing playback behavior. Physical microphone capture and macOS recording still require checking on the client machine.

```bash
node --test playground/cascade/frontend/orb/*.test.mjs
```
