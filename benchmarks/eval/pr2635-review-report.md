# MiniCPM-o PR 2635 review verification

This report supersedes the MiniCPM measurements in frontend-parity-report.md.
Production commit: cf0272d6172e62217d08002e141f48496ce3c766, based on
9ab7942c14f3a1df52ba79989c7a6c9b6fb00439. Shared production modules are unchanged.

## Changes and official implementation

- MiniCPM file/URL videos use the same local reference loader through inline and
  top-level entry points, including video without audio. Predecoded frame tensors
  remain supported; mixed path/tensor input uses the local loader for paths.
- Image encoder and thinker cache keys include the effective default/video
  processing policy. Video applies max_slice_nums=1 and use_image_id=False to
  all images in that request, including mixed image/video content.
- Local decode uses the existing eight-worker media pool and waits for decoder
  cleanup before propagating cancellation, including repeated cancellation.
- Whole-conversation omni concatenation is retained. Frozen official chat()
  uses "".join(cur_msgs) for each message when omni_mode or stream_input is true;
  otherwise it uses newline separators. Scoping this only to frame/audio
  boundaries would change the official prompt for earlier multipart text turns.
- Shared abstraction and backend replacement are deferred to a separate PR.

Frozen model revision: 503e754207c94da6bb26850b4469f367c9ea3582.
The README delegates video decoding to minicpmo-utils>=1.0.5. This investigation
pins that external package separately to 1.0.5; the model revision alone does
not freeze its code. Utility utils.py SHA256:
5c9ef5ee046302da3b2b2b4ff0b162c017f215aa26e9f62d546f6838c38a922c.

The default utility samples integer seconds for short clips and uniformly caps
a 0.1-second candidate timeline at 64 frames for long clips. Indices are
int(requested_time * average_fps); duration is frame_count / average_fps.
Librosa produces 16 kHz mono audio, split at requested sample times, with a
minimum 1600-sample final segment. Requested times are not decoded frame PTS.
Frames remain native resolution; the model's image processor performs PIL
bicubic resize with scale_resolution=448 and patch_size=14. The shared Qwen
sampler/resize and PyAV audio policy are different. The utility also supports
FFmpeg, so Decord itself is not a model architecture requirement.

All 17 benchmark clips match official frames, audio samples and timelines
exactly. TorchCodec at the same Decord-derived indices matches native frames
and image-processor tensors on all 17 clips. Shared PyAV audio has identical
segment lengths but different samples on all 17. Requested times differ from
source PTS by up to 0.041583 seconds. These observations support a future shared
native-frame decoder with explicit indices and timing metadata, while preserving
MiniCPM's sampling, audio and processor resize policy. They do not establish
backend equivalence for variable-rate, unusual-codec or damaged media.

Sources:

- [Official chat concatenation](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/modeling_minicpmo.py#L1168-L1171)
- [Official README](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/README.md)
- [Official processor](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/processing_minicpmo.py)

## Tests and input parity

- Final focused frontend/preprocessing tests: 145 passed.
- Broader regression: 1762 passed, 15 environment-dependent skips. This run
  preceded four additional empty-content parametrizations; those passed in the
  final focused run, together with the affected code paths.
- Enabled live inline/top-level ASR API tests: 2 passed.
- Full pre-commit checks passed, including Rust formatting.
- Four cache collision cases and cancellation reproduce failures before fixing.
  A real processor produces five regular-image slices versus one video slice
  for identical 896x896 pixels. Four alternating requests execute the counting
  encoder twice and reuse only compatible entries.
- Eight standard input fixtures and single/multi-turn audio-video fixtures match
  official prompts, token IDs, bounds, image inputs and audio features. A preceding
  text-only multipart turn exercises the global omni concatenation rule.
- Inline and top-level video inputs match each other with and without audio;
  silent single/multi-turn captures also match official prompts, IDs, pixels,
  target sizes and bounds. Empty top-level final text adds no trailing separator.
- All 50 final inline audio/video outputs match the earlier PR video run.
  A separate final top-level run also completes 50 cases with zero errors,
  scores 72%, and matches every inline generated response.
  Input parity is not a claim that arbitrary generated sentences are identical.

## Final generation benchmark

The final production commit completed 350 text/image/audio cases and 50
audio/video questions with zero request errors. Official results use the same
frozen inputs and checkpoint, in a separately recorded reference runtime.

| Task | Cases | Official HF | Final PR |
| --- | ---: | ---: | ---: |
| MMMU accuracy | 50 | 22% | 20% |
| MMSU accuracy | 100 | 69% | 67% |
| English raw WER | 100 | 0.6865% | 44.9275% |
| English extracted WER | 100 | 0.6865% | 3.8902% |
| Chinese CER | 100 | 0.5405% | 0.8108% |
| Audio/video accuracy | 50 | 74% | 72% |

Earlier PR runs recorded MMSU 68%. The final run records 67%; mmsu-0068 changed
from D to A. A targeted rerun on the untouched earlier PR also answered A three
times. This does not prove a frontend cause or a GPU cause, and complete
generation parity is not claimed. The observed final result is retained.

English raw WER includes a ||| JSON suffix emitted on 45 samples. Extracted WER
removes only that suffix; this execution difference remains unresolved. Strict
option extraction counts missing answers as incorrect; MMMU has 39 missing
answers on both official and final PR within the 256-token limit. One video
answer differs from official: video-020-2, official D versus PR C.

## Decode-policy ablation

| Configuration | Correct / 50 | Accuracy | Request errors |
| --- | ---: | ---: | ---: |
| Original baseline, inline input | 22 | 44% | 0 |
| New ordering with old decode policy | 35 | 70% | 3 |
| Final PR reference policy | 36 | 72% | 0 |
| Original baseline, top-level input | 36 | 72% | 0 |
| Official HF | 37 | 74% | 0 |

Original baseline: b9aa02d411c27f1509711cfc90f43a3f6c384ac5. It drops inline
media during normalization, so 44% to 72% is not evidence of a decoder accuracy
gain. Old top-level video used an explicit 64-frame cap. The old-policy adapter
keeps the new renderer, slicing options, weights and audio turn ownership, but
uses Qwen/TorchCodec frame sampling/resize, actual sampled PTS and PyAV audio.
It splits the waveform at consecutive PTS with reference final padding.

All three errors concern HIjX8OPuf-w: 1135 audio placeholders versus 1131 pooled
embedding rows. A real-processor probe reproduces the mismatch, with merged mel
lengths [3000, 3000, 3000, 2318]. Individual segment placeholder rounding and
merged feature rounding differ. Errors count as incorrect. This compares whole
decode policies, not decoder engines alone; blindly reverting would reintroduce
an input correctness issue.

Additional baseline 350-case runs observed inline/top-level MMMU 6%/20% and
MMSU 43%/68%. Earlier published baseline 14%/42% was not reproduced by these
runs and is not presented as the current comparison.

## One-clip decode microbenchmark

fFjv93ACGo8.mp4: about 74.3 seconds, native 640x360, 29.97 FPS, audio enabled,
64-frame cap. Separate processes, two warmups and 31 measured repeats. Timing
covers frame and audio decode, excluding network, model and processor features.
Peak RSS includes initialization, warmups and measured repeats; process-tree
RSS samples descendants every 5 ms.

| Policy | Frame size | Median | P95 | Peak process RSS | Peak process-tree RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Existing Qwen/TorchCodec/PyAV | 644x364 | 566.10 ms | 599.92 ms | 1584.07 MiB | 1584.07 MiB |
| MiniCPM Decord/librosa | 640x360 | 641.23 ms | 653.67 ms | 1080.66 MiB | 1109.39 MiB |

The reference policy was 13.27% slower at the median and used 31.78% less peak
process RSS on this clip. The direct old loader decodes audio/video sequentially;
its async serving path can overlap them. This is not an end-to-end latency,
throughput or concurrency claim. Pixel budgets resize sampled frames after
decoding and do not cap initial native-resolution allocation.

## Runtime, evidence and reproduction

- Image: docker.io/hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df.
- Host docker is Podman 4.9.3; two NVIDIA H200, driver 595.71.05.
- Serving: SGLang 0.5.21, Torch 2.13.0+cu130, Transformers 5.12.1.
  The image's SGLang 0.5.19 was upgraded inside the container.
- Reference: Torch 2.8.0, Transformers 4.51.0; both use Decord 0.6.0,
  librosa 0.11.0 and minicpmo-utils 1.0.5. The utility was installed without
  dependencies because its librosa 0.9.0 pin conflicts with this runtime.
- OMP_NUM_THREADS=8, MKL_NUM_THREADS=8; greedy, 256 new tokens, repetition
  penalty 1, text output, thinking disabled; existing audio TTS template retained.

| Dataset | Frozen revision | Cases |
| --- | --- | ---: |
| MMMU CI50 | ff72fd69cc7e0719e04a0ddb12d160de89c6fefe | 50 |
| MMSU CI2000 | 5ae6ed4343a89566be6dd90023d529f1bd97802d | 100 |
| SeedTTS EN/ZH | 81d1901582dee1293a537a6d945d084301712c41 | 100 each |
| Video-MME CI | 833bd815c628ff277911bea3b1563545b21d5e27 | 50 / 17 clips |

Small aggregate/profile/probe results are preserved beside this report under
pr2635-results/. Raw generation captures, input captures, manifests, media and
the isolated ablation/profile/probe scripts remain locally in benchmark-data/
and benchmark-tools/. The final raw files are minicpm-only-final-multimodal.json
and minicpm-only-final-video.json; official captures are final-official-*.

Use the validation scripts with production imported first. Run official capture
in the Transformers 4.51 reference environment. Reuse the frozen manifests and
checkpoint directories, with the production server started at the tested commit.

```bash
PYTHONPATH=PRODUCTION:VALIDATION python -m benchmarks.eval.minicpm_input_parity --backend omni --model-path MODEL --video CLIP --input-style inline --output inline-input.json
PYTHONPATH=PRODUCTION:VALIDATION python -m benchmarks.eval.minicpm_input_parity --backend omni --model-path MODEL --video CLIP --input-style top-level --output top-input.json
PYTHONPATH=PRODUCTION:VALIDATION python -m benchmarks.eval.minicpm_input_parity --backend official --model-path MODEL --video CLIP --output official-input.json
# Repeat the three captures with --no-video-audio for silent-video policy.
python -m benchmarks.eval.omni_frontend_parity --manifest FIXTURES/manifest.json --backend omni --api-url http://127.0.0.1:18004 --output multimodal.json
python -m benchmarks.eval.omni_frontend_parity --manifest VIDEO_FIXTURES/manifest.json --backend omni --api-url http://127.0.0.1:18004 --output video.json
python -m benchmarks.eval.omni_frontend_parity --manifest VIDEO_FIXTURES/manifest.json --backend omni --input-style top-level --api-url http://127.0.0.1:18004 --output top-video.json
python -m benchmarks.eval.frontend_parity_metrics multimodal.json video.json
OMNI_PARITY_API_URL=http://127.0.0.1:18004 OMNI_PARITY_MANIFEST=FIXTURES/manifest.json pytest tests/unit_test/serve/test_openai_api.py -k live_chat_frontend
```

Metrics use NFKC, lowercase and removal of Unicode punctuation/symbols; English
whitespace is collapsed and Chinese whitespace removed. Original output text
is retained in local raw captures.
