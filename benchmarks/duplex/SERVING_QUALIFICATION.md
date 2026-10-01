# Native serving qualification (2026-10-02)

Benchmark base revision: `4dbac033141d252e024b6d24543f96b3ac8eb180`, with
the native metrics changes on `feat/duplex-serving-native-metrics`.
Tests and captures ran in `docker.io/hongccc/sglang-omni:dev`, image ID
`2bac261779c813672df92a8ce1cab74a4eaa18681b5c645ee89e4dc8765c6119`.
The image's SGLang 0.5.19 was upgraded to the checkouts' pinned 0.5.20;
MiniCPM-o's optional `onnx` dependency was installed (1.23.1).
PyTorch was 2.13.0+cu130 and Transformers 5.12.1. Each server used one H200.

VoiceChat server SHA: `c3c33cfa91111217b7a88bb416b1259fbe5b2af8` (#2188).
Model revision: `443794ea956ef0065f001967ffd00e77f519cb39`.
Server command, from that checkout:

```bash
python examples/run_nemotron_voicechat_duplex.py \
  --model-path nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --serve --port 8097
```

MiniCPM-o server SHA: `167f16580acada0e6e144f06ba4cef007a747171` (#2377).
Model revision: `503e754207c94da6bb26850b4469f367c9ea3582`.
The unmodified `examples/full_duplex/minicpmo.yaml` supplies `max_sessions=2`,
sampled decoding, `force_listen_count=3`, and stage memory fractions
0.12/0.52/0.15/0.15. Server command, from that checkout:

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang_omni.cli serve \
  --config examples/full_duplex/minicpmo.yaml \
  --model-path openbmb/MiniCPM-o-4_5 --enable-realtime
```

The input was the VoiceChat checkpoint's `turn_taking.wav`, normalized by the
benchmark to mono 16 kHz PCM16, duration 41.08 s. Captures used the following
command with the corresponding profile, endpoint, revisions and config JSON:

```bash
python -m benchmarks.duplex.serving --url "$URL" --audio "$INPUT" \
  --profile "$PROFILE" --concurrencies "$LEVELS" --output-dir "$OUTPUT" \
  --model-id "$MODEL" --model-revision "$MODEL_SHA" \
  --server-sha "$SERVER_SHA" --server-config "$CONFIG_JSON" --timeout-s 180
```

VoiceChat used levels `1` and a separate `2,4` sweep; MiniCPM-o used `1,2,4`.
Each sweep included one excluded warmup and one measured repeat per level.
Raw traces, input receipts, config JSON and provenance summaries are retained
in the workspace's `duplex-qualification/` directory, outside the repository.

| Model | C | Admitted/attempted | Success/admitted | Unit lag p95 (ms) | Response TTFA p95 (ms) | Response gap-excess p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| VoiceChat | 1 | 1/1 | 1/1 | 3239.8 | 0.1 | 21.0 |
| VoiceChat | 2 | 1/2 | 1/1 | 3596.6 | 0.1 | 22.4 |
| VoiceChat | 4 | 1/4 | 1/1 | 3434.3 | 0.1 | 23.9 |
| MiniCPM-o | 1 | 1/1 | 1/1 | 3.4 | 0.3 | N/A |
| MiniCPM-o | 2 | 2/2 | 2/2 | 544.5 | 0.3 | 256.0 |
| MiniCPM-o | 4 | 2/4 | 2/2 | 515.6 | 1076.1 | 426.1 |

All measured runs had valid client timing. Loop lag p99 was 1.4-1.5 ms;
send lateness p95 was 1.6-1.8 ms and p99 was 2.0-2.1 ms, below the 20 ms
limit. VoiceChat contributed 513 full-unit samples per run; MiniCPM-o
contributed 41 per admitted session. The terminal partial units were excluded.
MiniCPM-o C1 produced one audio packet, so no within-response gap sample exists;
C2 and C4 contributed 20 and 7 gaps, and 4 and 6 audio responses respectively.
These are single-repeat observations, not a stable capacity estimate.

VoiceChat C1 additionally measured session TTFA 102.3 ms, coverage 100.1%
(terminal padding included), final/required-buffer drift 3352.2 ms and underrun
duration 3080.7 ms (7.5%). This drift shows the default serving configuration
did not keep pace throughout the recording despite valid client pacing.
MiniCPM-o continuous-only fields were N/A in every session and aggregate.
C4 rejected exactly two MiniCPM-o attempts; VoiceChat C2/C4 rejected one/three.
All admitted sessions drained input and closed without protocol errors.

Validation in the same container:

```bash
python -m pytest tests/unit_test/benchmarks/test_duplex_serving.py \
  tests/unit_test/benchmarks/test_duplex_client.py \
  tests/unit_test/benchmarks/test_duplex_oracle.py \
  tests/unit_test/benchmarks/test_duplex_profiles.py -q
python -m pre_commit run --all-files
```

Outcome: 125 tests passed; repository-wide pre-commit checks passed.
The final N/A fallback test was additionally rerun with the serving suite:
11 passed. Container startup alone was not counted as qualification.
