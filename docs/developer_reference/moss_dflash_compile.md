# MOSS-TD DFlash compilation investigation

## Finding

The SGLang 0.5.23 regression described in [PR #3606](https://github.com/sgl-project/sglang-omni/pull/3606) comes from compiler-generated copies around the draft KV-cache write. Disabling all decoder compilation recovers performance, but the narrower fix is to keep draft attention outside compiler tracing and inside the existing CUDA graph. The target decoder and surrounding draft projections can still compile.

SGLang's `MHATokenToKVPool` allocates matching K/V shapes as two views of one allocation. The registered `sglang.store_cache` operator mutates both views. On the measured Torch 2.14.1 / SGLang 0.5.23 stack, functionalization around that operator generates a full shared-storage clone and a full V-cache copy back. The copy size depends on allocated cache capacity, even when a request writes only eight tokens. The [0.5.22 allocator](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/mem_cache/memory_pool.py#L2575) uses separate K/V allocations, while the [measured 0.5.23 source](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/memory_pool.py#L2630) shares them. The cache-layout ablation below isolates that change.

For the measured 645,556-token pool, generated code takes K/V views of shape `[645557, 8, 128]`, allocates 1,322,100,736 BF16 elements, calls `sglang.store_cache`, and copies back 661,050,368 elements. The extra allocation is about 2.46 GiB. The clone and copy run in each of the five draft layers on every compiled speculative step. They move about 39.7 GB per step, including reads and writes, rather than only the proposed tokens.

A 25-batch CUDA trace attributes 153.217 ms to 125 `triton_poi_fused_as_strided_clone_0` launches and 76.916 ms to 125 `triton_poi_fused_as_strided_copy__2` launches. Together they consume 82.9% of the trace's 277.629 ms of kernel time. Neither kernel appears in the compilation-disabled trace or the fixed trace. The fixed trace has 46.503 ms of kernel time; this is diagnostic evidence, not an end-to-end throughput measurement.

## Change

`ReducedVocabularyDFlashDraftModel` marks each draft `RadixAttention.forward` with `torch.compiler.disable`. This creates a tracing boundary around the attention/cache update. CUDA graph capture still records the direct cache writes and attention kernels, so serving replays them without a Python call per layer. Other draft operations, the target decoder and the captured reduced-vocabulary sampler retain their existing behavior.

This is a narrower alternative to the broad compilation workaround proposed in #3606. It does not change weights, proposal width, target verification or sampling settings. Compiled arithmetic can still change near-tied token decisions; transcript identity is not assumed. The shared reduced-vocabulary adapter also serves Qwen3-ASR; the performance measurements here concern MOSS-TD.

[SpecPilot](https://github.com/StargazerX0/SpecPilot) motivates profiling the actual deployment, isolating memory traffic and validating the resulting outputs. Its model-specific wide trees, W12 weights and two-level verification were not transplanted: this bottleneck is a compiler-generated cache copy, and removing it needs no new drafting algorithm or weight format.

## Reproduce the cache-write bottleneck

Run each mode in a fresh process on an idle CUDA GPU. The probe uses the actual SGLang cache writer, compiles before timing, captures a CUDA graph and times replays with CUDA events. It checks the written slots, reserved zero slot and every untouched slot exactly. Each mode disables compiler artifact reuse so storage-layout ablations cannot reuse a specialization from another mode.

```bash
for tokens in 65536 262144; do
  for mode in compiled_shared compiled_separate boundary_shared; do
    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_dflash_cache_write \
      --cache-tokens "$tokens" --mode "$mode" \
      --output "cache-${tokens}-${mode}.json"
  done
done
```

`compiled_shared` reproduces the allocator's aliased K/V storage. `compiled_separate` isolates storage aliasing while keeping compilation. `boundary_shared` retains aliasing but excludes the writer from tracing, matching the fix's mechanism. This is a cache-write microbenchmark, not an ASR speedup estimate.

| Cache token slots | Compiled shared storage | Compiled separate storage | Shared storage with boundary |
|---:|---:|---:|---:|
| 65,536 | 194.81 µs | 3.44 µs | 3.48 µs |
| 262,144 | 755.83 µs | 3.76 µs | 3.61 µs |

Each row writes eight BF16 tokens with 1,024 elements per token. The separate-storage ablation and the tracing boundary both remove the capacity-dependent cost.

## End-to-end validation

The measurements use BF16 MOSS-TD on one H200, FA3 attention, eight draft tokens, graph capacity 16, `torch_compile_max_bs=4`, `mem_fraction_static=0.60`, synchronous scheduling and disabled prefill graphs. The final comparisons disable the prefix cache, retain the default encoder cache, start a fresh server per leg, and use the same GPU in compilation-disabled / fixed / fixed / compilation-disabled order. Startup and warmup are excluded from request throughput; retaining compilation retains its startup overhead. A separate no-draft server measures the autoregressive reference with compilation enabled.

Target revision: `10304d926740cf5a5cee05a268086e3265197910`. Draft revision: `2e3f00b0932112e4f92a4ef67a4cd60ee855056c`. Dataset revision: `382168daa4e9d764318a20c1365fd0af0d00d785`. Final source base: `61b70c063`. The earlier regression reproduction used `f4944c570`; its target model and reduced-vocabulary draft modules match the final base.

For transcription and diarization accuracy, run the first 120 `movies800times` clips after three warmup requests:

```bash
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times --concurrency 1 --max-samples 120 --warmup 3 \
  --use-existing-server --port 18102 --output-dir evaluation
```

The concurrency comparison additionally uses the first 160 locally materialized Movies800 clips with the SeedTTS-style transcription client, two measured repeats per concurrency and a discarded warmup at each concurrency. Its language override and output format differ from the diarization client, so the two workloads' absolute request rates must be interpreted separately.

## Results

The final same-GPU comparison follows the protocol above. Compilation-disabled and fixed values are means of two fresh-server legs, each with two measured concurrency repeats. The original compiled control is a later fresh-server leg with one repeat, warmed by the preceding 120-clip accuracy run. No profiler is active in these measured runs.

| Concurrency | Original compiled requests/s | Compilation disabled requests/s | Fixed requests/s | Fixed / original | Fixed vs disabled |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.96 | 10.15 | 10.27 | 3.46× | +1.1% |
| 4 | 9.64 | 25.94 | 25.65 | 2.66× | -1.1% |
| 16 | 54.37 | 60.51 | 59.81 | 1.10× | -1.2% |

This fixes the large regression against the original compiled path. It does not establish a throughput advantage over disabling compilation: the small differences against that workaround vary with concurrency and run. Each of the four ABBA legs completes 120 diarization requests and 960 measured transcription requests; the original compiled control completes 120 and 480 respectively.

| 120-clip diarization workload | Requests/s | CER | DER, collar 0 |
|---|---:|---:|---:|
| Original compiled | 2.90 | 5.68% | 21.86% |
| Compilation disabled | 8.43 | 5.62% | 22.00% |
| Fixed | 8.61 | 5.69% | 21.94% |
| No draft, compiled | 3.45 | 5.67% | 22.05% |

All 120 requests complete in every accuracy run. CER/DER use 116 clips; four clips have no reference segments. The SeedTTS-style client has seven unscorable references per 160 requests. Small transcript differences remain, consistent with near-tied decisions under compiled arithmetic; no accuracy improvement or token identity is claimed.

## Checks and evidence

The focused speculative/MOSS suite passes 302 non-accelerator cases, and the checkpoint-backed encoder graph suite passes all 20 accelerator cases on H200. The new compiler-boundary regression fails on the original source with `observed_compilation=[True]` and passes with the fix. All applicable pre-commit hooks pass, and `git diff --check` is clean.

The [machine-readable evidence](moss_dflash_compile.json) includes per-leg configs, concurrency metrics, latency ranges, CER/DER, package and GPU provenance, cache probe results, trace hashes and the tested source hash. Raw profiler traces are intentionally excluded from the branch.

The results cover one H200, concurrency through 16 and short clips. They do not establish H100 performance, Qwen3-ASR performance, long-audio memory-pressure behavior or stochastic-sampling behavior. The change keeps the existing CUDA-graph and eager fallback routes.
