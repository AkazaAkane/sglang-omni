# Realtime serving benchmark

Run against a server exposing `/v1/realtime`:

```bash
python -m benchmarks.duplex.serving \
  --url ws://127.0.0.1:8097/v1/realtime \
  --audio input.wav --profile nemotron-voicechat-pr2188 \
  --concurrencies 1,2,4,8 --repeats 3
```

The input is normalized to mono 16 kHz PCM16 using the existing duplex audio
loader. Each concurrency level starts only after its sessions finish negotiation.
By default, all configured sessions use one monotonic start deadline and fixed
80 ms input deadlines. A late sender sends buffered frames as soon as it can;
subsequent deadlines remain anchored to the original start. Use `--stagger-ms 80`
to spread session starts evenly across an 80 ms phase. Each session has its own
JSONL wire trace and input-send-receipts.json; the latter
records scheduled, send-start, and send-completion times. Trace serialization
and file writes run on a worker thread; each level awaits trace flush, receipts,
and the WebSocket close path before starting the next level. Failed and rejected
attempts retain their own artifacts and count toward attempted concurrency.

Both `nemotron-voicechat-pr2188` and `minicpmo-native-pr2377` use this runner.
Transport packets stay at 80 ms for both; native units are respectively 80 ms
and 1000 ms. No profile flags are needed beyond `continuous_output`.

The common table emphasizes model, concurrency, admitted/attempted,
success/admitted, unit lag p95, response TTFA p95 and response gap-excess p99.
All timing distributions include p50/p75/p95/p99/max with linear interpolation.
Missing measurements have no samples and display `-`; they never become zero.

Unit lag uses the client receipt timestamp of `sglang.unit.done unit_<k>` minus
`session_start + (k + 1) * native_unit_ms / 1000`. It includes transport to the
client. The event is emitted after unit output passes through the runtime/output
buffer, so it measures **unit fully emitted relative to its native-unit deadline**,
not pure GPU compute latency. Audio packet indices do not define this metric.
Only full native units within the unpadded input duration contribute. Terminal
partial/padded EOS units are retained as `excluded_terminal_units`, since their
nominal full-unit deadline would make an early EOS flush appear artificially early.

Response TTFA is first `response.output_audio.delta` receipt minus
`response.created` receipt, grouped by `response_id`. Text before audio contributes
to this interval. Responses without audio have no TTFA sample. Within each
response, gap excess is the positive arrival gap minus the preceding packet's
decoded PCM duration. Drift is arrival minus first arrival minus all earlier PCM
durations; maximum drift and required buffer (positive maximum drift) are reported
per response and aggregated across responses. Gaps between response IDs never
contribute. Raw gaps are unsuitable for comparing MiniCPM-o's one-second bursts
with VoiceChat output.

Late-send rate is the fraction of sent input frames starting more than 20 ms
after their deadline. The threshold is a load-generator diagnostic, not a
server SLO. A 10 ms event-loop ticker records p99 loop lag; runs exceeding
`--loop-lag-limit-ms` (20 ms by default) are marked `client_timing_valid=false`.
Additionally, every admitted session must have send receipts and send-lateness
p99 at most 20 ms. Loop lag p99 and send lateness p95/p99 are reported separately;
a run without admitted sessions is not timing-valid. These checks measure the
benchmark process, not the remote server, and prevent late input from being
silently interpreted as a valid server-capacity measurement.

The following session-wide metrics apply only to `continuous_output=True`
(VoiceChat): session TTFA, output gap/excess, coverage, continuous/final drift,
required playout buffer, underrun count/duration/ratio. MiniCPM-o returns `None`
for all of them, even for failed or rejected sessions; aggregates remain empty
or N/A. Listening and turn-taking time are not session serving latency.

For consecutive output packets, gap excess is the positive part of the receive
gap minus the preceding packet's decoded PCM duration. Output drift at packet
i is its receive time minus the first audio receive time and the duration of
all earlier packets. The first drift is zero; negative values mean audio arrived
ahead of that ideal schedule. Final drift is the last packet's drift, not an
input-to-output causal latency. These quantities use actual decoded PCM lengths,
not an assumed model-frame size. No deadline-miss SLO is defined in v0.

The largest positive drift is reported per session as the required playout
buffer for avoiding gaps between received packets. Its p50 and p95 are reported
across sessions with output audio; `n` shows how many contributed.

When output audio carries `sglang.media_time.t_start_ms`, the first packet for
each matching input media time also contributes a receive-minus-scheduled-send
and a receive-minus-actual-send lag. Repeated chunks for one media time are
counted once. These are wire-observed alignment offsets, not proof that an
individual input frame caused that output packet or pure server latency.
Absent or unmatched metadata produces no lag sample and remains visible in `n`.

Playback underrun is a simple client simulation. Playback starts 80 ms after
the first audio receipt by default (`--startup-reserve-ms` changes this), using
all audio received during that reserve. It consumes PCM time until the common
input deadline plus the reserve. Each interval that exhausts the buffer counts
as an underrun. A session without audio, or whose first audio arrives after the
input window, is assigned one full-input-duration underrun. Underrun ratio is
total underrun duration divided by the fixed input observation duration; the
aggregate ratio includes admitted continuous-output sessions, including failures,
and excludes rejected attempts and non-continuous profiles. This does
not model network jitter, device buffering, or the server's internal stages.
These observations alone do not establish a sustainable concurrency threshold.

One warmup session runs before measured levels and is excluded from the measured
summary. `--repeats` runs every level repeatedly, with separate traces and
per-run summaries. Percentile tables display sample counts. The top-level JSON
also combines all repeats at each concurrency in `levels`, so a C1 p95 based on
three repeats has `n=3`.

Optional `--gpu-index` records local GPU utilization through ResourceMonitor;
without it, resource metrics are unavailable. The top-level summary records local
benchmark provenance and the normalized input SHA-256. Supply `--model-id`,
`--model-revision`, `--server-sha`, and `--server-config config.json` to record
remote server details. The local repository SHA is the benchmark client SHA,
not the remote server SHA. Pass repeatable `--gpu-process-pid` values for
process-level GPU memory and CPU attribution; these must be host PIDs visible
to NVML.

An exhausted HTTP 503 handshake is `status=rejected`, with `admitted=false`,
not an admitted failure. Three short 250 ms retries accommodate closing-session
teardown races; the final denial is explicit in the trace. Summaries report
attempted, admitted, rejected and successful admitted sessions. Lifecycle checks
for admitted sessions still require input drain, session close and no protocol
error; VoiceChat also requires audio. Silence is valid for MiniCPM-o.

For VoiceChat #2188 qualify C1, and treat C2/C4 as admission observations when
the server limits connections. For MiniCPM-o #2377 use the default
`examples/full_duplex/minicpmo.yaml` with `max_sessions=2` and run C1/C2/C4.
C1/C2 measure serving performance; C4 measures admission and should report two
admitted, two rejected, and success 2/2 admitted when both admitted sessions pass.
Record the actual server SHA and config using provenance arguments. Do not
increase admission capacity for qualification.
