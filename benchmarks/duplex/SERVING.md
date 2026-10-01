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
records scheduled, send-start, and send-completion times. Failed attempts retain
their own traces and count toward the requested concurrency.

The summary reports client send lateness (send start minus deadline), TTFA (first
audio receipt minus that session's start), gaps between received audio events, and PCM
output duration divided by input duration. Timing distributions include p50,
p75, p95, p99 and max, with linear interpolation. Missing TTFA and output timing
have no samples; missing output has zero coverage and remains in the success
denominator.

Late-send rate is the fraction of sent input frames starting more than 20 ms
after their deadline. The threshold is a load-generator diagnostic, not a
server SLO. A 10 ms event-loop ticker records p99 loop lag; runs exceeding
`--loop-lag-limit-ms` (20 ms by default) are marked `client_timing_valid=false`.
This measures the benchmark process, not the remote server.

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
aggregate ratio includes all attempted sessions, including failures. This does
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

The serving benchmark requires continuous output and rejects other profiles.
The native endpoint documented by #2331 admitted one session at a time and
returned HTTP 503 for additional connections. Confirm multi-session support
before interpreting C>1 as a capacity result; admission failures remain in the
requested-concurrency denominator.
