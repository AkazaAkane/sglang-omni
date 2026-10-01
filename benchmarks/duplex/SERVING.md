# Realtime serving benchmark

Run against a server exposing `/v1/realtime`:

```bash
python -m benchmarks.duplex.serving \
  --url ws://127.0.0.1:8097/v1/realtime \
  --audio input.wav --profile nemotron --concurrencies 1,2,4,8
```

The input is normalized to mono 16 kHz PCM16 using the existing duplex audio
loader. Each concurrency level starts only after its sessions finish negotiation.
All configured sessions use one monotonic start deadline and fixed 80 ms input
deadlines. A late sender keeps at least 80 ms between later append starts. Each
session has its own JSONL wire trace and input-send-receipts.json; the latter
records scheduled, send-start, and send-completion times. Failed attempts retain
their own traces and count toward the requested concurrency.

The summary reports client send lateness (send start minus deadline), TTFA (first
audio receipt minus common start), gaps between received audio events, and PCM
output duration divided by input duration. Timing distributions include p50,
p75, p95, p99 and max, with linear interpolation. Missing TTFA and output timing
have no samples; missing output has zero coverage and remains in the success
denominator.

Late-send rate is the fraction of sent input frames starting more than 20 ms
after their deadline. The threshold is a load-generator diagnostic, not a
server SLO. Strict 80 ms minimum spacing means small client scheduling delays
can accumulate into send lateness over a long run.

For consecutive output packets, gap excess is the positive part of the receive
gap minus the preceding packet's decoded PCM duration. Output drift at packet
i is its receive time minus the first audio receive time and the duration of
all earlier packets. The first drift is zero; negative values mean audio arrived
ahead of that ideal schedule. Final drift is the last packet's drift, not an
input-to-output causal latency. These quantities use actual decoded PCM lengths,
not an assumed model-frame size. No deadline-miss SLO is defined in v0.

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
