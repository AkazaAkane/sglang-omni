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
output duration divided by input duration. Percentiles use linear interpolation
over observed values. Missing TTFA and gaps have no samples; missing output has
zero coverage and remains in the success denominator.

Playback underrun is a simple client simulation. Playback starts 80 ms after
the first audio receipt by default (`--startup-reserve-ms` changes this), using
all audio received during that reserve. It consumes PCM time until the common
input deadline plus the reserve. Each interval that exhausts the buffer counts
as an underrun. A session without audio, or whose first audio arrives after the
input window, is assigned one full-input-duration underrun. This does not model
network jitter, device buffering, or the server's internal stages. These
observations alone do not establish a sustainable concurrency threshold.
