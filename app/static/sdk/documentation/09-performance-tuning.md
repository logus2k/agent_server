# Performance Tuning

- **Latency**: `new AudioContext({ sampleRate: 48000, latencyHint: "interactive" })`. Batch ~50–100 ms (`BATCH_TARGET = fs / 20` to `fs / 10`).
- **Bandwidth**: Always downsample to **mono 16 kHz PCM16** before sending to STT.
- **Resampler**: Use `AudioResampler` (linear) for lowest CPU; anti-alias FIR can be swapped in later if desired.
- **UI**: Prefer `onStream` global handlers so server-initiated runs (from STT) render without extra per-run wiring.
