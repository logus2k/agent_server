# FAQ

**Can I use only text?** Yes. Do not call `sttSubscribe`; just use `runText`.

**How do I keep memory?** Reuse the same `threadId` across messages.

**Can I change agents mid-session?** Yes for text (`runText`). For STT, re-subscribe with the new agent.

**Why 16 kHz?** Matches most ASR models; reduces bandwidth and latency.

