# Architecture Overview

```
Browser (UI)
  ├─ AgentClient (Socket.IO) ───────────────→ Agent Server
  │        ▲                                     │
  │        └──────── RunStarted/Chunk/Done ──────┘
  │
  ├─ Mic → AudioWorklet → Resampler → (binary) → STT Server
  │                                           ▲
  └───────────────────────────── JoinSTT ←────┘ via Agent Server (multiplex)
```

**Flows**

- **Text chat** → Browser calls `Chat` with `{text, agent, thread_id}` → Agent server streams `RunStarted` / `ChatChunk` / `ChatDone` back.
- **Voice chat** → Browser sends **binary PCM16** to STT. Agent server subscribes via `JoinSTT` and receives **transcriptions**, then triggers a normal chat run for the same client.
- **Memory** → Use a persistent `threadId` (UUID) to enable thread-scoped memory strategies on the server.

**Client runtime**

- `agentClient.js` – minimal SDK: connect, runText, cancel, sttSubscribe, sttUnsubscribe, onStream.
- `audioResampler.js` – tiny resampler class converting Float32@48k → PCM16@16k.
- `recorder.worklet.js` – AudioWorklet that posts raw Float32 mic frames to JS.
