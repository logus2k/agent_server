# Agent SDK (JS) — Text + Voice (STT) Quickstart

A tiny JavaScript SDK for streaming LLM chat with optional **server‑side STT multiplexing**. Use it for:
- **Text chat** with token streaming
- **Voice chat**: browser streams **binary PCM16** to your STT; **agent_server** listens to transcripts and triggers runs
- **Agents** (e.g., `topic`, `router`) and **threaded memory** via `threadId`
- **Cancel/interrupt** and robust auto‑reconnect

---

## Architecture (at a glance)

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

- **Text chat**: `runText(text, { agent, threadId })`
- **Voice**: Browser only uploads audio; agent_server owns the STT transcript subscription (`sttSubscribe` / `sttUnsubscribe`)

---

## Requirements

- **Browser**: Socket.IO client v4, WebAudio (AudioWorklet), getUserMedia
- **Agent server**: FastAPI + python‑socketio (already in your repo)
- **STT multiplex (server→server)**: install the async extra on the agent_server:
  ```bash
  pip install "python-socketio[asyncio_client]~=5.11"
  ```
- **STT server**: emits `transcription { text, duration, client_id }` and accepts `audio_data { clientId, audioData }`

---

## Quickstart — Text

```html
<script src="./socket.io.min.js"></script>
<script type="module">
  import { AgentClient } from "./agentClient.js";

  const client = new AgentClient({ url: location.origin });
  await client.connect();
  client.onStream({ onText: t => console.log(t) });

  const threadId = crypto.randomUUID();
  await client.runText("Hello!", { agent: "topic", threadId });
</script>
```

## Quickstart — Voice + server‑side STT

```html
<script src="./socket.io.min.js"></script>
<script type="module">
  import { AgentClient } from "./agentClient.js";
  import { AudioResampler } from "./audioResampler.js";

  const client = new AgentClient({ url: location.origin });
  await client.connect();
  client.onStream({ onText: t => console.log(t) });

  const threadId = crypto.randomUUID();
  await client.sttSubscribe({ sttUrl: "http://localhost:2700", clientId: "demo", agent: "topic", threadId });

  const ctx = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
  await ctx.audioWorklet.addModule("./recorder.worklet.js");
  const src = ctx.createMediaStreamSource(await navigator.mediaDevices.getUserMedia({ audio: true }));
  const worklet = new AudioWorkletNode(ctx, "recorder-worklet");
  const resampler = new AudioResampler(ctx.sampleRate, 16000);

  const stt = io("http://localhost:2700", { transports: ["websocket"] });
  worklet.port.onmessage = e => {
    const pcm16 = resampler.pushFloat32(e.data);
    if (pcm16) stt.emit("audio_data", { clientId: "demo", audioData: pcm16.buffer });
  };
  src.connect(worklet);
</script>
```

> **Note:** For text runs, `Chat` expects `thread_id` (snake case). For STT subscription, `JoinSTT` expects `threadId` (camel case). The SDK maps these for you.

---

## Examples (ready to run)

- Text only: [examples/text-only.html](examples/text-only.html)
- Voice + STT: [examples/voice+stt.html](examples/voice+stt.html)
- Mixed chat (text + voice UI): [examples/mixed-chat.html](examples/mixed-chat.html)

Serve over a local web server (e.g., `python -m http.server`) so AudioWorklet modules can load.

---

## Documentation

- [01 — Architecture](docs/01-architecture.md)
- [02 — Quickstart (Text)](docs/02-quickstart-text.md)
- [03 — Quickstart (Voice + STT)](docs/03-quickstart-voice.md)
- [04 — Concepts](docs/04-concepts.md)
- [05 — JS API](docs/05-js-api.md)
- [06 — Recipes](docs/06-recipes.md)
- [07 — Events & Wire Protocol](docs/07-events-wire-protocol.md)
- [08 — Troubleshooting](docs/08-troubleshooting.md)
- [09 — Performance Tuning](docs/09-performance-tuning.md)
- [10 — FAQ](docs/10-faq.md)

---

## Common pitfalls

- **THREAD_REQUIRED**: Generate a `threadId` and pass it for `runText` and `sttSubscribe`.
- **STT_CONNECT**: Ensure `aiohttp` is installed on agent_server (see Requirements) and the STT URL is reachable from the server’s network context.
- **Agent not changing for STT**: Re‑subscribe with `sttUnsubscribe()` → `sttSubscribe()` after choosing a new agent.

---

## License

Add your license here (e.g., MIT).

