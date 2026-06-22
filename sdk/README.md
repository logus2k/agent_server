# agent_server SDKs

Drop-in, installable SDKs so a **new client integrates with agent_server in
minutes, not by copy-pasting and getting the streaming/parsing wrong**. Two
packages, by where the client runs:

| SDK | For | Path | Covers |
|---|---|---|---|
| **Python** (`agent_server_sdk`) | server-side: backends, scripts, workers | [`python/`](python/) | REST chat (stream + non-stream), channel parsing, **thinking on/off**, discovery, active-model switching, optional async Socket.IO interactive client (+ mediated TTS/STT) |
| **JavaScript** (`agent-server-client`) | client-side: browsers, widgets | [`javascript/`](javascript/) | REST + Socket.IO chat, the tested `<think>`/`<voice>`/answer streaming parser + TTS sanitiser, **thinking on/off + show/hide**, optional **TTS / STT / avatar** integrations |

Both share one guarantee about agent_server's wire format — reasoning and the
spoken summary arrive **inside** the content stream as
`<think>…</think><voice>…</voice>answer` — and both ship the *same tested
parser* for it (Python: 10 tests; JS: 69 assertions; identical semantics) so no
client re-implements it. That parser exists because a hand-rolled version once
read a whole answer aloud and spoke a literal `</voice>` tag.

## Two things every integrator needs, handled

**Thinking mode** is two independent controls:
* **generate** reasoning or not — a per-request toggle (`thinking=true/false`),
  mapped to the correct `chat_template_kwargs` per model family;
* **show** reasoning or not — a UI choice; both SDKs always *separate* the
  reasoning channel so you can collapse/hide it without re-requesting.

**Optional services fail soft.** TTS, STT and avatar are independent add-ons.
If a service isn't running, that one capability degrades and **chat (and every
other available capability) keeps working**. The JS `AgentClient` exposes a
`capabilities` map; `enableVoice()/enableMic()/enableAvatar()` return
`true`/`false`.

## Quick start

Python (server-side):
```bash
cd python && pip install -e .
python examples/hello_agent.py "Who is António?"
```

JavaScript (browser): serve `sdk/` and open `javascript/examples/index.html`
(chat, thinking on/off, show/hide reasoning, and TTS/STT/avatar toggles).

## Endpoints these wrap

* `POST /v1/chat/completions` — `model` = agent name **or** model id; stream/non-stream.
* `GET /v1/models` — discovery (active model + family + agent names).
* `GET /v1/agents/{name}` — a preset.
* `POST /admin/api/active-model` — switch the resident model.
* Socket.IO `Chat` → `RunStarted`/`ChatChunk`/`ChatDone`; `JoinTTS`/`JoinSTT`.
* Voice services (browser): tts_server `/tts/socket.io`, stt_server
  `/stt/socket.io`, avatar_server `/avatar/socket.io`.

See [`../documents/how_to.md`](../documents/how_to.md) for agent creation and
the raw REST/Socket.IO reference.
