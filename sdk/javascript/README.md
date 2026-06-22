# agent-server-client (JavaScript) — client-side SDK

The error-proof way for a **browser / widget** to talk to agent_server. Owns
the streaming `<think>`/`<voice>`/answer parser (the thing the CV widget once
hand-rolled into a voice bug), the thinking toggle, and **optional, fail-soft**
TTS / STT / avatar integrations.

> Server-side / backend integration (REST chat, discovery, model switching)
> lives in the sibling **Python SDK** at `../python/`.

## Design rule: services fail soft

Chat (REST or Socket.IO) is the always-available baseline. TTS, STT and avatar
are **independent and optional** — if a service is down, that one feature
degrades and **nothing else breaks**. `enableVoice()/enableMic()/enableAvatar()`
each return `true`/`false`; `client.capabilities` reflects what's live.

## Install / load

ES modules, zero bundled deps. `socket.io-client`'s `io` is **injected** (only
needed for TTS/STT/avatar — chat works without it):

```html
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script type="module">
  import { AgentClient, newClientId } from './agent-server-client/src/index.js';
  const client = new AgentClient({ io: window.io, baseUrl: '', clientId: newClientId() });
</script>
```

## Chat + channels (REST streaming)

```js
client.ask('cv_assistant_e2b', 'Tell me about Vision-Box',
  { thinking: true },                       // thinking ON/OFF per request
  {
    onAnswer:   (t) => render(t),            // user-visible answer
    onThinking: (t) => reasoningEl.append(t),// reasoning (show or hide — your call)
    onVoice:    (t) => {/* already spoken if TTS enabled */},
    onDone:     () => {},
  });
```

`onVoice` only fires with the **complete, closed** spoken summary, already safe
for TTS (`sanitizeForTTS` strips even malformed tags). An unclosed `<voice>`
becomes answer text and is never spoken.

## Thinking mode — two separate things

* **Generate it or not** — `{ thinking: true|false }` on `ask()`/`chatStream()`.
  The correct `chat_template_kwargs` per model family is chosen for you.
* **Show it or hide it** — pure UI. The parser always separates reasoning
  (`onThinking`); render it in a collapsible panel and toggle visibility without
  re-requesting. The demo wires both: a "Thinking on" checkbox (generation) and
  a "Show reasoning" checkbox (display).

## Optional voice / avatar

```js
await client.enableVoice();            // TTS: spoken replies (false if down)
await client.enableMic(onTranscript);  // STT: mic -> text (false if no mic/svc)
await client.enableAvatar(videoEl);    // avatar video; routes TTS audio to it
client.bargeIn();                      // stop speaking immediately
```

Service paths default to `/tts/socket.io`, `/stt/socket.io`,
`/avatar/socket.io` (same origin), overridable via the `AgentClient` options.

## Interactive Socket.IO chat (alternative to REST)

```js
import { AgentSocketClient } from './agent-server-client/src/socket.js';
const sock = new AgentSocketClient({ io: window.io, url: '' });
await sock.connect();
sock.chat('cv_assistant_e2b', 'Hello', {}, { onAnswer: render, onVoice: speak });
```

(Over Socket.IO, thinking on/off follows the agent **preset**; use the REST
client for per-request control.)

## Low-level pieces

`StreamParser`, `splitResponse`, `sanitizeForTTS` (from `./src/parser.js`),
`AgentRestClient`, `TtsClient`, `SttClient`, `AvatarClient` are all exported
individually if you don't want the `AgentClient` facade.

## Example & test

* `examples/index.html` — full demo: chat, thinking on/off, show/hide reasoning,
  TTS/STT/avatar toggles. Serve the `sdk/` folder and open it.
* `npm test` (or `node test/parser.test.mjs`) — 69 parser/sanitiser assertions,
  identical semantics to the Python SDK.
