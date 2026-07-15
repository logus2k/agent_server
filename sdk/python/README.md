# agent_server_sdk (Python) — server-side SDK

The error-proof way for a **backend / script / worker** to talk to
agent_server. It owns the parts hand-rolled clients get wrong: channel
parsing (`<think>` / `<voice>` / answer), the thinking toggle, retries,
discovery, and active-model switching.

> Client-side / browser integration (interactive Socket.IO, TTS, STT, avatar)
> lives in the sibling **JavaScript SDK** at `../javascript/`.

## Install

```bash
pip install -e .                 # core (httpx only)
pip install -e '.[interactive]'  # + async Socket.IO interactive client
```

## Quick start

```python
from agent_server_sdk import AgentServerClient

with AgentServerClient("http://localhost:7701") as ac:   # :7701 = agent_server
    r = ac.chat("cv_assistant", "Who is António?")
    print(r.answer)        # user-visible answer — NO <think>/<voice>
    print(r.thinking)      # reasoning channel (empty if off/absent)
    print(r.voice)         # spoken-summary channel
```

`model` accepts an **agent name** (`cv_assistant`, applies the preset's
prompt+sampling) **or** a **model id** (`gemma-4-e2b`). Discover both with
`ac.list_models()`.

## Streaming (answer vs reasoning vs voice)

```python
for ev in ac.chat_stream("cv_assistant", "Tell me about Vision-Box"):
    if ev.kind == "answer":
        print(ev.text, end="", flush=True)     # render live
    elif ev.kind == "thinking":
        log_reasoning(ev.text)                  # keep separate / hide
    elif ev.kind == "voice" and ev.final:
        speak(sanitize_for_tts(ev.text))        # the only text for TTS
```

The parser is the same code the tests fuzz: a voice block is **buffered until
`</voice>`** and emitted once (so a partial/forgotten voice block is never
handed to TTS), and `sanitize_for_tts()` strips even malformed/truncated tags.

## Thinking mode (on/off)

Thinking is the model's reasoning channel. Two independent things:

* **Generate it or not** — per request, via this SDK:

  ```python
  ac.chat("cv_assistant", "2+2?", thinking=False)   # no reasoning produced
  ```

  The right `chat_template_kwargs` is chosen for you. Pass `thinking_family=`
  (from `ModelInfo.family`) so granite/ministral are handled correctly:

  ```python
  active = ac.active_model()
  ac.chat(active.id, "…", thinking=True, thinking_family=active.family)
  ```

* **Show it or hide it** — a UI choice. The SDK always *separates* reasoning
  (`r.thinking` / `thinking` events); your client decides whether to render it
  (e.g. a collapsible "Show reasoning" toggle). The browser SDK's example wires
  both toggles up.

## Discovery & active-model switching

```python
ac.list_models()                       # chat models (+active/family) and agents
ac.active_model()                      # the resident chat model
ac.get_agent("cv_assistant")       # a preset (prompt/sampling/memory)

ac.set_active_model("gemma-4")         # switch; blocks ~30–45s until back up
ac.wait_until_ready(expect_active="gemma-4")
```

Requesting an **inactive** local model raises `ModelNotActiveError` — switch
first.

## Interactive (optional, push streaming)

```python
from agent_server_sdk.interactive import AsyncInteractiveClient

ac = AsyncInteractiveClient("http://localhost:7701", client_id="my-bot")
await ac.connect()
await ac.enable_tts(voice="af_heart")          # mediated TTS — fails soft if down
async for ev in ac.chat("cv_assistant", "Hello"):
    if ev.kind == "answer":
        print(ev.text, end="")
await ac.close()
```

`enable_tts` / `join_stt` **fail soft**: if the TTS/STT services aren't
available they no-op and plain chat keeps working.

## Examples

* `examples/hello_agent.py` — smallest correct integration.
* `examples/streaming_and_thinking.py` — streaming + thinking on/off + TTS-safe
  voice extraction.

## Test

```bash
pytest -q                              # parser + sanitiser (10 tests)
```
