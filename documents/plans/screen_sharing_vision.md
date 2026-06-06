# Plan — Share the desktop screen with a vision model

**Goal:** let a user "share their screen" with a model that understands it — ask questions about
what's on screen (errors, UI, documents, charts), with answers in the existing chat UX.

**Status:** plan / not built. Discussed 2026-06-06.

---

## 1. How "screen sharing to an LLM" actually works

The model does **not** watch a live video feed. Sharing a screen = a **screenshot pipeline**: the
client captures frames and sends them as **images** in ordinary chat requests. A VLM is
*image-in, text-out* — exactly like the existing vision-capable `gemma-4` (GGUF + `mmproj`).

```
[capture screen] -> [grab a still frame] -> [encode jpeg/png] -> [POST as image in a chat msg]
        ^                                                                    |
        └──────────────── repeat on a cadence for "continuous" ─────────────┘   (sampled stills, NOT video)
```

The vision tower (`mmproj`) encodes each image into tokens prepended to the LLM context; the model
reads "image + question" and replies in text. "Continuous" = re-send a frame periodically / on
change; the model sees a *sequence of stills*.

## 2. Capture (client side) — the only genuinely new code

Browser path (fits the cv widget / avatar stack; mirrors the existing `getUserMedia` mic flow):

```js
// 1. user grants a screen/window/tab via the Screen Capture API
const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 1 } });
// 2. draw the current frame to a canvas and export a still
//    (downscale to a sane width, e.g. <=1536px, to bound tokens; crop a region for dense text)
const dataUrl = canvas.toDataURL('image/jpeg', 0.8);  // "data:image/jpeg;base64,..."
// 3. send as an OpenAI-style image message (see §3)
```

Capture cadence options:
- **On-demand (recommended first):** one frame when the user asks a question. Cheapest, clearest.
- **Periodic:** a frame every N seconds while sharing (e.g. 1 fps), optionally only when the frame
  changed (perceptual diff) to save tokens.
- Native/desktop alternative: a small helper using an OS screenshot API instead of the browser.

## 3. Sending the frame — already supported by agent_server

agent_server is OpenAI-compatible and the active model can be vision-capable, so **no new server
contract** is needed — a multimodal message:

```jsonc
{
  "model": "<vision model or agent>",
  "messages": [{ "role": "user", "content": [
    { "type": "text", "text": "What's on my screen? Read any error text." },
    { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
  ]}]
}
```

llama-vision routes to the active model; its `mmproj` encodes the image. (gemma-4 today carries
`vision:true` + `projector:/agent_server_models/mmproj-F16.gguf`.)

## 4. Which model — and the VRAM/single-resident catch

We enforce a **single resident chat model** (only the active model is declared in the router
preset — see [[agent-server-model-switch]] / `active_model_switching_sdk.md`). That constrains the
choice:

| Option | Quality on screens | Fit with single-resident rule |
|--------|--------------------|-------------------------------|
| **gemma-4** (already active, vision-capable) | basic; weaker on dense small-text OCR | **Best fit** — works *today* whenever gemma-4 is the active model; zero new infra |
| **Qwen2.5-VL** (3B/7B) — best on-prem for screens/UI/OCR + grounding | strong | needs ITS OWN `mmproj` (gemma's is incompatible); and it must be **resident** (see below) |
| **Claude** (cloud, already in noted) | strongest overall + computer-use | no local VRAM; routed via noted, not agent_server |

**The catch:** a dedicated local vision model (Qwen2.5-VL) is *another chat model*. Under the
single-resident invariant it can only be used while it is the **active** model — so either:
- **(A) Reuse the active model when it's vision-capable** (gemma-4). Screen-share is available only
  while gemma-4 is active; if a non-vision model (qwen3.5/granite/…) is active, the feature is
  greyed out or triggers a switch. Simplest, no new infra. **Recommended for v1.**
- **(B) Switch-to-vision on demand** — calling screen-share switches the active model to a vision
  one (the ~38-45s restart from the switch API). Heavy/disruptive; not for interactive use.
- **(C) Dedicated always-on vision instance** — run a *separate* llama-server (own container/port)
  hosting the vision model, outside the single-resident router, always resident. Costs steady VRAM
  (a 3B-VL Q4/Q8 + KV) but gives instant, model-independent screen understanding. The right answer
  if screen-share becomes a first-class feature. (This is the same "split a model out of the shared
  router" idea noted for the no-restart switch.)

**Recommendation:** v1 = **Option A with gemma-4** (ship the capture UX against the model we already
run). If quality on dense screens is insufficient, do **Option C** with **Qwen2.5-VL-3B** (source
its mmproj, dedicated instance) — don't put a 2nd vision model into the shared router.

## 5. Practical realities
- **Resolution vs small text:** full-screen shots are high-res; VLMs downscale/tile and lose small
  text. Downscale to a bounded width and/or let the user crop a region; prefer screen-tuned models
  (Qwen2.5-VL dynamic tiling) for dense UIs. Prefer higher quant (Q8/Q6 of a 3B) over Q4-of-7B for
  OCR if VRAM forces the choice.
- **Token cost/latency:** each image is hundreds–thousands of tokens. Stream frames sparingly
  (on-demand or low fps), not continuously.
- **Privacy:** screen contents are sent to the model. On-prem (gemma/qwen-vl) keeps it local;
  Claude is cloud — gate accordingly and make "what gets captured" explicit to the user.
- **Understanding vs acting:** this plan is *understand the screen*. To also click/type (a screen
  **agent**) needs element **grounding** (coords) + an automation layer — Qwen2.5-VL/Claude support
  the grounding; out of scope for v1.

## 6. Phases
1. **v1 — on-demand screenshot Q&A (Option A / gemma-4):** add a "Share screen" control to the
   widget (mirror the mic button): `getDisplayMedia` → capture one frame on send → downscale →
   `image_url` in the chat request to the active (vision) model. Gate/disable when the active model
   isn't vision-capable. Deliverable: "ask about my screen" works end-to-end, fully on-prem.
2. **v2 — continuous/periodic frames:** sample at low fps or on-change while sharing; keep a short
   rolling context; manage token budget.
3. **v3 (optional) — dedicated vision instance (Option C) + Qwen2.5-VL** for better OCR/UI, and/or
   **grounding/computer-use** if interaction (not just understanding) is wanted.

## 7. Open decisions (for the user)
- v1 model: ride on **gemma-4** (Option A, no new infra) — confirm.
- Where the UI lives: the **cv widget**, the **noted** assistant, or a standalone tool?
- On-prem only, or allow **Claude** for highest quality (privacy trade-off)?
- Eventually need **acting** (computer-use), or only **understanding**?

## 8. Tooling already in place
- agent_server vision path is proven (gemma-4 + mmproj via llama-vision; OpenAI-compatible image
  messages). The new work is **client-side capture** + a model/VRAM decision (§4).
- For evaluating Qwen2.5-VL later: reuse `/tmp/gguf_meta.py` + the isolated-load harness pattern
  (`/tmp/ministral_check.sh`), plus sourcing its `mmproj`.
