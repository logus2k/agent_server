"""
DEPRECATED in forwarding mode (the current default).
Kept as reference / for the in-process LlamaCppEngine rollback path
(Dockerfile.v2.fat builds with llama-cpp-python, where this handler
would be active again). Vision in forwarding mode is delivered by
llama-server's own chat template handling — this Python subclass is
dead code in the slim image.

Will be removed in the cleanup PR after the in-process rollback path
is retired (see migration plan, post-Phase 5).

----------------------------------------------------------------------
Gemma4VisionChatHandler — drives Gemma 4 vision through llama-cpp-python's
mtmd C-API, with a chat template that uses Gemma's native turn markers
instead of Qwen's `<|im_start|>` / `<|im_end|>` format.

Empirical finding 2026-05-02: `Qwen25VLChatHandler` already drives Gemma 4
vision correctly because the actual image-token embedding happens at the
mtmd C-API layer, and Gemma is robust enough to interpret Qwen's chat
markers as benign text. But for production-quality multi-turn chat with
heavy system prompts, the format mismatch is worth removing — hence this
subclass that overrides only the chat template, leaving all mtmd
plumbing inherited.

Design:
  - Subclass `Qwen25VLChatHandler` (which itself subclasses
    `Llava15ChatHandler`). The parent does state clearing and defers
    to the grandparent's __call__ for the actual generation pipeline.
  - Override `CHAT_FORMAT` (Jinja template) with Gemma's native turn
    markers: `<|turn>{role}\n...<turn|>\n`, role mapping
    assistant→model, system role supported via `<|turn>system\n...`.
  - For `image_url` content blocks, render the URL as raw text just
    like Qwen does. `Llava15ChatHandler.__call__` scans the rendered
    prompt for `data:image/...;base64,...` URIs, invokes mtmd to
    encode each image, and substitutes the resulting embedding tokens
    in place. mtmd inserts model-specific image marker tokens at
    substitution time, so we don't emit `<|image|>` ourselves in the
    template.
  - Override `DEFAULT_SYSTEM_MESSAGE` to empty string. Gemma 4 has no
    canonical default persona; system text is owned by the calling
    application.

Limitations of this first cut:
  - No tool-call / tool-response handling (the native Gemma 4 template
    has rich `<|tool_call>` / `<|tool_response>` support; we strip it
    here for simplicity). Tool calls in v2 are still expected to flow
    through agent_server's existing parser at a higher layer, not via
    template rendering. Add later if a use case emerges.
  - No `<|think|>` / `strip_thinking()` handling. Gemma's reasoning
    output goes to `reasoning_content` in OpenAI-compat responses; the
    handler returns it untouched.
"""
from __future__ import annotations

from llama_cpp.llama_chat_format import Qwen25VLChatHandler


class Gemma4VisionChatHandler(Qwen25VLChatHandler):
    """Vision-capable chat handler tuned for Gemma 4."""

    DEFAULT_SYSTEM_MESSAGE = ""

    CHAT_FORMAT = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
        "<|turn>{{ role }}\n"
        "{% if message['content'] is string %}"
        "{{ message['content'] | trim }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'text' %}"
        "{{ content['text'] | trim }}"
        "{% elif content['type'] == 'image_url' %}"
        "{% if content.image_url is string %}"
        "{{ content.image_url }}"
        "{% else %}"
        "{{ content.image_url.url }}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "<turn|>\n"
        "{% endfor %}"
        "<|turn>model\n"
    )
