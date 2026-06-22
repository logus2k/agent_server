"""Typed results returned by the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ChatResult", "ModelInfo", "AgentPreset", "thinking_kwargs"]


@dataclass
class ChatResult:
    """A complete (non-streamed) chat response, already split into channels.

    ``answer`` is the user-visible text (no ``<think>``/``<voice>``).
    ``thinking`` is the reasoning channel (empty if thinking was off/absent).
    ``voice`` is the spoken-summary channel (empty if absent). ``raw`` is the
    untouched content string. ``usage`` carries token counts when present.
    """

    answer: str
    thinking: str = ""
    voice: str = ""
    raw: str = ""
    model: str = ""
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # convenience
        return self.answer


@dataclass
class ModelInfo:
    """One entry from ``GET /v1/models``."""

    id: str
    active: bool = False
    kind: str = ""          #: "model" or "agent"
    family: str = ""        #: e.g. "gemma", "qwen", "granite" (for chat models)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPreset:
    """One agent preset from ``GET /v1/agents/{name}``."""

    name: str
    system_prompt: str = ""
    params_override: Dict[str, Any] = field(default_factory=dict)
    memory_policy: str = "none"
    raw: Dict[str, Any] = field(default_factory=dict)


# Families whose chat template uses a non-default kwarg name to toggle
# reasoning. Everything else uses ``enable_thinking``; ``ministral`` ignores
# the kwarg entirely (its [THINK] block lives in the template's default system
# message). Verified set — see documents/active_model_switching_sdk.md §7b.
_THINKING_KWARG_BY_FAMILY = {"granite": "thinking"}
_THINKING_UNSUPPORTED = {"ministral"}


def thinking_kwargs(value: Optional[bool], family: str = "") -> Dict[str, Any]:
    """Return the ``chat_template_kwargs`` dict that toggles reasoning for a
    model ``family`` (e.g. from :class:`ModelInfo.family`).

    * ``value is None`` -> ``{}`` (leave the model's default).
    * gemma / qwen / smollm / nemotron -> ``{"enable_thinking": value}``.
    * granite -> ``{"thinking": value}``.
    * ministral -> ``{}`` (toggle unsupported; caller should not rely on it).
    """
    if value is None:
        return {}
    fam = (family or "").strip().lower()
    if fam in _THINKING_UNSUPPORTED:
        return {}
    key = _THINKING_KWARG_BY_FAMILY.get(fam, "enable_thinking")
    return {key: bool(value)}
