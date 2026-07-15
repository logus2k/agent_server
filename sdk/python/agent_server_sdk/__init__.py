"""agent_server_sdk — server-side Python SDK for agent_server.

The error-proof way for a backend/script/worker to talk to agent_server:

    from agent_server_sdk import AgentServerClient

    with AgentServerClient("http://agent_server:7701") as ac:
        # one shot, channels already separated
        r = ac.chat("cv_assistant", "Who is António?")
        print(r.answer)          # user-visible answer (no <think>/<voice>)
        print(r.thinking)        # reasoning channel (if any)

        # streaming
        for ev in ac.chat_stream("cv_assistant", "Tell me more"):
            if ev.kind == "answer":
                print(ev.text, end="")

        # thinking on/off (per request)
        ac.chat("cv_assistant", "2+2?", thinking=False)

Optional interactive Socket.IO client (Chat push + mediated TTS/STT) lives in
:mod:`agent_server_sdk.interactive` and needs the ``[interactive]`` extra.
"""

from .client import AgentServerClient
from .errors import (
    AgentServerError,
    AuthError,
    ModelNotActiveError,
    NotFoundError,
    TransportError,
)
from .parser import StreamEvent, StreamParser, sanitize_for_tts, split_response
from .types import AgentPreset, ChatResult, ModelInfo, thinking_kwargs

__version__ = "0.1.0"

__all__ = [
    "AgentServerClient",
    "StreamParser",
    "StreamEvent",
    "split_response",
    "sanitize_for_tts",
    "ChatResult",
    "ModelInfo",
    "AgentPreset",
    "thinking_kwargs",
    "AgentServerError",
    "AuthError",
    "NotFoundError",
    "ModelNotActiveError",
    "TransportError",
]
