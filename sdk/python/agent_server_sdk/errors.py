"""Exception types raised by the agent_server SDK."""

from __future__ import annotations

from typing import Optional

__all__ = [
    "AgentServerError",
    "AuthError",
    "NotFoundError",
    "ModelNotActiveError",
    "TransportError",
]


class AgentServerError(Exception):
    """Base class for all SDK errors."""

    def __init__(self, message: str, *, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


class AuthError(AgentServerError):
    """401 — missing or invalid Bearer API key."""


class NotFoundError(AgentServerError):
    """404 — unknown agent name or model id (``model_not_found``)."""


class ModelNotActiveError(NotFoundError):
    """A specific, common 404: an inactive (undeclared) chat model was

    requested. Switch the active model first, then retry. See
    :meth:`AgentServerClient.set_active_model`.
    """


class TransportError(AgentServerError):
    """Network / connection failure talking to agent_server."""
