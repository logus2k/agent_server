"""Synchronous client for agent_server's OpenAI-compatible REST API.

Designed for server-side integrators (FastAPI backends, scripts, workers).
Wraps the four things every client needs and tends to get wrong:

* **chat** — by agent name *or* model id, streaming or not, with retries,
  timeouts and optional Bearer auth.
* **channel parsing** — reasoning / spoken-summary / answer separated for you
  (via :mod:`agent_server_sdk.parser`).
* **thinking toggle** — ``thinking=True/False`` mapped to the correct
  per-family ``chat_template_kwargs``.
* **discovery + active-model switching** — list models/agents, switch the
  resident model, wait until the stack is back up.

Only dependency: ``httpx``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx

from .errors import (
    AgentServerError,
    AuthError,
    ModelNotActiveError,
    NotFoundError,
    TransportError,
)
from .parser import StreamParser, StreamEvent, split_response
from .types import AgentPreset, ChatResult, ModelInfo, thinking_kwargs

__all__ = ["AgentServerClient"]

Message = Dict[str, str]


class AgentServerClient:
    """A thin, robust client for one agent_server instance.

    Args:
        base_url: e.g. ``http://agent_server:7701`` (container network) or
            ``http://localhost:7701`` (from the host).
        api_key: Bearer token, only if the server sets one (most internal
            deployments don't).
        timeout: per-request seconds (read timeout for streams is generous).
        max_retries: transient-failure retries (connection errors / 5xx /
            model-reload races).
    """

    def __init__(
        self,
        base_url: str = "http://agent_server:7701",
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "AgentServerClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- headers / errors --------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        body: Dict[str, Any] = {}
        try:
            body = resp.json()
        except Exception:
            pass
        err = (body.get("error") or body) if isinstance(body, dict) else {}
        msg = (err.get("message") if isinstance(err, dict) else None) or resp.text or "request failed"
        code = err.get("code") if isinstance(err, dict) else None
        if resp.status_code == 401:
            raise AuthError(msg, status=401, code=code)
        if resp.status_code == 404:
            if code == "model_not_found" or "not_found" in str(code):
                raise ModelNotActiveError(msg, status=404, code=code)
            raise NotFoundError(msg, status=404, code=code)
        raise AgentServerError(msg, status=resp.status_code, code=code)

    # -- payload builder ---------------------------------------------------
    @staticmethod
    def _build_payload(
        model: str,
        messages: List[Message],
        *,
        stream: bool,
        thinking: Optional[bool],
        thinking_family: str,
        sampling: Optional[Dict[str, Any]],
        extra_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if sampling:
            payload.update(sampling)
        ctk = dict(payload.get("chat_template_kwargs") or {})
        ctk.update(thinking_kwargs(thinking, thinking_family))
        if ctk:
            payload["chat_template_kwargs"] = ctk
        if extra_body:
            payload.update(extra_body)
        return payload

    # -- non-streaming chat ------------------------------------------------
    def chat(
        self,
        model: str,
        messages: Union[str, List[Message]],
        *,
        thinking: Optional[bool] = None,
        thinking_family: str = "",
        sampling: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """One non-streaming completion, returned as a parsed :class:`ChatResult`.

        ``model`` is an agent name (e.g. ``"cv_assistant"``) or a model id.
        ``messages`` may be a plain string (treated as a single user turn) or a
        full OpenAI ``messages`` list. ``thinking`` toggles reasoning generation
        (see :func:`agent_server_sdk.types.thinking_kwargs`); pass
        ``thinking_family`` when the active model is granite/ministral.
        """
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        payload = self._build_payload(
            model, msgs, stream=False, thinking=thinking, thinking_family=thinking_family,
            sampling=sampling, extra_body=extra_body,
        )
        data = self._post_json("/v1/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        # If the server kept reasoning in a separate field, fold it in so the
        # parser sees a uniform shape.
        rc = msg.get("reasoning_content")
        if rc and "<think>" not in content:
            content = f"<think>{rc}</think>{content}"
        think, voice, answer = split_response(content)
        return ChatResult(
            answer=answer, thinking=think, voice=voice, raw=content,
            model=data.get("model") or model,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
        )

    # -- streaming chat ----------------------------------------------------
    def chat_stream(
        self,
        model: str,
        messages: Union[str, List[Message]],
        *,
        thinking: Optional[bool] = None,
        thinking_family: str = "",
        sampling: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StreamEvent]:
        """Stream a completion as parsed :class:`StreamEvent` objects.

        Yields ``thinking`` / ``voice`` / ``answer`` events as soon as each is
        known. A ``voice`` event with ``final=True`` marks the spoken summary
        complete and safe to hand to TTS (use
        :func:`agent_server_sdk.parser.sanitize_for_tts`)::

            for ev in client.chat_stream("cv_assistant", "Hi"):
                if ev.kind == "answer":
                    print(ev.text, end="")
        """
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        payload = self._build_payload(
            model, msgs, stream=True, thinking=thinking, thinking_family=thinking_family,
            sampling=sampling, extra_body=extra_body,
        )
        parser = StreamParser()
        url = f"{self.base_url}/v1/chat/completions"
        try:
            with self._client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._raise_for(resp)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
                    piece = delta.get("content")
                    if delta.get("reasoning_content") and not piece:
                        # Reasoning-only delta: wrap so the parser routes it.
                        piece = f"<think>{delta['reasoning_content']}</think>"
                    if not piece:
                        continue
                    for ev in parser.feed(piece):
                        yield ev
        except httpx.HTTPError as e:
            raise TransportError(f"stream failed: {e}") from e
        for ev in parser.flush():
            yield ev

    # -- discovery ---------------------------------------------------------
    def list_models(self) -> List[ModelInfo]:
        """All values accepted in the ``model`` field: chat models (with
        ``active``/``family``) and agent names (``kind == "agent"``)."""
        data = self._get_json("/v1/models")
        out: List[ModelInfo] = []
        for m in data.get("data") or []:
            out.append(ModelInfo(
                id=m.get("id") or "",
                active=bool(m.get("active")),
                kind=m.get("kind") or ("agent" if m.get("kind") == "agent" else "model"),
                family=m.get("family") or "",
                raw=m,
            ))
        return out

    def active_model(self) -> Optional[ModelInfo]:
        """The currently resident chat model, or ``None``."""
        for m in self.list_models():
            if m.active:
                return m
        return None

    def get_agent(self, name: str) -> AgentPreset:
        """Fetch one agent preset (system prompt + sampling + memory policy)."""
        data = self._get_json(f"/v1/agents/{name}")
        return AgentPreset(
            name=data.get("name") or name,
            system_prompt=data.get("system_prompt") or "",
            params_override=data.get("params_override") or {},
            memory_policy=data.get("memory_policy") or "none",
            raw=data,
        )

    # -- active-model switching -------------------------------------------
    def set_active_model(self, model_id: str, *, wait: bool = True, wait_timeout: float = 90.0) -> Dict[str, Any]:
        """Switch the resident chat model (``POST /admin/api/active-model``).

        The stack restarts (~30–45s). With ``wait=True`` this blocks until the
        server answers again and the active flag has flipped to ``model_id``.
        """
        resp = self._request("POST", "/admin/api/active-model", json={"model_id": model_id})
        self._raise_for(resp)
        result = resp.json()
        if wait:
            self.wait_until_ready(expect_active=model_id, timeout=wait_timeout)
        return result

    def wait_until_ready(self, *, expect_active: Optional[str] = None, timeout: float = 90.0, poll: float = 2.0) -> bool:
        """Poll ``/v1/models`` until the server responds (and, if
        ``expect_active`` is given, that model is the active one)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                models = self.list_models()
                if expect_active is None:
                    return True
                if any(m.active and m.id == expect_active for m in models):
                    return True
            except (TransportError, AgentServerError, httpx.HTTPError):
                pass
            time.sleep(poll)
        raise TransportError(
            f"agent_server not ready within {timeout}s"
            + (f" (waiting for active={expect_active})" if expect_active else "")
        )

    # -- low-level HTTP ----------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, url, headers=self._headers(), **kw)
                if resp.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return resp
            except httpx.HTTPError as e:
                last = e
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
        raise TransportError(f"{method} {path} failed: {last}")

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._request("POST", path, json=payload)
        self._raise_for(resp)
        return resp.json()

    def _get_json(self, path: str) -> Dict[str, Any]:
        resp = self._request("GET", path)
        self._raise_for(resp)
        return resp.json()
