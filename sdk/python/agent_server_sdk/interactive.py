"""Optional async interactive client over agent_server's Socket.IO interface.

For server-side consumers that want *push* streaming (voice bots, bridges)
rather than request/response REST. Requires the ``[interactive]`` extra
(``python-socketio[asyncio_client]``).

Maps the server's event protocol — ``Chat`` -> ``RunStarted`` /
``ChatChunk`` / ``ChatDone`` / ``Error`` / ``Interrupted`` — into an
``async for`` of parsed :class:`~agent_server_sdk.parser.StreamEvent`.

TTS and STT here are agent_server-*mediated* (``JoinTTS`` / ``JoinSTT``): the
server relays to the tts/stt services. They FAIL SOFT — if those services are
down, enabling them raises nothing fatal and plain text chat keeps working.
Rich browser-side TTS/STT/avatar wiring lives in the JavaScript SDK.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import socketio  # type: ignore
except Exception as _e:  # pragma: no cover - import guard
    socketio = None  # type: ignore
    _IMPORT_ERR = _e
else:
    _IMPORT_ERR = None

from .parser import StreamEvent, StreamParser

__all__ = ["AsyncInteractiveClient"]


class AsyncInteractiveClient:
    """Async Socket.IO client for one agent_server.

    Usage::

        ac = AsyncInteractiveClient("http://agent_server:7701")
        await ac.connect()
        async for ev in ac.chat("cv_assistant_e2b", "Hello"):
            if ev.kind == "answer":
                print(ev.text, end="")
        await ac.close()
    """

    def __init__(self, base_url: str = "http://agent_server:7701", *, client_id: Optional[str] = None):
        if socketio is None:
            raise RuntimeError(
                "python-socketio is required for the interactive client: "
                f"pip install 'agent_server_sdk[interactive]' ({_IMPORT_ERR})"
            )
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.sio = socketio.AsyncClient(reconnection=True, logger=False, engineio_logger=False)
        self._queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
        self._tts_enabled = False
        self._register()

    # -- lifecycle ---------------------------------------------------------
    async def connect(self) -> None:
        await self.sio.connect(self.base_url, transports=["websocket", "polling"])

    async def close(self) -> None:
        try:
            await self.sio.disconnect()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return bool(self.sio.connected)

    # -- chat --------------------------------------------------------------
    async def chat(
        self,
        agent: str,
        text: str,
        *,
        thread_id: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn; yields parsed events until ``ChatDone``/``Error``.

        Thinking on/off over Socket.IO is governed by the *agent preset*
        (``params_override.chat_template_kwargs``), not per-call — use the REST
        client's ``thinking=`` for per-request control.
        """
        # Drain any stale items.
        while not self._queue.empty():
            self._queue.get_nowait()
        payload: Dict[str, Any] = {"agent": agent, "text": text}
        if thread_id:
            payload["thread_id"] = thread_id
        if memory is not None:
            payload["memory"] = memory
        await self.sio.emit("Chat", payload)
        parser = StreamParser()
        while True:
            item = await self._queue.get()
            if item is None:
                continue
            kind = item.get("_event")
            if kind == "ChatChunk":
                for ev in parser.feed(item.get("chunk") or ""):
                    yield ev
            elif kind == "ChatDone":
                for ev in parser.flush():
                    yield ev
                return
            elif kind == "Interrupted":
                for ev in parser.flush():
                    yield ev
                return
            elif kind == "Error":
                for ev in parser.flush():
                    yield ev
                raise RuntimeError(f"agent_server Error: {item.get('code')}: {item.get('message')}")

    async def interrupt(self) -> None:
        await self.sio.emit("Interrupt")

    # -- mediated TTS (fail soft) -----------------------------------------
    async def enable_tts(self, *, voice: Optional[str] = None, speed: Optional[float] = None) -> bool:
        """Ask agent_server to forward this run's output to the TTS service.

        Returns True if the request was sent. If TTS is unavailable the server
        simply won't produce audio — chat is unaffected. Requires a
        ``client_id`` (set at construction)."""
        if not self.client_id:
            return False
        try:
            payload: Dict[str, Any] = {"clientId": self.client_id}
            if voice is not None:
                payload["voice"] = voice
            if speed is not None:
                payload["speed"] = speed
            await self.sio.emit("JoinTTS", payload)
            self._tts_enabled = True
            return True
        except Exception:
            return False

    # -- mediated STT (fail soft) -----------------------------------------
    async def join_stt(self, *, agent: str, thread_id: Optional[str] = None, transcript_only: bool = False) -> bool:
        """Subscribe agent_server to STT transcripts for this client_id. Fail
        soft: returns False if STT can't be joined; chat still works."""
        if not self.client_id:
            return False
        try:
            await self.sio.emit("JoinSTT", {
                "clientId": self.client_id,
                "agent": agent,
                "threadId": thread_id,
                "transcriptOnly": transcript_only,
            })
            return True
        except Exception:
            return False

    async def leave_stt(self) -> None:
        if self.client_id:
            try:
                await self.sio.emit("LeaveSTT", {"clientId": self.client_id})
            except Exception:
                pass

    # -- internal event wiring --------------------------------------------
    def _register(self) -> None:
        for name in ("RunStarted", "ChatChunk", "ChatDone", "Interrupted", "Error", "UserTranscript"):
            self.sio.on(name, self._make_handler(name))

    def _make_handler(self, name: str):
        async def _handler(data: Any = None) -> None:
            item = dict(data) if isinstance(data, dict) else {"data": data}
            item["_event"] = name
            await self._queue.put(item)
        return _handler
