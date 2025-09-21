# app/stt_manager.py
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set
import socketio  # python-socketio

TranscriptHandler = Callable[[str, str, float, str], asyncio.Future] | Callable[[str, str, float, str], None]
# handler signature: (client_id, text, duration, stt_url)

class STTConnection:
	"""
	Manages a single AsyncClient connection to one STT server URL.
	Allows subscribing to multiple clientId "rooms" on the same connection.
	"""
	def __init__(self, url: str, on_transcript: TranscriptHandler, *, socketio_path: str = "socket.io"):
		self.url = url
		self.socketio_path = socketio_path  # make path explicit
		self._on_transcript = on_transcript

		# Turn on client logging just during connect errors; quiet otherwise.
		self._client = socketio.AsyncClient(logger=False, engineio_logger=False)
		self._connected = asyncio.Event()
		self._wanted_rooms: Set[str] = set()  # clientIds we want to be in
		self._lock = asyncio.Lock()

		@self._client.event
		async def connect():
			print(f"[stt-link] connected → {self.url} (path='/{self.socketio_path}')")
			self._connected.set()
			# resubscribe rooms after reconnect
			for cid in list(self._wanted_rooms):
				try:
					await self._client.emit("subscribe_transcripts", {"clientId": cid})
					print(f"[stt-link] resubscribed room '{cid}'")
				except Exception as e:
					print(f"[stt-link] resubscribe failed for '{cid}': {e!r}")

		@self._client.event
		async def connect_error(err):
			# This event fires with the underlying cause of the failed namespace handshake
			print(f"[stt-link] connect_error to {self.url} (path='/{self.socketio_path}'): {err!r}")

		@self._client.event
		async def disconnect():
			print(f"[stt-link] disconnected ← {self.url}")
			self._connected.clear()

		@self._client.on("transcription")
		async def on_transcription(payload):
			# Expected shape: { text, duration, client_id }
			try:
				text = (payload.get("text") or "").strip()
				client_id = (payload.get("client_id") or "").strip()
				dur = float(payload.get("duration") or 0.0)
				if text and client_id:
					maybe_coro = self._on_transcript(client_id, text, dur, self.url)
					if asyncio.iscoroutine(maybe_coro):
						await maybe_coro
			except Exception as e:
				print(f"[stt-link] transcript dispatch error: {e!r}")

	async def ensure_connected(self):
		"""Idempotent: connect if needed, with explicit path and namespaces."""
		if self._client.connected:
			return
		async with self._lock:
			if self._client.connected:
				return
			try:
				# Make path explicit and restrict to root namespace
				await self._client.connect(
					self.url,
					transports=["websocket"],
					socketio_path=self.socketio_path,  # e.g., "socket.io"
					namespaces=["/"],
				)
				# wait for the connect event (set by handler)
				await asyncio.wait_for(self._connected.wait(), timeout=10.0)
			except Exception as e:
				# Wrap with URL + path for clarity (bubbles up to JoinSTT as STT_CONNECT)
				raise RuntimeError(f"STT connect failed to {self.url} (path='/{self.socketio_path}'): {e}")

	async def subscribe(self, client_id: str):
		await self.ensure_connected()
		self._wanted_rooms.add(client_id)
		await self._client.emit("subscribe_transcripts", {"clientId": client_id})
		print(f"[stt-link] subscribed room '{client_id}'")

	async def unsubscribe(self, client_id: str):
		if client_id in self._wanted_rooms:
			self._wanted_rooms.remove(client_id)
		if self._client.connected:
			try:
				await self._client.emit("unsubscribe_transcripts", {"clientId": client_id})
				print(f"[stt-link] unsubscribed room '{client_id}'")
			except Exception as e:
				print(f"[stt-link] unsubscribe error for '{client_id}': {e!r}")

	async def aclose(self):
		try:
			await self._client.disconnect()
		except Exception:
			pass


class STTManager:
	"""
	Holds one STTConnection per stt_url. Provides subscribe/unsubscribe by (url, clientId).
	"""
	def __init__(self, on_transcript: TranscriptHandler, *, socketio_path: str = "socket.io"):
		self._on_transcript = on_transcript
		self._conns: Dict[str, STTConnection] = {}
		self._lock = asyncio.Lock()
		self._socketio_path = socketio_path

	async def ensure(self, url: str) -> STTConnection:
		async with self._lock:
			if url not in self._conns:
				self._conns[url] = STTConnection(url, self._on_transcript, socketio_path=self._socketio_path)
			return self._conns[url]

	async def subscribe(self, url: str, client_id: str):
		conn = await self.ensure(url)
		await conn.subscribe(client_id)

	async def unsubscribe(self, url: str, client_id: str):
		conn = await self.ensure(url)
		await conn.unsubscribe(client_id)

	async def aclose(self):
		for conn in list(self._conns.values()):
			try:
				await conn.aclose()
			except Exception:
				pass
		self._conns.clear()
