# /mnt/data/tts_manager.py
# tabs for indentation (width 4)

import asyncio
import socketio

class TTSManager:
	def __init__(self, tts_url: str):
		self.url = tts_url.rstrip("/")
		self._client = socketio.AsyncClient()
		self._connected = asyncio.Event()

		@self._client.event
		async def connect():
			self._connected.set()

		@self._client.event
		async def disconnect():
			self._connected.clear()

	async def ensure_connected(self):
		if self._connected.is_set():
			return
		# Connect as the AGENT SERVER, use binary format
		q = "type=agent_server&format=binary"
		await self._client.connect(f"{self.url}/socket.io/?{q}", transports=["websocket"])
		await self._connected.wait()

	async def send_text_chunk(self, *, target_client_id: str, chunk: str, final: bool = False):
		await self.ensure_connected()
		payload = {"chunk": chunk, "target_client_id": target_client_id}
		if final:
			payload["final"] = True
		await self._client.emit("tts_text_chunk", payload)

	async def stop_generation(self, *, client_id: str):
		await self.ensure_connected()
		await self._client.emit("stop_generation", {"client_id": client_id})

	async def configure_client(self, *, client_id: str, voice: str | None = None, speed: float | None = None):
		await self.ensure_connected()
		payload = {"client_id": client_id}
		if voice is not None:
			payload["voice"] = voice
		if speed is not None:
			payload["speed"] = speed
		await self._client.emit("tts_configure_client", payload)
