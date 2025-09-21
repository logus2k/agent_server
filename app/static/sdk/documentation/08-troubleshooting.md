# Troubleshooting

- **THREAD_REQUIRED**: Provide a `threadId` when subscribing or sending text.
- **STT_CONNECT: One or more namespaces failed to connect**: Ensure `aiohttp` is installed in agent server environment (`pip install "python-socketio[asyncio_client]"`) and the STT URL/path is reachable (`path='/socket.io'` by default).
- **Changed agent but still using old one**: Re-subscribe STT with the new agent via `sttUnsubscribe()` → `sttSubscribe()`.
- **Localhost in containers**: If agent server is containerized, use `http://host.docker.internal:2700` (Mac/Win) or host LAN IP (Linux).
