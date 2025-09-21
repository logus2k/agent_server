# Events & Wire Protocol

## Client → Agent Server

### `Chat`
```json
{ "text": "Hello", "agent": "topic", "thread_id": "uuid-..." }
```

### `Interrupt`
```json
{ "runId": "..." }
```

### `JoinSTT`
```json
{ "sttUrl": "http://localhost:2700", "clientId": "abc", "agent": "router", "threadId": "uuid-..." }
```

### `LeaveSTT`
```json
{ "sttUrl": "http://localhost:2700", "clientId": "abc" }
```

## Agent Server → Client

- `RunStarted` → `{ "runId": "..." }`
- `ChatChunk`  → `{ "chunk": "text delta" }`
- `ChatDone`   → `{}`
- `Interrupted`→ `{}`
- `Error`      → `{ "code": "...", "message": "...", "runId": null }`

## Browser → STT Server (binary)

- `audio_data` → `{ "clientId": "abc", "audioData": <ArrayBuffer PCM16 mono @16k> }`

## STT Server → Agent Server

- `transcription` → `{ "text": "hello there", "duration": 0.7, "client_id": "abc" }`
