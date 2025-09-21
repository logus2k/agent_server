# Concepts

- **Agent**: named preset with system prompt and sampling overrides (e.g., `topic`, `router`). Choose per run, or per STT subscription.
- **Thread**: logical conversation id (UUID). Required by some memory strategies.
- **Server-side STT multiplex**: the agent server holds one Socket.IO client per STT URL and subscribes to many `clientId` rooms.
- **Events**: see `07-events-wire-protocol.md`.
