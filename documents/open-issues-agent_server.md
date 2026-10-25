# agent_server open issues

Items that could not be closed when found. Each entry says what closes it.

## 1. Host-wide container restart loop, 2026-10-21 (cause unknown)

- **Observed:** 2026-10-21 04:56–07:18 UTC, `llama-vision` restarted ~250 times, every ~40 s, each run
  dying with no error in its log. `agent_server`, `logus2k-cv`, `proxy_server` and `agent-bus-app`
  restarted in the same rhythm (e.g. all within 05:05:01–05:05:49), so the trigger was the Docker
  daemon / WSL, not llama.cpp or agent_server code. No admin write calls (POST/PUT /admin) were logged.
- **Why still open:** the host rebooted right after (oldest journal boot starts 07:24 UTC); that boot's
  journal is not retained, so the cause cannot be recovered from logs.
- **Closes when:** it recurs and is caught live — then capture `journalctl -u docker`, `dmesg`
  (look for OOM kills) and `docker events` during the loop, and record the cause here.

## 2. noted-rag crash loop since 2026-10-13 (agent_server client; lives in the noted project)

- **Observed 2026-10-25:** `noted-rag` (compose project `services`, `~/env/assets/noted/services`) has
  failed to start 7,164 times since 2026-10-13T10:13Z: `_probe_upstream_models` → `httpx.ConnectError:
  [Errno -3] Temporary failure in name resolution` → "Application startup failed", exit 3.
  `HostConfig.NetworkMode=noted-network`, but `NetworkSettings.Networks` is EMPTY — the container is not
  attached to any network, so it cannot resolve `llama-vision` / `embeddings-server`. `noted-network`
  itself exists (id 5aea68e0…, created 2026-03-20) and peers such as `noted-graph` are attached to it.
- **Cause:** unproven — why Docker lost the attachment is not known.
- **Closes when:** the user approves recreating it (`docker compose up -d --force-recreate noted-rag` in
  `~/env/assets/noted/services`), it starts, and a real RAG query through it returns chunks.
