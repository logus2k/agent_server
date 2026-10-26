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
- **Cause (two layers):**
  1. The empty attachment: why Docker lost it is unproven. Recreated 2026-10-25 with the user's approval
     (`docker compose -f docker-compose.yml up -d --no-build --no-deps --force-recreate noted-rag`, same
     image `1496080a…`) → now attached to `noted-network` (172.27.0.26). Fixed.
  2. Still failing after that, now with `[Errno -5] No address associated with hostname`: `embeddings-server`
     answers (`/health` 200), but `llama-vision` is not on `noted-network`. Commit `5b6fea3` (2026-10-07)
     replaced `noted-network` with `cortex-kb` in llama-vision's networks in `docker-compose.adapter.yml`,
     so noted-rag cannot reach `http://llama-vision:8500` for embeddings.
- **CLOSED 2026-10-25:** per the user, llama-vision and agent_server are on all three shared networks
  (`cortex-kb`, `logus2k_network`, `noted-network`) in `docker-compose.adapter.yml`; llama-vision recreated.
  embeddings-server got `logus2k_network` in its git-ignored `compose/docker-compose.local.yml` (attached live
  with `docker network connect`). Verified: noted-rag "Application startup complete" (bge-m3=loaded, rerank OK);
  `POST :8201/search` with `collection=national_ai_agendas__corpus` returned 3 relevant chunks (top score 0.98).
  Note: the legacy default collection `noted_corpus` does not exist — callers must pass a per-domain collection.
