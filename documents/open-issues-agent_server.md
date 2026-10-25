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
