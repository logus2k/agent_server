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

- **CAUSE FOUND 2026-11-12 (the same day): NOT a stall - the WSL clock runs 10% slow (see the entry after this one).** As first written: two long prompts sent to gemma-4 at the same instant sometimes take about 6.3 s
  instead of about 3 s.** Measured with Cortex `documentation/code-analysis/pair_vram.py` (two cold 13,400-token
  prompts, both slots, 200 tokens out): b11870 2 slow of 24 pairs; b12110 on CUDA 12.8 2 of 14 and on CUDA 13.4 1 of 3
  full benchmark runs. In a slow pair the worker's log is silent between `launch_slot_` and `print_timing` for the
  whole time, while its own timers count the usual 2.0 s and 2.8 s of work: about 3.3 s is spent inside the worker
  and counted nowhere. Ruled out by measurement: the upgrade (the old build does it), the CUDA version, GPU memory
  (flat at 18.7 and at 20.2 GB during slow pairs), a picture in the slot before (0 of 10). Cause unknown.
  PAGING tested on purpose the same day (the owner: "it could be paging"), with Windows' own counters read from WSL
  five times a second (Cortex `documentation/code-analysis/gpu_counters.ps1`, `pair_paging.py`): over 40 pairs the
  DEDICATED memory never moved (18.7 of 24 GB) and SHARED rose by up to 232 MB in 18 pairs - 16 of them fast, and one
  of the 3 slow pairs had no rise: the two do not go together. With the card filled deliberately (a 4.5 GB dummy
  allocation: Windows kept dedicated near 19.9 GB and put up to 2.1 GB in shared - real paging) a request hit by it
  showed 6.2 s INSIDE the worker's own timers; the stall never shows there. Slow pairs under pressure: 1 of 14 at
  22.2 GB dedicated, 3 of 14 while paging (7-8% without pressure: too few to call a difference). So paging is a
  different thing from this stall; it is not excluded that pressure makes the stall more frequent.
  Closes with: the cause found (the worker with `--verbose`, or a profile of a slow pair) or an upstream report.


- **OPEN - HOST (found 2026-11-12): the Linux (WSL 2) clock runs at 90.0% of real time; Windows steps its wall clock
  forward by 3.6 s every 35.8 s.** This is what looked like a "3-second stall" in the model server (entry above).
  Measured: a loop that only sleeps 10 ms sees the wall clock jump 3.59-3.60 s every 35.8 s while the monotonic clock
  advances 0.01 s across the jump (Cortex `documentation/code-analysis/freeze_ticks.py`); over one minute Windows'
  clock counted 66.82 s and Linux's monotonic clock 60.15 s; against an internet time source (HTTP Date) 167 s passed
  while Linux counted 150.3 s; 800 requests timed from Windows by Windows' clock showed no hang during three of the
  jumps. So no process pauses: durations measured by the wall clock across a jump read 3.6 s too long, and durations
  measured by the monotonic clock read 10% too SHORT.
  Why: at boot the kernel logged `tsc: Detected 2995.198 MHz processor` and `Switched to clocksource tsc` (current:
  `tsc`; also available: `hyperv_clocksource_tsc_page`). The timer really ticks about 11% faster than that figure
  (i9-13900K on an ASUS ROG STRIX Z790-E, BIOS 3001; WSL 3.0.5, kernel 6.18.58.1; up since 2026-11-11 14:02). Why the
  reported frequency is wrong is NOT established (a base-clock setting in the BIOS, or what Hyper-V reports).
  What it touches: every `sleep`, timeout and interval inside WSL lasts about 11% longer than written; every
  tokens-a-second figure and latency taken from the model server's own timers is about 11% too high (comparisons
  between two settings stay valid); logs' timestamps jump every 36 s; a deadline computed from the wall clock can pass
  3.6 s early.
  Closes with (needs root in WSL - not done, a password is asked): `echo hyperv_clocksource_tsc_page | sudo tee
  /sys/devices/system/clocksource/clocksource0/current_clocksource` (at once, no restart; back with `tsc`), then
  `python3 -I <cortex>/documentation/code-analysis/freeze_ticks.py 120` showing no jump and the one-minute comparison
  at 100%; to keep it after a WSL restart, `kernelCommandLine = clocksource=hyperv_clocksource_tsc_page` under
  `[wsl2]` in `%UserProfile%\.wslconfig`. And the reason the frequency is misreported looked for in the BIOS.

  **CORRECTED 2026-11-12 20:40 - the cause is two time services fighting, not a wrong timer frequency.** The owner
  switched the clocksource to `hyperv_clocksource_tsc_page` (it took effect) and NOTHING changed: still 90.0% against
  Windows (167.55 s / 150.78 s) and against internet time (168 s / 151.2 s), still a 3.60 s jump every 35.8 s. Then
  measured: the kernel's clock adjustment is `tick = 9000` (adjtimex; normal 10000, 9000 the lowest allowed) - the
  kernel is TOLD to run at exactly 90%; Windows measures the timer right (`~MHz` 2995, its high-resolution timer at
  100.00% of its wall clock) but its own clock is 2.6 s BEHIND internet time and is not synchronised (`w32tm /query
  /status`: "not synchronized", source "Local CMOS Clock"); `systemd-timesyncd` is active inside WSL (poll 32 s,
  3,082 packets, `Offset: +3.582908s` at each poll) and steps Linux forward to internet time, which puts Linux ahead
  of Windows; the host's time sync (`hv_utils.timesync_implicit=1` on the kernel command line) then slows the kernel
  to its floor, and Linux loses 3.58 s in the next 32 s. Same family as microsoft/WSL #11790 (an i9-13900K among its
  reports) and the "two time authorities" write-ups. The sentence above about the timer ticking faster than reported
  is WRONG and withdrawn; the clocksource change is neither needed nor harmful (it is not kept across a WSL restart).
  Closes with (root): `sudo systemctl disable --now systemd-timesyncd` (the host stays the one authority), then
  `tick` back at 10000 by itself or with adjtimex, `freeze_ticks.py 120` showing no jump and the one-minute comparison
  at 100%; and Windows' own clock synchronised (Windows Time service, `w32tm /resync`) so the one authority is right.
  **FIXED AND VERIFIED 2026-11-12 22:40.** The owner ran `sudo systemctl disable --now systemd-timesyncd`. Then
  measured (Cortex `documentation/code-analysis/clock_check.py`): the kernel's tick went 9357 -> 10833 (catching up
  with Windows) -> 10000 by itself within a minute and stayed there; over 180 s no wall-clock jump, Linux's monotonic
  clock at 100.00% of Windows' (180.36 s / 180.36 s) and 100.0% of internet time (181.0 s / 181 s); Linux now within
  0.1 s of Windows. Windows' own clock is still about 3 s behind internet time (its time service is not
  synchronised): Linux follows it. NOT done: Windows' clock synchronised; the model server's real speed re-measured on
  the right clock with the model free (at 22:45 an agent job of Cortex held both slots: the figures taken then - 245
  tok/s on the list request - are a shared model's, not a baseline). `disable` holds across restarts; the clocksource
  (`hyperv_clocksource_tsc_page`, set by hand) goes back to `tsc` at the next WSL restart, which is fine.
