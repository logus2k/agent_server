"""llama-vision's hold switch (Cortex register #344, decided by the owner 2026-11-08).

A movie render in Cortex needs the GPU's memory: its GPU ledger unloads the chat model first. Any other application's
request then made llama-server load the model again ON TOP of the render: the card passed ~22 GiB, WSL paged, and both
crawled (measured: render 67 s alone vs 240 s; a chat answer 12 s alone vs 186 s). This proxy sits in front of
llama-server (8500 -> 8501): while a HOLD is on, inference requests WAIT here and run, at full speed, when it ends.

The hold is a LEASE: `POST /admin/hold {"seconds": 120, "cap_seconds": 10800 | null, "by": "cortex"}` takes or renews it
for `seconds`; the holder renews it (Cortex: every 30 s) and ends it with `DELETE /admin/hold`. A holder that dies stops
renewing: the hold lapses by itself. `cap_seconds` (from the first take) bounds even a renewing hold - a holder alive
but stuck cannot make others wait longer; null = no cap. A restart of this container clears it. `GET /admin/hold` says
its state. With HOLD_TOKEN set, the admin routes need `Authorization: Bearer <token>`.

What waits: every POST but the router's own model controls (/models/load, /models/unload - the ledger's unload and
load) and these admin routes. GETs (health, /models, /slots) always pass. Every request and response is sent with
`Connection: close` so each request is seen here (a pooled keep-alive connection would otherwise carry later requests
past the hold). Standard library only: the image has no pip packages.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

HOLD = {"until": 0.0, "cap_at": None, "by": "", "since": 0.0}
TOKEN = os.environ.get("HOLD_TOKEN", "")
PASS_POST = {"/models/load", "/models/unload"}
waiting = 0


def log(msg: str) -> None:
    print(f"[hold] {time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def held(now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    return HOLD["until"] > now and (HOLD["cap_at"] is None or now < HOLD["cap_at"])


def state() -> dict:
    now = time.monotonic()
    on = held(now)
    return {"held": on, "by": HOLD["by"] if on else "", "seconds_left": round(max(0.0, HOLD["until"] - now), 1) if on else 0,
            "cap_seconds_left": None if not on or HOLD["cap_at"] is None else round(HOLD["cap_at"] - now, 1),
            "held_for": round(now - HOLD["since"], 1) if on else 0, "waiting": waiting}


def take(seconds: float, cap_seconds: float | None, by: str) -> dict:
    now = time.monotonic()
    if not held(now):                              # a new hold: its cap counts from now
        HOLD.update(since=now, cap_at=None if cap_seconds is None else now + max(0.0, cap_seconds), by=by)
        log(f"hold taken by {by or '?'} for {seconds:.0f} s (cap {'none' if cap_seconds is None else f'{cap_seconds:.0f} s'})")
    HOLD["until"] = now + max(1.0, min(seconds, 600.0))
    return state()


def release() -> dict:
    if held():
        log(f"hold released by {HOLD['by'] or '?'} after {time.monotonic() - HOLD['since']:.0f} s")
    HOLD.update(until=0.0, cap_at=None, by="")
    return state()


async def read_head(reader: asyncio.StreamReader) -> bytes:
    return await reader.readuntil(b"\r\n\r\n")


def closed_head(head: bytes) -> bytes:
    """The request or response head with `Connection: close` (any Connection/Keep-Alive header replaced)."""
    lines = head.decode("latin-1").split("\r\n")
    out = [lines[0]] + [l for l in lines[1:] if l and l.split(":", 1)[0].strip().lower() not in ("connection", "keep-alive")]
    return ("\r\n".join(out + ["Connection: close", "", ""])).encode("latin-1")


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass


async def answer(writer: asyncio.StreamWriter, status: str, body: dict) -> None:
    data = json.dumps(body).encode()
    writer.write(f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
    await writer.drain()


async def admin(method: str, head: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    headers = {l.split(":", 1)[0].strip().lower(): l.split(":", 1)[1].strip() for l in head.decode("latin-1").split("\r\n")[1:] if ":" in l}
    if TOKEN and headers.get("authorization", "") != f"Bearer {TOKEN}":
        return await answer(writer, "401 Unauthorized", {"error": "the hold needs its token"})
    if method == "GET":
        return await answer(writer, "200 OK", state())
    if method == "DELETE":
        return await answer(writer, "200 OK", release())
    if method == "POST":
        n = int(headers.get("content-length") or 0)
        try:
            body = json.loads((await reader.readexactly(n)).decode() or "{}") if n else {}
            cap = body.get("cap_seconds")
            return await answer(writer, "200 OK", take(float(body.get("seconds") or 120), None if cap in (None, 0, "") else float(cap), str(body.get("by") or "")))
        except (ValueError, TypeError) as e:
            return await answer(writer, "400 Bad Request", {"error": f"bad hold request: {e}"})
    return await answer(writer, "405 Method Not Allowed", {"error": "GET, POST or DELETE"})


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, upstream: tuple[str, int]) -> None:
    global waiting
    up_writer = None
    try:
        head = await read_head(reader)
        method, path = (head.split(b"\r\n", 1)[0].decode("latin-1").split(" ") + ["", ""])[:2]
        path = path.split("?", 1)[0]
        if path == "/admin/hold":
            return await admin(method, head, reader, writer)
        if method == "POST" and path not in PASS_POST and held():
            waiting += 1
            t0 = time.monotonic()
            log(f"{path} waits for the hold ({waiting} waiting)")
            try:
                while held():
                    await asyncio.sleep(0.5)
            finally:
                waiting -= 1
            log(f"{path} goes on after {time.monotonic() - t0:.0f} s")
        up_reader, up_writer = await asyncio.open_connection(*upstream)
        up_writer.write(closed_head(head))
        await up_writer.drain()
        sending = asyncio.create_task(pipe(reader, up_writer))           # the body, as the client sends it
        try:
            rhead = await read_head(up_reader)
            writer.write(closed_head(rhead))
            await writer.drain()
            await pipe(up_reader, writer)
        finally:
            sending.cancel()
    except (asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
        pass
    finally:
        for w in (up_writer, writer):
            if w is not None:
                try:
                    w.close()
                except Exception:  # noqa: BLE001 - closing a connection already gone
                    pass


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8500)
    ap.add_argument("--upstream", type=int, default=8501)
    ap.add_argument("--pid", type=int, default=0, help="llama-server's pid: when it exits, so does this proxy (the container restarts)")
    a = ap.parse_args()
    server = await asyncio.start_server(lambda r, w: handle(r, w, ("127.0.0.1", a.upstream)), "0.0.0.0", a.listen, limit=1 << 20)
    log(f"listening on {a.listen}, llama-server on {a.upstream}")

    async def watch() -> None:
        # llama-server is this process's child (the entrypoint exec'd it in): reaped here, so its exit is seen (a
        # zombie still answers kill(pid, 0))
        while a.pid:
            try:
                done, _status = os.waitpid(a.pid, os.WNOHANG)
            except ChildProcessError:           # not our child: whether it still exists
                try:
                    os.kill(a.pid, 0)
                    done = 0
                except ProcessLookupError:
                    done = a.pid
            if done:
                log("llama-server exited: the proxy stops (the container restarts)")
                os._exit(1)
            await asyncio.sleep(2)
    asyncio.create_task(watch())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
