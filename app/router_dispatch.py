# app/router_dispatch.py
from __future__ import annotations
import asyncio, json, logging, uuid, threading
from typing import Any, Dict

logger = logging.getLogger("router_dispatch")

# One shared event that is never set — satisfies generate_stream(cancel=...)
_NEVER_CANCEL = threading.Event()

class RouterDispatcher:
    """
    Fire-and-forget router calls.
    - Uses the 'router' agent preset
    - Memory OFF
    - Emits the agent's JSON output *as-is* to 'RouterResult' on the same sid.
    """
    def __init__(self, sio, pool, agents: Dict[str, Any]):
        self.sio = sio
        self.pool = pool
        self.agents = agents
        logger.info("RouterDispatcher initialized")

    def _preset(self):
        p = self.agents.get("router")
        if not p:
            raise ValueError("RouterDispatcher: agent preset 'router' not found")
        return p

    def dispatch(self, sid: str, text: str) -> None:
        run_id = f"rtr-{uuid.uuid4().hex[:8]}"
        text = (text or "").strip()
        if not text:
            logger.warning("[%s] empty text; ignoring (sid=%s)", run_id, sid)
            return

        logger.info("[%s] accept sid=%s text=%r", run_id, sid, text)
        preset = self._preset()

        async def _run():
            logger.debug("[%s] start", run_id)
            try:
                chunks: list[str] = []
                async with self.pool.acquire() as worker:
                    logger.debug("[%s] acquired engine=%s", run_id, type(worker.engine).__name__)
                    async for ch in worker.engine.generate_stream(
                        text,
                        cancel=_NEVER_CANCEL,  # ← FIX: provide an object with .is_set()
                        system_prompt_path=preset.system_prompt_path,
                        sampling_overrides=preset.params_override,
                        preamble=None,  # memory OFF
                    ):
                        chunks.append(ch)

                full = "".join(chunks).strip()
                logger.info("[%s] model_out len=%d", run_id, len(full))

                obj = json.loads(full)  # expecting exact JSON from the model, e.g. {"Operation":"LOCATE","Term":"Panama"}
                await self.sio.emit("RouterResult", obj, to=sid)
                logger.info("[%s] emitted RouterResult to sid=%s keys=%s", run_id, sid, list(obj.keys()))
            except Exception as e:
                logger.exception("[%s] router run failed: %s", run_id, e)
                try:
                    await self.sio.emit("RouterResult", {"Operation": "ERROR", "Reason": str(e)}, to=sid)
                except Exception:
                    logger.exception("[%s] also failed emitting error payload", run_id)

        asyncio.create_task(_run())
        logger.debug("[%s] scheduled task", run_id)
