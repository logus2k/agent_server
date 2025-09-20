# worker_pool.py


from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class Worker:
    wid: int
    engine: object  # LlamaCppEngine (kept generic to avoid circular imports)


class WorkerPool:
    """
    Minimal async worker pool.

    - Create N workers with the provided factory.
    - Acquire returns a Worker; release happens automatically via context manager.
    - Thread-safe for async use (asyncio.Queue).
    """

    def __init__(self, factory: Callable[[], object], size: int) -> None:
        if size < 1:
            raise ValueError("pool size must be >= 1")

        self._workers: List[Worker] = []
        self._queue: asyncio.Queue[Worker] = asyncio.Queue()

        for i in range(size):
            engine = factory()
            w = Worker(wid=i, engine=engine)
            self._workers.append(w)
            self._queue.put_nowait(w)

    @asynccontextmanager
    async def acquire(self):
        w = await self._queue.get()
        try:
            yield w
        finally:
            self._queue.put_nowait(w)

    def size(self) -> int:
        return len(self._workers)
