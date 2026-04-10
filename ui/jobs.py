"""Job manager, progress bridge, and WebModelManager for real-time progress streaming."""

from __future__ import annotations
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import WebSocket

from core.config import Config
from core.models import ModelManager


# ── Data Classes ───────────────────────────────────────────────────────


@dataclass
class Job:
    id: str
    type: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    result: dict | None = None


# ── Job Manager ────────────────────────────────────────────────────────


class JobManager:
    """Manages WebSocket connections and in-memory job state."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.connections: set[WebSocket] = set()

    def create_job(self, job_type: str, metadata: dict | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, type=job_type, metadata=metadata or {})
        self.jobs[job_id] = job
        return job

    async def broadcast(self, message: dict):
        dead: set[WebSocket] = set()
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        self.connections -= dead

    async def send_progress(self, job_id: str, msg_type: str, **kwargs):
        await self.broadcast({"type": msg_type, "job_id": job_id, **kwargs})


# ── Progress Bridge (thread → async) ──────────────────────────────────


class ProgressBridge:
    """Thread-safe bridge: sync worker pushes messages, async drain broadcasts them."""

    def __init__(self, job_manager: JobManager, job_id: str):
        self.jm = job_manager
        self.job_id = job_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def push(self, msg_type: str, **kwargs):
        """Call from worker thread — thread-safe."""
        msg = {"type": msg_type, "job_id": self.job_id, **kwargs}
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    async def drain(self):
        """Run on event loop. Broadcasts queued messages until sentinel (None)."""
        while True:
            msg = await self._queue.get()
            if msg is None:
                break
            await self.jm.broadcast(msg)

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)


# ── WebModelManager ───────────────────────────────────────────────────


class WebModelManager(ModelManager):
    """ModelManager subclass that forwards download progress to WebSocket via ProgressBridge."""

    def __init__(self, cfg: Config, bridge: ProgressBridge | None = None):
        super().__init__(cfg)
        self._bridge = bridge

    def _download(self, url: str, dest: Path):
        headers = {}
        if "civitai.com" in url:
            token = self.cfg.civitai_api_token or os.getenv("CIVITAI_API_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        if "huggingface.co" in url:
            token = self.cfg.hf_token or ""
            if token:
                headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=httpx.Timeout(10, read=3600), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                start_time = time.time()
                last_report = 0.0

                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if self._bridge and (now - last_report > 0.5):
                            elapsed = max(now - start_time, 0.01)
                            speed = downloaded / elapsed
                            eta = int((total - downloaded) / speed) if speed > 0 and total > 0 else 0
                            self._bridge.push(
                                "job:download_progress",
                                filename=dest.name,
                                downloaded=downloaded,
                                total=total,
                                speed=int(speed),
                                eta=eta,
                            )
                            last_report = now

                # Final 100% progress
                if self._bridge and total > 0:
                    self._bridge.push(
                        "job:download_progress",
                        filename=dest.name,
                        downloaded=total,
                        total=total,
                        speed=0,
                        eta=0,
                    )


# ── Singleton ──────────────────────────────────────────────────────────

job_manager = JobManager()
