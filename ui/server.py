"""ComfyForge Web Dashboard -- FastAPI server entry point.

Modules:
  ui/models.py      — Pydantic request models
  ui/jobs.py        — Job, JobManager, ProgressBridge, WebModelManager
  ui/comfyui_ws.py  — ComfyUI WebSocket progress listener
  ui/runners.py     — Async job runners (generate, clone, download)
  ui/routes.py      — All FastAPI route handlers
  ui/dashboard.html — Frontend HTML/CSS/JS template
"""

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="ComfyForge")
app.include_router(router)
