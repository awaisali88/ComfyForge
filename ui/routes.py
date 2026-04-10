"""FastAPI route handlers for the ComfyForge web dashboard."""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from core.config import Config
from core.models import ModelManager
from core.workflows import (
    text2img_sdxl, text2img_flux,
    img2vid_svd, img2vid_animatediff,
    text2vid_wan, img2vid_wan,
)

from .models import GenerateRequest, CloneRequest, WorkflowExportRequest
from .jobs import job_manager
from .runners import run_generate_job, run_clone_job, run_model_download_job

router = APIRouter()

# Resolve path to the HTML template file (alongside this module)
_HTML_PATH = Path(__file__).parent / "dashboard.html"


# ── WebSocket ──────────────────────────────────────────────────────────


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    job_manager.connections.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.connections.discard(ws)


# ── Generation ─────────────────────────────────────────────────────────


@router.post("/api/generate")
async def generate(req: GenerateRequest):
    job = job_manager.create_job("generate", metadata={"prompt": req.prompt, "pipeline": req.pipeline})
    asyncio.create_task(run_generate_job(job, req))
    return {"job_id": job.id, "status": "started"}


# ── Models ─────────────────────────────────────────────────────────────


@router.get("/api/models")
async def list_models():
    cfg = Config.load()
    mm = ModelManager(cfg)
    models = []
    for m in mm.registry.all_models():
        dest = mm.registry.dest_path(m)
        models.append({
            **m,
            "downloaded": dest.exists(),
            "path": str(dest),
        })
    return {"models": models}


@router.post("/api/models/download/{model_id}")
async def download_model(model_id: str):
    job = job_manager.create_job("model_download", metadata={"model_id": model_id})
    asyncio.create_task(run_model_download_job(job, model_id))
    return {"job_id": job.id, "status": "started"}


@router.get("/api/stacks")
async def list_stacks():
    cfg = Config.load()
    return {"stacks": cfg.raw.get("stacks", {})}


# ── CivitAI Clone ─────────────────────────────────────────────────────


@router.post("/api/clone-civitai")
async def clone_civitai(req: CloneRequest):
    """Clone a CivitAI image -- fetch metadata, download models, generate workflow."""
    job = job_manager.create_job("clone", metadata={"url": req.url})
    asyncio.create_task(run_clone_job(job, req))
    return {"job_id": job.id, "status": "started"}


@router.post("/api/clone-civitai/preview")
async def clone_civitai_preview(req: CloneRequest):
    """Fetch CivitAI image metadata without downloading models or generating workflow."""
    from core.civitai import parse_civitai_url, fetch_image_metadata, _fetch_post_images

    try:
        content_type, content_id = parse_civitai_url(req.url)

        if content_type == "post":
            images = _fetch_post_images(content_id)
            image_data = None
            for img in images:
                if img.get("meta"):
                    image_data = img
                    break
            if not image_data:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "No images with metadata in this post"},
                )
            image_id = image_data["id"]
        else:
            image_id = content_id

        loop = asyncio.get_event_loop()
        meta = await loop.run_in_executor(None, lambda: fetch_image_metadata(image_id))

        return {
            "success": True,
            "image_id": meta.image_id,
            "prompt": meta.prompt,
            "negative_prompt": meta.negative_prompt,
            "steps": meta.steps,
            "sampler": meta.sampler,
            "scheduler": meta.scheduler,
            "cfg_scale": meta.cfg_scale,
            "seed": meta.seed,
            "width": meta.width,
            "height": meta.height,
            "clip_skip": meta.clip_skip,
            "base_model": meta.base_model,
            "base_model_raw": meta.base_model_raw,
            "checkpoint_name": meta.checkpoint_name,
            "checkpoint_filename": meta.checkpoint_filename,
            "vae_name": meta.vae_name,
            "loras": [
                {"name": lr.name, "weight": lr.weight, "version_id": lr.version_id}
                for lr in meta.loras
            ],
            "hires_upscale": meta.hires_upscale,
            "hires_upscaler": meta.hires_upscaler,
            "hires_denoising": meta.hires_denoising,
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


# ── Workflow Export ────────────────────────────────────────────────────


@router.post("/api/workflow-export")
async def workflow_export(req: WorkflowExportRequest):
    """Export a ComfyUI workflow JSON (API format)."""
    try:
        template = req.template
        kwargs: dict[str, Any] = {}

        if template in ("text2img", "text2img_sdxl"):
            kwargs["prompt"] = req.prompt
            kwargs["negative"] = req.negative
            if req.checkpoint:
                kwargs["checkpoint"] = req.checkpoint
            if req.vae:
                kwargs["vae"] = req.vae
            if req.width:
                kwargs["width"] = req.width
            if req.height:
                kwargs["height"] = req.height
            if req.steps:
                kwargs["steps"] = req.steps
            if req.cfg:
                kwargs["cfg"] = req.cfg
            if req.seed != -1:
                kwargs["seed"] = req.seed
            wf = text2img_sdxl(**kwargs)

        elif template == "text2img_flux":
            kwargs["prompt"] = req.prompt
            if req.checkpoint:
                kwargs["checkpoint"] = req.checkpoint
            if req.width:
                kwargs["width"] = req.width
            if req.height:
                kwargs["height"] = req.height
            if req.steps:
                kwargs["steps"] = req.steps
            if req.cfg:
                kwargs["guidance"] = req.cfg
            if req.seed != -1:
                kwargs["seed"] = req.seed
            wf = text2img_flux(**kwargs)

        elif template == "text2vid_wan":
            kwargs["prompt"] = req.prompt
            kwargs["negative"] = req.negative
            if req.checkpoint:
                kwargs["checkpoint"] = req.checkpoint
            if req.width:
                kwargs["width"] = req.width
            if req.height:
                kwargs["height"] = req.height
            if req.steps:
                kwargs["steps"] = req.steps
            if req.cfg:
                kwargs["cfg"] = req.cfg
            if req.seed != -1:
                kwargs["seed"] = req.seed
            if req.frames:
                kwargs["frames"] = req.frames
            wf = text2vid_wan(**kwargs)

        elif template == "img2vid_svd":
            kwargs["image_path"] = req.image_path or "input.png"
            if req.frames:
                kwargs["frames"] = req.frames
            if req.fps:
                kwargs["fps"] = req.fps
            if req.steps:
                kwargs["steps"] = req.steps
            if req.cfg:
                kwargs["cfg"] = req.cfg
            if req.seed != -1:
                kwargs["seed"] = req.seed
            wf = img2vid_svd(**kwargs)

        elif template == "img2vid_animatediff":
            kwargs["prompt"] = req.prompt
            kwargs["image_path"] = req.image_path or "input.png"
            kwargs["negative"] = req.negative
            if req.frames:
                kwargs["frames"] = req.frames
            if req.fps:
                kwargs["fps"] = req.fps
            if req.steps:
                kwargs["steps"] = req.steps
            if req.cfg:
                kwargs["cfg"] = req.cfg
            if req.seed != -1:
                kwargs["seed"] = req.seed
            wf = img2vid_animatediff(**kwargs)

        elif template == "img2vid_wan":
            kwargs["prompt"] = req.prompt
            kwargs["image_path"] = req.image_path or "input.png"
            kwargs["negative"] = req.negative
            if req.frames:
                kwargs["frames"] = req.frames
            if req.fps:
                kwargs["fps"] = req.fps
            if req.steps:
                kwargs["steps"] = req.steps
            if req.cfg:
                kwargs["cfg"] = req.cfg
            if req.seed != -1:
                kwargs["seed"] = req.seed
            wf = img2vid_wan(**kwargs)

        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Unknown template: {template}"},
            )

        # Save to file
        out_dir = Path("exported_workflows")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        dest = out_dir / f"{template}_{ts}.json"
        dest.write_text(json.dumps(wf, indent=2))

        return {
            "success": True,
            "workflow_path": str(dest),
            "workflow_filename": dest.name,
            "workflow_json": wf,
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


# ── File Serving ───────────────────────────────────────────────────────


@router.get("/api/files/outputs/{path:path}")
async def serve_output_file(path: str):
    cfg = Config.load()
    file_path = (cfg.output_dir / path).resolve()
    if not file_path.is_relative_to(cfg.output_dir.resolve()):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path)


@router.get("/api/files/comfyui/{path:path}")
async def serve_comfyui_output(path: str):
    cfg = Config.load()
    base = (cfg.comfyui_path / "output").resolve()
    file_path = (base / path).resolve()
    if not file_path.is_relative_to(base):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path)


@router.get("/api/files/workflows/{path:path}")
async def serve_workflow_file(path: str):
    base = Path("exported_workflows").resolve()
    file_path = (base / path).resolve()
    if not file_path.is_relative_to(base):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(
        file_path,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={file_path.name}"},
    )


# ── Gallery ────────────────────────────────────────────────────────────


@router.get("/api/gallery")
async def gallery():
    cfg = Config.load()
    runs = []
    out_dir = cfg.output_dir
    if out_dir.exists():
        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov"}
        for run_dir in sorted(out_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                files = []
                for f in sorted(run_dir.iterdir()):
                    ext = f.suffix.lower()
                    if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                        ftype = "video" if ext in VIDEO_EXTS else "image"
                        files.append({
                            "name": f.name,
                            "type": ftype,
                            "url": f"/api/files/outputs/{run_dir.name}/{f.name}",
                            "size": f.stat().st_size,
                        })
                if files:
                    runs.append({"name": run_dir.name, "files": files})
    return {"runs": runs[:50]}


# ── Job Status ─────────────────────────────────────────────────────────


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = job_manager.jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return {"job_id": job.id, "type": job.type, "status": job.status, "result": job.result}


# ── Dashboard HTML ─────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PATH.read_text(encoding="utf-8")
