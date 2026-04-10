"""ComfyForge Web Dashboard -- FastAPI server."""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import Config
from core.models import ModelManager
from core.pipeline import Pipeline

app = FastAPI(title="ComfyForge")


# ── Request Models ──────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    pipeline: str = "text2img"
    negative: str = "blurry, low quality, watermark, text"
    image_path: str | None = None
    model_stack: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    cfg: float | None = None
    seed: int = -1
    video_backend: str = "svd"
    frames: int | None = None
    fps: int | None = None
    use_flux: bool = False


class CloneRequest(BaseModel):
    url: str
    no_download: bool = False
    output_path: str | None = None


class WorkflowExportRequest(BaseModel):
    template: str = "text2img"
    prompt: str = ""
    negative: str = "blurry, low quality, watermark, text, deformed"
    checkpoint: str | None = None
    vae: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    cfg: float | None = None
    seed: int = -1
    use_flux: bool = False
    image_path: str | None = None
    frames: int | None = None
    fps: int | None = None
    video_backend: str = "svd"


# ── API Endpoints ───────────────────────────────────────────────────────


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    cfg = Config.load()
    pipe = Pipeline(cfg)

    overrides: dict[str, Any] = {
        "width": req.width,
        "height": req.height,
        "seed": req.seed,
        "video_backend": req.video_backend,
    }
    if req.steps:
        overrides["steps"] = req.steps
    if req.cfg:
        overrides["cfg"] = req.cfg
    if req.frames:
        overrides["frames"] = req.frames
    if req.fps:
        overrides["fps"] = req.fps
    if req.use_flux:
        overrides["use_flux"] = True

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipe.run(
            prompt=req.prompt,
            pipeline=req.pipeline,
            negative=req.negative,
            image_path=req.image_path,
            model_stack=req.model_stack,
            overrides=overrides,
        ),
    )

    return {
        "success": result.success,
        "pipeline": result.pipeline,
        "stages": result.stages,
        "outputs": {k: [str(p) for p in v] for k, v in result.outputs.items()},
        "timings": result.timings,
        "errors": result.errors,
    }


@app.get("/api/models")
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


@app.post("/api/models/download/{model_id}")
async def download_model(model_id: str):
    cfg = Config.load()
    mm = ModelManager(cfg)
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, lambda: mm.ensure_model(model_id))
    return {"success": True, "path": str(path)}


@app.get("/api/stacks")
async def list_stacks():
    cfg = Config.load()
    return {"stacks": cfg.raw.get("stacks", {})}


@app.post("/api/clone-civitai")
async def clone_civitai(req: CloneRequest):
    """Clone a CivitAI image -- fetch metadata, download models, generate workflow."""
    from core.civitai import clone_from_civitai

    cfg = Config.load()
    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(
            None,
            lambda: clone_from_civitai(
                req.url,
                output_path=req.output_path,
                no_download=req.no_download,
                config=cfg,
            ),
        )
        return {"success": True, "workflow_path": str(path)}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/clone-civitai/preview")
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


@app.post("/api/workflow-export")
async def workflow_export(req: WorkflowExportRequest):
    """Export a ComfyUI workflow JSON."""
    import sys
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    from workflow_export import (
        make_text2img_sdxl, make_text2img_flux,
        make_img2vid_svd, make_img2vid_animatediff,
        make_text2vid_wan, make_img2vid_wan,
    )

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
            wf = make_text2img_sdxl(**kwargs)

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
            wf = make_text2img_flux(**kwargs)

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
            if req.fps:
                kwargs["fps"] = req.fps
            wf = make_text2vid_wan(**kwargs)

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
            wf = make_img2vid_svd(**kwargs)

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
            wf = make_img2vid_animatediff(**kwargs)

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
            wf = make_img2vid_wan(**kwargs)

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
        wf.save(str(dest))

        return {
            "success": True,
            "workflow_path": str(dest),
            "workflow_json": wf.export(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


# ── Inline Dashboard HTML ────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ComfyForge</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #14141f;
    --surface2: #1a1a2a;
    --border: #1e1e30;
    --border-light: #2a2a40;
    --text: #e0e0f0;
    --muted: #6b6b8a;
    --accent: #7c5cfc;
    --accent2: #fc5c7c;
    --success: #3cda7c;
    --warn: #f0c040;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Header ── */
  .header {
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .header h1 {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .tag {
    font-size: 0.6rem;
    padding: 2px 8px;
    border: 1px solid var(--accent);
    border-radius: 3px;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Tabs ── */
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    gap: 0;
  }
  .tab {
    padding: 0.8rem 1.5rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    letter-spacing: 0.5px;
  }
  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  /* ── Tab Panels ── */
  .tab-content { display: none; min-height: calc(100vh - 110px); }
  .tab-content.active { display: grid; }

  /* ── Layouts ── */
  .two-panel {
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
  }
  .single-panel {
    grid-template-columns: 1fr;
    background: var(--bg);
  }
  .panel {
    background: var(--bg);
    padding: 2rem;
    overflow-y: auto;
    max-height: calc(100vh - 110px);
  }
  .panel h2 {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 1.5rem;
  }

  /* ── Forms ── */
  textarea, input[type="text"], input[type="number"], input[type="url"], select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.65rem 0.8rem;
    border-radius: 6px;
    font-family: inherit;
    font-size: 0.85rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s;
  }
  textarea { height: 110px; resize: vertical; }
  textarea:focus, input:focus, select:focus {
    outline: none;
    border-color: var(--accent);
  }
  label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    display: block;
    margin-bottom: 0.3rem;
  }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
  .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; }

  /* ── Buttons ── */
  button {
    font-family: inherit;
    cursor: pointer;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    transition: all 0.2s;
  }
  button.primary {
    width: 100%;
    padding: 0.85rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    font-size: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }
  button.primary:hover { opacity: 0.9; transform: translateY(-1px); }
  button.primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  button.secondary {
    padding: 0.65rem 1.2rem;
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    font-size: 0.8rem;
  }
  button.secondary:hover { border-color: var(--accent); }

  /* ── Log ── */
  .log {
    font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
    font-size: 0.75rem;
    line-height: 1.7;
    white-space: pre-wrap;
    color: var(--muted);
    min-height: 100px;
  }
  .log .ok { color: var(--success); }
  .log .err { color: var(--accent2); }
  .log .info { color: var(--accent); }
  .log .warn { color: var(--warn); }

  /* ── Cards ── */
  .meta-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
  }
  .meta-card h3 {
    font-size: 0.75rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
  }
  .meta-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    font-size: 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  .meta-row:last-child { border-bottom: none; }
  .meta-row .label { color: var(--muted); }
  .meta-row .value { color: var(--text); font-weight: 500; text-align: right; max-width: 60%; word-break: break-all; }

  .prompt-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem;
    font-size: 0.8rem;
    line-height: 1.5;
    margin-bottom: 0.8rem;
    max-height: 150px;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  .lora-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: var(--surface2);
    border: 1px solid var(--border-light);
    border-radius: 4px;
    font-size: 0.75rem;
    margin: 0.2rem;
  }
  .lora-tag .weight { color: var(--accent); margin-left: 0.3rem; }

  /* ── Models Table ── */
  .model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  .model-table th {
    text-align: left;
    padding: 0.6rem;
    color: var(--muted);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid var(--border);
  }
  .model-table td {
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid var(--border);
  }
  .model-table tr:hover { background: var(--surface); }
  .badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
  }
  .badge.yes { background: rgba(60,218,124,0.15); color: var(--success); }
  .badge.no { background: rgba(252,92,124,0.1); color: var(--accent2); }
  .badge.type { background: rgba(124,92,252,0.15); color: var(--accent); }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Utility ── */
  .hidden { display: none !important; }
  .mt { margin-top: 1rem; }
  .mb { margin-bottom: 1rem; }
  .text-muted { color: var(--muted); }
  .text-sm { font-size: 0.75rem; }

  /* ── Checkbox ── */
  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.9rem;
  }
  .checkbox-row input[type="checkbox"] {
    width: auto;
    margin: 0;
  }
  .checkbox-row label {
    margin: 0;
    text-transform: none;
    font-size: 0.8rem;
    color: var(--text);
  }
</style>
</head>
<body>

<div class="header">
  <h1>ComfyForge</h1>
  <span class="tag">v0.1</span>
</div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('generate')">Generate</div>
  <div class="tab" onclick="switchTab('clone')">Clone CivitAI</div>
  <div class="tab" onclick="switchTab('export')">Export Workflow</div>
  <div class="tab" onclick="switchTab('models')">Models</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TAB 1: GENERATE                                                     -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<div id="tab-generate" class="tab-content two-panel active">
  <div class="panel">
    <h2>Generate</h2>

    <label>Prompt</label>
    <textarea id="gen-prompt" placeholder="A cinematic dragon soaring over Lahore at golden hour..."></textarea>

    <label>Negative Prompt</label>
    <input type="text" id="gen-negative" value="blurry, low quality, watermark, text">

    <div class="row">
      <div>
        <label>Pipeline</label>
        <select id="gen-pipeline">
          <option value="text2img">Text to Image</option>
          <option value="text2vid">Text to Image to Video</option>
          <option value="text2vid_wan">Text to Video (Wan 2.1)</option>
          <option value="full">Full Pipeline (+ Audio)</option>
          <option value="img2vid">Image to Video</option>
        </select>
      </div>
      <div>
        <label>Video Backend</label>
        <select id="gen-video-backend">
          <option value="svd">SVD</option>
          <option value="animatediff">AnimateDiff</option>
          <option value="wan">Wan 2.1</option>
        </select>
      </div>
    </div>

    <div class="row3">
      <div><label>Width</label><input type="number" id="gen-width" value="1024"></div>
      <div><label>Height</label><input type="number" id="gen-height" value="1024"></div>
      <div><label>Steps</label><input type="number" id="gen-steps" value="25"></div>
    </div>

    <div class="row3">
      <div><label>CFG</label><input type="number" id="gen-cfg" value="7" step="0.5"></div>
      <div><label>Seed</label><input type="number" id="gen-seed" value="-1"></div>
      <div><label>Frames</label><input type="number" id="gen-frames" value="24"></div>
    </div>

    <div class="row">
      <div class="checkbox-row">
        <input type="checkbox" id="gen-flux">
        <label for="gen-flux">Use Flux</label>
      </div>
      <div>
        <label>Image Path (for img2vid)</label>
        <input type="text" id="gen-image" placeholder="path/to/image.png">
      </div>
    </div>

    <button class="primary" id="btn-generate" onclick="doGenerate()">Generate</button>
  </div>

  <div class="panel">
    <h2>Output</h2>
    <div id="gen-log" class="log"></div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TAB 2: CLONE CIVITAI                                                -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<div id="tab-clone" class="tab-content two-panel">
  <div class="panel">
    <h2>Clone from CivitAI</h2>
    <p class="text-sm text-muted mb">Paste a CivitAI image or post URL. ComfyForge will extract the prompt, models, and all generation parameters, then produce a ready-to-import ComfyUI workflow.</p>

    <label>CivitAI URL</label>
    <input type="url" id="clone-url" placeholder="https://civitai.com/images/12345">

    <div class="row">
      <div class="checkbox-row">
        <input type="checkbox" id="clone-no-download">
        <label for="clone-no-download">Skip model downloads</label>
      </div>
      <div></div>
    </div>

    <div class="row" style="gap: 0.5rem">
      <button class="secondary" id="btn-preview" onclick="doPreview()">Preview Metadata</button>
      <button class="primary" id="btn-clone" onclick="doClone()" style="margin-top:0">Clone & Build Workflow</button>
    </div>

    <!-- Preview card (hidden until fetched) -->
    <div id="clone-preview" class="hidden mt">
      <div class="meta-card">
        <h3>Generation Metadata</h3>
        <div id="clone-meta-rows"></div>
      </div>
      <div id="clone-prompt-section" class="meta-card">
        <h3>Prompt</h3>
        <div id="clone-prompt-text" class="prompt-box"></div>
      </div>
      <div id="clone-neg-section" class="meta-card hidden">
        <h3>Negative Prompt</h3>
        <div id="clone-neg-text" class="prompt-box"></div>
      </div>
      <div id="clone-lora-section" class="meta-card hidden">
        <h3>LoRAs</h3>
        <div id="clone-loras"></div>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>Status</h2>
    <div id="clone-log" class="log"></div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TAB 3: EXPORT WORKFLOW                                              -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<div id="tab-export" class="tab-content two-panel">
  <div class="panel">
    <h2>Export Workflow</h2>
    <p class="text-sm text-muted mb">Generate a ComfyUI-native JSON workflow you can drag into the ComfyUI web UI. No execution happens.</p>

    <label>Workflow Template</label>
    <select id="exp-template" onchange="onTemplateChange()">
      <option value="text2img">Text to Image (SDXL)</option>
      <option value="text2img_flux">Text to Image (Flux)</option>
      <option value="text2vid_wan">Text to Video (Wan 2.1)</option>
      <option value="img2vid_svd">Image to Video (SVD)</option>
      <option value="img2vid_animatediff">Image to Video (AnimateDiff)</option>
      <option value="img2vid_wan">Image to Video (Wan 2.1)</option>
    </select>

    <label>Prompt</label>
    <textarea id="exp-prompt" placeholder="A dragon over Badshahi Mosque at golden hour..."></textarea>

    <div id="exp-neg-row">
      <label>Negative Prompt</label>
      <input type="text" id="exp-negative" value="blurry, low quality, watermark, text, deformed">
    </div>

    <div class="row">
      <div><label>Width</label><input type="number" id="exp-width" value="1024"></div>
      <div><label>Height</label><input type="number" id="exp-height" value="1024"></div>
    </div>

    <div class="row3">
      <div><label>Steps</label><input type="number" id="exp-steps" value="25"></div>
      <div><label>CFG / Guidance</label><input type="number" id="exp-cfg" value="7" step="0.5"></div>
      <div><label>Seed</label><input type="number" id="exp-seed" value="-1"></div>
    </div>

    <div id="exp-video-opts" class="hidden">
      <div class="row">
        <div><label>Frames</label><input type="number" id="exp-frames" value="24"></div>
        <div><label>FPS</label><input type="number" id="exp-fps" value="8"></div>
      </div>
    </div>

    <div id="exp-image-row" class="hidden">
      <label>Input Image Path</label>
      <input type="text" id="exp-image" placeholder="path/to/image.png">
    </div>

    <button class="primary" id="btn-export" onclick="doExport()">Export Workflow</button>
  </div>

  <div class="panel">
    <h2>Result</h2>
    <div id="exp-log" class="log"></div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- TAB 4: MODELS                                                       -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<div id="tab-models" class="tab-content single-panel">
  <div class="panel">
    <h2>Model Registry</h2>
    <button class="secondary mb" onclick="loadModels()">Refresh</button>
    <table class="model-table" id="model-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Filename</th>
          <th>Type</th>
          <th>Size</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody id="model-tbody"></tbody>
    </table>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<!-- JavaScript                                                          -->
<!-- ═══════════════════════════════════════════════════════════════════ -->
<script>
/* ── Tab switching ── */
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab-content#tab-${name}`).classList.add('active');
  // Find the tab button by matching text
  document.querySelectorAll('.tab').forEach(t => {
    if (t.textContent.toLowerCase().replace(/\s+/g,'').includes(name.replace('-','')))
      t.classList.add('active');
  });
  // Auto-load models tab
  if (name === 'models') loadModels();
}

/* ── Logging helper ── */
function log(elId, msg, cls) {
  const el = document.getElementById(elId);
  const line = document.createElement('div');
  line.className = cls || '';
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
function clearLog(elId) {
  document.getElementById(elId).innerHTML = '';
}

/* ── TAB 1: Generate ── */
async function doGenerate() {
  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  clearLog('gen-log');

  const body = {
    prompt: document.getElementById('gen-prompt').value,
    negative: document.getElementById('gen-negative').value,
    pipeline: document.getElementById('gen-pipeline').value,
    video_backend: document.getElementById('gen-video-backend').value,
    width: +document.getElementById('gen-width').value,
    height: +document.getElementById('gen-height').value,
    steps: +document.getElementById('gen-steps').value || null,
    cfg: +document.getElementById('gen-cfg').value || null,
    seed: +document.getElementById('gen-seed').value,
    frames: +document.getElementById('gen-frames').value || null,
    use_flux: document.getElementById('gen-flux').checked,
    image_path: document.getElementById('gen-image').value || null,
  };

  log('gen-log', 'Starting pipeline: ' + body.pipeline, 'info');
  log('gen-log', 'Prompt: ' + body.prompt);

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (data.success) {
      log('gen-log', '\nPipeline complete!', 'ok');
      for (const [stage, t] of Object.entries(data.timings))
        log('gen-log', `  ${stage}: ${t.toFixed(1)}s`, 'ok');
      for (const [stage, files] of Object.entries(data.outputs))
        for (const f of files)
          log('gen-log', `  > ${f}`, 'ok');
    } else {
      log('gen-log', '\nPipeline failed', 'err');
      data.errors.forEach(e => log('gen-log', '  ' + e, 'err'));
    }
  } catch (e) {
    log('gen-log', 'Error: ' + e.message, 'err');
  }

  btn.disabled = false;
}

/* ── TAB 2: Clone CivitAI ── */
async function doPreview() {
  const url = document.getElementById('clone-url').value.trim();
  if (!url) return;

  const btn = document.getElementById('btn-preview');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Fetching...';
  clearLog('clone-log');
  document.getElementById('clone-preview').classList.add('hidden');

  try {
    log('clone-log', 'Fetching metadata from CivitAI...', 'info');
    const res = await fetch('/api/clone-civitai/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    if (!data.success) {
      log('clone-log', 'Error: ' + data.error, 'err');
      btn.disabled = false;
      btn.textContent = 'Preview Metadata';
      return;
    }

    log('clone-log', 'Metadata fetched for image #' + data.image_id, 'ok');

    // Populate preview card
    const rows = document.getElementById('clone-meta-rows');
    rows.innerHTML = '';
    const fields = [
      ['Base Model', `${data.base_model_raw || '?'} (${data.base_model})`],
      ['Checkpoint', data.checkpoint_filename || data.checkpoint_name],
      ['Size', `${data.width} x ${data.height}`],
      ['Steps', data.steps],
      ['CFG Scale', data.cfg_scale],
      ['Sampler', `${data.sampler} / ${data.scheduler}`],
      ['Seed', data.seed],
    ];
    if (data.clip_skip > 1) fields.push(['Clip Skip', data.clip_skip]);
    if (data.vae_name) fields.push(['VAE', data.vae_name]);
    if (data.hires_upscale) fields.push(['Hires Fix', `${data.hires_upscale}x (${data.hires_upscaler})`]);

    for (const [label, value] of fields) {
      const row = document.createElement('div');
      row.className = 'meta-row';
      row.innerHTML = `<span class="label">${label}</span><span class="value">${value}</span>`;
      rows.appendChild(row);
    }

    document.getElementById('clone-prompt-text').textContent = data.prompt;

    if (data.negative_prompt) {
      document.getElementById('clone-neg-section').classList.remove('hidden');
      document.getElementById('clone-neg-text').textContent = data.negative_prompt;
    } else {
      document.getElementById('clone-neg-section').classList.add('hidden');
    }

    const loraSection = document.getElementById('clone-lora-section');
    const loraContainer = document.getElementById('clone-loras');
    if (data.loras && data.loras.length > 0) {
      loraSection.classList.remove('hidden');
      loraContainer.innerHTML = data.loras.map(l =>
        `<span class="lora-tag">${l.name}<span class="weight">${l.weight}</span></span>`
      ).join('');
    } else {
      loraSection.classList.add('hidden');
    }

    document.getElementById('clone-preview').classList.remove('hidden');
  } catch (e) {
    log('clone-log', 'Error: ' + e.message, 'err');
  }

  btn.disabled = false;
  btn.textContent = 'Preview Metadata';
}

async function doClone() {
  const url = document.getElementById('clone-url').value.trim();
  if (!url) return;

  const btn = document.getElementById('btn-clone');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Cloning...';
  clearLog('clone-log');

  const noDownload = document.getElementById('clone-no-download').checked;

  try {
    log('clone-log', 'Starting clone from CivitAI...', 'info');
    log('clone-log', 'URL: ' + url);
    if (noDownload) log('clone-log', 'Model downloads: SKIPPED', 'warn');

    const res = await fetch('/api/clone-civitai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, no_download: noDownload }),
    });
    const data = await res.json();

    if (data.success) {
      log('clone-log', '\nWorkflow generated!', 'ok');
      log('clone-log', 'Saved to: ' + data.workflow_path, 'ok');
      log('clone-log', '\nLoad in ComfyUI: Menu > Load > select the JSON file', 'info');
    } else {
      log('clone-log', '\nClone failed: ' + data.error, 'err');
    }
  } catch (e) {
    log('clone-log', 'Error: ' + e.message, 'err');
  }

  btn.disabled = false;
  btn.textContent = 'Clone & Build Workflow';
}

/* ── TAB 3: Export Workflow ── */
function onTemplateChange() {
  const tpl = document.getElementById('exp-template').value;
  const isVideo = tpl.includes('vid');
  const needsImage = tpl.startsWith('img2vid');
  const isFlux = tpl.includes('flux');

  document.getElementById('exp-video-opts').classList.toggle('hidden', !isVideo);
  document.getElementById('exp-image-row').classList.toggle('hidden', !needsImage);
  document.getElementById('exp-neg-row').style.display = isFlux ? 'none' : '';
}

async function doExport() {
  const btn = document.getElementById('btn-export');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Exporting...';
  clearLog('exp-log');

  const body = {
    template: document.getElementById('exp-template').value,
    prompt: document.getElementById('exp-prompt').value,
    negative: document.getElementById('exp-negative').value,
    width: +document.getElementById('exp-width').value,
    height: +document.getElementById('exp-height').value,
    steps: +document.getElementById('exp-steps').value || null,
    cfg: +document.getElementById('exp-cfg').value || null,
    seed: +document.getElementById('exp-seed').value,
    frames: +document.getElementById('exp-frames').value || null,
    fps: +document.getElementById('exp-fps').value || null,
    image_path: document.getElementById('exp-image').value || null,
  };

  try {
    log('exp-log', 'Exporting ' + body.template + ' workflow...', 'info');

    const res = await fetch('/api/workflow-export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (data.success) {
      log('exp-log', '\nWorkflow exported!', 'ok');
      log('exp-log', 'Saved to: ' + data.workflow_path, 'ok');
      log('exp-log', '\nLoad in ComfyUI: Menu > Load > select the JSON file', 'info');

      // Show node count
      if (data.workflow_json && data.workflow_json.nodes) {
        log('exp-log', `Nodes: ${data.workflow_json.nodes.length}  Links: ${data.workflow_json.links.length}`, 'info');
      }
    } else {
      log('exp-log', 'Export failed: ' + data.error, 'err');
    }
  } catch (e) {
    log('exp-log', 'Error: ' + e.message, 'err');
  }

  btn.disabled = false;
  btn.textContent = 'Export Workflow';
}

/* ── TAB 4: Models ── */
async function loadModels() {
  const tbody = document.getElementById('model-tbody');
  tbody.innerHTML = '<tr><td colspan="6" class="text-muted">Loading...</td></tr>';

  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    tbody.innerHTML = '';

    for (const m of data.models) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${m.id}</td>
        <td style="font-family: monospace; font-size: 0.75rem">${m.filename}</td>
        <td><span class="badge type">${m.type}</span></td>
        <td>${m.size_gb || '?'} GB</td>
        <td>${m.downloaded
          ? '<span class="badge yes">Downloaded</span>'
          : '<span class="badge no">Missing</span>'}</td>
        <td>${!m.downloaded
          ? `<button class="secondary" onclick="downloadModel('${m.id}', this)">Download</button>`
          : ''}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6" class="err">Error: ${e.message}</td></tr>`;
  }
}

async function downloadModel(modelId, btn) {
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const res = await fetch(`/api/models/download/${modelId}`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      btn.textContent = 'Done';
      setTimeout(() => loadModels(), 1000);
    } else {
      btn.textContent = 'Failed';
    }
  } catch (e) {
    btn.textContent = 'Error';
  }
}
</script>
</body>
</html>
"""
