"""ComfyForge Web Dashboard — FastAPI server."""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import Config
from core.models import ModelManager
from core.pipeline import Pipeline

app = FastAPI(title="ComfyForge")


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

    # Run in thread to not block
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


# ── Inline Dashboard HTML ────────────────

DASHBOARD_HTML = """
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
    --border: #1e1e30;
    --text: #e0e0f0;
    --muted: #6b6b8a;
    --accent: #7c5cfc;
    --accent2: #fc5c7c;
    --success: #3cda7c;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  .header {
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .header h1 {
    font-size: 1.3rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .tag {
    font-size: 0.65rem;
    padding: 2px 8px;
    border: 1px solid var(--accent);
    border-radius: 3px;
    color: var(--accent);
  }
  .main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    min-height: calc(100vh - 60px);
  }
  .panel {
    background: var(--bg);
    padding: 2rem;
    overflow-y: auto;
  }
  .panel h2 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 1.5rem;
  }
  textarea, input, select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.7rem;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.85rem;
    margin-bottom: 1rem;
  }
  textarea { height: 120px; resize: vertical; }
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
  button.primary {
    width: 100%;
    padding: 0.9rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    border: none;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: bold;
    cursor: pointer;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.5rem;
  }
  button.primary:hover { opacity: 0.9; }
  button.primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .log {
    font-size: 0.75rem;
    line-height: 1.6;
    white-space: pre-wrap;
    color: var(--muted);
  }
  .log .ok { color: var(--success); }
  .log .err { color: var(--accent2); }
  .log .info { color: var(--accent); }
  .output-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  .output-grid img, .output-grid video {
    width: 100%;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
</style>
</head>
<body>

<div class="header">
  <h1>ComfyForge</h1>
  <span class="tag">v0.1</span>
</div>

<div class="main">
  <div class="panel">
    <h2>Generate</h2>

    <label>Prompt</label>
    <textarea id="prompt" placeholder="A cinematic dragon soaring over Lahore at golden hour..."></textarea>

    <label>Negative</label>
    <input id="negative" value="blurry, low quality, watermark, text">

    <div class="row">
      <div>
        <label>Pipeline</label>
        <select id="pipeline">
          <option value="text2img">Text → Image</option>
          <option value="text2vid">Text → Image → Video</option>
          <option value="text2vid_wan">Text → Video (Wan)</option>
          <option value="full">Full (+ Audio)</option>
          <option value="img2vid">Image → Video</option>
        </select>
      </div>
      <div>
        <label>Video Backend</label>
        <select id="video_backend">
          <option value="svd">SVD</option>
          <option value="animatediff">AnimateDiff</option>
          <option value="wan">Wan 2.1</option>
        </select>
      </div>
    </div>

    <div class="row3">
      <div><label>Width</label><input id="width" type="number" value="1024"></div>
      <div><label>Height</label><input id="height" type="number" value="1024"></div>
      <div><label>Steps</label><input id="steps" type="number" value="25"></div>
    </div>

    <div class="row3">
      <div><label>CFG</label><input id="cfg" type="number" value="7" step="0.5"></div>
      <div><label>Seed</label><input id="seed" type="number" value="-1"></div>
      <div><label>Frames</label><input id="frames" type="number" value="24"></div>
    </div>

    <div class="row">
      <div>
        <label>
          <input type="checkbox" id="use_flux"> Use Flux
        </label>
      </div>
      <div>
        <label>Image Path (img2vid)</label>
        <input id="image_path" placeholder="path/to/image.png">
      </div>
    </div>

    <button class="primary" id="btn-generate" onclick="doGenerate()">Generate</button>
  </div>

  <div class="panel">
    <h2>Output</h2>
    <div id="log" class="log"></div>
    <div id="outputs" class="output-grid"></div>
  </div>
</div>

<script>
function log(msg, cls) {
  const el = document.getElementById('log');
  const line = document.createElement('div');
  line.className = cls || '';
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

async function doGenerate() {
  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  document.getElementById('log').innerHTML = '';
  document.getElementById('outputs').innerHTML = '';

  const body = {
    prompt: document.getElementById('prompt').value,
    negative: document.getElementById('negative').value,
    pipeline: document.getElementById('pipeline').value,
    video_backend: document.getElementById('video_backend').value,
    width: +document.getElementById('width').value,
    height: +document.getElementById('height').value,
    steps: +document.getElementById('steps').value || null,
    cfg: +document.getElementById('cfg').value || null,
    seed: +document.getElementById('seed').value,
    frames: +document.getElementById('frames').value || null,
    use_flux: document.getElementById('use_flux').checked,
    image_path: document.getElementById('image_path').value || null,
  };

  log('Starting pipeline: ' + body.pipeline, 'info');
  log('Prompt: ' + body.prompt);

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (data.success) {
      log('\\n✓ Pipeline complete!', 'ok');
      for (const [stage, time] of Object.entries(data.timings)) {
        log(`  ${stage}: ${time.toFixed(1)}s`, 'ok');
      }
      for (const [stage, files] of Object.entries(data.outputs)) {
        for (const f of files) {
          log(`  → ${f}`, 'ok');
        }
      }
    } else {
      log('\\n✗ Pipeline failed', 'err');
      data.errors.forEach(e => log('  ' + e, 'err'));
    }
  } catch (e) {
    log('Error: ' + e.message, 'err');
  }

  btn.disabled = false;
}
</script>
</body>
</html>
"""
