"""Async job runners — execute pipelines, clones, and downloads with progress streaming."""

from __future__ import annotations
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from core.config import Config
from core.comfy_client import ComfyClient
from core.models import ModelManager
from core.workflows import (
    text2img_sdxl, text2img_flux,
    img2vid_svd, img2vid_animatediff,
    text2vid_wan, img2vid_wan,
    save_workflow,
)

from .jobs import Job, JobManager, ProgressBridge, WebModelManager, job_manager
from .comfyui_ws import comfyui_ws_wait
from .models import GenerateRequest, CloneRequest


# ── Helpers ────────────────────────────────────────────────────────────


def _get_comfyui_outputs(cfg: Config, history_entry: dict) -> list[Path]:
    """Extract output file paths from a ComfyUI history entry."""
    outputs = []
    for node_id, node_output in history_entry.get("outputs", {}).items():
        for img in node_output.get("images", []):
            filename = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            p = cfg.comfyui_path / "output"
            if subfolder:
                p = p / subfolder
            p = p / filename
            if p.exists():
                outputs.append(p)
        for vid in node_output.get("gifs", []):
            filename = vid.get("filename", "")
            subfolder = vid.get("subfolder", "")
            p = cfg.comfyui_path / "output"
            if subfolder:
                p = p / subfolder
            p = p / filename
            if p.exists():
                outputs.append(p)
    return outputs


def _collect_to_output_dir(cfg: Config, pipeline_name: str, stage_outputs: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Copy outputs to timestamped run directory. Returns updated paths."""
    out = cfg.output_dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out / f"{pipeline_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[Path]] = {}
    for stage, files in stage_outputs.items():
        collected[stage] = []
        for f in files:
            dest = run_dir / f"{stage}_{f.name}"
            shutil.copy2(f, dest)
            collected[stage].append(dest)
    return collected


def _output_file_url(cfg: Config, file_path: Path) -> str:
    """Build a URL for serving an output file."""
    try:
        rel = file_path.relative_to(cfg.output_dir)
        return f"/api/files/outputs/{rel.as_posix()}"
    except ValueError:
        pass
    try:
        rel = file_path.relative_to(cfg.comfyui_path / "output")
        return f"/api/files/comfyui/{rel.as_posix()}"
    except ValueError:
        return f"/api/files/outputs/{file_path.name}"


def _file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".mp4", ".webm", ".avi", ".mov", ".gif"):
        return "video"
    return "image"


# ── Generate Job ───────────────────────────────────────────────────────


async def run_generate_job(job: Job, req: GenerateRequest):
    """Run the full generation pipeline with progress streaming."""
    cfg = Config.load()
    bridge = ProgressBridge(job_manager, job.id)
    bridge.set_loop(asyncio.get_event_loop())
    drain_task = asyncio.create_task(bridge.drain())
    loop = asyncio.get_event_loop()

    try:
        job.status = "running"
        bridge.push("job:stage", stage="init", status="started", message="Initializing pipeline...")

        # Ensure ComfyUI is running
        comfy = ComfyClient(cfg)
        bridge.push("job:stage", stage="comfyui", status="checking", message="Checking ComfyUI...")
        running = await loop.run_in_executor(None, comfy.ensure_running)
        if not running:
            raise RuntimeError("ComfyUI is not available. Start it or set auto_start: true in config.")

        # Build overrides
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

        stage_outputs: dict[str, list[Path]] = {}
        timings: dict[str, float] = {}
        pipeline = req.pipeline

        # Helper: run a single ComfyUI workflow with progress
        async def _execute_workflow(wf: dict, stage: str, wf_timeout: int = 600):
            bridge.push("job:stage", stage=stage, status="queuing", message=f"Queuing {stage} workflow...")
            prompt_id = await loop.run_in_executor(None, lambda: comfy.queue(wf))

            bridge.push("job:stage", stage=stage, status="executing", message=f"Executing {stage} in ComfyUI...")
            history_entry = await comfyui_ws_wait(cfg.comfyui_url, comfy.client_id, prompt_id, bridge, timeout=wf_timeout)

            outputs = _get_comfyui_outputs(cfg, history_entry)
            return outputs

        # ── Stage: text2img ──
        if pipeline in ("text2img", "text2vid", "full"):
            stage = "text2img"
            t0 = time.time()

            stack_name = req.model_stack or "text2img"
            use_flux = "flux" in stack_name.lower() or overrides.get("use_flux", False)
            if use_flux:
                stack_name = "text2img_flux"

            bridge.push("job:stage", stage=stage, status="models", message="Downloading models...")
            mm = WebModelManager(cfg, bridge)
            await loop.run_in_executor(None, lambda: mm.ensure_stack(stack_name))

            # Also download any override models
            override_model_keys = {"checkpoint": "checkpoint", "vae": "vae"}
            for key, mtype in override_model_keys.items():
                if fname := overrides.get(key):
                    if isinstance(fname, str) and fname.endswith((".safetensors", ".ckpt", ".pth")):
                        await loop.run_in_executor(None, lambda f=fname, t=mtype: mm.ensure_or_search(f, t))

            bridge.push("job:stage", stage=stage, status="building", message="Building workflow...")
            stack_cfg = cfg.stack(stack_name)
            defaults = cfg.default("image")

            if use_flux:
                wf = text2img_flux(
                    prompt=req.prompt,
                    checkpoint=overrides.get("checkpoint", stack_cfg.get("checkpoint", "flux1-dev.safetensors")),
                    clip_files=stack_cfg.get("clip"),
                    vae=overrides.get("vae", stack_cfg.get("vae", "ae.safetensors")),
                    width=overrides.get("width", defaults.get("width", 1024)),
                    height=overrides.get("height", defaults.get("height", 1024)),
                    steps=overrides.get("steps", defaults.get("steps", 20)),
                    guidance=overrides.get("guidance", 3.5),
                    seed=overrides.get("seed", -1),
                )
            else:
                wf = text2img_sdxl(
                    prompt=req.prompt,
                    negative=req.negative,
                    checkpoint=overrides.get("checkpoint", stack_cfg.get("checkpoint", "sd_xl_base_1.0.safetensors")),
                    vae=overrides.get("vae", stack_cfg.get("vae")),
                    loras=overrides.get("loras", stack_cfg.get("loras", [])),
                    width=overrides.get("width", defaults.get("width", 1024)),
                    height=overrides.get("height", defaults.get("height", 1024)),
                    steps=overrides.get("steps", defaults.get("steps", 25)),
                    cfg=overrides.get("cfg", defaults.get("cfg", 7.0)),
                    sampler=overrides.get("sampler", defaults.get("sampler", "euler_ancestral")),
                    scheduler=overrides.get("scheduler", defaults.get("scheduler", "normal")),
                    seed=overrides.get("seed", -1),
                    batch_size=overrides.get("batch_size", defaults.get("batch_size", 1)),
                )

            save_workflow(wf, cfg.temp_dir / f"{stage}_workflow.json")
            outputs = await _execute_workflow(wf, stage)
            stage_outputs[stage] = outputs
            timings[stage] = time.time() - t0

            file_list = [{"url": _output_file_url(cfg, p), "type": _file_type(p), "name": p.name} for p in outputs]
            bridge.push("job:output", stage=stage, files=file_list)

        # ── Stage: img2vid ──
        if pipeline in ("img2vid", "text2vid", "full"):
            stage = "img2vid"
            t0 = time.time()

            if pipeline == "img2vid":
                img_path = req.image_path
                if not img_path:
                    raise RuntimeError("img2vid requires image_path")
            else:
                t2i_outputs = stage_outputs.get("text2img", [])
                if not t2i_outputs:
                    raise RuntimeError("text2img stage failed, cannot continue to img2vid")
                img_path = str(t2i_outputs[0])

            backend = overrides.get("video_backend", "svd")
            vid_stack_name = f"img2vid_{backend}" if backend != "svd" else "img2vid"

            bridge.push("job:stage", stage=stage, status="models", message=f"Downloading {backend} models...")
            mm = WebModelManager(cfg, bridge)
            await loop.run_in_executor(None, lambda: mm.ensure_stack(vid_stack_name))

            bridge.push("job:stage", stage=stage, status="building", message="Building video workflow...")
            stack_cfg = cfg.stack(vid_stack_name)
            vid_defaults = cfg.default("video")

            if backend == "animatediff":
                wf = img2vid_animatediff(
                    prompt=req.prompt, image_path=img_path, negative=req.negative,
                    checkpoint=stack_cfg.get("checkpoint", "sd_xl_base_1.0.safetensors"),
                    motion_module=stack_cfg.get("motion_module", "mm_sdxl_v10_beta.safetensors"),
                    frames=overrides.get("frames", vid_defaults.get("frames", 16)),
                    fps=overrides.get("fps", vid_defaults.get("fps", 8)),
                    steps=overrides.get("video_steps", 20),
                    cfg=overrides.get("video_cfg", 7.0),
                    seed=overrides.get("seed", -1),
                )
            elif backend == "wan":
                wf = img2vid_wan(
                    prompt=req.prompt, image_path=img_path, negative=req.negative,
                    checkpoint=stack_cfg.get("checkpoint"),
                    frames=overrides.get("frames", 81),
                    steps=overrides.get("video_steps", 30),
                    cfg=overrides.get("video_cfg", 6.0),
                    seed=overrides.get("seed", -1),
                )
            else:
                wf = img2vid_svd(
                    image_path=img_path,
                    model=stack_cfg.get("model", "svd_xt_1_1.safetensors"),
                    frames=overrides.get("frames", vid_defaults.get("frames", 25)),
                    fps=overrides.get("fps", vid_defaults.get("fps", 8)),
                    steps=overrides.get("video_steps", 20),
                    cfg=overrides.get("video_cfg", 2.5),
                    seed=overrides.get("seed", -1),
                )

            save_workflow(wf, cfg.temp_dir / f"{stage}_workflow.json")
            outputs = await _execute_workflow(wf, stage, wf_timeout=900)
            stage_outputs[stage] = outputs
            timings[stage] = time.time() - t0

            file_list = [{"url": _output_file_url(cfg, p), "type": _file_type(p), "name": p.name} for p in outputs]
            bridge.push("job:output", stage=stage, files=file_list)

        # ── Stage: text2vid_wan (direct) ──
        if pipeline == "text2vid_wan":
            stage = "text2vid_wan"
            t0 = time.time()

            bridge.push("job:stage", stage=stage, status="models", message="Downloading Wan 2.1 models...")
            mm = WebModelManager(cfg, bridge)
            await loop.run_in_executor(None, lambda: mm.ensure_stack("text2vid_wan"))

            bridge.push("job:stage", stage=stage, status="building", message="Building text2vid workflow...")
            stack_cfg = cfg.stack("text2vid_wan")

            wf = text2vid_wan(
                prompt=req.prompt, negative=req.negative,
                checkpoint=stack_cfg.get("checkpoint"),
                width=overrides.get("width", 832),
                height=overrides.get("height", 480),
                frames=overrides.get("frames", 81),
                steps=overrides.get("steps", 30),
                cfg=overrides.get("cfg", 6.0),
                seed=overrides.get("seed", -1),
            )

            save_workflow(wf, cfg.temp_dir / f"{stage}_workflow.json")
            outputs = await _execute_workflow(wf, stage, wf_timeout=1200)
            stage_outputs[stage] = outputs
            timings[stage] = time.time() - t0

            file_list = [{"url": _output_file_url(cfg, p), "type": _file_type(p), "name": p.name} for p in outputs]
            bridge.push("job:output", stage=stage, files=file_list)

        # Collect outputs to output directory
        collected = _collect_to_output_dir(cfg, pipeline, stage_outputs)
        all_output_urls = []
        for stage_name, files in collected.items():
            for f in files:
                all_output_urls.append({"url": _output_file_url(cfg, f), "type": _file_type(f), "name": f.name})

        job.status = "complete"
        job.result = {"success": True, "timings": timings, "outputs": all_output_urls}
        bridge.push("job:complete", success=True, timings=timings, outputs=all_output_urls, errors=[])

    except Exception as e:
        job.status = "error"
        job.result = {"success": False, "error": str(e)}
        bridge.push("job:error", error=str(e))
    finally:
        bridge.close()
        await drain_task


# ── Clone Job ──────────────────────────────────────────────────────────


async def run_clone_job(job: Job, req: CloneRequest):
    """Run CivitAI clone with progress streaming."""
    from core.civitai import (
        parse_civitai_url, fetch_image_metadata, _fetch_post_images,
        resolve_and_download_models, generate_clone_workflow,
    )

    cfg = Config.load()
    bridge = ProgressBridge(job_manager, job.id)
    bridge.set_loop(asyncio.get_event_loop())
    drain_task = asyncio.create_task(bridge.drain())
    loop = asyncio.get_event_loop()

    try:
        job.status = "running"

        # 1. Parse URL
        bridge.push("job:stage", stage="parse", status="started", message="Parsing CivitAI URL...")
        content_type, content_id = parse_civitai_url(req.url)

        # 2. Fetch metadata
        bridge.push("job:stage", stage="metadata", status="started", message="Fetching image metadata...")
        if content_type == "post":
            images = await loop.run_in_executor(None, lambda: _fetch_post_images(content_id, cfg))
            image_data = None
            for img in images:
                if img.get("meta"):
                    image_data = img
                    break
            if not image_data:
                raise RuntimeError("No images with generation metadata in this post")
            image_id = image_data["id"]
        else:
            image_id = content_id

        meta = await loop.run_in_executor(None, lambda: fetch_image_metadata(image_id, cfg))
        bridge.push("job:stage", stage="metadata", status="done",
                     message=f"Found: {meta.checkpoint_name or 'unknown'} ({meta.base_model})")

        # 3. Download models
        if not req.no_download:
            bridge.push("job:stage", stage="models", status="started", message="Downloading models...")
            mm = WebModelManager(cfg, bridge)
            model_filenames = await loop.run_in_executor(
                None, lambda: resolve_and_download_models(meta, mm, no_download=False, config=cfg)
            )
        else:
            bridge.push("job:stage", stage="models", status="skipped", message="Model downloads skipped")
            mm = ModelManager(cfg)
            model_filenames = await loop.run_in_executor(
                None, lambda: resolve_and_download_models(meta, mm, no_download=True, config=cfg)
            )

        # 4. Generate workflow
        bridge.push("job:stage", stage="workflow", status="started", message="Generating workflow...")
        wf = await loop.run_in_executor(
            None, lambda: generate_clone_workflow(meta, model_filenames, cfg)
        )

        # 5. Save workflow
        out_dir = Path(req.output_path).parent if req.output_path else Path("exported_workflows")
        out_dir.mkdir(parents=True, exist_ok=True)

        if req.output_path:
            dest = Path(req.output_path)
        else:
            safe_name = (meta.checkpoint_name or "clone").replace(" ", "_")[:40]
            dest = out_dir / f"clone_{safe_name}_{image_id}.json"

        dest.write_text(json.dumps(wf, indent=2))

        workflow_filename = dest.name
        bridge.push("job:output", stage="workflow",
                     files=[{"url": f"/api/files/workflows/{workflow_filename}", "type": "workflow", "name": workflow_filename}])

        job.status = "complete"
        job.result = {"success": True, "workflow_path": str(dest), "workflow_filename": workflow_filename}
        bridge.push("job:complete", success=True, workflow_path=str(dest), workflow_filename=workflow_filename, errors=[])

    except Exception as e:
        job.status = "error"
        job.result = {"success": False, "error": str(e)}
        bridge.push("job:error", error=str(e))
    finally:
        bridge.close()
        await drain_task


# ── Model Download Job ─────────────────────────────────────────────────


async def run_model_download_job(job: Job, model_id: str):
    """Download a single model with progress streaming."""
    cfg = Config.load()
    bridge = ProgressBridge(job_manager, job.id)
    bridge.set_loop(asyncio.get_event_loop())
    drain_task = asyncio.create_task(bridge.drain())
    loop = asyncio.get_event_loop()

    try:
        job.status = "running"
        bridge.push("job:stage", stage="download", status="started", message=f"Downloading {model_id}...")

        mm = WebModelManager(cfg, bridge)
        path = await loop.run_in_executor(None, lambda: mm.ensure_model(model_id))

        job.status = "complete"
        job.result = {"success": True, "path": str(path)}
        bridge.push("job:complete", success=True, path=str(path), errors=[])

    except Exception as e:
        job.status = "error"
        job.result = {"success": False, "error": str(e)}
        bridge.push("job:error", error=str(e))
    finally:
        bridge.close()
        await drain_task
