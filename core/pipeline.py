"""Pipeline Orchestrator — chains generation stages and handles the full flow."""

from __future__ import annotations
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comfy_client import ComfyClient
from .config import Config
from .models import ModelManager
from .workflows import (
    WORKFLOW_MAP,
    img2vid_animatediff,
    img2vid_svd,
    img2vid_wan,
    save_workflow,
    text2img_flux,
    text2img_sdxl,
    text2vid_wan,
)

console = Console()


@dataclass
class PipelineResult:
    pipeline: str
    stages: list[str] = field(default_factory=list)
    outputs: dict[str, list[Path]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def all_files(self) -> list[Path]:
        files = []
        for v in self.outputs.values():
            files.extend(v)
        return files

    def summary(self):
        table = Table(title=f"Pipeline: {self.pipeline}")
        table.add_column("Stage")
        table.add_column("Time")
        table.add_column("Outputs")
        table.add_column("Status")

        for stage in self.stages:
            t = self.timings.get(stage, 0)
            outs = self.outputs.get(stage, [])
            status = "[green]✓[/]" if outs else "[red]✗[/]"
            table.add_row(stage, f"{t:.1f}s", str(len(outs)), status)

        console.print(table)
        if self.errors:
            for err in self.errors:
                console.print(f"  [red]✗ {err}[/]")


class Pipeline:
    """Orchestrates multi-stage generation pipelines."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.models = ModelManager(self.cfg)
        self.comfy = ComfyClient(self.cfg)

    def run(
        self,
        prompt: str,
        pipeline: str = "full",
        negative: str = "blurry, low quality, watermark, text",
        image_path: str | None = None,
        model_stack: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run a named pipeline.

        pipeline options:
          - text2img       : prompt → image
          - img2vid        : image → video (needs image_path)
          - text2vid       : prompt → image → video
          - full           : prompt → image → video → audio
          - text2vid_wan   : prompt → video (Wan2.1 direct)
        """
        result = PipelineResult(pipeline=pipeline)
        overrides = overrides or {}

        console.print(Panel(f"[bold]{pipeline}[/]\n{prompt[:80]}…", title="ComfyForge"))

        # Ensure ComfyUI is running
        if not self.comfy.ensure_running():
            result.errors.append("ComfyUI not available")
            return result

        try:
            if pipeline == "text2img":
                self._run_text2img(prompt, negative, model_stack, overrides, result)

            elif pipeline == "img2vid":
                if not image_path:
                    result.errors.append("img2vid requires image_path")
                    return result
                self._run_img2vid(prompt, negative, image_path, model_stack, overrides, result)

            elif pipeline == "text2vid":
                # Stage 1: text → image
                self._run_text2img(prompt, negative, model_stack, overrides, result)
                if not result.outputs.get("text2img"):
                    result.errors.append("text2img stage failed, cannot continue")
                    return result
                # Stage 2: image → video
                img = str(result.outputs["text2img"][0])
                self._run_img2vid(prompt, negative, img, model_stack, overrides, result)

            elif pipeline == "text2vid_wan":
                self._run_text2vid_wan(prompt, negative, overrides, result)

            elif pipeline == "full":
                # Stage 1: text → image
                self._run_text2img(prompt, negative, model_stack, overrides, result)
                if not result.outputs.get("text2img"):
                    result.errors.append("text2img failed")
                    return result
                # Stage 2: image → video
                img = str(result.outputs["text2img"][0])
                self._run_img2vid(prompt, negative, img, model_stack, overrides, result)
                # Stage 3: audio (placeholder — uses external engine)
                if result.outputs.get("img2vid"):
                    self._run_audio(prompt, result.outputs["img2vid"][0], result)

            else:
                result.errors.append(f"Unknown pipeline: {pipeline}")

        except Exception as e:
            result.errors.append(str(e))
            console.print(f"[red]Pipeline error: {e}[/]")

        # Copy outputs to output dir
        self._collect_outputs(result)
        result.summary()
        return result

    # ── stage runners ────────────────────

    def _ensure_override_models(self, overrides: dict):
        """Download any custom model files specified in overrides."""
        override_model_keys = {
            "checkpoint": "checkpoint",
            "vae": "vae",
            "motion_module": "motion",
        }
        for key, model_type in override_model_keys.items():
            if filename := overrides.get(key):
                if isinstance(filename, str) and filename.endswith((".safetensors", ".ckpt", ".pth")):
                    self.models.ensure_or_search(filename, model_type)

        # Handle lora overrides
        if loras := overrides.get("loras"):
            for lora in (loras if isinstance(loras, list) else [loras]):
                name = lora if isinstance(lora, str) else lora.get("filename", "")
                if name:
                    self.models.ensure_or_search(name, "lora")

    def _run_text2img(
        self, prompt: str, negative: str, stack: str | None,
        overrides: dict, result: PipelineResult,
    ):
        stage = "text2img"
        result.stages.append(stage)
        console.print(f"\n[bold cyan]Stage: {stage}[/]")

        stack_name = stack or "text2img"
        # Determine if flux
        use_flux = "flux" in stack_name.lower() or overrides.get("use_flux", False)
        if use_flux:
            stack_name = "text2img_flux"

        # Download models
        console.print("  Ensuring models…")
        self.models.ensure_stack(stack_name)
        self._ensure_override_models(overrides)

        # Build workflow
        stack_cfg = self.cfg.stack(stack_name)
        defaults = self.cfg.default("image")

        t0 = time.time()
        if use_flux:
            wf = text2img_flux(
                prompt=prompt,
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
                prompt=prompt,
                negative=negative,
                checkpoint=overrides.get("checkpoint", stack_cfg.get("checkpoint", "sd_xl_base_1.0.safetensors")),
                vae=overrides.get("vae", stack_cfg.get("vae", "sdxl_vae.safetensors")),
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

        # Save workflow for debugging
        save_workflow(wf, self.cfg.temp_dir / f"{stage}_workflow.json")

        # Execute
        outputs = self.comfy.run_workflow(wf)
        result.outputs[stage] = outputs
        result.timings[stage] = time.time() - t0

    def _run_img2vid(
        self, prompt: str, negative: str, image_path: str,
        stack: str | None, overrides: dict, result: PipelineResult,
    ):
        stage = "img2vid"
        result.stages.append(stage)
        console.print(f"\n[bold cyan]Stage: {stage}[/]")

        # Pick video backend
        backend = overrides.get("video_backend", "svd")
        stack_name = f"img2vid_{backend}" if backend != "svd" else "img2vid"

        console.print(f"  Backend: {backend}")
        self.models.ensure_stack(stack_name)
        self._ensure_override_models(overrides)

        stack_cfg = self.cfg.stack(stack_name)
        vid_defaults = self.cfg.default("video")

        t0 = time.time()
        if backend == "animatediff":
            wf = img2vid_animatediff(
                prompt=prompt,
                image_path=image_path,
                negative=negative,
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
                prompt=prompt,
                image_path=image_path,
                negative=negative,
                checkpoint=stack_cfg.get("checkpoint"),
                frames=overrides.get("frames", 81),
                steps=overrides.get("video_steps", 30),
                cfg=overrides.get("video_cfg", 6.0),
                seed=overrides.get("seed", -1),
            )
        else:  # svd
            wf = img2vid_svd(
                image_path=image_path,
                model=stack_cfg.get("model", "svd_xt_1_1.safetensors"),
                frames=overrides.get("frames", vid_defaults.get("frames", 25)),
                fps=overrides.get("fps", vid_defaults.get("fps", 8)),
                steps=overrides.get("video_steps", 20),
                cfg=overrides.get("video_cfg", 2.5),
                seed=overrides.get("seed", -1),
            )

        save_workflow(wf, self.cfg.temp_dir / f"{stage}_workflow.json")
        outputs = self.comfy.run_workflow(wf, timeout=900)
        result.outputs[stage] = outputs
        result.timings[stage] = time.time() - t0

    def _run_text2vid_wan(
        self, prompt: str, negative: str, overrides: dict, result: PipelineResult,
    ):
        stage = "text2vid_wan"
        result.stages.append(stage)
        console.print(f"\n[bold cyan]Stage: {stage}[/]")

        self.models.ensure_stack("text2vid_wan")
        self._ensure_override_models(overrides)
        stack_cfg = self.cfg.stack("text2vid_wan")

        t0 = time.time()
        wf = text2vid_wan(
            prompt=prompt,
            negative=negative,
            checkpoint=stack_cfg.get("checkpoint"),
            width=overrides.get("width", 832),
            height=overrides.get("height", 480),
            frames=overrides.get("frames", 81),
            steps=overrides.get("steps", 30),
            cfg=overrides.get("cfg", 6.0),
            seed=overrides.get("seed", -1),
        )

        save_workflow(wf, self.cfg.temp_dir / f"{stage}_workflow.json")
        outputs = self.comfy.run_workflow(wf, timeout=1200)
        result.outputs[stage] = outputs
        result.timings[stage] = time.time() - t0

    def _run_audio(self, prompt: str, video_path: Path, result: PipelineResult):
        """Generate audio for a video. Uses Bark/F5-TTS via subprocess."""
        stage = "audio"
        result.stages.append(stage)
        console.print(f"\n[bold cyan]Stage: {stage}[/]")

        audio_cfg = self.cfg.default("audio")
        engine = audio_cfg.get("engine", "bark")

        t0 = time.time()
        try:
            if engine == "bark":
                output = self._run_bark_audio(prompt, video_path)
            elif engine == "f5tts":
                output = self._run_f5tts_audio(prompt, video_path)
            elif engine == "kokoro":
                output = self._run_kokoro_audio(prompt, video_path)
            else:
                console.print(f"  [yellow]⚠ Unknown audio engine: {engine}[/]")
                return

            if output and output.exists():
                result.outputs[stage] = [output]
        except Exception as e:
            console.print(f"  [yellow]⚠ Audio generation failed: {e}[/]")
            result.errors.append(f"Audio: {e}")

        result.timings[stage] = time.time() - t0

    def _run_bark_audio(self, text: str, video_path: Path) -> Path | None:
        """Generate speech with Bark and merge into video."""
        import subprocess

        audio_out = video_path.with_suffix(".wav")
        final_out = video_path.with_name(video_path.stem + "_with_audio.mp4")

        # Generate audio via bark CLI
        script = f"""
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile

preload_models()
audio = generate_audio("{text[:500]}")
wavfile.write("{audio_out}", SAMPLE_RATE, audio)
"""
        script_path = self.cfg.temp_dir / "bark_gen.py"
        script_path.write_text(script)

        subprocess.run(["python", str(script_path)], check=True, timeout=300)

        # Merge with ffmpeg
        if audio_out.exists():
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_out),
                "-c:v", "copy", "-c:a", "aac",
                "-shortest",
                str(final_out),
            ], check=True, timeout=120)
            return final_out
        return None

    def _run_f5tts_audio(self, text: str, video_path: Path) -> Path | None:
        """Generate speech with F5-TTS."""
        import subprocess
        audio_out = video_path.with_suffix(".wav")
        final_out = video_path.with_name(video_path.stem + "_with_audio.mp4")

        subprocess.run([
            "f5-tts_infer-cli",
            "--model", "F5-TTS",
            "--gen_text", text[:500],
            "--output_dir", str(audio_out.parent),
            "--output_file", audio_out.name,
        ], check=True, timeout=300)

        if audio_out.exists():
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_out),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(final_out),
            ], check=True, timeout=120)
            return final_out
        return None

    def _run_kokoro_audio(self, text: str, video_path: Path) -> Path | None:
        """Generate speech with Kokoro TTS."""
        import subprocess
        audio_out = video_path.with_suffix(".wav")
        final_out = video_path.with_name(video_path.stem + "_with_audio.mp4")

        script = f"""
from kokoro import KPipeline
pipe = KPipeline(lang_code='a')
audio, sr = pipe("{text[:500]}")
import scipy.io.wavfile as wavfile
wavfile.write("{audio_out}", sr, audio)
"""
        script_path = self.cfg.temp_dir / "kokoro_gen.py"
        script_path.write_text(script)
        subprocess.run(["python", str(script_path)], check=True, timeout=300)

        if audio_out.exists():
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_out),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(final_out),
            ], check=True, timeout=120)
            return final_out
        return None

    # ── output collection ────────────────

    def _collect_outputs(self, result: PipelineResult):
        """Copy all outputs to the configured output directory."""
        out = self.cfg.output_dir
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = out / f"{result.pipeline}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        collected: dict[str, list[Path]] = {}
        for stage, files in result.outputs.items():
            collected[stage] = []
            for f in files:
                dest = run_dir / f"{stage}_{f.name}"
                shutil.copy2(f, dest)
                collected[stage].append(dest)
                console.print(f"  [green]→ {dest}[/]")

        result.outputs = collected
