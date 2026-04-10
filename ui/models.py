"""Pydantic request models for the ComfyForge web API."""

from __future__ import annotations

from pydantic import BaseModel


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
