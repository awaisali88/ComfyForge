"""Workflow Generator — builds ComfyUI API-format JSON workflows programmatically."""

from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any

from .config import Config


class Node:
    """Single ComfyUI node."""

    def __init__(self, node_id: str, class_type: str, inputs: dict[str, Any] | None = None):
        self.id = node_id
        self.class_type = class_type
        self.inputs = inputs or {}

    def link(self, input_name: str, source_node: "Node", output_index: int = 0):
        """Link an output from another node into this node's input."""
        self.inputs[input_name] = [source_node.id, output_index]
        return self

    def to_dict(self) -> dict:
        return {
            "class_type": self.class_type,
            "inputs": self.inputs,
        }


class WorkflowBuilder:
    """Fluent builder for ComfyUI workflows."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._counter = 0

    def add(self, class_type: str, inputs: dict[str, Any] | None = None, node_id: str | None = None) -> Node:
        self._counter += 1
        nid = node_id or str(self._counter)
        n = Node(nid, class_type, inputs or {})
        self._nodes[nid] = n
        return n

    def to_json(self) -> str:
        return json.dumps(
            {nid: n.to_dict() for nid, n in self._nodes.items()},
            indent=2,
        )

    def to_dict(self) -> dict:
        return {nid: n.to_dict() for nid, n in self._nodes.items()}


# ── Pre-built workflow factories ─────────────


def text2img_sdxl(
    prompt: str,
    negative: str = "blurry, low quality, watermark, text",
    checkpoint: str = "sd_xl_base_1.0.safetensors",
    vae: str = "sdxl_vae.safetensors",
    loras: list[dict] | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg: float = 7.0,
    sampler: str = "euler_ancestral",
    scheduler: str = "normal",
    seed: int = -1,
    batch_size: int = 1,
) -> dict:
    """SDXL text-to-image workflow."""
    b = WorkflowBuilder()

    # Checkpoint loader
    ckpt = b.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    # Optional VAE
    vae_node = b.add("VAELoader", {"vae_name": vae})

    # Current model/clip outputs
    model_out = ckpt
    model_idx = 0
    clip_out = ckpt
    clip_idx = 1

    # LoRA chain
    for lora_info in (loras or []):
        lora_name = lora_info if isinstance(lora_info, str) else lora_info["filename"]
        strength = 0.8 if isinstance(lora_info, str) else lora_info.get("strength", 0.8)
        lora_node = b.add("LoraLoader", {
            "lora_name": lora_name,
            "strength_model": strength,
            "strength_clip": strength,
        })
        lora_node.link("model", model_out, model_idx)
        lora_node.link("clip", clip_out, clip_idx)
        model_out = lora_node
        model_idx = 0
        clip_out = lora_node
        clip_idx = 1

    # CLIP encode
    pos = b.add("CLIPTextEncode", {"text": prompt})
    pos.link("clip", clip_out, clip_idx)

    neg = b.add("CLIPTextEncode", {"text": negative})
    neg.link("clip", clip_out, clip_idx)

    # Empty latent
    latent = b.add("EmptyLatentImage", {
        "width": width,
        "height": height,
        "batch_size": batch_size,
    })

    # KSampler
    sampler_node = b.add("KSampler", {
        "seed": seed if seed >= 0 else _random_seed(),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0,
    })
    sampler_node.link("model", model_out, model_idx)
    sampler_node.link("positive", pos)
    sampler_node.link("negative", neg)
    sampler_node.link("latent_image", latent)

    # VAE decode
    decode = b.add("VAEDecode")
    decode.link("samples", sampler_node)
    decode.link("vae", vae_node)

    # Save
    save = b.add("SaveImage", {"filename_prefix": "comfyforge"})
    save.link("images", decode)

    return b.to_dict()


def text2img_flux(
    prompt: str,
    checkpoint: str = "flux1-dev.safetensors",
    clip_files: list[str] | None = None,
    vae: str = "ae.safetensors",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 3.5,
    seed: int = -1,
) -> dict:
    """Flux text-to-image workflow."""
    b = WorkflowBuilder()
    clip_files = clip_files or ["t5xxl_fp16.safetensors", "clip_l.safetensors"]

    unet = b.add("UNETLoader", {"unet_name": checkpoint, "weight_dtype": "fp8_e4m3fn"})

    clip = b.add("DualCLIPLoader", {
        "clip_name1": clip_files[0],
        "clip_name2": clip_files[1] if len(clip_files) > 1 else clip_files[0],
        "type": "flux",
    })

    vae_node = b.add("VAELoader", {"vae_name": vae})

    cond = b.add("CLIPTextEncode", {"text": prompt})
    cond.link("clip", clip)

    # Flux uses guidance node instead of CFG in sampler
    guidance_node = b.add("FluxGuidance", {"guidance": guidance})
    guidance_node.link("conditioning", cond)

    latent = b.add("EmptySD3LatentImage", {
        "width": width,
        "height": height,
        "batch_size": 1,
    })

    sampler_node = b.add("KSampler", {
        "seed": seed if seed >= 0 else _random_seed(),
        "steps": steps,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
    })
    sampler_node.link("model", unet)
    sampler_node.link("positive", guidance_node)
    sampler_node.link("negative", b.add("ConditioningZeroOut").link("conditioning", cond))
    sampler_node.link("latent_image", latent)

    decode = b.add("VAEDecode")
    decode.link("samples", sampler_node)
    decode.link("vae", vae_node)

    save = b.add("SaveImage", {"filename_prefix": "comfyforge_flux"})
    save.link("images", decode)

    return b.to_dict()


def img2vid_svd(
    image_path: str,
    model: str = "svd_xt_1_1.safetensors",
    frames: int = 25,
    fps: int = 8,
    motion_bucket: int = 127,
    augmentation_level: float = 0.0,
    steps: int = 20,
    cfg: float = 2.5,
    seed: int = -1,
) -> dict:
    """SVD image-to-video workflow."""
    b = WorkflowBuilder()

    ckpt = b.add("ImageOnlyCheckpointLoader", {"ckpt_name": model})

    load_img = b.add("LoadImage", {"image": image_path})

    cond = b.add("SVD_img2vid_Conditioning", {
        "width": 1024,
        "height": 576,
        "video_frames": frames,
        "motion_bucket_id": motion_bucket,
        "fps": fps,
        "augmentation_level": augmentation_level,
    })
    cond.link("clip_vision", ckpt, 1)
    cond.link("init_image", load_img)
    cond.link("vae", ckpt, 2)

    sampler_node = b.add("KSampler", {
        "seed": seed if seed >= 0 else _random_seed(),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "karras",
        "denoise": 1.0,
    })
    sampler_node.link("model", ckpt, 0)
    sampler_node.link("positive", cond, 0)
    sampler_node.link("negative", cond, 1)
    sampler_node.link("latent_image", cond, 2)

    decode = b.add("VAEDecode")
    decode.link("samples", sampler_node)
    decode.link("vae", ckpt, 2)

    combine = b.add("VHS_VideoCombine", {
        "frame_rate": fps,
        "loop_count": 0,
        "filename_prefix": "comfyforge_svd",
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
    })
    combine.link("images", decode)

    return b.to_dict()


def img2vid_animatediff(
    prompt: str,
    image_path: str,
    negative: str = "blurry, low quality, static",
    checkpoint: str = "sd_xl_base_1.0.safetensors",
    motion_module: str = "mm_sdxl_v10_beta.safetensors",
    frames: int = 16,
    fps: int = 8,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
) -> dict:
    """AnimateDiff image-to-video workflow."""
    b = WorkflowBuilder()

    ckpt = b.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    motion = b.add("ADE_LoadAnimateDiffModel", {"model_name": motion_module})

    apply_motion = b.add("ADE_ApplyAnimateDiffModelSimple", {"motion_scale": 1.0})
    apply_motion.link("motion_model", motion)
    apply_motion.link("model", ckpt)

    pos = b.add("CLIPTextEncode", {"text": prompt})
    pos.link("clip", ckpt, 1)

    neg = b.add("CLIPTextEncode", {"text": negative})
    neg.link("clip", ckpt, 1)

    load_img = b.add("LoadImage", {"image": image_path})

    # Encode image to latent for init
    vae_encode = b.add("VAEEncode")
    vae_encode.link("pixels", load_img)
    vae_encode.link("vae", ckpt, 2)

    # Repeat latent for batch frames
    repeat = b.add("RepeatLatentBatch", {"amount": frames})
    repeat.link("samples", vae_encode)

    sampler_node = b.add("KSampler", {
        "seed": seed if seed >= 0 else _random_seed(),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "denoise": 0.7,
    })
    sampler_node.link("model", apply_motion)
    sampler_node.link("positive", pos)
    sampler_node.link("negative", neg)
    sampler_node.link("latent_image", repeat)

    decode = b.add("VAEDecode")
    decode.link("samples", sampler_node)
    decode.link("vae", ckpt, 2)

    combine = b.add("VHS_VideoCombine", {
        "frame_rate": fps,
        "loop_count": 0,
        "filename_prefix": "comfyforge_animdiff",
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
    })
    combine.link("images", decode)

    return b.to_dict()


def text2vid_wan(
    prompt: str,
    negative: str = "blurry, low quality, watermark",
    checkpoint: str = "wan2.1_t2v_14B_fp16.safetensors",
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    steps: int = 30,
    cfg: float = 6.0,
    seed: int = -1,
) -> dict:
    """Wan 2.1 text-to-video workflow."""
    b = WorkflowBuilder()

    ckpt = b.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    pos = b.add("CLIPTextEncode", {"text": prompt})
    pos.link("clip", ckpt, 1)

    neg = b.add("CLIPTextEncode", {"text": negative})
    neg.link("clip", ckpt, 1)

    latent = b.add("EmptyLatentImage", {
        "width": width,
        "height": height,
        "batch_size": frames,
    })

    sampler_node = b.add("KSampler", {
        "seed": seed if seed >= 0 else _random_seed(),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
    })
    sampler_node.link("model", ckpt, 0)
    sampler_node.link("positive", pos)
    sampler_node.link("negative", neg)
    sampler_node.link("latent_image", latent)

    decode = b.add("VAEDecode")
    decode.link("samples", sampler_node)
    decode.link("vae", ckpt, 2)

    combine = b.add("VHS_VideoCombine", {
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "comfyforge_wan_t2v",
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
    })
    combine.link("images", decode)

    return b.to_dict()


def img2vid_wan(
    prompt: str,
    image_path: str,
    negative: str = "blurry, low quality",
    checkpoint: str = "wan2.1_i2v_480p_14B_fp16.safetensors",
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    steps: int = 30,
    cfg: float = 6.0,
    seed: int = -1,
) -> dict:
    """Wan 2.1 image-to-video workflow."""
    b = WorkflowBuilder()

    ckpt = b.add("CheckpointLoaderSimple", {"ckpt_name": checkpoint})

    load_img = b.add("LoadImage", {"image": image_path})

    pos = b.add("CLIPTextEncode", {"text": prompt})
    pos.link("clip", ckpt, 1)

    neg = b.add("CLIPTextEncode", {"text": negative})
    neg.link("clip", ckpt, 1)

    # Encode init image
    vae_encode = b.add("VAEEncode")
    vae_encode.link("pixels", load_img)
    vae_encode.link("vae", ckpt, 2)

    sampler_node = b.add("KSampler", {
        "seed": seed if seed >= 0 else _random_seed(),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 0.85,
    })
    sampler_node.link("model", ckpt, 0)
    sampler_node.link("positive", pos)
    sampler_node.link("negative", neg)
    sampler_node.link("latent_image", vae_encode)

    decode = b.add("VAEDecode")
    decode.link("samples", sampler_node)
    decode.link("vae", ckpt, 2)

    combine = b.add("VHS_VideoCombine", {
        "frame_rate": 16,
        "loop_count": 0,
        "filename_prefix": "comfyforge_wan_i2v",
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
    })
    combine.link("images", decode)

    return b.to_dict()


# ── Helpers ──────────────────────────────────

def _random_seed() -> int:
    import random
    return random.randint(0, 2**63)


# ── Workflow selector ────────────────────────

WORKFLOW_MAP = {
    "text2img": text2img_sdxl,
    "text2img_sdxl": text2img_sdxl,
    "text2img_flux": text2img_flux,
    "img2vid": img2vid_svd,
    "img2vid_svd": img2vid_svd,
    "img2vid_animatediff": img2vid_animatediff,
    "img2vid_wan": img2vid_wan,
    "text2vid": text2vid_wan,
    "text2vid_wan": text2vid_wan,
}


def save_workflow(workflow: dict, path: str | Path) -> Path:
    """Save a workflow dict to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(workflow, indent=2))
    return p
