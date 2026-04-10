"""
ComfyForge Workflow Exporter
─────────────────────────────
Generates complete ComfyUI-native workflow JSON files that you can
drag-and-drop into the ComfyUI web interface.  No generation, no
downloads — just the .json files ready to import and tweak.

Usage:
    python workflow_export.py text2img "A dragon over Lahore at golden hour"
    python workflow_export.py text2img_flux "Photorealistic portrait of a warrior"
    python workflow_export.py img2vid_svd --image input.png
    python workflow_export.py img2vid_animatediff "Camera slowly panning" --image input.png
    python workflow_export.py text2vid_wan "Cinematic ocean waves crashing"
    python workflow_export.py img2vid_wan "Slow zoom into a painting" --image input.png
    python workflow_export.py full "A cyberpunk city at night"

All outputs go to ./exported_workflows/
"""

from __future__ import annotations
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════
# ComfyUI Native Workflow Builder (UI format)
# ═══════════════════════════════════════════════════
# The UI format has:
#   - "nodes": list of node objects with id, type, pos, size, widgets_values, inputs/outputs
#   - "links": list of [link_id, from_node, from_slot, to_node, to_slot, type]
#   - "last_node_id", "last_link_id"
# This is what you get when you click "Save" in ComfyUI.


@dataclass
class Slot:
    name: str
    type: str  # MODEL, CLIP, VAE, CONDITIONING, LATENT, IMAGE, etc.


@dataclass
class UINode:
    id: int
    type: str
    title: str | None = None
    pos: tuple[float, float] = (0, 0)
    size: tuple[float, float] = (300, 120)
    widgets_values: list[Any] = field(default_factory=list)
    inputs: list[Slot] = field(default_factory=list)
    outputs: list[Slot] = field(default_factory=list)
    color: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "pos": list(self.pos),
            "size": {"0": self.size[0], "1": self.size[1]},
            "flags": {},
            "order": self.id,
            "mode": 0,
            "inputs": [
                {"name": s.name, "type": s.type, "link": None} for s in self.inputs
            ],
            "outputs": [
                {"name": s.name, "type": s.type, "links": [], "slot_index": i}
                for i, s in enumerate(self.outputs)
            ],
            "properties": {"Node name for S&R": self.type},
            "widgets_values": self.widgets_values,
        }
        if self.title:
            d["title"] = self.title
        if self.color:
            d["color"] = self.color
            d["bgcolor"] = self.color
        return d


class UIWorkflow:
    """Builds a ComfyUI-native workflow JSON (the UI/export format)."""

    def __init__(self, title: str = "ComfyForge Workflow"):
        self.title = title
        self.nodes: list[UINode] = []
        self.links: list[list[Any]] = []  # [link_id, from_node, from_slot, to_node, to_slot, type]
        self._nid = 0
        self._lid = 0
        self._col = 0  # layout column tracker

    def add_node(
        self,
        node_type: str,
        widgets: list[Any] | None = None,
        inputs: list[Slot] | None = None,
        outputs: list[Slot] | None = None,
        title: str | None = None,
        color: str | None = None,
        width: float = 315,
        height: float = 120,
    ) -> UINode:
        self._nid += 1
        # Auto-layout: place nodes in a grid
        col = self._col
        row_in_col = sum(1 for n in self.nodes if abs(n.pos[0] - col * 380) < 10)
        x = 50 + col * 380
        y = 50 + row_in_col * (height + 40)

        node = UINode(
            id=self._nid,
            type=node_type,
            title=title,
            pos=(x, y),
            size=(width, height),
            widgets_values=widgets or [],
            inputs=inputs or [],
            outputs=outputs or [],
            color=color,
        )
        self.nodes.append(node)
        return node

    def next_column(self):
        """Move layout cursor to next column."""
        self._col += 1

    def link(self, from_node: UINode, from_slot: int, to_node: UINode, to_slot: int, link_type: str = "*"):
        """Create a link between two nodes."""
        self._lid += 1
        self.links.append([self._lid, from_node.id, from_slot, to_node.id, to_slot, link_type])

        # Update node metadata
        node_dict_from = None
        node_dict_to = None
        for n in self.nodes:
            if n.id == from_node.id:
                node_dict_from = n
            if n.id == to_node.id:
                node_dict_to = n

    def export(self) -> dict:
        """Export as ComfyUI-native JSON."""
        nodes_data = []
        for n in self.nodes:
            nd = n.to_dict()
            nodes_data.append(nd)

        # Patch link references into node inputs/outputs
        for lnk in self.links:
            lid, fnode, fslot, tnode, tslot, ltype = lnk
            # Set input link
            for nd in nodes_data:
                if nd["id"] == tnode and tslot < len(nd["inputs"]):
                    nd["inputs"][tslot]["link"] = lid
                if nd["id"] == fnode and fslot < len(nd["outputs"]):
                    if lid not in nd["outputs"][fslot]["links"]:
                        nd["outputs"][fslot]["links"].append(lid)

        return {
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": nodes_data,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {
                "ds": {"scale": 0.8, "offset": [0, 0]},
                "info": {"title": self.title, "generator": "ComfyForge"},
            },
            "version": 0.4,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.export(), indent=2))
        return p


# ═══════════════════════════════════════════════════
# Workflow Templates
# ═══════════════════════════════════════════════════

def _seed() -> int:
    return random.randint(0, 2**53)


def make_text2img_sdxl(
    prompt: str,
    negative: str = "blurry, low quality, watermark, text, deformed",
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
) -> UIWorkflow:
    wf = UIWorkflow("SDXL Text-to-Image")
    seed = seed if seed >= 0 else _seed()

    # ── Column 0: Loaders ──
    ckpt = wf.add_node(
        "CheckpointLoaderSimple",
        widgets=[checkpoint],
        outputs=[Slot("MODEL", "MODEL"), Slot("CLIP", "CLIP"), Slot("VAE", "VAE")],
        title="Load Checkpoint",
        color="#2a363b",
        height=100,
    )

    vae_node = wf.add_node(
        "VAELoader",
        widgets=[vae],
        outputs=[Slot("VAE", "VAE")],
        title="Load VAE",
        color="#2a363b",
        height=80,
    )

    # LoRA chain
    model_src, model_slot = ckpt, 0
    clip_src, clip_slot = ckpt, 1
    for lora_info in (loras or []):
        lname = lora_info if isinstance(lora_info, str) else lora_info.get("filename", "")
        strength = 0.8 if isinstance(lora_info, str) else lora_info.get("strength", 0.8)
        lora_node = wf.add_node(
            "LoraLoader",
            widgets=[lname, strength, strength],
            inputs=[Slot("model", "MODEL"), Slot("clip", "CLIP")],
            outputs=[Slot("MODEL", "MODEL"), Slot("CLIP", "CLIP")],
            title=f"LoRA: {Path(lname).stem}",
            color="#3b2a36",
            height=130,
        )
        wf.link(model_src, model_slot, lora_node, 0, "MODEL")
        wf.link(clip_src, clip_slot, lora_node, 1, "CLIP")
        model_src, model_slot = lora_node, 0
        clip_src, clip_slot = lora_node, 1

    wf.next_column()

    # ── Column 1: CLIP Encode ──
    pos_clip = wf.add_node(
        "CLIPTextEncode",
        widgets=[prompt],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Positive Prompt",
        color="#232",
        width=400,
        height=180,
    )
    wf.link(clip_src, clip_slot, pos_clip, 0, "CLIP")

    neg_clip = wf.add_node(
        "CLIPTextEncode",
        widgets=[negative],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Negative Prompt",
        color="#322",
        width=400,
        height=150,
    )
    wf.link(clip_src, clip_slot, neg_clip, 0, "CLIP")

    latent = wf.add_node(
        "EmptyLatentImage",
        widgets=[width, height, batch_size],
        outputs=[Slot("LATENT", "LATENT")],
        title="Empty Latent",
        height=100,
    )

    wf.next_column()

    # ── Column 2: Sampler ──
    ksampler = wf.add_node(
        "KSampler",
        widgets=[seed, "fixed", steps, cfg, sampler, scheduler, 1.0],
        inputs=[
            Slot("model", "MODEL"),
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent_image", "LATENT"),
        ],
        outputs=[Slot("LATENT", "LATENT")],
        title="KSampler",
        color="#335",
        width=320,
        height=200,
    )
    wf.link(model_src, model_slot, ksampler, 0, "MODEL")
    wf.link(pos_clip, 0, ksampler, 1, "CONDITIONING")
    wf.link(neg_clip, 0, ksampler, 2, "CONDITIONING")
    wf.link(latent, 0, ksampler, 3, "LATENT")

    wf.next_column()

    # ── Column 3: Decode + Save ──
    decode = wf.add_node(
        "VAEDecode",
        inputs=[Slot("samples", "LATENT"), Slot("vae", "VAE")],
        outputs=[Slot("IMAGE", "IMAGE")],
        title="VAE Decode",
        height=80,
    )
    wf.link(ksampler, 0, decode, 0, "LATENT")
    wf.link(vae_node, 0, decode, 1, "VAE")

    save = wf.add_node(
        "SaveImage",
        widgets=["comfyforge/sdxl"],
        inputs=[Slot("images", "IMAGE")],
        title="Save Image",
        color="#252",
        height=80,
    )
    wf.link(decode, 0, save, 0, "IMAGE")

    # Also add a preview
    preview = wf.add_node(
        "PreviewImage",
        inputs=[Slot("images", "IMAGE")],
        title="Preview",
        height=80,
    )
    wf.link(decode, 0, preview, 0, "IMAGE")

    return wf


def make_text2img_flux(
    prompt: str,
    checkpoint: str = "flux1-dev.safetensors",
    clip_1: str = "t5xxl_fp16.safetensors",
    clip_2: str = "clip_l.safetensors",
    vae: str = "ae.safetensors",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 3.5,
    seed: int = -1,
) -> UIWorkflow:
    wf = UIWorkflow("Flux Text-to-Image")
    seed = seed if seed >= 0 else _seed()

    # ── Column 0: Loaders ──
    unet = wf.add_node(
        "UNETLoader",
        widgets=[checkpoint, "fp8_e4m3fn"],
        outputs=[Slot("MODEL", "MODEL")],
        title="Load UNET (Flux)",
        color="#2a363b",
        height=100,
    )

    clip = wf.add_node(
        "DualCLIPLoader",
        widgets=[clip_1, clip_2, "flux"],
        outputs=[Slot("CLIP", "CLIP")],
        title="Dual CLIP Loader",
        color="#2a363b",
        height=100,
    )

    vae_node = wf.add_node(
        "VAELoader",
        widgets=[vae],
        outputs=[Slot("VAE", "VAE")],
        title="Load VAE",
        color="#2a363b",
        height=80,
    )

    wf.next_column()

    # ── Column 1: Conditioning ──
    clip_encode = wf.add_node(
        "CLIPTextEncode",
        widgets=[prompt],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Prompt",
        color="#232",
        width=400,
        height=180,
    )
    wf.link(clip, 0, clip_encode, 0, "CLIP")

    flux_guidance = wf.add_node(
        "FluxGuidance",
        widgets=[guidance],
        inputs=[Slot("conditioning", "CONDITIONING")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Flux Guidance",
        height=80,
    )
    wf.link(clip_encode, 0, flux_guidance, 0, "CONDITIONING")

    # Zero-out negative
    zero_cond = wf.add_node(
        "ConditioningZeroOut",
        inputs=[Slot("conditioning", "CONDITIONING")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Zero Negative",
        height=70,
    )
    wf.link(clip_encode, 0, zero_cond, 0, "CONDITIONING")

    latent = wf.add_node(
        "EmptySD3LatentImage",
        widgets=[width, height, 1],
        outputs=[Slot("LATENT", "LATENT")],
        title="Empty Latent (SD3/Flux)",
        height=100,
    )

    wf.next_column()

    # ── Column 2: Sampler ──
    ksampler = wf.add_node(
        "KSampler",
        widgets=[seed, "fixed", steps, 1.0, "euler", "simple", 1.0],
        inputs=[
            Slot("model", "MODEL"),
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent_image", "LATENT"),
        ],
        outputs=[Slot("LATENT", "LATENT")],
        title="KSampler",
        color="#335",
        width=320,
        height=200,
    )
    wf.link(unet, 0, ksampler, 0, "MODEL")
    wf.link(flux_guidance, 0, ksampler, 1, "CONDITIONING")
    wf.link(zero_cond, 0, ksampler, 2, "CONDITIONING")
    wf.link(latent, 0, ksampler, 3, "LATENT")

    wf.next_column()

    # ── Column 3: Decode + Save ──
    decode = wf.add_node(
        "VAEDecode",
        inputs=[Slot("samples", "LATENT"), Slot("vae", "VAE")],
        outputs=[Slot("IMAGE", "IMAGE")],
        title="VAE Decode",
        height=80,
    )
    wf.link(ksampler, 0, decode, 0, "LATENT")
    wf.link(vae_node, 0, decode, 1, "VAE")

    save = wf.add_node(
        "SaveImage",
        widgets=["comfyforge/flux"],
        inputs=[Slot("images", "IMAGE")],
        title="Save Image",
        color="#252",
        height=80,
    )
    wf.link(decode, 0, save, 0, "IMAGE")

    preview = wf.add_node(
        "PreviewImage",
        inputs=[Slot("images", "IMAGE")],
        title="Preview",
        height=80,
    )
    wf.link(decode, 0, preview, 0, "IMAGE")

    return wf


def make_img2vid_svd(
    image_path: str = "input.png",
    model: str = "svd_xt_1_1.safetensors",
    width: int = 1024,
    height: int = 576,
    frames: int = 25,
    fps: int = 8,
    motion_bucket: int = 127,
    augmentation: float = 0.0,
    steps: int = 20,
    cfg: float = 2.5,
    seed: int = -1,
) -> UIWorkflow:
    wf = UIWorkflow("SVD Image-to-Video")
    seed = seed if seed >= 0 else _seed()

    # ── Column 0 ──
    ckpt = wf.add_node(
        "ImageOnlyCheckpointLoader",
        widgets=[model],
        outputs=[Slot("MODEL", "MODEL"), Slot("CLIP_VISION", "CLIP_VISION"), Slot("VAE", "VAE")],
        title="SVD Checkpoint",
        color="#2a363b",
        height=100,
    )

    load_img = wf.add_node(
        "LoadImage",
        widgets=[image_path],
        outputs=[Slot("IMAGE", "IMAGE"), Slot("MASK", "MASK")],
        title="Load Input Image",
        height=100,
    )

    wf.next_column()

    # ── Column 1 ──
    cond = wf.add_node(
        "SVD_img2vid_Conditioning",
        widgets=[width, height, frames, motion_bucket, fps, augmentation],
        inputs=[
            Slot("clip_vision", "CLIP_VISION"),
            Slot("init_image", "IMAGE"),
            Slot("vae", "VAE"),
        ],
        outputs=[
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent", "LATENT"),
        ],
        title="SVD Conditioning",
        height=180,
    )
    wf.link(ckpt, 1, cond, 0, "CLIP_VISION")
    wf.link(load_img, 0, cond, 1, "IMAGE")
    wf.link(ckpt, 2, cond, 2, "VAE")

    wf.next_column()

    # ── Column 2 ──
    ksampler = wf.add_node(
        "KSampler",
        widgets=[seed, "fixed", steps, cfg, "euler", "karras", 1.0],
        inputs=[
            Slot("model", "MODEL"),
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent_image", "LATENT"),
        ],
        outputs=[Slot("LATENT", "LATENT")],
        title="KSampler",
        color="#335",
        width=320,
        height=200,
    )
    wf.link(ckpt, 0, ksampler, 0, "MODEL")
    wf.link(cond, 0, ksampler, 1, "CONDITIONING")
    wf.link(cond, 1, ksampler, 2, "CONDITIONING")
    wf.link(cond, 2, ksampler, 3, "LATENT")

    wf.next_column()

    # ── Column 3 ──
    decode = wf.add_node(
        "VAEDecode",
        inputs=[Slot("samples", "LATENT"), Slot("vae", "VAE")],
        outputs=[Slot("IMAGE", "IMAGE")],
        title="VAE Decode",
        height=80,
    )
    wf.link(ksampler, 0, decode, 0, "LATENT")
    wf.link(ckpt, 2, decode, 1, "VAE")

    video = wf.add_node(
        "VHS_VideoCombine",
        widgets=[fps, 0, "comfyforge/svd", "video/h264-mp4", False, True, None],
        inputs=[Slot("images", "IMAGE")],
        title="Save Video (MP4)",
        color="#252",
        height=160,
    )
    wf.link(decode, 0, video, 0, "IMAGE")

    return wf


def make_img2vid_animatediff(
    prompt: str,
    image_path: str = "input.png",
    negative: str = "blurry, low quality, static, jitter",
    checkpoint: str = "sd_xl_base_1.0.safetensors",
    motion_module: str = "mm_sdxl_v10_beta.safetensors",
    frames: int = 16,
    fps: int = 8,
    steps: int = 20,
    cfg: float = 7.0,
    denoise: float = 0.7,
    seed: int = -1,
) -> UIWorkflow:
    wf = UIWorkflow("AnimateDiff Image-to-Video")
    seed = seed if seed >= 0 else _seed()

    # ── Column 0: Loaders ──
    ckpt = wf.add_node(
        "CheckpointLoaderSimple",
        widgets=[checkpoint],
        outputs=[Slot("MODEL", "MODEL"), Slot("CLIP", "CLIP"), Slot("VAE", "VAE")],
        title="Checkpoint",
        color="#2a363b",
        height=100,
    )

    motion = wf.add_node(
        "ADE_LoadAnimateDiffModel",
        widgets=[motion_module],
        outputs=[Slot("MOTION_MODEL", "MOTION_MODEL_ADE")],
        title="Load Motion Model",
        color="#3b2a36",
        height=80,
    )

    apply_motion = wf.add_node(
        "ADE_ApplyAnimateDiffModelSimple",
        widgets=[1.0],
        inputs=[Slot("motion_model", "MOTION_MODEL_ADE"), Slot("model", "MODEL")],
        outputs=[Slot("MODEL", "MODEL")],
        title="Apply AnimateDiff",
        height=100,
    )
    wf.link(motion, 0, apply_motion, 0, "MOTION_MODEL_ADE")
    wf.link(ckpt, 0, apply_motion, 1, "MODEL")

    load_img = wf.add_node(
        "LoadImage",
        widgets=[image_path],
        outputs=[Slot("IMAGE", "IMAGE"), Slot("MASK", "MASK")],
        title="Input Image",
        height=100,
    )

    wf.next_column()

    # ── Column 1: Conditioning ──
    pos = wf.add_node(
        "CLIPTextEncode",
        widgets=[prompt],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Positive Prompt",
        color="#232",
        width=400,
        height=160,
    )
    wf.link(ckpt, 1, pos, 0, "CLIP")

    neg = wf.add_node(
        "CLIPTextEncode",
        widgets=[negative],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Negative Prompt",
        color="#322",
        width=400,
        height=120,
    )
    wf.link(ckpt, 1, neg, 0, "CLIP")

    # Encode image → latent → repeat for frames
    vae_encode = wf.add_node(
        "VAEEncode",
        inputs=[Slot("pixels", "IMAGE"), Slot("vae", "VAE")],
        outputs=[Slot("LATENT", "LATENT")],
        title="Encode Image",
        height=80,
    )
    wf.link(load_img, 0, vae_encode, 0, "IMAGE")
    wf.link(ckpt, 2, vae_encode, 1, "VAE")

    repeat = wf.add_node(
        "RepeatLatentBatch",
        widgets=[frames],
        inputs=[Slot("samples", "LATENT")],
        outputs=[Slot("LATENT", "LATENT")],
        title=f"Repeat × {frames} frames",
        height=80,
    )
    wf.link(vae_encode, 0, repeat, 0, "LATENT")

    wf.next_column()

    # ── Column 2: Sampler ──
    ksampler = wf.add_node(
        "KSampler",
        widgets=[seed, "fixed", steps, cfg, "euler_ancestral", "normal", denoise],
        inputs=[
            Slot("model", "MODEL"),
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent_image", "LATENT"),
        ],
        outputs=[Slot("LATENT", "LATENT")],
        title="KSampler",
        color="#335",
        width=320,
        height=200,
    )
    wf.link(apply_motion, 0, ksampler, 0, "MODEL")
    wf.link(pos, 0, ksampler, 1, "CONDITIONING")
    wf.link(neg, 0, ksampler, 2, "CONDITIONING")
    wf.link(repeat, 0, ksampler, 3, "LATENT")

    wf.next_column()

    # ── Column 3: Decode + Save ──
    decode = wf.add_node(
        "VAEDecode",
        inputs=[Slot("samples", "LATENT"), Slot("vae", "VAE")],
        outputs=[Slot("IMAGE", "IMAGE")],
        title="VAE Decode",
        height=80,
    )
    wf.link(ksampler, 0, decode, 0, "LATENT")
    wf.link(ckpt, 2, decode, 1, "VAE")

    video = wf.add_node(
        "VHS_VideoCombine",
        widgets=[fps, 0, "comfyforge/animdiff", "video/h264-mp4", False, True, None],
        inputs=[Slot("images", "IMAGE")],
        title="Save Video (MP4)",
        color="#252",
        height=160,
    )
    wf.link(decode, 0, video, 0, "IMAGE")

    return wf


def make_text2vid_wan(
    prompt: str,
    negative: str = "blurry, low quality, watermark, deformed",
    checkpoint: str = "wan2.1_t2v_14B_fp16.safetensors",
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    fps: int = 16,
    steps: int = 30,
    cfg: float = 6.0,
    seed: int = -1,
) -> UIWorkflow:
    wf = UIWorkflow("Wan 2.1 Text-to-Video")
    seed = seed if seed >= 0 else _seed()

    ckpt = wf.add_node(
        "CheckpointLoaderSimple",
        widgets=[checkpoint],
        outputs=[Slot("MODEL", "MODEL"), Slot("CLIP", "CLIP"), Slot("VAE", "VAE")],
        title="Wan 2.1 T2V Checkpoint",
        color="#2a363b",
        height=100,
    )

    wf.next_column()

    pos = wf.add_node(
        "CLIPTextEncode",
        widgets=[prompt],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Positive Prompt",
        color="#232",
        width=400,
        height=160,
    )
    wf.link(ckpt, 1, pos, 0, "CLIP")

    neg = wf.add_node(
        "CLIPTextEncode",
        widgets=[negative],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Negative Prompt",
        color="#322",
        width=400,
        height=120,
    )
    wf.link(ckpt, 1, neg, 0, "CLIP")

    latent = wf.add_node(
        "EmptyLatentImage",
        widgets=[width, height, frames],
        outputs=[Slot("LATENT", "LATENT")],
        title=f"Empty Latent ({frames} frames)",
        height=100,
    )

    wf.next_column()

    ksampler = wf.add_node(
        "KSampler",
        widgets=[seed, "fixed", steps, cfg, "euler", "normal", 1.0],
        inputs=[
            Slot("model", "MODEL"),
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent_image", "LATENT"),
        ],
        outputs=[Slot("LATENT", "LATENT")],
        title="KSampler",
        color="#335",
        width=320,
        height=200,
    )
    wf.link(ckpt, 0, ksampler, 0, "MODEL")
    wf.link(pos, 0, ksampler, 1, "CONDITIONING")
    wf.link(neg, 0, ksampler, 2, "CONDITIONING")
    wf.link(latent, 0, ksampler, 3, "LATENT")

    wf.next_column()

    decode = wf.add_node(
        "VAEDecode",
        inputs=[Slot("samples", "LATENT"), Slot("vae", "VAE")],
        outputs=[Slot("IMAGE", "IMAGE")],
        title="VAE Decode",
        height=80,
    )
    wf.link(ksampler, 0, decode, 0, "LATENT")
    wf.link(ckpt, 2, decode, 1, "VAE")

    video = wf.add_node(
        "VHS_VideoCombine",
        widgets=[fps, 0, "comfyforge/wan_t2v", "video/h264-mp4", False, True, None],
        inputs=[Slot("images", "IMAGE")],
        title="Save Video (MP4)",
        color="#252",
        height=160,
    )
    wf.link(decode, 0, video, 0, "IMAGE")

    return wf


def make_img2vid_wan(
    prompt: str,
    image_path: str = "input.png",
    negative: str = "blurry, low quality, watermark",
    checkpoint: str = "wan2.1_i2v_480p_14B_fp16.safetensors",
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    fps: int = 16,
    steps: int = 30,
    cfg: float = 6.0,
    denoise: float = 0.85,
    seed: int = -1,
) -> UIWorkflow:
    wf = UIWorkflow("Wan 2.1 Image-to-Video")
    seed = seed if seed >= 0 else _seed()

    ckpt = wf.add_node(
        "CheckpointLoaderSimple",
        widgets=[checkpoint],
        outputs=[Slot("MODEL", "MODEL"), Slot("CLIP", "CLIP"), Slot("VAE", "VAE")],
        title="Wan 2.1 I2V Checkpoint",
        color="#2a363b",
        height=100,
    )

    load_img = wf.add_node(
        "LoadImage",
        widgets=[image_path],
        outputs=[Slot("IMAGE", "IMAGE"), Slot("MASK", "MASK")],
        title="Input Image",
        height=100,
    )

    wf.next_column()

    pos = wf.add_node(
        "CLIPTextEncode",
        widgets=[prompt],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Positive Prompt",
        color="#232",
        width=400,
        height=160,
    )
    wf.link(ckpt, 1, pos, 0, "CLIP")

    neg = wf.add_node(
        "CLIPTextEncode",
        widgets=[negative],
        inputs=[Slot("clip", "CLIP")],
        outputs=[Slot("CONDITIONING", "CONDITIONING")],
        title="Negative Prompt",
        color="#322",
        width=400,
        height=120,
    )
    wf.link(ckpt, 1, neg, 0, "CLIP")

    vae_encode = wf.add_node(
        "VAEEncode",
        inputs=[Slot("pixels", "IMAGE"), Slot("vae", "VAE")],
        outputs=[Slot("LATENT", "LATENT")],
        title="Encode Init Image",
        height=80,
    )
    wf.link(load_img, 0, vae_encode, 0, "IMAGE")
    wf.link(ckpt, 2, vae_encode, 1, "VAE")

    wf.next_column()

    ksampler = wf.add_node(
        "KSampler",
        widgets=[seed, "fixed", steps, cfg, "euler", "normal", denoise],
        inputs=[
            Slot("model", "MODEL"),
            Slot("positive", "CONDITIONING"),
            Slot("negative", "CONDITIONING"),
            Slot("latent_image", "LATENT"),
        ],
        outputs=[Slot("LATENT", "LATENT")],
        title="KSampler",
        color="#335",
        width=320,
        height=200,
    )
    wf.link(ckpt, 0, ksampler, 0, "MODEL")
    wf.link(pos, 0, ksampler, 1, "CONDITIONING")
    wf.link(neg, 0, ksampler, 2, "CONDITIONING")
    wf.link(vae_encode, 0, ksampler, 3, "LATENT")

    wf.next_column()

    decode = wf.add_node(
        "VAEDecode",
        inputs=[Slot("samples", "LATENT"), Slot("vae", "VAE")],
        outputs=[Slot("IMAGE", "IMAGE")],
        title="VAE Decode",
        height=80,
    )
    wf.link(ksampler, 0, decode, 0, "LATENT")
    wf.link(ckpt, 2, decode, 1, "VAE")

    video = wf.add_node(
        "VHS_VideoCombine",
        widgets=[fps, 0, "comfyforge/wan_i2v", "video/h264-mp4", False, True, None],
        inputs=[Slot("images", "IMAGE")],
        title="Save Video (MP4)",
        color="#252",
        height=160,
    )
    wf.link(decode, 0, video, 0, "IMAGE")

    return wf


# ═══════════════════════════════════════════════════
# Chained / Full pipeline (multiple JSONs)
# ═══════════════════════════════════════════════════

def make_full_pipeline(
    prompt: str,
    negative: str = "blurry, low quality, watermark, text, deformed",
    use_flux: bool = False,
    video_backend: str = "svd",
    **kwargs,
) -> dict[str, UIWorkflow]:
    """Generate all workflow files for a full pipeline.
    Returns dict of {name: workflow} — each saved as a separate JSON.
    """
    workflows = {}

    # Stage 1: Text → Image
    if use_flux:
        workflows["01_text2img_flux"] = make_text2img_flux(prompt=prompt, **kwargs)
    else:
        workflows["01_text2img_sdxl"] = make_text2img_sdxl(
            prompt=prompt, negative=negative, **kwargs,
        )

    # Stage 2: Image → Video
    img_placeholder = "comfyforge_output.png"  # user replaces with actual output
    if video_backend == "svd":
        workflows["02_img2vid_svd"] = make_img2vid_svd(image_path=img_placeholder)
    elif video_backend == "animatediff":
        workflows["02_img2vid_animatediff"] = make_img2vid_animatediff(
            prompt=prompt, image_path=img_placeholder, negative=negative,
        )
    elif video_backend == "wan":
        workflows["02_img2vid_wan"] = make_img2vid_wan(
            prompt=prompt, image_path=img_placeholder, negative=negative,
        )

    return workflows


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

TEMPLATES = {
    "text2img":            make_text2img_sdxl,
    "text2img_sdxl":       make_text2img_sdxl,
    "text2img_flux":       make_text2img_flux,
    "img2vid":             make_img2vid_svd,
    "img2vid_svd":         make_img2vid_svd,
    "img2vid_animatediff": make_img2vid_animatediff,
    "img2vid_wan":         make_img2vid_wan,
    "text2vid":            make_text2vid_wan,
    "text2vid_wan":        make_text2vid_wan,
}


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  ComfyForge Workflow Exporter                                ║
║  Generates ComfyUI-native JSON — no execution, just files    ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  python workflow_export.py <template> "prompt" [options]

Templates:
  text2img             SDXL text-to-image
  text2img_flux        Flux text-to-image
  img2vid_svd          SVD image-to-video
  img2vid_animatediff  AnimateDiff image-to-video
  img2vid_wan          Wan 2.1 image-to-video
  text2vid_wan         Wan 2.1 text-to-video
  full                 All stages as separate JSONs

Options:
  --image PATH         Input image (for img2vid templates)
  --width N            Image/video width  (default 1024)
  --height N           Image/video height (default 1024)
  --steps N            Sampling steps (default varies)
  --cfg N              CFG scale (default varies)
  --seed N             Seed (-1 = random)
  --frames N           Video frames (default 25)
  --fps N              Video FPS (default 8)
  --negative "text"    Negative prompt
  --checkpoint FILE    Override checkpoint filename
  --flux               Use Flux for text2img in full pipeline
  --video-backend X    svd | animatediff | wan (for full)
  --output DIR         Output directory (default: ./exported_workflows)

Examples:
  python workflow_export.py text2img "A dragon over Lahore"
  python workflow_export.py text2img_flux "Photorealistic warrior" --steps 30
  python workflow_export.py img2vid_svd --image photo.png --frames 30
  python workflow_export.py full "Cyberpunk city" --flux --video-backend wan
""")


def parse_args(argv: list[str]) -> dict:
    opts: dict[str, Any] = {}
    i = 0
    positional = []
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:].replace("-", "_")
            if key in ("flux",):
                opts[key] = True
            elif i + 1 < len(argv):
                val = argv[i + 1]
                # Try numeric
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                opts[key] = val
                i += 1
        else:
            positional.append(a)
        i += 1
    return {"positional": positional, **opts}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        return

    args = parse_args(sys.argv[1:])
    pos = args.pop("positional", [])

    if not pos:
        print_help()
        return

    template = pos[0]
    prompt = pos[1] if len(pos) > 1 else "A beautiful landscape"

    out_dir = Path(args.pop("output", "exported_workflows"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    if template == "full":
        use_flux = args.pop("flux", False)
        vb = args.pop("video_backend", "svd")
        workflows = make_full_pipeline(prompt=prompt, use_flux=use_flux, video_backend=vb, **args)
        for name, wf in workflows.items():
            p = wf.save(out_dir / f"{name}_{ts}.json")
            print(f"  ✓ {p}")
        print(f"\n✓ Exported {len(workflows)} workflow files to {out_dir}/")
        print("  Load them in ComfyUI: Menu → Load → select the JSON")
        print("  For stage 2 (img2vid), update the LoadImage node with your actual output image.")
    else:
        if template not in TEMPLATES:
            print(f"✗ Unknown template: {template}")
            print(f"  Available: {', '.join(TEMPLATES.keys())}")
            return

        fn = TEMPLATES[template]

        # Map CLI args to function params
        fn_kwargs: dict[str, Any] = {}
        if "prompt" in fn.__code__.co_varnames:
            fn_kwargs["prompt"] = prompt
        for k in ("image", "image_path", "negative", "checkpoint", "width", "height",
                   "steps", "cfg", "seed", "frames", "fps", "sampler", "scheduler",
                   "motion_module", "model", "guidance", "denoise", "batch_size",
                   "motion_bucket", "augmentation"):
            if k in args:
                # image → image_path mapping
                actual_key = "image_path" if k == "image" else k
                if actual_key in fn.__code__.co_varnames:
                    fn_kwargs[actual_key] = args[k]

        wf = fn(**fn_kwargs)
        filename = f"{template}_{ts}.json"
        p = wf.save(out_dir / filename)
        print(f"\n  ✓ Exported: {p}")
        print(f"  Load in ComfyUI: Menu → Load → {filename}")


if __name__ == "__main__":
    main()
