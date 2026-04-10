"""CivitAI clone -- scrape generation metadata and produce a ready-to-run workflow."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

CIVITAI_API = "https://civitai.com/api/v1"

# ── Sampler mapping (A1111 / CivitAI -> ComfyUI) ────────────────────────

SAMPLER_MAP: dict[str, tuple[str, str]] = {
    "Euler":                          ("euler", "normal"),
    "Euler a":                        ("euler_ancestral", "normal"),
    "LMS":                            ("lms", "normal"),
    "LMS Karras":                     ("lms", "karras"),
    "Heun":                           ("heun", "normal"),
    "DPM2":                           ("dpm_2", "normal"),
    "DPM2 a":                         ("dpm_2_ancestral", "normal"),
    "DPM2 Karras":                    ("dpm_2", "karras"),
    "DPM2 a Karras":                  ("dpm_2_ancestral", "karras"),
    "DPM++ 2S a":                     ("dpmpp_2s_ancestral", "normal"),
    "DPM++ 2S a Karras":              ("dpmpp_2s_ancestral", "karras"),
    "DPM++ 2M":                       ("dpmpp_2m", "normal"),
    "DPM++ 2M Karras":                ("dpmpp_2m", "karras"),
    "DPM++ 2M SDE":                   ("dpmpp_2m_sde", "normal"),
    "DPM++ 2M SDE Karras":            ("dpmpp_2m_sde", "karras"),
    "DPM++ 2M SDE Exponential":       ("dpmpp_2m_sde", "exponential"),
    "DPM++ SDE":                      ("dpmpp_sde", "normal"),
    "DPM++ SDE Karras":               ("dpmpp_sde", "karras"),
    "DPM++ 3M SDE":                   ("dpmpp_3m_sde", "normal"),
    "DPM++ 3M SDE Karras":            ("dpmpp_3m_sde", "karras"),
    "DPM++ 3M SDE Exponential":       ("dpmpp_3m_sde", "exponential"),
    "DDIM":                           ("ddim", "normal"),
    "DDPM":                           ("ddpm", "normal"),
    "UniPC":                          ("uni_pc", "normal"),
    "UniPC BH2":                      ("uni_pc_bh2", "normal"),
    "LCM":                            ("lcm", "normal"),
    "PLMS":                           ("plms", "normal"),
    "DPM fast":                       ("dpm_fast", "normal"),
    "DPM adaptive":                   ("dpm_adaptive", "normal"),
}

# ── Base-model routing ──────────────────────────────────────────────────

BASE_MODEL_MAP: dict[str, str] = {
    "SD 1.4":      "sd15",
    "SD 1.5":      "sd15",
    "SD 1.5 LCM":  "sd15",
    "SD 2.1":      "sd15",   # same node graph
    "SDXL 0.9":    "sdxl",
    "SDXL 1.0":    "sdxl",
    "SDXL 1.0 LCM":"sdxl",
    "SDXL Turbo":  "sdxl",
    "SDXL Lightning":"sdxl",
    "Pony":        "sdxl",
    "Flux.1 D":    "flux",
    "Flux.1 S":    "flux",
}

_LORA_RE = re.compile(r"<lora:([^:>]+):([0-9.]+)>")

# ── Dataclasses ─────────────────────────────────────────────────────────


@dataclass
class LoraRef:
    name: str
    weight: float
    version_id: int | None = None
    filename: str | None = None


@dataclass
class CivitaiGenMeta:
    image_id: int
    prompt: str
    negative_prompt: str
    steps: int
    sampler: str       # ComfyUI sampler name
    scheduler: str     # ComfyUI scheduler name
    cfg_scale: float
    seed: int
    width: int
    height: int
    clip_skip: int
    base_model: str    # internal key: "sd15", "sdxl", "flux"
    base_model_raw: str  # original CivitAI string

    # Model references
    checkpoint_name: str
    checkpoint_version_id: int | None = None
    checkpoint_filename: str | None = None
    loras: list[LoraRef] = field(default_factory=list)
    vae_name: str | None = None

    # Hires fix (optional)
    hires_upscale: float | None = None
    hires_upscaler: str | None = None
    hires_denoising: float | None = None


# ── URL Parsing ─────────────────────────────────────────────────────────

_IMAGE_URL_RE = re.compile(r"civitai\.com/images/(\d+)")
_POST_URL_RE = re.compile(r"civitai\.com/posts/(\d+)")


def parse_civitai_url(url: str) -> tuple[str, int]:
    """Parse a CivitAI URL and return (content_type, id).

    Supports civitai.com/images/{id}, civitai.com/posts/{id}, or raw numeric IDs.
    """
    url = url.strip()

    m = _IMAGE_URL_RE.search(url)
    if m:
        return ("image", int(m.group(1)))

    m = _POST_URL_RE.search(url)
    if m:
        return ("post", int(m.group(1)))

    # Raw numeric ID
    if url.isdigit():
        return ("image", int(url))

    raise ValueError(
        f"Cannot parse CivitAI URL: {url}\n"
        "  Expected: https://civitai.com/images/12345 or a numeric image ID"
    )


# ── Prompt Parsing ──────────────────────────────────────────────────────


def parse_loras_from_prompt(
    prompt: str,
    resources: list[dict] | None = None,
) -> tuple[str, list[LoraRef]]:
    """Extract <lora:name:weight> tags from prompt.

    Returns (clean_prompt, lora_refs).
    Merges with Civitai resources array to fill in version_id.
    """
    loras: list[LoraRef] = []
    for match in _LORA_RE.finditer(prompt):
        name = match.group(1)
        weight = float(match.group(2))
        loras.append(LoraRef(name=name, weight=weight))

    clean = _LORA_RE.sub("", prompt).strip()
    # collapse multiple spaces left behind
    clean = re.sub(r"  +", " ", clean)

    # Merge version IDs from Civitai resources
    if resources:
        for res in resources:
            if res.get("type", "").lower() in ("lora", "locon", "lycoris"):
                res_name = res.get("modelName", "") or res.get("name", "")
                vid = res.get("modelVersionId")
                # Match by substring (CivitAI resource names may differ slightly)
                for lr in loras:
                    if lr.version_id is None and _fuzzy_name_match(lr.name, res_name):
                        lr.version_id = vid
                        break
                else:
                    # Resource not found in prompt tags -- add it anyway
                    loras.append(LoraRef(
                        name=res_name,
                        weight=1.0,
                        version_id=vid,
                    ))

    return clean, loras


def _fuzzy_name_match(prompt_name: str, resource_name: str) -> bool:
    """Check if a LoRA name from a prompt tag matches a CivitAI resource name."""
    a = prompt_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    b = resource_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    return a in b or b in a


# ── Sampler Mapping ─────────────────────────────────────────────────────


def map_sampler(civitai_sampler: str | None) -> tuple[str, str]:
    """Map a CivitAI/A1111 sampler name to (comfyui_sampler, comfyui_scheduler)."""
    if not civitai_sampler:
        return ("euler", "normal")

    result = SAMPLER_MAP.get(civitai_sampler)
    if result:
        return result

    # Try case-insensitive lookup
    lower = civitai_sampler.lower()
    for k, v in SAMPLER_MAP.items():
        if k.lower() == lower:
            return v

    console.print(f"  [yellow]WARNING: Unknown sampler '{civitai_sampler}' -- using euler/normal[/]")
    return ("euler", "normal")


# ── CivitAI API ─────────────────────────────────────────────────────────


def _api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("CIVITAI_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_model_version(version_id: int) -> dict:
    """GET /api/v1/model-versions/{id}."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{CIVITAI_API}/model-versions/{version_id}",
            headers=_api_headers(),
        )
        resp.raise_for_status()
        return resp.json()


def _fetch_image(image_id: int) -> dict:
    """GET /api/v1/images/{id}."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{CIVITAI_API}/images/{image_id}",
            headers=_api_headers(),
        )
        if resp.status_code == 403:
            raise RuntimeError(
                f"CivitAI returned 403 for image {image_id}.\n"
                "  Set CIVITAI_API_TOKEN env var with your API key.\n"
                "  Get one at: https://civitai.com/user/account -> API Keys"
            )
        resp.raise_for_status()
        return resp.json()


def _fetch_post_images(post_id: int) -> list[dict]:
    """GET /api/v1/images?postId={id} to get images from a post."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{CIVITAI_API}/images",
            params={"postId": post_id, "limit": 20},
            headers=_api_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])


# ── Metadata Extraction ─────────────────────────────────────────────────


def fetch_image_metadata(image_id: int) -> CivitaiGenMeta:
    """Fetch a CivitAI image and parse all generation metadata."""
    data = _fetch_image(image_id)
    meta = data.get("meta")
    if not meta:
        raise ValueError(
            f"CivitAI image {image_id} has no generation metadata.\n"
            "  This image may have been uploaded without generation info."
        )

    # Parse dimensions from meta or top-level
    width = data.get("width", 512)
    height = data.get("height", 512)
    if "Size" in meta:
        try:
            parts = meta["Size"].split("x")
            width, height = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass

    # Parse prompt and LoRAs
    raw_prompt = meta.get("prompt", "")
    civitai_resources = meta.get("civitaiResources") or meta.get("resources") or []
    clean_prompt, loras = parse_loras_from_prompt(raw_prompt, civitai_resources)

    # Map sampler
    sampler_name, scheduler_name = map_sampler(meta.get("sampler"))

    # Identify checkpoint
    checkpoint_name = meta.get("Model", "unknown")
    checkpoint_vid = None

    # Try to find checkpoint in resources
    for res in civitai_resources:
        rtype = res.get("type", "").lower()
        if rtype in ("checkpoint", "model"):
            checkpoint_vid = res.get("modelVersionId")
            if res.get("modelName"):
                checkpoint_name = res["modelName"]
            break

    # Also check modelVersionIds from the image response
    model_version_ids = data.get("modelVersionIds") or []
    if not checkpoint_vid and model_version_ids:
        checkpoint_vid = model_version_ids[0]

    # Fetch checkpoint details to get baseModel and filename
    base_model_raw = ""
    checkpoint_filename = None
    if checkpoint_vid:
        try:
            mv = _fetch_model_version(checkpoint_vid)
            base_model_raw = mv.get("baseModel", "")
            files = mv.get("files", [])
            if files:
                checkpoint_filename = files[0].get("name")
        except Exception as e:
            console.print(f"  [yellow]WARNING: Could not fetch checkpoint version {checkpoint_vid}: {e}[/]")

    # Resolve base model key
    base_model = BASE_MODEL_MAP.get(base_model_raw, "")
    if not base_model:
        # Try guessing from checkpoint name
        ckpt_lower = checkpoint_name.lower()
        if "flux" in ckpt_lower:
            base_model = "flux"
        elif "xl" in ckpt_lower or "sdxl" in ckpt_lower or "pony" in ckpt_lower:
            base_model = "sdxl"
        else:
            base_model = "sdxl"  # safe default -- same node graph as SD 1.5

    return CivitaiGenMeta(
        image_id=image_id,
        prompt=clean_prompt,
        negative_prompt=meta.get("negativePrompt", ""),
        steps=int(meta.get("steps", 20)),
        sampler=sampler_name,
        scheduler=scheduler_name,
        cfg_scale=float(meta.get("cfgScale", 7.0)),
        seed=int(meta.get("seed", -1)),
        width=width,
        height=height,
        clip_skip=int(meta.get("clipSkip") or meta.get("clip_skip") or 1),
        base_model=base_model,
        base_model_raw=base_model_raw,
        checkpoint_name=checkpoint_name,
        checkpoint_version_id=checkpoint_vid,
        checkpoint_filename=checkpoint_filename,
        loras=loras,
        vae_name=meta.get("VAE"),
        hires_upscale=_float_or_none(meta.get("Hires upscale")),
        hires_upscaler=meta.get("Hires upscaler"),
        hires_denoising=_float_or_none(meta.get("Denoising strength")),
    )


def _float_or_none(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Model Resolution & Download ─────────────────────────────────────────


def resolve_and_download_models(
    meta: CivitaiGenMeta,
    mm,  # ModelManager -- avoid circular import
    no_download: bool = False,
) -> dict[str, str | list[dict]]:
    """Resolve all model references to local filenames and download as needed.

    Returns dict ready to be passed as kwargs to workflow factory:
      {
        "checkpoint": "filename.safetensors",
        "vae": "filename.safetensors" | None,
        "loras": [{"filename": "...", "strength": 0.8}, ...],
      }
    """
    result: dict[str, str | list[dict] | None] = {}

    # ── Checkpoint ──
    ckpt_filename = meta.checkpoint_filename
    if not ckpt_filename and meta.checkpoint_version_id:
        try:
            mv = _fetch_model_version(meta.checkpoint_version_id)
            files = mv.get("files", [])
            if files:
                ckpt_filename = files[0]["name"]
                download_url = files[0].get("downloadUrl", "")
                if download_url and not no_download:
                    mm._auto_register(ckpt_filename, "checkpoint", [download_url])
        except Exception as e:
            console.print(f"  [yellow]WARNING: Could not fetch checkpoint info: {e}[/]")

    if not ckpt_filename:
        # Fall back to the model name with .safetensors
        ckpt_filename = meta.checkpoint_name.replace(" ", "_") + ".safetensors"

    if not no_download:
        console.print(f"\n  [bold]Checkpoint:[/] {ckpt_filename}")
        try:
            mm.ensure_or_search(ckpt_filename, "checkpoint")
        except Exception as e:
            console.print(f"  [red]✗ Could not download checkpoint: {e}[/]")

    result["checkpoint"] = ckpt_filename

    # ── VAE ──
    vae_filename = meta.vae_name
    if vae_filename and vae_filename.lower() not in ("automatic", "none", "default", ""):
        if not no_download:
            console.print(f"  [bold]VAE:[/] {vae_filename}")
            try:
                mm.ensure_or_search(vae_filename, "vae")
            except Exception as e:
                console.print(f"  [yellow]WARNING: Could not download VAE: {e} -- using checkpoint VAE[/]")
                vae_filename = None
        result["vae"] = vae_filename
    else:
        result["vae"] = None

    # ── LoRAs ──
    lora_entries: list[dict] = []
    for lr in meta.loras:
        lora_filename = lr.filename

        # Fetch filename from version ID if we don't have it
        if not lora_filename and lr.version_id:
            try:
                mv = _fetch_model_version(lr.version_id)
                files = mv.get("files", [])
                if files:
                    lora_filename = files[0]["name"]
                    download_url = files[0].get("downloadUrl", "")
                    if download_url and not no_download:
                        mm._auto_register(lora_filename, "lora", [download_url])
            except Exception as e:
                console.print(f"  [yellow]WARNING: Could not fetch LoRA '{lr.name}' info: {e}[/]")

        if not lora_filename:
            lora_filename = lr.name.replace(" ", "_") + ".safetensors"

        if not no_download:
            console.print(f"  [bold]LoRA:[/] {lora_filename} (strength: {lr.weight})")
            try:
                mm.ensure_or_search(lora_filename, "lora")
            except Exception as e:
                console.print(f"  [yellow]WARNING: Could not download LoRA '{lr.name}': {e}[/]")

        lora_entries.append({"filename": lora_filename, "strength": lr.weight})

    result["loras"] = lora_entries
    return result


# ── Workflow Generation ──────────────────────────────────────────────────


def generate_clone_workflow(meta: CivitaiGenMeta, model_filenames: dict):
    """Generate a UIWorkflow that reproduces the CivitAI image."""
    # Add parent dir to path so we can import workflow_export
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from workflow_export import make_text2img_sdxl, make_text2img_flux

    if meta.base_model in ("sd15", "sdxl"):
        kwargs = {
            "prompt": meta.prompt,
            "negative": meta.negative_prompt or "blurry, low quality, watermark, text, deformed",
            "checkpoint": model_filenames["checkpoint"],
            "vae": model_filenames.get("vae"),
            "loras": model_filenames.get("loras") or None,
            "width": meta.width,
            "height": meta.height,
            "steps": meta.steps,
            "cfg": meta.cfg_scale,
            "sampler": meta.sampler,
            "scheduler": meta.scheduler,
            "seed": meta.seed,
        }
        wf = make_text2img_sdxl(**kwargs)

        if meta.hires_upscale:
            console.print(
                f"  [yellow]WARNING: Hires fix was used (upscale: {meta.hires_upscale}x, "
                f"upscaler: {meta.hires_upscaler}) -- not yet supported in workflow export. "
                f"Results may differ.[/]"
            )

        return wf

    elif meta.base_model == "flux":
        if model_filenames.get("loras"):
            console.print(
                "  [yellow]WARNING: Flux LoRAs detected but not yet supported in workflow export. "
                "LoRAs will be skipped.[/]"
            )

        kwargs = {
            "prompt": meta.prompt,
            "checkpoint": model_filenames["checkpoint"],
            "width": meta.width,
            "height": meta.height,
            "steps": meta.steps,
            "guidance": meta.cfg_scale,
            "seed": meta.seed,
        }
        return make_text2img_flux(**kwargs)

    else:
        supported = sorted(set(BASE_MODEL_MAP.values()))
        raise ValueError(
            f"Unsupported base model: '{meta.base_model_raw}'\n"
            f"  Supported: {', '.join(supported)}"
        )


# ── Summary Display ──────────────────────────────────────────────────────


def _print_summary(meta: CivitaiGenMeta):
    """Print a Rich table summarizing the extracted metadata."""
    table = Table(title=f"CivitAI Image #{meta.image_id}", show_lines=True)
    table.add_column("Field", style="bold", width=20)
    table.add_column("Value", max_width=80)

    table.add_row("Base Model", f"{meta.base_model_raw} -> {meta.base_model}")
    table.add_row("Checkpoint", meta.checkpoint_name)
    if meta.checkpoint_filename:
        table.add_row("Checkpoint File", meta.checkpoint_filename)
    table.add_row("Prompt", meta.prompt[:200] + ("..." if len(meta.prompt) > 200 else ""))
    if meta.negative_prompt:
        table.add_row("Negative", meta.negative_prompt[:150] + ("..." if len(meta.negative_prompt) > 150 else ""))
    table.add_row("Size", f"{meta.width} x {meta.height}")
    table.add_row("Steps", str(meta.steps))
    table.add_row("CFG Scale", str(meta.cfg_scale))
    table.add_row("Sampler", f"{meta.sampler} / {meta.scheduler}")
    table.add_row("Seed", str(meta.seed))
    if meta.clip_skip > 1:
        table.add_row("Clip Skip", str(meta.clip_skip))
    if meta.vae_name:
        table.add_row("VAE", meta.vae_name)
    if meta.loras:
        lora_str = "\n".join(
            f"  {lr.name} (weight: {lr.weight})" + (f" [vid: {lr.version_id}]" if lr.version_id else "")
            for lr in meta.loras
        )
        table.add_row("LoRAs", lora_str)
    if meta.hires_upscale:
        table.add_row("Hires Fix", f"{meta.hires_upscale}x ({meta.hires_upscaler})")

    console.print(table)


# ── Main Orchestrator ────────────────────────────────────────────────────


def clone_from_civitai(
    url: str,
    output_dir: str = "exported_workflows",
    output_path: str | None = None,
    no_download: bool = False,
    config=None,
) -> Path:
    """End-to-end: URL -> metadata -> download models -> generate workflow -> save JSON.

    Returns path to saved workflow JSON.
    """
    from .config import Config
    from .models import ModelManager

    cfg = config or Config.load()

    # 1. Parse URL
    content_type, content_id = parse_civitai_url(url)

    # 2. Get image metadata
    if content_type == "post":
        console.print(f"  Fetching post #{content_id}...")
        images = _fetch_post_images(content_id)
        # Find the first image with generation metadata
        image_data = None
        for img in images:
            if img.get("meta"):
                image_data = img
                break
        if not image_data:
            raise ValueError(f"No images with generation metadata found in post #{content_id}")
        image_id = image_data["id"]
        console.print(f"  Found image #{image_id} with metadata")
    else:
        image_id = content_id

    console.print(f"  Fetching metadata for image #{image_id}...")
    meta = fetch_image_metadata(image_id)

    # 3. Print summary
    _print_summary(meta)

    # 4. Resolve and download models
    mm = ModelManager(cfg)
    console.print("\n[bold]Resolving models...[/]")
    model_filenames = resolve_and_download_models(meta, mm, no_download=no_download)

    # 5. Generate workflow
    console.print("\n[bold]Generating workflow...[/]")
    wf = generate_clone_workflow(meta, model_filenames)

    # 6. Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        dest = Path(output_path)
    else:
        dest = out_dir / f"clone_{image_id}.json"

    wf.save(str(dest))
    return dest
