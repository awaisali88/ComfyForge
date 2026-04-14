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

console = Console(force_terminal=True)

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


# BASE_MODEL_MAP is now in configs/config.yaml under "architectures".
# Use Config.get_architecture(base_model_name) to look up.

_LORA_RE = re.compile(r"<lora:([^:>]+):([0-9.]+)>")

# ── Dataclasses ─────────────────────────────────────────────────────────


@dataclass
class LoraRef:
    name: str
    weight: float
    version_id: int | None = None
    filename: str | None = None
    hash: str | None = None  # SHA256/AutoV2 — used for CivitAI by-hash lookup


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
    checkpoint_hash: str | None = None
    loras: list[LoraRef] = field(default_factory=list)
    vae_name: str | None = None
    vae_hash: str | None = None

    # Hires fix (optional)
    hires_upscale: float | None = None
    hires_upscaler: str | None = None
    hires_denoising: float | None = None

    # Video (optional — set when the CivitAI item is a video)
    media_type: str = "image"                     # "image" | "video"
    frames: int | None = None
    fps: int | None = None
    duration: float | None = None
    motion_module_name: str | None = None
    motion_module_hash: str | None = None
    motion_module_version_id: int | None = None
    init_image_url: str | None = None             # source still for i2v
    init_image_local: str | None = None           # filled after download


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
    hashes: dict[str, str] | None = None,
) -> tuple[str, list[LoraRef]]:
    """Extract <lora:name:weight> tags from prompt.

    Returns (clean_prompt, lora_refs).
    Merges with Civitai resources array (for version_id/hash) and the
    A1111-style ``hashes`` dict (keys like ``lora:NAME``).
    """
    loras: list[LoraRef] = []
    for match in _LORA_RE.finditer(prompt):
        name = match.group(1)
        weight = float(match.group(2))
        loras.append(LoraRef(name=name, weight=weight))

    clean = _LORA_RE.sub("", prompt).strip()
    # collapse multiple spaces left behind
    clean = re.sub(r"  +", " ", clean)

    # Apply hashes from the A1111-style hashes dict (e.g. {"lora:foo": "abc..."})
    if isinstance(hashes, dict):
        for key, h in hashes.items():
            if not isinstance(key, str) or not h:
                continue
            if key.lower().startswith("lora:"):
                tag_name = key.split(":", 1)[1]
                for lr in loras:
                    if lr.hash is None and _fuzzy_name_match(lr.name, tag_name):
                        lr.hash = h
                        break

    # Merge version IDs / hashes from Civitai resources (new + legacy formats)
    if resources:
        for res in resources:
            if res.get("type", "").lower() not in ("lora", "locon", "lycoris"):
                continue
            res_name = res.get("modelName", "") or res.get("name", "")
            vid = res.get("modelVersionId")
            rhash = res.get("hash")
            rweight = res.get("weight")

            # Match by substring (CivitAI resource names may differ slightly)
            matched = False
            for lr in loras:
                if _fuzzy_name_match(lr.name, res_name):
                    if lr.version_id is None and vid:
                        lr.version_id = vid
                    if lr.hash is None and rhash:
                        lr.hash = rhash
                    matched = True
                    break

            if not matched:
                # Resource not found in prompt tags -- add it anyway
                loras.append(LoraRef(
                    name=res_name or "unknown_lora",
                    weight=float(rweight) if rweight is not None else 1.0,
                    version_id=vid,
                    hash=rhash,
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


def _api_headers(config=None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = ""
    if config:
        token = config.civitai_api_token or ""
    if not token:
        token = os.getenv("CIVITAI_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _nsfw_params(config=None) -> dict[str, str | int]:
    """Return NSFW query params if an API token is available."""
    headers = _api_headers(config)
    if "Authorization" in headers:
        # browsingLevel bitmask: 1=PG | 2=PG13 | 4=R | 8=X | 16=XXX → 31 = all
        return {"nsfw": "X", "browsingLevel": 31}
    return {}


def _fetch_model_version(version_id: int, config=None) -> dict:
    """GET /api/v1/model-versions/{id}."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{CIVITAI_API}/model-versions/{version_id}",
            headers=_api_headers(config),
        )
        resp.raise_for_status()
        return resp.json()


def _fetch_model_version_by_hash(file_hash: str, config=None) -> dict | None:
    """GET /api/v1/model-versions/by-hash/{hash}.

    Returns the model-version JSON, or None on 404 / error. CivitAI accepts
    AutoV1, AutoV2, SHA256, CRC32 and Blake3 hashes here.
    """
    if not file_hash:
        return None
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{CIVITAI_API}/model-versions/by-hash/{file_hash}",
                headers=_api_headers(config),
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        console.print(
            f"  [dim]CivitAI by-hash lookup failed for {file_hash[:12]}…: {e}[/]"
        )
        return None


def _resolve_civitai_by_hash(
    file_hash: str, config=None
) -> tuple[str | None, str | None]:
    """Resolve a file hash to (filename, downloadUrl) via CivitAI."""
    mv = _fetch_model_version_by_hash(file_hash, config)
    if not mv:
        return None, None
    files = mv.get("files", [])
    # Prefer the file whose hash matches; fall back to first file
    target_hash = file_hash.lower()
    for f in files:
        f_hashes = f.get("hashes", {}) or {}
        for h in f_hashes.values():
            if isinstance(h, str) and h.lower() == target_hash:
                return f.get("name"), f.get("downloadUrl")
    if files:
        return files[0].get("name"), files[0].get("downloadUrl")
    return None, None


def _fetch_image(image_id: int, config=None) -> dict:
    """Fetch a single image by ID via GET /api/v1/images?imageId={id}.

    Includes nsfw and browsingLevel params so NSFW/R-rated/X-rated images
    are returned when using an authenticated API token.
    """
    headers = _api_headers(config)
    has_token = "Authorization" in headers

    params: dict[str, str | int] = {"imageId": image_id}
    params.update(_nsfw_params(config))

    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{CIVITAI_API}/images",
            params=params,
            headers=headers,
        )
        if resp.status_code == 403:
            raise RuntimeError(
                f"CivitAI returned 403 for image {image_id}.\n"
                "  Set CIVITAI_API_TOKEN in config.yaml or as env var.\n"
                "  Get one at: https://civitai.com/user/account -> API Keys"
            )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            hint = ""
            if not has_token:
                hint = (
                    "\n  If the image is NSFW, set CIVITAI_API_TOKEN in config.yaml or env.\n"
                    "  Get one at: https://civitai.com/user/account -> API Keys"
                )
            raise ValueError(
                f"CivitAI image {image_id} not found.{hint}\n"
                "  The image may have been removed or the ID may be incorrect."
            )
        return items[0]


def _download_init_image(
    url: str,
    image_id: int,
    comfy_input_dir: Path,
    config=None,
) -> str | None:
    """Download a source still into ComfyUI's ``input/`` directory.

    Returns the basename (no path) so ``LoadImage`` nodes can reference it.
    ``LoadImage`` resolves filenames relative to ``<comfyui>/input``.
    """
    if not url:
        return None
    try:
        comfy_input_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the source extension when possible
        ext = ".png"
        lower = url.lower().split("?", 1)[0]
        for candidate in (".png", ".jpg", ".jpeg", ".webp"):
            if lower.endswith(candidate):
                ext = candidate
                break
        filename = f"comfyforge_clone_{image_id}{ext}"
        dest = comfy_input_dir / filename

        if dest.exists():
            console.print(f"  [dim]Init image already cached: {filename}[/]")
            return filename

        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url, headers=_api_headers(config))
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        console.print(f"  [green]✓[/] Init image saved: {filename}")
        return filename
    except Exception as e:
        console.print(f"  [yellow]WARNING: Could not download init image: {e}[/]")
        return None


def _fetch_post_images(post_id: int, config=None) -> list[dict]:
    """GET /api/v1/images?postId={id} to get images from a post."""
    params: dict[str, str | int] = {"postId": post_id, "limit": 20}
    params.update(_nsfw_params(config))
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{CIVITAI_API}/images",
            params=params,
            headers=_api_headers(config),
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])


# ── Metadata Extraction ─────────────────────────────────────────────────


def fetch_image_metadata(image_id: int, config=None) -> CivitaiGenMeta:
    """Fetch a CivitAI image (or video) and parse all generation metadata."""
    data = _fetch_image(image_id, config)

    # CivitAI returns both images and videos from /api/v1/images. The ``type``
    # field distinguishes them ("image" | "video"). Videos use the same meta
    # shape but with extra fields (frameRate, numFrames, duration, motion modules).
    media_type = (data.get("type") or "image").lower()

    meta = data.get("meta")
    if not meta:
        kind = "video" if media_type == "video" else "image"
        raise ValueError(
            f"CivitAI {kind} {image_id} has no generation metadata.\n"
            f"  This {kind} may have been uploaded without generation info."
        )

    # CivitAI API sometimes double-nests: meta: { id, meta: { prompt, seed, ... } }
    if isinstance(meta.get("meta"), dict):
        meta = meta["meta"]
    elif not meta.get("prompt") and not meta.get("seed") and not meta.get("steps"):
        # meta has no generation fields — might be wrapper only
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
    hashes_dict = meta.get("hashes") if isinstance(meta.get("hashes"), dict) else {}
    clean_prompt, loras = parse_loras_from_prompt(
        raw_prompt, civitai_resources, hashes_dict
    )

    # Map sampler
    sampler_name, scheduler_name = map_sampler(meta.get("sampler"))

    # Identify checkpoint
    checkpoint_name = meta.get("Model", "unknown")
    checkpoint_vid = None
    checkpoint_hash: str | None = None
    vae_hash: str | None = None

    # Pull hashes from the A1111-style hashes dict
    if hashes_dict:
        checkpoint_hash = hashes_dict.get("model") or hashes_dict.get("Model")
        vae_hash = hashes_dict.get("vae") or hashes_dict.get("VAE")

    # Try to find checkpoint in resources
    for res in civitai_resources:
        rtype = res.get("type", "").lower()
        if rtype in ("checkpoint", "model"):
            checkpoint_vid = res.get("modelVersionId")
            if res.get("modelName"):
                checkpoint_name = res["modelName"]
            if not checkpoint_hash and res.get("hash"):
                checkpoint_hash = res["hash"]
            break
        if rtype == "vae" and not vae_hash and res.get("hash"):
            vae_hash = res["hash"]

    # Also check modelVersionIds from the image response.
    # Look up each version to properly identify checkpoint vs LoRA, and
    # capture baseModel from whichever resource reports it — even a LoRA's
    # baseModel tells us which architecture to target.
    base_model_raw = ""
    model_version_ids = data.get("modelVersionIds") or []
    if model_version_ids:
        for vid in model_version_ids:
            try:
                mv = _fetch_model_version(vid, config)
                mv_type = (mv.get("model", {}).get("type", "") or "").upper()
                mv_base = mv.get("baseModel", "")
                mv_files = mv.get("files", [])
                mv_filename = mv_files[0].get("name") if mv_files else None

                # Any resource can tell us the architecture's base model
                if mv_base and not base_model_raw:
                    base_model_raw = mv_base

                if mv_type in ("CHECKPOINT", "MODEL") and not checkpoint_vid:
                    checkpoint_vid = vid
                    checkpoint_name = mv.get("model", {}).get("name", checkpoint_name)
                elif mv_type in ("LORA", "LOCON", "LYCORIS"):
                    # Add as LoRA if not already found via prompt tags
                    lora_name = mv.get("model", {}).get("name", "")
                    already_found = any(
                        lr.version_id == vid or _fuzzy_name_match(lr.name, lora_name)
                        for lr in loras
                    )
                    if not already_found:
                        loras.append(LoraRef(
                            name=lora_name,
                            weight=1.0,
                            version_id=vid,
                            filename=mv_filename,
                        ))
            except Exception as e:
                console.print(f"  [yellow]WARNING: Could not fetch model version {vid}: {e}[/]")

        # NOTE: do NOT promote the first model_version_id to checkpoint.
        # Many CivitAI posts (especially Wan/AnimateDiff videos) ship only
        # LoRAs and rely on a base model that isn't on CivitAI at all. In
        # that case we leave checkpoint_vid=None and fall through to the
        # factory's built-in default (e.g. wan2.1_i2v_480p_14B_fp16.safetensors).

    # Fetch checkpoint details to get filename. If no checkpoint was found,
    # base_model_raw should already have been captured from a sibling LoRA
    # above, so we can still route the architecture correctly.
    checkpoint_filename = None
    if checkpoint_vid:
        try:
            mv = _fetch_model_version(checkpoint_vid, config)
            if not base_model_raw:
                base_model_raw = mv.get("baseModel", "")
            files = mv.get("files", [])
            if files:
                checkpoint_filename = files[0].get("name")
        except Exception as e:
            console.print(f"  [yellow]WARNING: Could not fetch checkpoint version {checkpoint_vid}: {e}[/]")

    # Resolve base model key via config architecture registry
    from .config import Config
    cfg = config or Config.load()
    base_model, arch = cfg.get_architecture(base_model_raw)

    # ── Video-specific fields ────────────────────────────────────────────
    frames: int | None = None
    fps: int | None = None
    duration: float | None = None
    motion_module_name: str | None = None
    motion_module_hash: str | None = None
    motion_module_version_id: int | None = None
    init_image_url: str | None = None

    if media_type == "video":
        # Frame count — CivitAI video meta uses various keys
        for key in ("numFrames", "frames", "Frame count", "length", "num_frames"):
            val = meta.get(key)
            if val is not None:
                try:
                    frames = int(val)
                    break
                except (ValueError, TypeError):
                    pass

        # Frame rate
        for key in ("frameRate", "fps", "Frame rate", "frame_rate"):
            val = meta.get(key)
            if val is not None:
                try:
                    fps = int(float(val))
                    break
                except (ValueError, TypeError):
                    pass

        # Duration (seconds)
        for key in ("duration", "Duration"):
            val = meta.get(key) or data.get(key)
            if val is not None:
                try:
                    duration = float(val)
                    break
                except (ValueError, TypeError):
                    pass

        # If we know fps + duration but not frames, derive it
        if frames is None and fps and duration:
            frames = int(round(fps * duration))

        # Detect motion module from civitaiResources (AnimateDiff style)
        for res in civitai_resources:
            rtype = (res.get("type", "") or "").lower()
            if rtype in ("motionmodule", "motion", "motion_module"):
                motion_module_name = res.get("modelName") or res.get("name")
                motion_module_version_id = res.get("modelVersionId")
                motion_module_hash = res.get("hash")
                break

        # Init image for i2v workflows — prefer explicit meta, fall back to
        # sibling still in the same post.
        for key in ("Initial image", "sourceImage", "source_image", "driver"):
            val = meta.get(key)
            if isinstance(val, str) and val.startswith("http"):
                init_image_url = val
                break

        if not init_image_url:
            post_id = data.get("postId")
            if post_id:
                try:
                    siblings = _fetch_post_images(post_id, config)
                    for sib in siblings:
                        if sib.get("id") == image_id:
                            continue
                        sib_type = (sib.get("type") or "image").lower()
                        sib_url = sib.get("url")
                        if sib_type == "image" and sib_url:
                            init_image_url = sib_url
                            break
                except Exception as e:
                    console.print(
                        f"  [dim]Could not fetch sibling post images for init image: {e}[/]"
                    )

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
        checkpoint_hash=checkpoint_hash,
        loras=loras,
        vae_name=meta.get("VAE"),
        vae_hash=vae_hash,
        hires_upscale=_float_or_none(meta.get("Hires upscale")),
        hires_upscaler=meta.get("Hires upscaler"),
        hires_denoising=_float_or_none(meta.get("Denoising strength")),
        media_type=media_type,
        frames=frames,
        fps=fps,
        duration=duration,
        motion_module_name=motion_module_name,
        motion_module_hash=motion_module_hash,
        motion_module_version_id=motion_module_version_id,
        init_image_url=init_image_url,
    )


def _float_or_none(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Model Resolution & Download ─────────────────────────────────────────


def _resolve_civitai_version(version_id: int, config=None) -> tuple[str | None, str | None]:
    """Fetch filename and direct download URL from a CivitAI model version ID.

    Returns (filename, download_url) or (None, None) on failure.
    """
    try:
        mv = _fetch_model_version(version_id, config)
        files = mv.get("files", [])
        if files:
            return files[0].get("name"), files[0].get("downloadUrl")
    except Exception as e:
        console.print(f"  [yellow]WARNING: Could not fetch CivitAI version {version_id}: {e}[/]")
    return None, None


def _ensure_civitai_model(
    mm,
    filename: str,
    model_type: str,
    download_url: str,
) -> None:
    """Register and download a model directly from CivitAI.

    Raises on failure. Callers should catch and fall back to
    ``mm.ensure_or_search`` (HF / CivitAI search / FireCrawl) on errors —
    most importantly 401 Unauthorized, which means the file is gated and
    the configured CivitAI token can't fetch it directly.
    """
    entry = mm._auto_register(filename, model_type, [download_url])
    mm._ensure(entry)


def resolve_and_download_models(
    meta: CivitaiGenMeta,
    mm,  # ModelManager -- avoid circular import
    no_download: bool = False,
    config=None,
) -> dict[str, str | list[dict]]:
    """Resolve all model references to local filenames and download as needed.

    When a CivitAI version ID is available, downloads directly from CivitAI
    using the model version's download URL. Falls back to ensure_or_search()
    only when no version ID is available.

    Returns dict ready to be passed as kwargs to workflow factory:
      {
        "checkpoint": "filename.safetensors",
        "vae": "filename.safetensors" | None,
        "loras": [{"filename": "...", "strength": 0.8}, ...],
      }
    """
    result: dict[str, str | list[dict] | None] = {}

    # ── Checkpoint ──
    # CivitAI doesn't always have a checkpoint in the post metadata — many
    # Wan / AnimateDiff videos ship only LoRAs and rely on a base model
    # that isn't on CivitAI. In that case we leave checkpoint=None and let
    # the workflow factory's built-in default kick in.
    ckpt_filename = meta.checkpoint_filename
    ckpt_download_url = None
    have_real_checkpoint = bool(meta.checkpoint_version_id or meta.checkpoint_hash)

    # 1) version ID is the strongest signal
    if meta.checkpoint_version_id:
        vid_filename, vid_url = _resolve_civitai_version(meta.checkpoint_version_id, config)
        if vid_filename:
            ckpt_filename = vid_filename
        if vid_url:
            ckpt_download_url = vid_url

    # 2) by-hash lookup if version ID didn't pan out
    if not ckpt_download_url and meta.checkpoint_hash:
        h_filename, h_url = _resolve_civitai_by_hash(meta.checkpoint_hash, config)
        if h_filename and not ckpt_filename:
            ckpt_filename = h_filename
        if h_url:
            ckpt_download_url = h_url
            console.print(
                f"  [cyan]↳ Resolved checkpoint via hash {meta.checkpoint_hash[:12]}…[/]"
            )

    if have_real_checkpoint:
        if not ckpt_filename:
            ckpt_filename = meta.checkpoint_name.replace(" ", "_") + ".safetensors"

        if not no_download:
            console.print(f"\n  [bold]Checkpoint:[/] {ckpt_filename}")
            try:
                mm.download_with_fallback(
                    ckpt_filename,
                    "checkpoint",
                    primary_url=ckpt_download_url,
                    search_hint=meta.checkpoint_name,
                )
            except Exception as e:
                console.print(f"  [red]x Could not download checkpoint: {e}[/]")
        result["checkpoint"] = ckpt_filename
    else:
        console.print(
            "\n  [yellow]No checkpoint in CivitAI metadata — "
            "workflow will use the factory's built-in default.[/]"
        )
        result["checkpoint"] = None

    # ── VAE ──
    vae_filename = meta.vae_name
    if vae_filename and vae_filename.lower() not in ("automatic", "none", "default", ""):
        if not no_download:
            console.print(f"  [bold]VAE:[/] {vae_filename}")
            try:
                # Try hash lookup first to get a direct CivitAI URL
                vae_url = None
                if meta.vae_hash:
                    h_filename, h_url = _resolve_civitai_by_hash(meta.vae_hash, config)
                    if h_url:
                        vae_url = h_url
                        if h_filename:
                            vae_filename = h_filename
                mm.download_with_fallback(
                    vae_filename, "vae", primary_url=vae_url, search_hint=vae_filename
                )
            except Exception as e:
                console.print(f"  [yellow]WARNING: Could not download VAE: {e} -- using checkpoint VAE[/]")
                vae_filename = None
        result["vae"] = vae_filename
    else:
        # No explicit VAE in metadata. For SDXL we still want to override the
        # checkpoint's bundled VAE: many community SDXL checkpoints (Cinenauts,
        # Juggernaut, RealVis, etc.) ship with an fp16-broken VAE that produces
        # rainbow-speckle NaN noise on decode. Routing through madebyollin's
        # fp16-fix VAE side-steps that bug entirely.
        result["vae"] = None
        try:
            from .config import Config
            cfg = config or Config.load()
            arch_key, _arch = cfg.get_architecture(meta.base_model_raw)
        except Exception:
            arch_key = ""
        if arch_key == "sdxl":
            fix_filename = "sdxl_vae_fp16_fix.safetensors"
            if not no_download:
                console.print(
                    f"  [bold]VAE:[/] {fix_filename} "
                    "[dim](auto-applied to avoid SDXL fp16 NaN bug)[/]"
                )
                # Use download_with_fallback so transient HF connection drops
                # (WinError 10054) fall through to HF/CivitAI alternative-source
                # search instead of leaving us with the broken checkpoint VAE.
                # Plus a small retry loop for plain network flakiness.
                primary_url = (
                    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/"
                    "resolve/main/sdxl_vae.safetensors"
                )
                downloaded = False
                last_err: Exception | None = None
                for attempt in range(3):
                    try:
                        mm.download_with_fallback(
                            fix_filename,
                            "vae",
                            primary_url=primary_url,
                            search_hint="sdxl vae fp16 fix madebyollin",
                        )
                        downloaded = True
                        break
                    except Exception as e:
                        last_err = e
                        msg = str(e)
                        # Only retry on transient network errors; bail
                        # immediately on auth / 404 / other hard failures.
                        transient = any(
                            t in msg
                            for t in (
                                "10054", "10060", "ConnectionReset",
                                "RemoteDisconnected", "ConnectionError",
                                "timed out", "Read timed out", "EOF",
                            )
                        )
                        if not transient:
                            break
                        console.print(
                            f"  [yellow]Transient error on attempt {attempt + 1}/3: "
                            f"{msg[:80]} — retrying…[/]"
                        )
                if downloaded:
                    result["vae"] = fix_filename
                else:
                    console.print(
                        f"  [yellow]WARNING: Could not fetch fp16-fix VAE: {last_err} "
                        "-- falling back to checkpoint VAE (output may show "
                        "rainbow-speckle noise on Cinenauts/Juggernaut/RealVis)[/]"
                    )
            else:
                result["vae"] = fix_filename

    # ── LoRAs ──
    # Dedupe by resolved filename. CivitAI metadata commonly lists the same
    # underlying LoRA file twice via different channels (civitaiResources vs
    # modelVersionIds), with different display names that defeat the upstream
    # fuzzy matcher. If we don't dedupe here, the workflow ends up with the
    # same file stacked through multiple LoraLoader nodes — which blows the
    # effective strength past 1.0 and turns the output into rainbow noise.
    # First occurrence wins so the explicit civitaiResources weight is kept
    # over the modelVersionIds default of 1.0.
    lora_entries: list[dict] = []
    seen_filenames: set[str] = set()
    for lr in meta.loras:
        lora_filename = lr.filename
        lora_download_url = None

        # 1) version ID is the strongest signal
        if lr.version_id:
            vid_filename, vid_url = _resolve_civitai_version(lr.version_id, config)
            if vid_filename:
                lora_filename = vid_filename
            if vid_url:
                lora_download_url = vid_url

        # 2) hash-based CivitAI lookup (works for legacy `resources` entries)
        if not lora_download_url and lr.hash:
            h_filename, h_url = _resolve_civitai_by_hash(lr.hash, config)
            if h_filename and not lora_filename:
                lora_filename = h_filename
            if h_url:
                lora_download_url = h_url
                console.print(
                    f"  [cyan]↳ Resolved LoRA '{lr.name}' via hash {lr.hash[:12]}…[/]"
                )

        if not lora_filename:
            lora_filename = lr.name.replace(" ", "_") + ".safetensors"

        dedupe_key = lora_filename.lower()
        if dedupe_key in seen_filenames:
            console.print(
                f"  [dim]↳ Skipping duplicate LoRA '{lr.name}' "
                f"(already loaded as {lora_filename})[/]"
            )
            continue
        seen_filenames.add(dedupe_key)

        if not no_download:
            console.print(f"  [bold]LoRA:[/] {lora_filename} (strength: {lr.weight})")
            try:
                mm.download_with_fallback(
                    lora_filename,
                    "lora",
                    primary_url=lora_download_url,
                    search_hint=lr.name,
                )
            except Exception as e:
                console.print(f"  [yellow]WARNING: Could not download LoRA '{lr.name}': {e}[/]")

        lora_entries.append({"filename": lora_filename, "strength": lr.weight})

    result["loras"] = lora_entries

    # ── Motion module (AnimateDiff) ──
    if meta.motion_module_name:
        mm_filename: str | None = None
        mm_url: str | None = None

        if meta.motion_module_version_id:
            vid_filename, vid_url = _resolve_civitai_version(
                meta.motion_module_version_id, config
            )
            if vid_filename:
                mm_filename = vid_filename
            if vid_url:
                mm_url = vid_url

        if not mm_url and meta.motion_module_hash:
            h_filename, h_url = _resolve_civitai_by_hash(
                meta.motion_module_hash, config
            )
            if h_filename and not mm_filename:
                mm_filename = h_filename
            if h_url:
                mm_url = h_url

        if not mm_filename:
            mm_filename = meta.motion_module_name.replace(" ", "_") + ".safetensors"

        if not no_download:
            console.print(f"  [bold]Motion module:[/] {mm_filename}")
            try:
                mm.download_with_fallback(
                    mm_filename,
                    "motion",
                    primary_url=mm_url,
                    search_hint=meta.motion_module_name,
                )
            except Exception as e:
                console.print(
                    f"  [yellow]WARNING: Could not download motion module: {e}[/]"
                )
        result["motion_module"] = mm_filename
    else:
        result["motion_module"] = None

    # ── Init image (i2v workflows) ──
    result["init_image"] = None
    if meta.init_image_url and not no_download:
        from .config import Config
        cfg = config or Config.load()
        try:
            comfy_input_dir = cfg.comfyui_data_path / "input"
        except Exception:
            comfy_input_dir = Path("input")

        console.print(f"  [bold]Init image:[/] {meta.init_image_url}")
        local_name = _download_init_image(
            meta.init_image_url,
            meta.image_id,
            comfy_input_dir,
            config=cfg,
        )
        if local_name:
            result["init_image"] = local_name
            meta.init_image_local = local_name

    return result


# ── Workflow Generation ──────────────────────────────────────────────────


def _resolve_checkpoint_from_arch(
    arch: dict,
    mm,
    config,
) -> str | None:
    """Find a usable checkpoint for a video architecture when CivitAI didn't
    supply one.

    Resolution order:
      1. Query ComfyUI's /object_info for installed checkpoint filenames.
      2. If any installed file contains one of arch['checkpoint_match']
         substrings (case-insensitive), return that filename.
      3. Otherwise, walk arch['checkpoint_model_ids'] and call
         ``mm.ensure_model(id)`` to download from the registry. Return the
         resulting filename on first success.
      4. Return None if nothing worked; caller should error clearly.
    """
    from .comfy_client import ComfyClient

    match_patterns = [p.lower() for p in arch.get("checkpoint_match") or []]
    model_ids = arch.get("checkpoint_model_ids") or []

    # 1) query live ComfyUI for installed checkpoints
    installed: list[str] = []
    try:
        client = ComfyClient(config)
        installed = client.get_model_filenames("CheckpointLoaderSimple", "ckpt_name")
    except Exception:
        installed = []

    if installed and match_patterns:
        for fname in installed:
            low = fname.lower()
            if any(p in low for p in match_patterns):
                console.print(
                    f"  [cyan]-> Found local checkpoint matching arch: {fname}[/]"
                )
                return fname

    # 2) try each registry model ID
    for mid in model_ids:
        try:
            console.print(
                f"  [bold]No matching checkpoint installed — downloading '{mid}' "
                "from registry...[/]"
            )
            path = mm.ensure_model(mid)
            return Path(path).name
        except Exception as e:
            console.print(f"  [yellow]WARNING: Could not ensure '{mid}': {e}[/]")

    return None


# SDXL was trained on bucketed ~1 MP resolutions. Sampling well above that
# range — especially with stacked LoRAs and fp16 UNet — produces NaN latents
# which decode to rainbow-speckle noise no matter what VAE is downstream.
# CivitAI's reported "Size" almost always reflects the *post-hires-fix* output,
# not the native sampling resolution, so we clamp to a trained bucket and add
# an automatic hires upscale to recover the requested final size.
_SDXL_BUCKETS: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1536, 640), (640, 1536),
]


def _clamp_sdxl_resolution(
    width: int, height: int, max_megapixels: float = 1.3
) -> tuple[int, int, float | None]:
    """Clamp (width, height) to SDXL's trained bucket range.

    Returns (safe_width, safe_height, hires_upscale_factor). hires factor is
    None when no clamping was needed.
    """
    if width * height <= int(max_megapixels * 1_000_000):
        return width, height, None
    target_aspect = width / height
    # Pick the bucket whose aspect ratio is closest to the target
    best = min(_SDXL_BUCKETS, key=lambda wh: abs((wh[0] / wh[1]) - target_aspect))
    safe_w, safe_h = best
    # Choose the larger of the two scale factors so the hires pass meets or
    # exceeds the requested final size on both axes
    upscale = max(width / safe_w, height / safe_h)
    # Round to 2 decimals — A1111-style hires fix factor
    upscale = round(upscale, 2)
    return safe_w, safe_h, upscale


def generate_clone_workflow(meta: CivitaiGenMeta, model_filenames: dict, config=None) -> dict:
    """Generate an API-format workflow dict that reproduces the CivitAI image.

    Uses the architecture registry from config to pick the right workflow function.
    """
    from .config import Config
    from . import workflows
    from .models import ModelManager

    cfg = config or Config.load()
    arch_key, arch = cfg.get_architecture(meta.base_model_raw)

    # ── Init image fallback ─────────────────────────────────────────────
    # If this arch requires an init image but CivitAI didn't expose one,
    # and the arch declares a t2v_fallback, swap to that instead of
    # leaving a dangling placeholder. The user gets a runnable workflow.
    init_image = model_filenames.get("init_image")
    if arch.get("requires_init_image") and not init_image:
        t2v_fallback = arch.get("t2v_fallback")
        if t2v_fallback and t2v_fallback in cfg._data.get("architectures", {}):
            console.print(
                "\n  [bold yellow]No init image in CivitAI metadata — "
                f"falling back to t2v workflow '{t2v_fallback}'.[/]\n"
                "  [yellow]  The source video's LoRAs were trained for i2v, so "
                "quality may differ from the original.[/]"
            )
            arch_key = t2v_fallback
            arch = cfg._data["architectures"][t2v_fallback]
        else:
            # No fallback defined — keep the legacy placeholder behaviour
            # so at least the JSON is well-formed and the user knows what
            # to drop into input/.
            init_image = "init_image.png"
            console.print(
                "\n  [bold yellow]Init image not found in CivitAI metadata.[/]\n"
                f"  [yellow]  Using placeholder filename '{init_image}'. Drop your source[/]\n"
                f"  [yellow]  still into {cfg.comfyui_data_path / 'input'} before running.[/]"
            )

    workflow_name = arch.get("workflow", "text2img_sdxl")

    # Get the workflow function by name
    workflow_fn = getattr(workflows, workflow_name, None)
    if workflow_fn is None:
        raise ValueError(
            f"Unknown workflow '{workflow_name}' for architecture '{arch_key}'.\n"
            f"  Check architectures.{arch_key}.workflow in config.yaml"
        )

    # Build a superset of kwargs covering image + video factories. The
    # inspect.signature() filter below drops anything a given factory doesn't
    # accept, so no per-backend branching is needed for most fields.
    checkpoint = model_filenames.get("checkpoint")

    # If CivitAI didn't give us a checkpoint and the arch uses a
    # CheckpointLoaderSimple, try to auto-resolve one from what's installed
    # locally or from the model registry. This avoids the factory's
    # hardcoded default filename ending up in the workflow when that file
    # doesn't exist in ComfyUI's checkpoints folder.
    if not checkpoint and arch.get("loader") == "checkpoint":
        mm = ModelManager(cfg)
        resolved = _resolve_checkpoint_from_arch(arch, mm, cfg)
        if resolved:
            checkpoint = resolved
            model_filenames["checkpoint"] = resolved
        else:
            console.print(
                "\n  [yellow]Could not auto-resolve a checkpoint for "
                f"architecture '{arch_key}'. The workflow will use the "
                "factory's built-in default, which may not exist in your "
                "checkpoints folder.[/]"
            )

    kwargs: dict = {
        "prompt": meta.prompt,
        # checkpoint/model are omitted when CivitAI had no real base model;
        # the None-drop below lets the factory's own default kick in.
        "checkpoint": checkpoint,
        "model": checkpoint,  # img2vid_svd uses 'model'
        "loras": model_filenames.get("loras") or None,
        "width": meta.width,
        "height": meta.height,
        "steps": meta.steps,
        "seed": meta.seed,
        # Video-specific (None unless CivitAI reported them)
        "frames": meta.frames,
        "fps": meta.fps,
        "motion_module": model_filenames.get("motion_module"),
        "image_path": init_image,
    }

    loader_type = arch.get("loader", "checkpoint")

    if loader_type == "checkpoint":
        # SDXL/SD1.5/Wan style -- uses CheckpointLoaderSimple
        kwargs["negative"] = meta.negative_prompt or "blurry, low quality, watermark, text, deformed"
        kwargs["cfg"] = meta.cfg_scale
        kwargs["sampler"] = meta.sampler
        kwargs["scheduler"] = meta.scheduler
        vae = model_filenames.get("vae")
        if vae:
            kwargs["vae"] = vae

        # ── SDXL native-resolution clamp ────────────────────────────────
        # SDXL's UNet (especially in fp16 with stacked LoRAs) explodes into
        # NaN latents when sampled well above ~1 MP, producing rainbow
        # speckle that no VAE can decode. CivitAI's reported "Size" usually
        # reflects post-upscale dimensions, so we clamp the native sampling
        # resolution to a trained SDXL bucket. We deliberately do NOT add
        # an auto hires-fix pass: the latent-space second sampler runs the
        # same fp16 UNet over the larger upscaled latent and reintroduces
        # the same NaN speckle (sparser, but still visible). Users who want
        # the full target resolution should upscale the clean PNG with an
        # external pixel-space upscaler (ESRGAN / Real-ESRGAN), which never
        # touches the UNet. Skip clamping if metadata explicitly asked for
        # hires fix — that's the user's responsibility to debug.
        if arch_key == "sdxl" and not meta.hires_upscale:
            safe_w, safe_h, _ = _clamp_sdxl_resolution(meta.width, meta.height)
            if (safe_w, safe_h) != (meta.width, meta.height):
                console.print(
                    f"  [cyan]Clamping SDXL native res {meta.width}x{meta.height} "
                    f"-> {safe_w}x{safe_h} (SDXL is unstable above ~1MP native; "
                    f"upscale the output PNG externally if you need a larger size)[/]"
                )
                kwargs["width"] = safe_w
                kwargs["height"] = safe_h

        if meta.hires_upscale:
            kwargs["hires_upscale"] = meta.hires_upscale
            kwargs["hires_denoising"] = meta.hires_denoising
            kwargs["hires_upscaler"] = meta.hires_upscaler
            console.print(
                f"  [cyan]Hires fix: {meta.hires_upscale}x ({meta.hires_upscaler}), "
                f"denoise: {meta.hires_denoising or 0.5}[/]"
            )

    elif loader_type == "unet":
        # Flux/ZImageTurbo style -- uses UNETLoader + separate CLIP/VAE
        kwargs["guidance"] = meta.cfg_scale

    elif loader_type == "image_only_checkpoint":
        # SVD — ImageOnlyCheckpointLoader bundles its own CLIP/VAE and
        # the factory owns its sampler/cfg defaults. Don't override them.
        pass

    # Drop None values so each factory keeps its own defaults when CivitAI
    # didn't report a field.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Filter kwargs to only params the workflow function accepts
    import inspect
    sig = inspect.signature(workflow_fn)
    valid_params = set(sig.parameters.keys())
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_keyword:
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return workflow_fn(**kwargs)


# ── Summary Display ──────────────────────────────────────────────────────


def _print_summary(meta: CivitaiGenMeta):
    """Print a Rich table summarizing the extracted metadata."""
    kind = "Video" if meta.media_type == "video" else "Image"
    table = Table(title=f"CivitAI {kind} #{meta.image_id}", show_lines=True)
    table.add_column("Field", style="bold", width=20)
    table.add_column("Value", max_width=80)

    table.add_row("Media Type", meta.media_type)
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

    if meta.media_type == "video":
        if meta.frames is not None:
            table.add_row("Frames", str(meta.frames))
        if meta.fps is not None:
            table.add_row("FPS", str(meta.fps))
        if meta.duration is not None:
            table.add_row("Duration", f"{meta.duration:.2f}s")
        if meta.motion_module_name:
            table.add_row("Motion Module", meta.motion_module_name)
        if meta.init_image_url:
            table.add_row("Init Image", meta.init_image_url[:80] + ("..." if len(meta.init_image_url) > 80 else ""))

    console.print(table)


# ── Workflow Preflight ───────────────────────────────────────────────────


# Mapping from node class_type → (custom-node repo name, git URL).
# Used to suggest what to install when a workflow references a node type
# that ComfyUI doesn't have registered. Prefixes match substrings from the
# left, so "ADE_*" routes to AnimateDiff-Evolved etc.
_NODE_HINTS: list[tuple[str, str, str]] = [
    ("VHS_",   "ComfyUI-VideoHelperSuite",    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ("ADE_",   "ComfyUI-AnimateDiff-Evolved", "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git"),
    ("IPAdapter", "ComfyUI_IPAdapter_plus",   "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"),
    ("Reactor", "ComfyUI-Reactor",            "https://github.com/Gourieff/comfyui-reactor-node.git"),
    ("Impact",  "ComfyUI-Impact-Pack",        "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"),
]


def _node_install_hint(class_type: str) -> tuple[str, str] | None:
    """Return (repo_name, git_url) for a known custom-node prefix, else None."""
    for prefix, repo, url in _NODE_HINTS:
        if class_type.startswith(prefix):
            return repo, url
    return None


def preflight_workflow(workflow: dict, config=None) -> dict:
    """Check that every node type in the workflow is registered in ComfyUI.

    Returns a report dict:
      {
        "ok": bool,
        "reachable": bool,       # whether we could talk to ComfyUI
        "required": set[str],    # all unique class_types in the workflow
        "missing": list[str],    # node types ComfyUI didn't report
        "hints": dict[str, tuple[str, str]],  # missing type → (repo, url)
      }

    Prints a Rich-formatted summary. Does not raise — downstream code can
    still save the workflow JSON; this is advisory.
    """
    from .comfy_client import ComfyClient

    required: set[str] = set()
    for node in workflow.values():
        if isinstance(node, dict):
            ct = node.get("class_type")
            if isinstance(ct, str):
                required.add(ct)

    report: dict = {
        "ok": False,
        "reachable": False,
        "required": required,
        "missing": [],
        "hints": {},
    }

    try:
        client = ComfyClient(config)
        installed = set(client.get_installed_nodes())
    except Exception as e:
        console.print(f"  [yellow]Preflight: could not reach ComfyUI ({e}) — skipping node check.[/]")
        return report

    if not installed:
        console.print(
            "  [yellow]Preflight: ComfyUI isn't running or returned no nodes. "
            "Start ComfyUI and re-run to validate node coverage.[/]"
        )
        return report

    report["reachable"] = True
    missing = sorted(n for n in required if n not in installed)
    report["missing"] = missing
    report["ok"] = not missing

    if report["ok"]:
        console.print(
            f"  [green]✓ Preflight: all {len(required)} node types are installed.[/]"
        )
        return report

    # Group missing nodes by suggested custom-node repo
    hints: dict[str, tuple[str, str]] = {}
    by_repo: dict[str, list[str]] = {}
    unknown: list[str] = []
    for n in missing:
        hint = _node_install_hint(n)
        if hint:
            hints[n] = hint
            by_repo.setdefault(hint[0], []).append(n)
        else:
            unknown.append(n)
    report["hints"] = hints

    table = Table(
        title=f"Missing ComfyUI nodes ({len(missing)})",
        show_lines=False,
    )
    table.add_column("Custom node", style="bold")
    table.add_column("Install hint")
    table.add_column("Missing types", max_width=40)

    for repo, nodes in by_repo.items():
        url = next(u for _, (r, u) in hints.items() if r == repo)
        table.add_row(repo, url, ", ".join(nodes))
    if unknown:
        table.add_row("(unknown)", "no hint — search ComfyUI-Manager", ", ".join(unknown))
    console.print(table)

    comfy_data = None
    try:
        from .config import Config
        cfg = config or Config.load()
        comfy_data = cfg.comfyui_data_path
    except Exception:
        pass

    if comfy_data and by_repo:
        custom_nodes_dir = comfy_data / "custom_nodes"
        # Split by_repo into two buckets: repos already present on disk
        # (ComfyUI just needs a restart to pick them up) versus repos that
        # need to be cloned.
        need_restart: list[tuple[str, list[str]]] = []
        need_clone: list[tuple[str, str, list[str]]] = []
        for repo, nodes in by_repo.items():
            url = next(u for _, (r, u) in hints.items() if r == repo)
            if (custom_nodes_dir / repo).exists():
                need_restart.append((repo, nodes))
            else:
                need_clone.append((repo, url, nodes))

        if need_restart:
            restart_list = "\n".join(
                f"    - {repo}  ({', '.join(nodes)})"
                for repo, nodes in need_restart
            )
            console.print(
                "\n  [bold yellow]These custom nodes are already on disk but ComfyUI "
                "hasn't loaded them:[/]\n"
                f"{restart_list}\n"
                "  [yellow]ComfyUI only scans custom_nodes/ at startup. Fix: fully quit "
                "ComfyUI (if using Desktop, right-click the system tray icon -> Quit) "
                "and relaunch it. Then reopen this workflow.[/]"
            )

        if need_clone:
            clone_lines = "\n".join(
                f"    git clone {url}" for _, url, _ in need_clone
            )
            console.print(
                "\n  [bold]To install the missing custom nodes:[/]\n"
                f"    cd {custom_nodes_dir}\n"
                f"{clone_lines}\n"
                "  Then restart ComfyUI."
            )

    return report


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
        images = _fetch_post_images(content_id, cfg)
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
    meta = fetch_image_metadata(image_id, cfg)

    # 3. Print summary
    _print_summary(meta)

    # 4. Resolve and download models
    mm = ModelManager(cfg)
    console.print("\n[bold]Resolving models...[/]")
    model_filenames = resolve_and_download_models(meta, mm, no_download=no_download, config=cfg)

    # 5. Generate workflow
    console.print("\n[bold]Generating workflow...[/]")
    wf = generate_clone_workflow(meta, model_filenames, config=cfg)

    # 5b. Preflight — check every node type the workflow references is
    # registered in the running ComfyUI instance. Advisory only; we still
    # save the JSON even if nodes are missing so the user can install them.
    console.print("\n[bold]Preflight: checking ComfyUI node coverage...[/]")
    preflight_workflow(wf, config=cfg)

    # 6. Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        dest = Path(output_path)
    else:
        dest = out_dir / f"clone_{image_id}.json"

    dest.parent.mkdir(parents=True, exist_ok=True)
    import json
    dest.write_text(json.dumps(wf, indent=2))
    return dest
