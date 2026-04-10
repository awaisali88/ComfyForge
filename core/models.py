"""Model Manager — auto-download and verify models from CivitAI, HuggingFace, Firebase."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from .config import Config

console = Console()

# ── Firebase integration (lazy) ─────────────
_firebase_app = None
_firestore_client = None


def _init_firebase(cfg: Config):
    global _firebase_app, _firestore_client
    if _firebase_app is not None:
        return
    cred_path = cfg.firebase_credentials
    if not cred_path or not Path(cred_path).exists():
        return
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(cred_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
        console.print("[green]✓ Firebase connected[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Firebase init failed: {e}[/]")


def _firebase_url(model_id: str, cfg: Config) -> str | None:
    """Look up a faster/private download URL from Firestore."""
    _init_firebase(cfg)
    if _firestore_client is None:
        return None
    try:
        doc = _firestore_client.collection(cfg.firebase_collection).document(model_id).get()
        if doc.exists:
            data = doc.to_dict()
            url = data.get("download_url") or data.get("url")
            if url:
                console.print(f"  [cyan]↳ Firebase mirror found[/]")
                return url
    except Exception:
        pass
    return None


# ── Registry ────────────────────────────────

class ModelRegistry:
    """Loads models.yaml and resolves what needs downloading."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._models: list[dict[str, Any]] = []
        self._type_dirs: dict[str, str] = {}
        self._load()

    def _load(self):
        p = Path("configs/models.yaml")
        if not p.exists():
            console.print("[red]✗ configs/models.yaml not found[/]")
            return
        data = yaml.safe_load(p.read_text()) or {}
        self._models = data.get("models", [])
        self._type_dirs = data.get("type_directories", {})

    def find(self, model_id: str) -> dict | None:
        for m in self._models:
            if m["id"] == model_id:
                return m
        return None

    def find_by_filename(self, filename: str) -> dict | None:
        for m in self._models:
            if m["filename"] == filename:
                return m
        return None

    def find_by_tags(self, tags: list[str]) -> list[dict]:
        """Find models matching ANY of the given tags, ranked by overlap."""
        results = []
        for m in self._models:
            mt = set(m.get("tags", []))
            overlap = len(mt & set(tags))
            if overlap > 0:
                results.append((overlap, m))
        results.sort(key=lambda x: -x[0])
        return [m for _, m in results]

    def dest_path(self, model: dict) -> Path:
        mtype = model["type"]
        subdir = self._type_dirs.get(mtype, mtype)
        explicit = self.cfg._data["models"].get(mtype, "")
        if explicit:
            base = Path(explicit)
        else:
            base = self.cfg.comfyui_path / "models" / subdir
        return base / model["filename"]

    def all_models(self) -> list[dict]:
        return self._models


# ── Downloader ──────────────────────────────

class ModelManager:
    """Download and manage models."""

    CIVITAI_API = "https://civitai.com/api/v1"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.registry = ModelRegistry(self.cfg)

    def ensure_model(self, model_id: str) -> Path:
        """Ensure a model is downloaded. Returns local path."""
        model = self.registry.find(model_id)
        if model is None:
            raise ValueError(f"Unknown model: {model_id}")
        return self._ensure(model)

    def ensure_filename(self, filename: str) -> Path:
        """Ensure a model by filename. Returns local path."""
        model = self.registry.find_by_filename(filename)
        if model is None:
            raise ValueError(f"Unknown model file: {filename}")
        return self._ensure(model)

    def ensure_stack(self, pipeline: str) -> dict[str, Path]:
        """Download all models needed for a pipeline stack."""
        stack = self.cfg.stack(pipeline)
        if not stack:
            raise ValueError(f"No stack config for pipeline: {pipeline}")

        paths: dict[str, Path] = {}

        # Checkpoint
        if ckpt := stack.get("checkpoint"):
            paths["checkpoint"] = self.ensure_filename(ckpt)

        # VAE
        if vae := stack.get("vae"):
            paths["vae"] = self.ensure_filename(vae)

        # CLIP (can be list)
        clips = stack.get("clip", [])
        if isinstance(clips, str):
            clips = [clips]
        for i, c in enumerate(clips):
            paths[f"clip_{i}"] = self.ensure_filename(c)

        # LoRAs
        for i, lora in enumerate(stack.get("loras", [])):
            name = lora if isinstance(lora, str) else lora["filename"]
            paths[f"lora_{i}"] = self.ensure_filename(name)

        # Motion module
        if mm := stack.get("motion_module"):
            paths["motion_module"] = self.ensure_filename(mm)

        # Generic model field
        if m := stack.get("model"):
            paths["model"] = self.ensure_filename(m)

        return paths

    def ensure_for_prompt(self, prompt: str, pipeline: str) -> dict[str, Path]:
        """Smart model selection based on prompt keywords + pipeline."""
        # Start with the default stack
        return self.ensure_stack(pipeline)

    def add_civitai_model(
        self,
        model_url_or_id: str,
        model_type: str = "lora",
        tags: list[str] | None = None,
    ) -> dict:
        """Add a model from CivitAI by URL or model version ID."""
        # Extract version ID from URL
        vid = model_url_or_id.split("/")[-1] if "/" in model_url_or_id else model_url_or_id

        # Fetch metadata
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{self.CIVITAI_API}/model-versions/{vid}")
            resp.raise_for_status()
            meta = resp.json()

        filename = meta["files"][0]["name"]
        download_url = meta["files"][0]["downloadUrl"]
        size_kb = meta["files"][0].get("sizeKB", 0)

        entry = {
            "id": f"civitai_{vid}",
            "filename": filename,
            "type": model_type,
            "size_gb": round(size_kb / 1024 / 1024, 2),
            "tags": tags or [],
            "sources": [download_url],
        }

        # Append to registry file
        reg_path = Path("configs/models.yaml")
        data = yaml.safe_load(reg_path.read_text()) or {}
        data.setdefault("models", []).append(entry)
        reg_path.write_text(yaml.dump(data, sort_keys=False))

        console.print(f"[green]✓ Added {filename} from CivitAI[/]")
        self.registry = ModelRegistry(self.cfg)  # reload
        return entry

    # ── internals ────────────────────────

    def _ensure(self, model: dict) -> Path:
        dest = self.registry.dest_path(model)
        if dest.exists():
            console.print(f"  [dim]✓ {model['filename']} already exists[/]")
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"  [bold]↓ Downloading {model['filename']}[/] ({model.get('size_gb', '?')} GB)")

        # Build source list: Firebase first, then registry sources
        sources = []
        fb_url = _firebase_url(model["id"], self.cfg)
        if fb_url:
            sources.append(fb_url)
        sources.extend(model.get("sources", []))

        for url in sources:
            try:
                self._download(url, dest)
                # Verify hash if available
                if sha := model.get("sha256"):
                    if not self._verify_sha256(dest, sha):
                        console.print(f"  [red]✗ SHA256 mismatch, retrying next source[/]")
                        dest.unlink()
                        continue
                console.print(f"  [green]✓ Saved → {dest}[/]")
                return dest
            except Exception as e:
                console.print(f"  [yellow]⚠ Failed from {url[:60]}…: {e}[/]")
                if dest.exists():
                    dest.unlink()

        raise RuntimeError(f"Failed to download {model['filename']} from all sources")

    def _download(self, url: str, dest: Path):
        headers = {}
        # CivitAI API token support
        if "civitai.com" in url:
            token = os.getenv("CIVITAI_API_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # HuggingFace token support
        if "huggingface.co" in url:
            token = os.getenv("HF_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=httpx.Timeout(10, read=3600), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))

                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task(dest.name, total=total)
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> bool:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
                h.update(chunk)
        return h.hexdigest().lower() == expected.lower()
