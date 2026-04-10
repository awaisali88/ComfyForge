"""Model Manager — auto-download and verify models from CivitAI, HuggingFace, Firebase."""

from __future__ import annotations
import hashlib
import json
import os
import re
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

# ── Model type inference from filename ─────
_TYPE_HINTS = {
    "lora": "lora",
    "locon": "lora",
    "lycoris": "lora",
    "vae": "vae",
    "clip": "clip",
    "t5xxl": "clip",
    "controlnet": "controlnet",
    "control": "controlnet",
    "upscale": "upscale",
    "esrgan": "upscale",
    "animatediff": "motion",
    "mm_sd": "motion",
    "mm_sdxl": "motion",
    "svd": "svd",
    "wan": "wan",
}


def _guess_model_type(filename: str) -> str:
    """Guess model type from filename patterns."""
    lower = filename.lower()
    for hint, mtype in _TYPE_HINTS.items():
        if hint in lower:
            return mtype
    # Default to checkpoint for .safetensors/.ckpt, upscale for .pth
    if lower.endswith(".pth"):
        return "upscale"
    return "checkpoint"

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
            base = self.cfg.comfyui_data_path / "models" / subdir
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

    def ensure_filename(self, filename: str, model_type: str | None = None) -> Path:
        """Ensure a model by filename. Searches online if not in registry."""
        model = self.registry.find_by_filename(filename)
        if model is not None:
            return self._ensure(model)
        # Not in registry — search and auto-download
        return self.ensure_or_search(filename, model_type)

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

    def ensure_or_search(self, filename: str, model_type: str | None = None) -> Path:
        """Ensure a model is available — search HuggingFace/CivitAI if not in registry.

        This is the main entry point for "download if missing" logic.
        1. Check if file already exists locally (any model dir)
        2. Try registry lookup
        3. Search HuggingFace API by filename
        4. Auto-register and download
        """
        # 1. Try registry first
        model = self.registry.find_by_filename(filename)
        if model is not None:
            return self._ensure(model)

        # 2. Check if file already exists locally
        mtype = model_type or _guess_model_type(filename)
        local_path = self._local_model_path(filename, mtype)
        if local_path.exists():
            console.print(f"  [dim]✓ {filename} already exists at {local_path}[/]")
            return local_path

        # 3. Search HuggingFace
        console.print(f"  [yellow]⚡ {filename} not in registry — searching HuggingFace…[/]")
        hf_url = self._search_huggingface(filename)

        if hf_url:
            # Auto-register and download
            entry = self._auto_register(filename, mtype, [hf_url])
            return self._ensure(entry)

        # 4. Search CivitAI
        console.print(f"  [yellow]⚡ Not found on HuggingFace — searching CivitAI…[/]")
        civitai_url = self._search_civitai(filename)

        if civitai_url:
            entry = self._auto_register(filename, mtype, [civitai_url])
            return self._ensure(entry)

        # 5. Search via FireCrawl (web scraping fallback)
        if self.cfg.firecrawl_api_key:
            console.print(f"  [yellow]⚡ Not found on CivitAI — searching the web via FireCrawl…[/]")
            fc_url = self._search_firecrawl(filename, mtype)
            if fc_url:
                entry = self._auto_register(filename, mtype, [fc_url])
                return self._ensure(entry)

        raise RuntimeError(
            f"Could not find {filename} anywhere.\n"
            f"  - Not in models.yaml registry\n"
            f"  - Not found on HuggingFace\n"
            f"  - Not found on CivitAI\n"
            f"  - Not found via FireCrawl web search\n"
            f"  Add it manually: python -m core add-civitai <url> --type {mtype}"
        )

    def _local_model_path(self, filename: str, model_type: str) -> Path:
        """Get the expected local path for a model file."""
        subdir = self.registry._type_dirs.get(model_type, model_type)
        explicit = self.cfg._data["models"].get(model_type, "")
        if explicit:
            return Path(explicit) / filename
        return self.cfg.comfyui_data_path / "models" / subdir / filename

    def _search_huggingface(self, filename: str) -> str | None:
        """Search HuggingFace for a model file by name."""
        try:
            # Use HF Hub API to search for the file
            search_name = filename.replace(".safetensors", "").replace(".ckpt", "").replace(".pth", "")
            with httpx.Client(timeout=30) as client:
                headers = {}
                token = self.cfg.hf_token or ""
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                # Search models API
                resp = client.get(
                    "https://huggingface.co/api/models",
                    params={"search": search_name, "limit": 10, "sort": "downloads"},
                    headers=headers,
                )
                resp.raise_for_status()
                results = resp.json()

                for repo in results:
                    repo_id = repo.get("id", "")
                    # Check if this repo has the exact file
                    try:
                        files_resp = client.get(
                            f"https://huggingface.co/api/models/{repo_id}",
                            headers=headers,
                            params={"blobs": False},
                        )
                        files_resp.raise_for_status()
                        siblings = files_resp.json().get("siblings", [])
                        for sib in siblings:
                            if sib.get("rfilename", "").endswith(filename):
                                url = f"https://huggingface.co/{repo_id}/resolve/main/{sib['rfilename']}"
                                console.print(f"  [green]✓ Found on HuggingFace: {repo_id}[/]")
                                return url
                    except Exception:
                        continue

        except Exception as e:
            console.print(f"  [dim]HuggingFace search failed: {e}[/]")
        return None

    def _search_civitai(self, filename: str) -> str | None:
        """Search CivitAI for a model by filename."""
        try:
            search_name = filename.replace(".safetensors", "").replace(".ckpt", "")
            with httpx.Client(timeout=30) as client:
                headers = {}
                token = os.getenv("CIVITAI_API_TOKEN", "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                resp = client.get(
                    f"{self.CIVITAI_API}/models",
                    params={"query": search_name, "limit": 5, "sort": "Most Downloaded"},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("items", []):
                    for version in item.get("modelVersions", []):
                        for f in version.get("files", []):
                            if f.get("name", "").lower() == filename.lower():
                                url = f.get("downloadUrl")
                                if url:
                                    console.print(f"  [green]✓ Found on CivitAI: {item.get('name')}[/]")
                                    return url
        except Exception as e:
            console.print(f"  [dim]CivitAI search failed: {e}[/]")
        return None

    def _search_firecrawl(self, filename: str, model_type: str) -> str | None:
        """Use FireCrawl to search the web for a model download link."""
        api_key = self.cfg.firecrawl_api_key
        if not api_key:
            return None

        try:
            search_name = filename.replace(".safetensors", "").replace(".ckpt", "").replace(".pth", "")

            # Search for the model on known sites
            queries = [
                f"{search_name} safetensors download site:huggingface.co",
                f"{search_name} safetensors download site:civitai.com",
                f"{search_name} {model_type} model download safetensors",
            ]

            with httpx.Client(timeout=60) as client:
                for query in queries:
                    resp = client.post(
                        "https://api.firecrawl.dev/v1/search",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query": query,
                            "limit": 5,
                        },
                    )
                    resp.raise_for_status()
                    results = resp.json()

                    for item in results.get("data", []):
                        url = item.get("url", "")
                        markdown = item.get("markdown", "")

                        # Extract direct download link from HuggingFace pages
                        if "huggingface.co" in url:
                            dl_url = self._extract_hf_download_url(url, filename, markdown)
                            if dl_url:
                                console.print(f"  [green]✓ Found via FireCrawl (HuggingFace): {url}[/]")
                                return dl_url

                        # Extract download link from CivitAI pages
                        if "civitai.com" in url:
                            dl_url = self._extract_civitai_download_url(url, filename, markdown)
                            if dl_url:
                                console.print(f"  [green]✓ Found via FireCrawl (CivitAI): {url}[/]")
                                return dl_url

        except Exception as e:
            console.print(f"  [dim]FireCrawl search failed: {e}[/]")
        return None

    def _extract_hf_download_url(self, page_url: str, filename: str, markdown: str) -> str | None:
        """Extract a direct download URL from a HuggingFace page/markdown."""
        # Pattern: https://huggingface.co/{org}/{repo}/resolve/main/{path}
        # If page URL is a model page, try to construct resolve URL
        import re as _re

        # Look for direct file links in markdown content
        pattern = rf'(https://huggingface\.co/[^/]+/[^/]+/(?:resolve|blob)/main/[^\s\)]*{_re.escape(filename)})'
        match = _re.search(pattern, markdown)
        if match:
            url = match.group(1).replace("/blob/", "/resolve/")
            return url

        # Try to construct from the page URL itself
        # e.g. https://huggingface.co/org/repo → resolve/main/filename
        hf_match = _re.match(r'https://huggingface\.co/([^/]+/[^/]+)', page_url)
        if hf_match and filename.lower() in markdown.lower():
            repo_id = hf_match.group(1)
            # Check if the filename appears in the file listing
            return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

        return None

    def _extract_civitai_download_url(self, page_url: str, filename: str, markdown: str) -> str | None:
        """Extract a download URL from a CivitAI page."""
        import re as _re

        # Look for direct API download links in the content
        pattern = r'(https://civitai\.com/api/download/models/\d+)'
        match = _re.search(pattern, markdown)
        if match:
            return match.group(1)

        # Extract model version ID from URL pattern like /models/12345/...
        # and construct API download URL
        vid_match = _re.search(r'civitai\.com/models/(\d+)', page_url)
        if vid_match and filename.lower() in markdown.lower():
            model_id = vid_match.group(1)
            # Try to get the download URL via CivitAI API
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.get(f"{self.CIVITAI_API}/models/{model_id}")
                    resp.raise_for_status()
                    data = resp.json()
                    for version in data.get("modelVersions", []):
                        for f in version.get("files", []):
                            if f.get("name", "").lower() == filename.lower():
                                return f.get("downloadUrl")
            except Exception:
                pass

        return None

    def _auto_register(self, filename: str, model_type: str, sources: list[str]) -> dict:
        """Add a discovered model to models.yaml and reload registry."""
        model_id = re.sub(r"[^a-zA-Z0-9_]", "_", filename.rsplit(".", 1)[0]).lower()

        entry = {
            "id": model_id,
            "filename": filename,
            "type": model_type,
            "size_gb": 0,
            "tags": [model_type, "auto-discovered"],
            "sources": sources,
        }

        # Append to registry file
        reg_path = Path("configs/models.yaml")
        data = yaml.safe_load(reg_path.read_text()) or {}

        # Check for duplicate
        existing = data.get("models", [])
        for m in existing:
            if m["filename"] == filename:
                console.print(f"  [dim]Already in registry: {filename}[/]")
                return m

        data.setdefault("models", []).append(entry)
        reg_path.write_text(yaml.dump(data, sort_keys=False, default_flow_style=False))

        console.print(f"  [green]✓ Auto-registered {filename} in models.yaml[/]")
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
            token = self.cfg.hf_token or ""
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
