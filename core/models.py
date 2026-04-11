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


def _norm(s: str | None) -> str:
    """Normalize a name for fuzzy comparison: lowercase, strip separators."""
    if not s:
        return ""
    return re.sub(r"[\s_\-\.]+", "", s).lower()


def _domain(url: str) -> str:
    """Extract a short host label from a URL for log output."""
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1) if m else url


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

    def download_with_fallback(
        self,
        filename: str,
        model_type: str,
        *,
        primary_url: str | None = None,
        search_hint: str | None = None,
    ) -> Path:
        """Download a model, with automatic fallback when the primary source fails.

        This is the function callers should use when they already have a
        candidate download URL (e.g. resolved via CivitAI version-id or
        by-hash) but want graceful recovery when that URL turns out to be
        gated (401), removed (404), rate-limited, or otherwise broken.

        Order of operations:
          1. If the file is already downloaded locally, return it.
          2. Try ``primary_url`` if provided.
          3. On failure (especially 401 Unauthorized), search HF / CivitAI /
             FireCrawl for alternative sources.
          4. Prepend any working alternatives to the registry entry's
             ``sources`` list and retry the download.
          5. As a final fallback, run the full ``ensure_or_search`` chain.
        """
        # 1) Already on disk?
        existing = self.registry.find_by_filename(filename)
        if existing:
            dest = self.registry.dest_path(existing)
            if dest.exists():
                console.print(f"  [dim]✓ {filename} already exists[/]")
                return dest

        last_error: Exception | None = None
        primary_failed_with_auth = False
        gated_page_url: str | None = None

        # 2) Try the primary URL
        if primary_url:
            try:
                entry = self._auto_register(filename, model_type, [primary_url])
                return self._ensure(entry)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "401" in msg or "Unauthorized" in msg:
                    primary_failed_with_auth = True
                    gated_page_url = self._civitai_page_url_for_download(primary_url)
                    console.print(
                        "  [yellow]⚡ Primary source returned 401 (gated / auth required) "
                        "— searching alternative sources…[/]"
                    )
                    if gated_page_url:
                        console.print(
                            f"  [cyan]→ Accept the model's terms here: {gated_page_url}[/]"
                        )
                else:
                    console.print(
                        f"  [yellow]⚡ Primary source failed ({msg[:80]}) "
                        "— searching alternative sources…[/]"
                    )

        # 3) Search HF / CivitAI / FireCrawl for alternatives
        alt_urls = self.search_alternative_urls(
            filename, model_type, search_hint=search_hint
        )
        # Drop the URL we already tried
        alt_urls = [u for u in alt_urls if u and u != primary_url]

        if alt_urls:
            console.print(
                f"  [cyan]Found {len(alt_urls)} alternative source(s) — retrying…[/]"
            )
            self._merge_sources(filename, model_type, alt_urls)
            entry = self.registry.find_by_filename(filename)
            try:
                return self._ensure(entry)
            except Exception as e:
                last_error = e
                console.print(f"  [yellow]Alternative sources also failed: {e}[/]")

        # 4) Last-ditch: run the full ensure_or_search chain (handles the case
        #    where there was no primary URL at all and no entry in the registry).
        if not primary_url:
            try:
                return self.ensure_or_search(
                    filename, model_type, search_hint=search_hint
                )
            except Exception as e:
                last_error = e

        # Surface a useful diagnostic
        if primary_failed_with_auth:
            page_line = (
                f"\n  → Open this URL, log in, and click 'I accept' / age-confirm:\n"
                f"      {gated_page_url}"
                if gated_page_url
                else ""
            )
            raise RuntimeError(
                f"Could not download {filename}.\n"
                f"  - CivitAI returned 401 Unauthorized for the direct URL.\n"
                f"    Your token IS configured, but the model is gated — your CivitAI\n"
                f"    account hasn't accepted its terms / age verification yet.\n"
                f"  - No alternative sources found on HuggingFace or via FireCrawl."
                f"{page_line}"
            )
        if last_error:
            raise last_error
        raise RuntimeError(f"Could not download {filename} — no sources available.")

    def search_alternative_urls(
        self,
        filename: str,
        model_type: str,
        *,
        search_hint: str | None = None,
    ) -> list[str]:
        """Run the HF → CivitAI search → FireCrawl chain and collect any
        download URLs found, without touching the registry. Use this to find
        fallback sources after a primary source has failed."""
        urls: list[str] = []

        try:
            hf = self._search_huggingface(filename, search_hint=search_hint)
            if hf:
                urls.append(hf)
        except Exception as e:
            console.print(f"  [dim]HuggingFace alt-search failed: {e}[/]")

        try:
            civitai_url, _ = self._search_civitai(filename, search_hint=search_hint)
            if civitai_url:
                urls.append(civitai_url)
        except Exception as e:
            console.print(f"  [dim]CivitAI alt-search failed: {e}[/]")

        if self.cfg.firecrawl_api_key:
            try:
                fc_url, _ = self._search_firecrawl(
                    filename, model_type, search_hint=search_hint
                )
                if fc_url:
                    urls.append(fc_url)
            except Exception as e:
                console.print(f"  [dim]FireCrawl alt-search failed: {e}[/]")

        return urls

    def _merge_sources(
        self, filename: str, model_type: str, new_sources: list[str]
    ) -> None:
        """Prepend ``new_sources`` to an existing registry entry's source list,
        or create a new entry if none exists. Persists to ``models.yaml`` and
        reloads the in-memory registry."""
        reg_path = Path("configs/models.yaml")
        data = yaml.safe_load(reg_path.read_text()) or {}
        models = data.get("models", [])

        target = None
        for m in models:
            if m["filename"] == filename:
                target = m
                break

        if target is None:
            self._auto_register(filename, model_type, new_sources)
            return

        existing = target.get("sources", []) or []
        merged: list[str] = []
        for s in list(new_sources) + list(existing):
            if s and s not in merged:
                merged.append(s)
        target["sources"] = merged

        reg_path.write_text(yaml.dump(data, sort_keys=False, default_flow_style=False))
        self.registry = ModelRegistry(self.cfg)  # reload

    def ensure_or_search(
        self,
        filename: str,
        model_type: str | None = None,
        *,
        search_hint: str | None = None,
    ) -> Path:
        """Ensure a model is available — search HuggingFace/CivitAI/FireCrawl if not in registry.

        ``search_hint`` is an optional human-readable name (e.g. ``"Korean Doll
        Likeness v2.0"``) to use when the filename alone is too cryptic for
        search engines. Used by the CivitAI clone path.

        Order of operations:
          1. Registry lookup
          2. Local filesystem check
          3. HuggingFace API search
          4. CivitAI API search (filename + hint, fuzzy match)
          5. FireCrawl web search → scrape → extract direct download URL
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
        hf_url = self._search_huggingface(filename, search_hint=search_hint)

        if hf_url:
            # Auto-register and download
            entry = self._auto_register(filename, mtype, [hf_url])
            return self._ensure(entry)

        # 4. Search CivitAI
        console.print(f"  [yellow]⚡ Not found on HuggingFace — searching CivitAI…[/]")
        civitai_url, civitai_filename = self._search_civitai(filename, search_hint=search_hint)

        if civitai_url:
            # If CivitAI returned a different (canonical) filename, use it so the
            # downloaded file matches what the workflow references via fuzzy resolution.
            entry = self._auto_register(civitai_filename or filename, mtype, [civitai_url])
            return self._ensure(entry)

        # 5. Search via FireCrawl (web scraping fallback)
        if self.cfg.firecrawl_api_key:
            console.print(f"  [yellow]⚡ Not found on CivitAI — searching the web via FireCrawl…[/]")
            fc_url, fc_filename = self._search_firecrawl(filename, mtype, search_hint=search_hint)
            if fc_url:
                entry = self._auto_register(fc_filename or filename, mtype, [fc_url])
                return self._ensure(entry)
        else:
            console.print(
                "  [dim]FireCrawl skipped (no FIRECRAWL_API_KEY configured).[/]"
            )

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

    def _search_huggingface(
        self, filename: str, search_hint: str | None = None
    ) -> str | None:
        """Search HuggingFace for a model file by name."""
        try:
            stem = filename.replace(".safetensors", "").replace(".ckpt", "").replace(".pth", "")
            queries: list[str] = []
            if search_hint:
                queries.append(search_hint)
            queries.append(stem)

            with httpx.Client(timeout=30) as client:
                headers = {}
                token = self.cfg.hf_token or ""
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                seen_repos: set[str] = set()
                for q in queries:
                    resp = client.get(
                        "https://huggingface.co/api/models",
                        params={"search": q, "limit": 10, "sort": "downloads"},
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        continue
                    results = resp.json()

                    for repo in results:
                        repo_id = repo.get("id", "")
                        if repo_id in seen_repos:
                            continue
                        seen_repos.add(repo_id)
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

    def _search_civitai(
        self, filename: str, search_hint: str | None = None
    ) -> tuple[str | None, str | None]:
        """Search CivitAI for a model by filename or display name.

        Returns ``(downloadUrl, canonical_filename)``. The canonical filename
        is what CivitAI actually serves the file as — may differ slightly from
        the requested ``filename`` (e.g. case, version suffix).
        """
        try:
            stem = filename.replace(".safetensors", "").replace(".ckpt", "")
            stem_lower = stem.lower()
            filename_lower = filename.lower()

            queries: list[str] = []
            if search_hint:
                queries.append(search_hint)
            queries.append(stem)

            with httpx.Client(timeout=30) as client:
                headers = {}
                token = self.cfg.civitai_api_token or os.getenv("CIVITAI_API_TOKEN", "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                # Collect candidates across queries
                exact: tuple[str, str, str] | None = None  # (url, fname, model_name)
                fuzzy: tuple[str, str, str] | None = None
                name_match: tuple[str, str, str] | None = None

                seen_model_ids: set[int] = set()
                for q in queries:
                    resp = client.get(
                        f"{self.CIVITAI_API}/models",
                        params={"query": q, "limit": 10, "sort": "Most Downloaded"},
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                    for item in data.get("items", []):
                        mid = item.get("id")
                        if mid in seen_model_ids:
                            continue
                        seen_model_ids.add(mid)
                        model_name = item.get("name", "") or ""
                        model_name_norm = _norm(model_name)
                        hint_norm = _norm(search_hint or stem)

                        for version in item.get("modelVersions", []):
                            for f in version.get("files", []):
                                fname = f.get("name", "") or ""
                                url = f.get("downloadUrl")
                                if not url:
                                    continue
                                fname_lower = fname.lower()
                                fstem = fname_lower.rsplit(".", 1)[0]

                                # Pass 1: exact filename
                                if fname_lower == filename_lower and exact is None:
                                    exact = (url, fname, model_name)
                                # Pass 2: filename stem substring either way
                                elif (
                                    fuzzy is None and (
                                        stem_lower == fstem
                                        or stem_lower in fstem
                                        or fstem in stem_lower
                                    )
                                ):
                                    fuzzy = (url, fname, model_name)
                                # Pass 3: model name fuzzy-matches the hint
                                elif (
                                    name_match is None
                                    and hint_norm
                                    and (
                                        hint_norm in model_name_norm
                                        or model_name_norm in hint_norm
                                    )
                                ):
                                    name_match = (url, fname, model_name)

                            if exact:
                                break
                        if exact:
                            break
                    if exact:
                        break

                hit = exact or fuzzy or name_match
                if hit:
                    url, fname, model_name = hit
                    label = "exact" if hit is exact else ("fuzzy" if hit is fuzzy else "name-match")
                    console.print(
                        f"  [green]✓ Found on CivitAI: {model_name} → {fname} ({label})[/]"
                    )
                    return url, fname
        except Exception as e:
            console.print(f"  [dim]CivitAI search failed: {e}[/]")
        return None, None

    def _search_firecrawl(
        self,
        filename: str,
        model_type: str,
        search_hint: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Search the web via FireCrawl, scrape candidate pages, and extract a
        direct download URL for the requested model.

        Returns ``(downloadUrl, canonical_filename)``.

        Strategy:
          1. Build search queries from both ``search_hint`` and ``filename``.
          2. Use FireCrawl ``/v1/search`` with ``scrapeOptions`` so each result
             comes back with full markdown — no separate scrape round-trip.
          3. For every result URL, route by domain:
               - civitai.com/models/...   → resolve via CivitAI API
               - huggingface.co/...       → construct /resolve/main/ URL
               - anything else            → regex-extract a .safetensors link
          4. As a last resort, ``/v1/scrape`` the most promising URL for a
             deeper look.
        """
        api_key = self.cfg.firecrawl_api_key
        if not api_key:
            return None, None

        stem = filename.replace(".safetensors", "").replace(".ckpt", "").replace(".pth", "")
        hint = search_hint or stem

        # Build queries — hint-based first (more likely to land on the right page)
        queries: list[str] = []
        if search_hint and search_hint != stem:
            queries.extend([
                f'"{search_hint}" {model_type} site:civitai.com',
                f'"{search_hint}" {model_type} safetensors site:huggingface.co',
                f'{search_hint} {model_type} download safetensors',
            ])
        queries.extend([
            f'"{stem}" site:civitai.com',
            f'"{stem}" safetensors site:huggingface.co',
            f'{stem} {model_type} safetensors download',
        ])

        seen_urls: set[str] = set()
        candidates: list[tuple[str, str]] = []  # (url, markdown)

        try:
            with httpx.Client(timeout=120) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                for query in queries:
                    try:
                        resp = client.post(
                            "https://api.firecrawl.dev/v1/search",
                            headers=headers,
                            json={
                                "query": query,
                                "limit": 5,
                                "scrapeOptions": {"formats": ["markdown"]},
                            },
                        )
                        if resp.status_code != 200:
                            continue
                        results = resp.json().get("data", []) or []
                    except Exception:
                        continue

                    for item in results:
                        url = item.get("url", "") or ""
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        markdown = item.get("markdown", "") or item.get("content", "") or ""
                        candidates.append((url, markdown))

                        dl_url, dl_filename = self._firecrawl_extract(
                            url, filename, hint, markdown, client, headers
                        )
                        if dl_url:
                            console.print(
                                f"  [green]✓ Found via FireCrawl ({_domain(url)}): {dl_filename or filename}[/]"
                            )
                            return dl_url, dl_filename

                # Deep fallback: explicitly /v1/scrape the top candidates that
                # we couldn't extract from on the first pass.
                for url, _ in candidates[:3]:
                    try:
                        resp = client.post(
                            "https://api.firecrawl.dev/v1/scrape",
                            headers=headers,
                            json={"url": url, "formats": ["markdown"]},
                        )
                        if resp.status_code != 200:
                            continue
                        scraped = (resp.json().get("data") or {}).get("markdown", "")
                        if not scraped:
                            continue
                        dl_url, dl_filename = self._firecrawl_extract(
                            url, filename, hint, scraped, client, headers
                        )
                        if dl_url:
                            console.print(
                                f"  [green]✓ Found via FireCrawl deep-scrape ({_domain(url)})[/]"
                            )
                            return dl_url, dl_filename
                    except Exception:
                        continue

        except Exception as e:
            console.print(f"  [dim]FireCrawl search failed: {e}[/]")
        return None, None

    def _firecrawl_extract(
        self,
        page_url: str,
        filename: str,
        hint: str,
        markdown: str,
        client: httpx.Client,
        headers: dict,
    ) -> tuple[str | None, str | None]:
        """Route a single FireCrawl result by domain and extract a download URL."""
        if "civitai.com" in page_url:
            return self._extract_civitai_download_url(page_url, filename, hint, markdown)
        if "huggingface.co" in page_url:
            url = self._extract_hf_download_url(page_url, filename, markdown)
            return (url, filename if url else None)

        # Generic page: look for any direct safetensors link in the markdown
        import re as _re
        for m in _re.finditer(r'(https?://[^\s\)\]"\'<>]+\.safetensors)', markdown):
            link = m.group(1)
            link_lower = link.lower()
            stem = filename.lower().rsplit(".", 1)[0]
            hint_norm = _norm(hint)
            if (
                filename.lower() in link_lower
                or stem in link_lower
                or (hint_norm and hint_norm in _norm(link))
            ):
                # Derive filename from the URL itself
                derived = link.rsplit("/", 1)[-1]
                return link, derived
        return None, None

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

    def _extract_civitai_download_url(
        self,
        page_url: str,
        filename: str,
        hint: str,
        markdown: str,
    ) -> tuple[str | None, str | None]:
        """Extract a download URL from a CivitAI model page.

        Returns ``(downloadUrl, canonical_filename)``.
        """
        import re as _re

        # 1. Direct API download link sitting in the page markdown
        api_match = _re.search(r'(https://civitai\.com/api/download/models/\d+)', markdown)
        if api_match:
            # We don't know the canonical filename in this case
            return api_match.group(1), None

        # 2. Land on /models/{id}, hit the CivitAI API for the full model
        vid_match = _re.search(r'civitai\.com/models/(\d+)', page_url)
        if not vid_match:
            return None, None
        model_id = vid_match.group(1)

        try:
            headers = {}
            token = self.cfg.civitai_api_token or os.getenv("CIVITAI_API_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{self.CIVITAI_API}/models/{model_id}", headers=headers
                )
                resp.raise_for_status()
                data = resp.json()

            model_name = data.get("name", "") or ""
            versions = data.get("modelVersions", []) or []
            stem = filename.lower().rsplit(".", 1)[0]
            hint_norm = _norm(hint)
            model_name_norm = _norm(model_name)

            exact: tuple[str, str] | None = None
            fuzzy: tuple[str, str] | None = None

            for version in versions:
                for f in version.get("files", []):
                    fname = f.get("name", "") or ""
                    fname_lower = fname.lower()
                    fstem = fname_lower.rsplit(".", 1)[0]
                    url = f.get("downloadUrl")
                    if not url:
                        continue
                    if fname_lower == filename.lower() and exact is None:
                        exact = (url, fname)
                    elif fuzzy is None and (
                        stem == fstem or stem in fstem or fstem in stem
                    ):
                        fuzzy = (url, fname)

            if exact:
                return exact
            if fuzzy:
                return fuzzy

            # 3. Last resort: trust that this page corresponds to the requested
            #    model (its name fuzzy-matches the hint) and return the latest
            #    version's primary file. We only do this when the model name is
            #    a confident match — otherwise we'd risk grabbing the wrong file.
            if (
                hint_norm
                and model_name_norm
                and (hint_norm in model_name_norm or model_name_norm in hint_norm)
                and versions
            ):
                files = versions[0].get("files", []) or []
                # Prefer the primary/Model file over VAE/config sidecars
                primary = next(
                    (f for f in files if (f.get("type", "") or "").lower() == "model"),
                    files[0] if files else None,
                )
                if primary and primary.get("downloadUrl"):
                    console.print(
                        f"  [yellow]Using latest-version match from CivitAI: {primary.get('name')}[/]"
                    )
                    return primary["downloadUrl"], primary.get("name")
        except Exception as e:
            console.print(f"  [dim]CivitAI page resolve failed: {e}[/]")

        return None, None

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

    def _civitai_page_url_for_download(self, download_url: str) -> str | None:
        """Resolve a CivitAI download URL (/api/download/models/{version_id})
        to the human-facing model page URL, so 401 errors can show the user
        exactly where to click 'accept terms'.
        """
        m = re.search(r"/api/download/models/(\d+)", download_url)
        if not m:
            return None
        version_id = m.group(1)
        try:
            headers = {}
            token = self.cfg.civitai_api_token or os.getenv("CIVITAI_API_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{self.CIVITAI_API}/model-versions/{version_id}", headers=headers
                )
                if resp.status_code != 200:
                    # Even without the model_id we can give a usable URL via query param
                    return f"https://civitai.com/models?modelVersionId={version_id}"
                mv = resp.json()
                model_id = (mv.get("modelId")
                            or (mv.get("model") or {}).get("id"))
                if model_id:
                    return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
                return f"https://civitai.com/models?modelVersionId={version_id}"
        except Exception:
            return f"https://civitai.com/models?modelVersionId={version_id}"

    def _download(self, url: str, dest: Path):
        headers = {}
        # CivitAI API token support
        if "civitai.com" in url:
            token = self.cfg.civitai_api_token or os.getenv("CIVITAI_API_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # HuggingFace token support
        if "huggingface.co" in url:
            token = self.cfg.hf_token or ""
            if token:
                headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=httpx.Timeout(10, read=3600), follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 401:
                    # Surface a clear, actionable 401 message so the caller's
                    # fallback can recognize the auth-gate case.
                    if "civitai.com" in url:
                        page_url = self._civitai_page_url_for_download(url)
                        page_hint = f"\n  Model page: {page_url}" if page_url else ""
                        raise RuntimeError(
                            "401 Unauthorized from CivitAI — model is gated. "
                            "Token is set in config, but the model requires you to accept "
                            "its terms / age verification on the website first." + page_hint
                        )
                    raise RuntimeError(f"401 Unauthorized for {url[:80]}")
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
