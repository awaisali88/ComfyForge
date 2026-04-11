"""ComfyUI API Client — queue workflows, track progress, retrieve outputs."""

from __future__ import annotations
import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from .config import Config

console = Console()


class ComfyClient:
    """Synchronous + async wrapper around ComfyUI's HTTP/WS API."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.base = self.cfg.comfyui_url
        self.client_id = str(uuid.uuid4())
        self._process: subprocess.Popen | None = None

    # ── lifecycle ────────────────────────

    def ensure_running(self) -> bool:
        """Check if ComfyUI is up; auto-start if configured."""
        try:
            r = httpx.get(f"{self.base}/system_stats", timeout=3)
            return r.status_code == 200
        except Exception:
            pass

        if not self.cfg.auto_start:
            console.print("[red]✗ ComfyUI not running. Start it or set auto_start: true[/]")
            return False

        return self._start_comfyui()

    def _start_comfyui(self) -> bool:
        comfy_path = self.cfg.comfyui_path
        main_py = comfy_path / "main.py"
        if not main_py.exists():
            console.print(f"[red]✗ ComfyUI not found at {comfy_path}[/]")
            return False

        console.print("[yellow]⟳ Starting ComfyUI…[/]")
        self._process = subprocess.Popen(
            [sys.executable, str(main_py), "--listen", "127.0.0.1", "--port",
             str(self.cfg._data["comfyui"]["port"])],
            cwd=str(comfy_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for startup
        for _ in range(60):
            time.sleep(1)
            try:
                r = httpx.get(f"{self.base}/system_stats", timeout=2)
                if r.status_code == 200:
                    console.print("[green]✓ ComfyUI started[/]")
                    return True
            except Exception:
                pass

        console.print("[red]✗ ComfyUI failed to start within 60s[/]")
        return False

    def shutdown(self):
        if self._process:
            self._process.terminate()
            self._process = None

    # ── API calls ────────────────────────

    def queue(self, workflow: dict) -> str:
        """Queue a workflow and return the prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        r = httpx.post(
            f"{self.base}/prompt",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()

        if "error" in data:
            raise RuntimeError(f"ComfyUI error: {data['error']}")
        if "node_errors" in data and data["node_errors"]:
            raise RuntimeError(f"Node errors: {json.dumps(data['node_errors'], indent=2)}")

        prompt_id = data["prompt_id"]
        console.print(f"  [cyan]⟳ Queued → {prompt_id[:12]}…[/]")
        return prompt_id

    def wait(self, prompt_id: str, timeout: int = 600) -> dict:
        """Poll until a prompt finishes. Returns history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.base}/history/{prompt_id}", timeout=5)
                r.raise_for_status()
                history = r.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False) or status.get("status_str") == "success":
                        return entry
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"Generation failed: {status}")
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError(f"Prompt {prompt_id} did not finish in {timeout}s")

    def get_outputs(self, history_entry: dict) -> list[Path]:
        """Extract output file paths from a history entry."""
        outputs = []
        for node_id, node_output in history_entry.get("outputs", {}).items():
            for img in node_output.get("images", []):
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                p = self.cfg.comfyui_path / "output"
                if subfolder:
                    p = p / subfolder
                p = p / filename
                outputs.append(p)

            # Video outputs (VHS nodes)
            for vid in node_output.get("gifs", []):
                filename = vid.get("filename", "")
                subfolder = vid.get("subfolder", "")
                p = self.cfg.comfyui_path / "output"
                if subfolder:
                    p = p / subfolder
                p = p / filename
                outputs.append(p)

        return [p for p in outputs if p.exists()]

    def run_workflow(self, workflow: dict, timeout: int = 600) -> list[Path]:
        """Queue → wait → return output paths. One-shot convenience."""
        pid = self.queue(workflow)
        entry = self.wait(pid, timeout)
        return self.get_outputs(entry)

    # ── info ─────────────────────────────

    def get_installed_nodes(self) -> list[str]:
        """List installed custom node class types."""
        try:
            r = httpx.get(f"{self.base}/object_info", timeout=10)
            r.raise_for_status()
            return list(r.json().keys())
        except Exception:
            return []

    def check_nodes(self, required: list[str]) -> list[str]:
        """Return list of missing required node types."""
        installed = set(self.get_installed_nodes())
        return [n for n in required if n not in installed]

    def get_model_filenames(
        self,
        node_type: str = "CheckpointLoaderSimple",
        input_name: str = "ckpt_name",
    ) -> list[str]:
        """Return the list of filenames ComfyUI advertises for a model input.

        ComfyUI exposes installed model files via /object_info: each loader
        node declares an enum-style input whose first element is the list
        of currently-scanned filenames. Useful for detecting which
        checkpoints / loras / vae files are actually present without
        reimplementing ComfyUI's folder-scan logic on our side.
        """
        try:
            r = httpx.get(f"{self.base}/object_info/{node_type}", timeout=10)
            r.raise_for_status()
            data = r.json().get(node_type, {})
            inputs = data.get("input", {})
            # "required" or "optional" — check both
            for section in ("required", "optional"):
                entry = inputs.get(section, {}).get(input_name)
                if entry and isinstance(entry, list) and entry:
                    first = entry[0]
                    if isinstance(first, list):
                        return [str(x) for x in first]
            return []
        except Exception:
            return []
