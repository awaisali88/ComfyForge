"""Configuration loader — merges defaults, user config, and CLI overrides."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

_DEFAULT_CONFIG_PATHS = [
    Path("configs/config.yaml"),
    Path.home() / ".comfyforge" / "config.yaml",
]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


class Config:
    """Singleton-ish config holder."""

    _instance: Config | None = None

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # ── accessors ────────────────────────
    @property
    def comfyui_path(self) -> Path:
        return Path(self._data["comfyui"]["path"])

    @property
    def comfyui_data_path(self) -> Path:
        """Data directory (models, custom_nodes, etc.) — falls back to comfyui_path."""
        return Path(self._data["comfyui"].get("data_path") or self._data["comfyui"]["path"])

    @property
    def comfyui_url(self) -> str:
        h = self._data["comfyui"]["host"]
        p = self._data["comfyui"]["port"]
        return f"http://{h}:{p}"

    @property
    def auto_start(self) -> bool:
        return self._data["comfyui"].get("auto_start", True)

    @property
    def output_dir(self) -> Path:
        return Path(self._data["paths"]["output"])

    @property
    def temp_dir(self) -> Path:
        return Path(self._data["paths"]["temp"])

    def model_dir(self, model_type: str) -> Path:
        explicit = self._data["models"].get(model_type, "")
        if explicit:
            return Path(explicit)
        # Load from models.yaml type_directories mapping
        registry = self._load_model_registry()
        subdir = registry.get("type_directories", {}).get(model_type, model_type)
        return self.comfyui_data_path / "models" / subdir

    def stack(self, pipeline: str) -> dict[str, Any]:
        return self._data.get("stacks", {}).get(pipeline, {})

    def default(self, section: str) -> dict[str, Any]:
        return self._data.get("defaults", {}).get(section, {})

    @property
    def firebase_credentials(self) -> str | None:
        return self._data.get("firebase", {}).get("credentials")

    @property
    def firebase_collection(self) -> str:
        return self._data.get("firebase", {}).get("collection", "comfyforge_models")

    @property
    def firebase_bucket(self) -> str | None:
        return self._data.get("firebase", {}).get("bucket")

    @property
    def hf_token(self) -> str | None:
        token = self._data.get("huggingface", {}).get("token", "")
        return token or os.getenv("HF_TOKEN")

    @property
    def firecrawl_api_key(self) -> str | None:
        key = self._data.get("firecrawl", {}).get("api_key", "")
        return key or os.getenv("FIRECRAWL_API_KEY")

    @property
    def raw(self) -> dict:
        return self._data

    # ── loaders ──────────────────────────
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if cls._instance is not None:
            return cls._instance

        data: dict[str, Any] = {}

        # Load example as base defaults
        example = Path("configs/config.example.yaml")
        if example.exists():
            data = yaml.safe_load(example.read_text()) or {}

        # Overlay user config
        config_path = Path(path) if path else None
        if config_path is None:
            for cp in _DEFAULT_CONFIG_PATHS:
                if cp.exists():
                    config_path = cp
                    break

        if config_path and config_path.exists():
            user = yaml.safe_load(config_path.read_text()) or {}
            data = _deep_merge(data, user)

        # Env overrides
        if ep := os.getenv("COMFYUI_PATH"):
            data.setdefault("comfyui", {})["path"] = ep

        cls._instance = cls(data)
        return cls._instance

    @staticmethod
    def _load_model_registry() -> dict:
        p = Path("configs/models.yaml")
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}
        return {}

    @classmethod
    def reset(cls):
        cls._instance = None
