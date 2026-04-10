# CLAUDE.md — ComfyForge

## Project Overview

ComfyForge is a local orchestration layer on top of ComfyUI that automates model downloads, workflow generation, and multi-stage AI media pipelines. Users describe what they want in plain text; ComfyForge handles the rest: text -> image -> video -> audio.

**Version:** 0.1.0  
**Python:** 3.10+  
**Platform:** Windows (primary), cross-platform compatible

## Quick Reference

```bash
# Run CLI commands
python -m core generate "prompt" -p text2img
python -m core generate "prompt" -p full --flux
python -m core serve                          # web dashboard on :7860

# Export workflows (no execution)
python workflow_export.py text2img "prompt"
python workflow_export.py full "prompt" --flux --video-backend wan

# Setup
python setup.py

# Model management
python -m core download sdxl_base
python -m core download-stack text2img_flux
python -m core add-civitai "https://civitai.com/models/..." --type lora
python -m core list-models
```

## Architecture

```
core/
  __main__.py      → Entry point dispatcher (routes to cli.py)
  cli.py           → Typer CLI (generate, download, serve, etc.)
  config.py        → Singleton config loader (YAML + env overrides)
  models.py        → ModelRegistry + ModelManager (download with multi-source fallback)
  comfy_client.py  → Sync HTTP client for ComfyUI REST API (:8188)
  workflows.py     → Builder pattern for ComfyUI API-format JSON workflows
  pipeline.py      → Pipeline orchestrator (chains stages: t2i → i2v → audio)
ui/
  server.py        → FastAPI web dashboard (inline HTML, dark theme)
scripts/
  firebase_models.py → Firebase/Firestore model mirror management
configs/
  config.yaml          → User config (gitignored)
  config.example.yaml  → Config template with defaults
  models.yaml          → Model registry (IDs, filenames, sources, hashes)
workflow_export.py     → Standalone UI-format workflow exporter (drag-drop into ComfyUI)
setup.py               → First-time setup (deps, ComfyUI, nodes, base models)
```

## Key Design Patterns

- **Config:** Singleton with lazy load. Merges `config.example.yaml` → `config.yaml` → env vars. Access via `Config.load()`.
- **Models:** Registry-backed (`models.yaml`). Download chain: Firebase mirrors → registry sources → HF search → CivitAI search → FireCrawl scraping. SHA256 verification when available.
- **Workflows:** `WorkflowBuilder` with fluent `Node.link()` chaining. Factory functions per workflow type (`text2img_sdxl`, `text2img_flux`, `img2vid_svd`, `img2vid_animatediff`, `text2vid_wan`, `img2vid_wan`).
- **Pipeline:** `Pipeline.run()` dispatches to stage runners (`_run_text2img`, `_run_img2vid`, `_run_text2vid_wan`, `_run_audio`). Returns `PipelineResult` dataclass with outputs, timings, errors.
- **ComfyClient:** Sync HTTP. `ensure_running()` → `queue()` → `wait()` → `get_outputs()`. Can auto-start ComfyUI subprocess.

## Pipelines

| Pipeline | Stages | Backend Options |
|----------|--------|-----------------|
| `text2img` | prompt → image | SDXL, Flux |
| `img2vid` | image → video | SVD, AnimateDiff, Wan 2.1 |
| `text2vid` | prompt → image → video | chained |
| `text2vid_wan` | prompt → video (direct) | Wan 2.1 |
| `full` | prompt → image → video → audio | complete chain |

## Configuration

Config files in `configs/`. User config is gitignored. Key sections:
- `comfyui` — path, host, port, auto_start
- `paths` — output, temp, workflows dirs
- `models` — per-type model directories
- `stacks` — model stacks per pipeline (checkpoint, vae, loras, clip)
- `defaults` — image/video/audio generation defaults
- `firebase` — optional private mirror credentials
- `firecrawl` — optional web search API key

Environment overrides: `COMFYUI_PATH`, `HF_TOKEN`, `CIVITAI_API_TOKEN`, `FIRECRAWL_API_KEY`

## Code Conventions

- Python 3.10+ type hints throughout (`dict[str, Any]`, `list[str] | None`)
- snake_case for modules/functions, PascalCase for classes, UPPER_CASE for constants
- Private methods prefixed with `_`
- Rich console for terminal output (progress bars, tables, colors)
- Typer for CLI, FastAPI for web, httpx for HTTP, pyyaml for config
- No formal test suite yet

## Secrets & Sensitive Files (gitignored)

- `configs/config.yaml` — user config with API keys
- `configs/firebase-service-account.json` — Firebase credentials
- `configs/*-credentials.json` — any auth files
- `.env` files
- `outputs/`, `temp/`, `exported_workflows/` — generated content

## Working with This Codebase

- ComfyUI runs as a separate process on localhost:8188; ComfyForge communicates via its REST API
- Two workflow formats: API format (`core/workflows.py` — for execution) and UI format (`workflow_export.py` — for drag-drop into ComfyUI web UI)
- Model downloads can be large (6-30+ GB); `ensure_model`/`ensure_stack` handles this transparently
- Audio engine is WIP (placeholder in pipeline)
- Web dashboard has no auth; assumes trusted network/localhost
- No WebSocket usage yet despite dependency being listed
