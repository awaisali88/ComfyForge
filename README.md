# ComfyForge — Automated AI Media Pipeline

**One command. Full pipeline. Text → Image → Video → Audio.**

ComfyForge is a local orchestration layer on top of ComfyUI that automates
everything: model downloads, workflow generation, and multi-stage pipelines.
You describe what you want in plain text; ComfyForge handles the rest.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ComfyForge CLI / Web UI            │
│  "A cinematic shot of a dragon over Lahore at dusk"  │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │    Pipeline Orchestrator   │
         │  Decides: T2I → I2V → TTS │
         └─────────────┬─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌───────────┐  ┌──────────┐
  │  Model   │  │ Workflow   │  │  Audio   │
  │ Manager  │  │ Generator  │  │ Engine   │
  │          │  │            │  │          │
  │ CivitAI  │  │ ComfyUI    │  │ Bark/    │
  │ HF Hub   │  │ JSON API   │  │ F5-TTS  │
  │ Firebase │  │ workflows  │  │          │
  └──────────┘  └───────────┘  └──────────┘
        │              │
        ▼              ▼
  ┌─────────────────────────┐
  │   ComfyUI Backend       │
  │   (localhost:8188)      │
  └─────────────────────────┘
```

## Quick Start

```bash
# 1. Clone & install
cd D:\repos
git clone <this-repo> comfy-forge
cd comfy-forge
pip install -r requirements.txt

# 2. Configure (edit your paths + Firebase creds)
cp configs/config.example.yaml configs/config.yaml

# 3. Run a generation
python -m core.cli "A photorealistic dragon flying over Badshahi Mosque at golden hour"

# 4. Or launch the web dashboard
python -m ui.server
```

## Pipelines

| Pipeline       | What it does                                          |
|---------------|-------------------------------------------------------|
| `text2img`    | Prompt → SDXL/Flux image                             |
| `img2vid`     | Image → AnimateDiff / SVD / Wan video                 |
| `text2vid`    | Prompt → Image → Video (chained)                     |
| `vid2audio`   | Video → AI narration / SFX / music track              |
| `full`        | Prompt → Image → Video → Audio (complete pipeline)    |

## Model Registry

Models are defined in `configs/models.yaml`. ComfyForge auto-downloads
what it needs based on your prompt. Firebase links override defaults
for faster/private mirrors.
