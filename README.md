# ComfyForge — Automated AI Media Pipeline

**One command. Full pipeline. Text -> Image -> Video -> Audio.**

ComfyForge is a local orchestration layer on top of ComfyUI that automates
everything: model downloads, workflow generation, and multi-stage pipelines.
You describe what you want in plain text; ComfyForge handles the rest.

---

## Architecture

```
+-----------------------------------------------------+
|                   ComfyForge CLI / Web UI            |
|  "A cinematic shot of a dragon over Lahore at dusk"  |
+------------------------+----------------------------+
                         |
           +-------------v-------------+
           |    Pipeline Orchestrator   |
           |  Decides: T2I -> I2V -> TTS|
           +-------------+-------------+
                         |
          +--------------+--------------+
          v              v              v
    +----------+   +-----------+   +----------+
    |  Model   |   | Workflow   |   |  Audio   |
    | Manager  |   | Generator  |   | Engine   |
    |          |   |            |   |          |
    | CivitAI  |   | ComfyUI    |   | Bark/    |
    | HF Hub   |   | JSON API   |   | F5-TTS   |
    | Firebase |   | workflows  |   |          |
    +----------+   +-----------+   +----------+
          |              |
          v              v
    +-------------------------+
    |   ComfyUI Backend       |
    |   (localhost:8188)      |
    +-------------------------+
```

---

## First-Time Setup

```bash
cd D:\repos\comfy-forge
python setup.py
```

This will:
- Install Python dependencies
- Detect or install ComfyUI
- Install custom nodes (AnimateDiff, VideoHelperSuite, Manager)
- Download base models (SDXL, VAE, RealESRGAN)

Then edit `configs/config.yaml` with your ComfyUI and model paths.

---

## Core Generation Commands

### Text to Image (SDXL)

```bash
python -m core generate "A dragon over Badshahi Mosque at golden hour"
```

### Text to Image (Flux)

```bash
python -m core generate "A dragon over Badshahi Mosque" --flux
```

### Full Pipeline: Prompt -> Image -> Video -> Audio

```bash
python -m core generate "Cinematic aerial shot of Lahore Fort" -p full
```

### Direct Text-to-Video via Wan 2.1

```bash
python -m core generate "A timelapse of clouds over mountains" -p text2vid_wan
```

### Web Dashboard

```bash
python -m core serve
```

---

## Workflow Export (No Execution)

Export ComfyUI-native JSON workflows that you can drag into the ComfyUI UI.
No execution happens — just workflow generation.

### Single Workflow Export

```bash
# SDXL text-to-image
python workflow_export.py text2img "A dragon over Badshahi Mosque at golden hour"

# Flux text-to-image with custom steps
python workflow_export.py text2img_flux "Photorealistic warrior" --steps 30

# Image-to-video with SVD
python workflow_export.py img2vid_svd --image photo.png --frames 30

# Image-to-video with AnimateDiff
python workflow_export.py img2vid_animatediff "Slow camera pan" --image photo.png

# Text-to-video with Wan 2.1
python workflow_export.py text2vid_wan "Ocean waves crashing on rocks"

# Image-to-video with Wan 2.1
python workflow_export.py img2vid_wan "Zoom into painting" --image art.png
```

### Full Pipeline Export

Exports multiple numbered JSONs for each stage:

```bash
python workflow_export.py full "Cyberpunk city at night" --flux --video-backend wan
```

This generates files like `01_text2img_...json`, `02_img2vid_...json`, etc.
in the `exported_workflows/` directory.

**To use in ComfyUI:** Menu -> Load -> pick the JSON. All nodes are pre-wired
with proper links, positioned in a clean left-to-right layout, with colored
groups (loaders, prompts, sampler, output). Run stage 1, grab the output image,
drop it into the LoadImage node in stage 2, and queue that.

---

## Adding CivitAI Models

```bash
python -m core add-civitai "https://civitai.com/models/12345/some-lora" --type lora --tags "anime,detail"
```

---

## Firebase Setup (Private Model Mirrors)

For storing your own download mirrors:

```bash
# Drop your service account JSON into configs/
python scripts/firebase_models.py populate configs/firebase-service-account.json
```

---

## What It Does Automatically

- Downloads checkpoints, LoRAs, VAEs, CLIP models on first use
- Checks Firebase first for your private mirrors, falls back to HuggingFace/CivitAI
- Generates ComfyUI API-format JSON workflows programmatically (no manual node wiring)
- Queues workflows to ComfyUI, polls for completion, collects outputs
- Chains stages: text2img -> img2vid -> audio merge via ffmpeg
- Saves everything to timestamped output folders

---

## Pipelines

| Pipeline             | What it does                                      |
|----------------------|---------------------------------------------------|
| `text2img`           | Prompt -> SDXL image                              |
| `text2img_flux`      | Prompt -> Flux image                              |
| `img2vid_svd`        | Image -> SVD video                                |
| `img2vid_animatediff`| Image -> AnimateDiff video                        |
| `text2vid_wan`       | Prompt -> Wan 2.1 video (direct)                  |
| `img2vid_wan`        | Image -> Wan 2.1 video                            |
| `full`               | Prompt -> Image -> Video -> Audio (complete chain) |

---

## Configuration

Key files to customize:

- `configs/config.yaml` — Your paths, ComfyUI settings, default generation params
- `configs/models.yaml` — Model registry (names, URLs, types, hashes)

Everything else is plug-and-play.

---

## Model Registry

Models are defined in `configs/models.yaml`. ComfyForge auto-downloads
what it needs based on your prompt. Firebase links override defaults
for faster/private mirrors.
