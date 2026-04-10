#!/usr/bin/env python3
"""ComfyForge first-time setup — installs ComfyUI, custom nodes, and base models."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
CUSTOM_NODES = {
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "ComfyUI-AnimateDiff-Evolved": "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
    "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
}


def run(cmd, **kw):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║       ComfyForge Setup               ║")
    print("╚══════════════════════════════════════╝\n")

    # 1. Install Python deps
    print("[1/4] Installing Python dependencies…")
    run(f"{sys.executable} -m pip install -r requirements.txt")

    # 2. Check / install ComfyUI
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        shutil.copy("configs/config.example.yaml", config_path)
        print(f"  Created {config_path} — edit it with your paths!")

    import yaml
    cfg = yaml.safe_load(config_path.read_text())
    comfy_path = Path(cfg["comfyui"]["path"])

    if not comfy_path.exists():
        print(f"\n[2/4] ComfyUI not found at {comfy_path}. Installing…")
        comfy_path.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone {COMFYUI_REPO} {comfy_path}")
        run(f"{sys.executable} -m pip install -r {comfy_path / 'requirements.txt'}")
    elif not (comfy_path / "main.py").exists():
        print(f"[2/4] ComfyUI directory exists at {comfy_path} but main.py not found.")
        print(f"  Installing ComfyUI requirements…")
        req = comfy_path / "requirements.txt"
        if req.exists():
            run(f"{sys.executable} -m pip install -r {req}")
        else:
            print(f"  ⚠ No requirements.txt found in {comfy_path}")
    else:
        print(f"[2/4] ComfyUI found at {comfy_path} ✓")

    # 3. Install custom nodes
    print("\n[3/4] Installing custom nodes…")
    data_path = Path(cfg["comfyui"].get("data_path") or cfg["comfyui"]["path"])
    cn_dir = data_path / "custom_nodes"
    cn_dir.mkdir(exist_ok=True)
    for name, repo in CUSTOM_NODES.items():
        node_path = cn_dir / name
        if node_path.exists():
            print(f"  ✓ {name} already installed")
        else:
            print(f"  ↓ Cloning {name}…")
            run(f"git clone {repo} {node_path}")
            req = node_path / "requirements.txt"
            if req.exists():
                run(f"{sys.executable} -m pip install -r {req}")

    # 4. Download base models
    print("\n[4/4] Downloading base models…")
    # Import after deps are installed
    sys.path.insert(0, str(Path(__file__).parent))
    from core.config import Config
    from core.models import ModelManager

    Config.reset()
    c = Config.load()
    mm = ModelManager(c)

    base_models = ["sdxl_base", "sdxl_vae", "realesrgan_x4"]
    for mid in base_models:
        try:
            mm.ensure_model(mid)
        except Exception as e:
            print(f"  ⚠ Could not download {mid}: {e}")

    # Create output dirs
    Path(cfg["paths"]["output"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["temp"]).mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 40)
    print("Setup complete!")
    print(f"  Config:   {config_path.resolve()}")
    print(f"  ComfyUI:  {comfy_path.resolve()}")
    print(f"  Output:   {cfg['paths']['output']}")
    print()
    print("Next steps:")
    print("  1. Edit configs/config.yaml with your paths")
    print("  2. (Optional) Add Firebase credentials")
    print("  3. Run:  python -m core generate 'your prompt here'")
    print("  4. Or:   python -m core serve  (web dashboard)")


if __name__ == "__main__":
    main()
