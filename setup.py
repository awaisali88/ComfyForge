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

    # Detect existing ComfyUI installation (Desktop or git clone)
    # ComfyUI Desktop won't have main.py — check for the models folder instead
    has_main = (comfy_path / "main.py").exists()
    has_models = (comfy_path / "models").exists()
    has_checkpoints = (comfy_path / "models" / "checkpoints").exists()

    if has_main or has_models or has_checkpoints:
        print(f"[2/4] ComfyUI found at {comfy_path} ✓")
        if has_models:
            # List what model subdirs exist
            model_dirs = [d.name for d in (comfy_path / "models").iterdir() if d.is_dir()]
            print(f"       Model folders: {', '.join(sorted(model_dirs))}")
        if not has_main:
            print(f"       (ComfyUI Desktop detected — skipping git/pip steps)")
    else:
        print(f"\n[2/4] ComfyUI not found at {comfy_path}")
        print(f"       Neither main.py nor models/ folder exists at that path.")
        print(f"       If you have ComfyUI Desktop installed elsewhere, update")
        print(f"       configs/config.yaml → comfyui.path to point at it.")
        print()
        resp = input("       Install fresh ComfyUI via git clone? [y/N] ").strip().lower()
        if resp == "y":
            comfy_path.parent.mkdir(parents=True, exist_ok=True)
            run(f"git clone {COMFYUI_REPO} {comfy_path}")
            run(f"{sys.executable} -m pip install -r {comfy_path / 'requirements.txt'}")
        else:
            print("       Skipping ComfyUI install. Fix the path and re-run.")

    # 3. Install custom nodes (only for git-based ComfyUI, skip for Desktop)
    print("\n[3/4] Custom nodes…")
    cn_dir = comfy_path / "custom_nodes"
    if has_main and cn_dir.exists():
        # Git-based ComfyUI — can install custom nodes directly
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
    elif cn_dir.exists():
        # Desktop but custom_nodes folder exists
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
    else:
        print("  ⚠ custom_nodes/ folder not found — install nodes via ComfyUI Manager instead")
        print("    Open ComfyUI → Manager → Install Missing Nodes")

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
