"""ComfyForge CLI — simple command-line interface."""

from __future__ import annotations
import json
from pathlib import Path

import typer
from rich.console import Console

from .config import Config
from .models import ModelManager
from .pipeline import Pipeline

app = typer.Typer(name="comfyforge", help="Automated AI Media Pipeline on ComfyUI")
console = Console()


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="What to generate"),
    pipeline: str = typer.Option("text2img", "--pipeline", "-p",
        help="Pipeline: text2img | img2vid | text2vid | text2vid_wan | full"),
    image: str = typer.Option(None, "--image", "-i", help="Input image path (for img2vid)"),
    stack: str = typer.Option(None, "--stack", "-s",
        help="Model stack override (e.g. text2img_flux)"),
    negative: str = typer.Option("blurry, low quality, watermark, text", "--neg", "-n"),
    width: int = typer.Option(1024, "--width", "-W"),
    height: int = typer.Option(1024, "--height", "-H"),
    steps: int = typer.Option(None, "--steps"),
    cfg_scale: float = typer.Option(None, "--cfg"),
    seed: int = typer.Option(-1, "--seed"),
    video_backend: str = typer.Option("svd", "--video-backend",
        help="Video backend: svd | animatediff | wan"),
    frames: int = typer.Option(None, "--frames"),
    fps: int = typer.Option(None, "--fps"),
    config: str = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    use_flux: bool = typer.Option(False, "--flux", help="Use Flux instead of SDXL"),
):
    """Generate images/videos from a text prompt."""
    cfg = Config.load(config)

    overrides = {"width": width, "height": height, "seed": seed, "video_backend": video_backend}
    if steps:
        overrides["steps"] = steps
    if cfg_scale:
        overrides["cfg"] = cfg_scale
    if frames:
        overrides["frames"] = frames
    if fps:
        overrides["fps"] = fps
    if use_flux:
        overrides["use_flux"] = True

    pipe = Pipeline(cfg)
    result = pipe.run(
        prompt=prompt,
        pipeline=pipeline,
        negative=negative,
        image_path=image,
        model_stack=stack,
        overrides=overrides,
    )

    if result.success:
        console.print(f"\n[bold green]✓ Done! {len(result.all_files)} file(s) generated[/]")
        for f in result.all_files:
            console.print(f"  {f}")
    else:
        console.print("[bold red]✗ Pipeline failed[/]")
        raise typer.Exit(1)


@app.command()
def download(
    model_id: str = typer.Argument(..., help="Model ID from models.yaml"),
    config: str = typer.Option(None, "--config", "-c"),
):
    """Download a specific model."""
    cfg = Config.load(config)
    mm = ModelManager(cfg)
    path = mm.ensure_model(model_id)
    console.print(f"[green]✓ {path}[/]")


@app.command()
def download_stack(
    pipeline: str = typer.Argument(..., help="Pipeline name: text2img, text2img_flux, img2vid, etc."),
    config: str = typer.Option(None, "--config", "-c"),
):
    """Download all models for a pipeline stack."""
    cfg = Config.load(config)
    mm = ModelManager(cfg)
    paths = mm.ensure_stack(pipeline)
    for k, v in paths.items():
        console.print(f"  [green]✓ {k}: {v}[/]")


@app.command()
def add_civitai(
    url: str = typer.Argument(..., help="CivitAI model URL or version ID"),
    model_type: str = typer.Option("lora", "--type", "-t"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    config: str = typer.Option(None, "--config", "-c"),
):
    """Add a model from CivitAI to the registry and download it."""
    cfg = Config.load(config)
    mm = ModelManager(cfg)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    entry = mm.add_civitai_model(url, model_type, tag_list)
    mm.ensure_model(entry["id"])


@app.command()
def list_models(
    config: str = typer.Option(None, "--config", "-c"),
):
    """List all models in the registry."""
    cfg = Config.load(config)
    mm = ModelManager(cfg)
    from rich.table import Table

    table = Table(title="Model Registry")
    table.add_column("ID")
    table.add_column("Filename")
    table.add_column("Type")
    table.add_column("Size")
    table.add_column("Downloaded")

    for m in mm.registry.all_models():
        dest = mm.registry.dest_path(m)
        status = "[green]✓[/]" if dest.exists() else "[dim]—[/]"
        table.add_row(m["id"], m["filename"], m["type"], f"{m.get('size_gb', '?')} GB", status)

    console.print(table)


@app.command()
def workflow(
    prompt: str = typer.Argument(...),
    pipeline: str = typer.Option("text2img", "--pipeline", "-p"),
    output: str = typer.Option("workflow.json", "--output", "-o"),
    use_flux: bool = typer.Option(False, "--flux"),
):
    """Generate a ComfyUI workflow JSON without executing it."""
    from .workflows import text2img_sdxl, text2img_flux, text2vid_wan, img2vid_svd

    cfg = Config.load()
    stack_cfg = cfg.stack(pipeline if not use_flux else "text2img_flux")

    if pipeline == "text2img" and not use_flux:
        wf = text2img_sdxl(prompt=prompt, **{k: v for k, v in stack_cfg.items() if k != "loras"})
    elif pipeline == "text2img" and use_flux:
        wf = text2img_flux(prompt=prompt, checkpoint=stack_cfg.get("checkpoint", "flux1-dev.safetensors"))
    elif pipeline == "text2vid":
        wf = text2vid_wan(prompt=prompt, checkpoint=stack_cfg.get("checkpoint"))
    else:
        wf = text2img_sdxl(prompt=prompt)

    Path(output).write_text(json.dumps(wf, indent=2))
    console.print(f"[green]✓ Saved workflow → {output}[/]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    config: str = typer.Option(None, "--config", "-c"),
):
    """Launch the web dashboard."""
    import uvicorn
    Config.load(config)
    uvicorn.run("ui.server:app", host=host, port=port, reload=True)


def main():
    app()


if __name__ == "__main__":
    main()
