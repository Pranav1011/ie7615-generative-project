"""Shared helpers: config loading, seeding, Drive-aware paths, JSON logging, image grids."""

from __future__ import annotations

import json
import os
import platform
import random
import textwrap
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

_TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pycocotools",
    "numpy",
    "pillow",
)


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML config (defaults to configs/config.yaml in the repo)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: str = "cpu"):
    """A torch.Generator seeded for one sampling call.

    Diffusers draws the initial latents from this generator, so the same seed
    gives the same image for a fixed model, scheduler and settings.
    """
    import torch

    return torch.Generator(device=device).manual_seed(seed)


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_colab() -> bool:
    """True inside a Google Colab runtime (Colab sets COLAB_RELEASE_TAG)."""
    return "COLAB_RELEASE_TAG" in os.environ


def _root_override() -> str | None:
    """IE7615_ARTIFACT_ROOT if it is set to something non-empty."""
    return os.environ.get("IE7615_ARTIFACT_ROOT") or None


def artifact_root(cfg: dict) -> Path:
    """Where large artifacts live: Google Drive on Colab, ./data in the repo otherwise.

    Override with the IE7615_ARTIFACT_ROOT environment variable.
    """
    override = _root_override()
    if override:
        root = Path(override)
    elif is_colab():
        root = Path(cfg["paths"]["drive_root"])
    else:
        root = REPO_ROOT / cfg["paths"]["local_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_paths(cfg: dict) -> dict[str, Path]:
    """Resolve every configured path. Large-artifact paths sit under artifact_root()."""
    root = artifact_root(cfg)
    p = cfg["paths"]
    scratch = Path(p["colab_scratch_root"]) if is_colab() and not _root_override() else root
    paths = {
        "root": root,
        "raw": root / p["raw"],
        "images": scratch / p["images"],
        "annotations": root / p["annotations"],
        "cache": root / p["cache"],
        "manifest": root / p["manifest"],
        "embeddings": root / p["embeddings"],
        "outputs": root / p["outputs"],
        "milestone1_samples": REPO_ROOT / p["milestone1_samples"],
    }
    for key, path in paths.items():
        if key != "manifest":
            path.mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def environment_info() -> dict[str, Any]:
    """Library versions and hardware, logged next to every result for reproducibility."""
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    info: dict[str, Any] = {"python": platform.python_version(), "packages": versions}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def save_json(obj: Any, path: str | os.PathLike, add_env: bool = False) -> Path:
    """Write obj as pretty JSON. With add_env, a dict also gets a timestamp and environment info."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if add_env and isinstance(obj, dict):
        obj = {
            **obj,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": environment_info(),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_image_grid(
    images: Sequence[Image.Image],
    captions: Sequence[str] | None = None,
    ncols: int = 5,
    cell_size: int = 256,
    caption_chars: int = 34,
    font_size: int = 14,
) -> Image.Image:
    """Tile images into a grid, with an optional wrapped caption under each one."""
    if not images:
        raise ValueError("make_image_grid needs at least one image")
    if captions is not None and len(captions) != len(images):
        raise ValueError("captions and images must have the same length")
    ncols = max(1, min(ncols, len(images)))
    nrows = (len(images) + ncols - 1) // ncols
    font = _font(font_size)
    line_h = font_size + 4
    wrapped = [textwrap.wrap(c, caption_chars) for c in captions] if captions else []
    text_h = (max(len(w) for w in wrapped) * line_h + 8) if wrapped else 0
    pad = 6
    cell_w, cell_h = cell_size + pad, cell_size + text_h + pad
    grid = Image.new("RGB", (ncols * cell_w + pad, nrows * cell_h + pad), "white")
    draw = ImageDraw.Draw(grid)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        x, y = pad + c * cell_w, pad + r * cell_h
        grid.paste(img.convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS), (x, y))
        for j, line in enumerate(wrapped[i] if wrapped else []):
            draw.text((x, y + cell_size + 4 + j * line_h), line, fill="black", font=font)
    return grid
