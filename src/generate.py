"""Stable Diffusion v1.5 generation with fully logged, seeded runs.

Pipeline: CLIP ViT-L/14 text encoder -> [77, 768] hidden states -> U-Net
cross-attention, which denoises 4x64x64 VAE latents over the scheduler's
steps. The VAE then decodes the result to 512x512 RGB.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch

from .utils import get_device, make_generator, save_json

SCHEDULERS = {
    "pndm": "PNDMScheduler",          # SD 1.5 default
    "ddim": "DDIMScheduler",
    "dpm": "DPMSolverMultistepScheduler",
    "euler": "EulerDiscreteScheduler",
}


def load_pipeline(cfg: dict, scheduler: str | None = None, device: str | None = None):
    """Load SD 1.5 in fp16 on GPU, with attention slicing on to fit a 16 GB T4."""
    from diffusers import StableDiffusionPipeline

    device = device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = cfg["models"]
    revision = (m.get("revisions") or {}).get("diffusion")
    kwargs = {"torch_dtype": dtype, "revision": revision}
    if dtype == torch.float16 and m.get("diffusion_variant"):
        kwargs["variant"] = m["diffusion_variant"]
    pipe = StableDiffusionPipeline.from_pretrained(m["diffusion"], **kwargs).to(device)
    pipe._ie7615_model_id = m["diffusion"]
    pipe._ie7615_revision = revision
    set_scheduler(pipe, scheduler or cfg["generation"]["scheduler"])
    if cfg["generation"].get("attention_slicing", True):
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def set_scheduler(pipe, name: str) -> None:
    """Swap the noise scheduler in place, keeping the model's trained beta schedule."""
    import diffusers

    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {name!r}; choose from {sorted(SCHEDULERS)}")
    cls = getattr(diffusers, SCHEDULERS[name])
    pipe.scheduler = cls.from_config(pipe.scheduler.config)
    pipe._ie7615_scheduler = name


def slugify(text: str, max_len: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:max_len] or "prompt"


def generate(
    pipe,
    prompt: str | None,
    seed: int,
    steps: int = 30,
    guidance_scale: float = 7.5,
    out_dir: str | Path | None = None,
    name: str | None = None,
    negative_prompt: str | None = None,
    height: int = 512,
    width: int = 512,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    extra_meta: dict | None = None,
):
    """Generate one image. If out_dir is given, write <name>.png plus a <name>.json sidecar.

    prompt_embeds ([1, 77, 768]) bypasses the pipeline's own text encoder. This
    is the hook for conditioning on externally computed embeddings in later
    milestones. Returns (PIL image, metadata dict).
    """
    if prompt is None and prompt_embeds is None:
        raise ValueError("Pass a prompt or prompt_embeds")
    if prompt_embeds is not None and negative_prompt is not None:
        raise ValueError("Pass negative_prompt_embeds (not negative_prompt) when using prompt_embeds")
    device = pipe.device.type
    generator = make_generator(seed, device="cpu" if device == "mps" else device)
    call = dict(
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )
    if prompt_embeds is not None:
        call["prompt_embeds"] = prompt_embeds.to(pipe.device, pipe.unet.dtype)
        if negative_prompt_embeds is not None:
            call["negative_prompt_embeds"] = negative_prompt_embeds.to(pipe.device, pipe.unet.dtype)
    else:
        call["prompt"] = prompt
        call["negative_prompt"] = negative_prompt

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = pipe(**call)
    elapsed = time.perf_counter() - t0
    image = result.images[0]
    nsfw = (result.nsfw_content_detected or [None])[0]

    meta = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "scheduler": getattr(pipe, "_ie7615_scheduler", type(pipe.scheduler).__name__),
        "scheduler_class": type(pipe.scheduler).__name__,
        "model_id": getattr(pipe, "_ie7615_model_id", None),
        "model_revision": getattr(pipe, "_ie7615_revision", None),
        "dtype": str(pipe.unet.dtype),
        "used_prompt_embeds": prompt_embeds is not None,
        "used_negative_prompt_embeds": negative_prompt_embeds is not None,
        "nsfw_flagged": nsfw,
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
        **(extra_meta or {}),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = name or f"{slugify(prompt or 'embeds')}_s{seed}"
        image.save(out_dir / f"{stem}.png")
        meta["file"] = f"{stem}.png"
        save_json(meta, out_dir / f"{stem}.json", add_env=True)
    return image, meta


def run_prompt_set(pipe, prompts: list[dict], cfg: dict, out_dir: str | Path, seed: int | None = None):
    """Generate one image per prompt entry ({id, prompt, source, ...}) with shared settings.

    Writes PNG + JSON per prompt, then prompts.json (the full run log) and grid.png.
    """
    from .utils import make_image_grid

    g = cfg["generation"]
    seed = cfg["seed"] if seed is None else seed
    images, metas = [], []
    for p in prompts:
        img, meta = generate(
            pipe, p["prompt"], seed=seed, steps=g["steps"], guidance_scale=g["guidance_scale"],
            out_dir=out_dir, name=p["id"], height=g["height"], width=g["width"],
            extra_meta={k: v for k, v in p.items() if k != "prompt"},
        )
        print(f"{p['id']}: {meta['elapsed_sec']}s, peak VRAM {meta['peak_vram_gb']} GB")
        images.append(img)
        metas.append(meta)
    out_dir = Path(out_dir)
    grid = make_image_grid(images, [p["prompt"] for p in prompts], ncols=len(images))
    grid.save(out_dir / "grid.png")
    save_json(
        {
            "settings": {
                "seed": seed,
                **g,
                # What the pipeline actually ran with, which may differ from the config default
                "scheduler": getattr(pipe, "_ie7615_scheduler", type(pipe.scheduler).__name__),
                "scheduler_class": type(pipe.scheduler).__name__,
                "model_id": getattr(pipe, "_ie7615_model_id", None),
                "model_revision": getattr(pipe, "_ie7615_revision", None),
            },
            "runs": metas,
        },
        out_dir / "prompts.json",
        add_env=True,
    )
    return images, metas, grid
