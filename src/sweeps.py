"""Milestone 2 sweeps: CLIP vs projected-BERT baseline, CFG scale, noise scheduler and step count.

Every image is written to Drive next to its JSON sidecar. A rerun skips any
image whose PNG and JSON already exist, so a Colab disconnect only costs the
image that was in progress. Delete the samples folder to regenerate from scratch.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .generate import generate, set_scheduler
from .utils import load_json, make_image_grid, save_json

LOG_FIELDS = [
    "sweep", "encoder", "prompt_id", "prompt_text", "scheduler", "steps", "guidance_scale", "seed",
    "elapsed_sec", "peak_vram_gb", "nsfw_flagged", "resumed", "path",
]


def write_csv(rows: list[dict], csv_path: Path, fields: list[str] | None = None) -> Path:
    """Write rows to CSV (overwriting), keeping only the given fields in a fixed order."""
    fields = fields or list(rows[0].keys())
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def select_prompts(cfg: dict) -> list[dict]:
    by_id = {p["id"]: p for p in cfg["milestone1_prompts"]}
    ids = cfg["milestone2_sweeps"].get("sweep_prompts") or list(by_id)
    return [by_id[i] for i in ids]


def _run_one(pipe, root: Path, rel_dir: str, name: str, *, sweep: str, encoder: str, prompt: dict,
             seed: int, steps: int, guidance: float, height: int, width: int,
             prompt_embeds=None, negative_prompt_embeds=None):
    """Generate one image, or reuse it if its PNG and JSON already exist."""
    out_dir = root / rel_dir
    png, js = out_dir / f"{name}.png", out_dir / f"{name}.json"
    rel_path = f"{rel_dir}/{name}.png"
    if png.exists() and js.exists():
        meta = load_json(js)
        meta["resumed"] = True
        with Image.open(png) as im:
            return im.convert("RGB"), meta
    extra = {"sweep": sweep, "encoder": encoder, "prompt_id": prompt["id"],
             "prompt_text": prompt["prompt"], "path": rel_path, "resumed": False}
    img, meta = generate(
        pipe, None if prompt_embeds is not None else prompt["prompt"], seed=seed, steps=steps,
        guidance_scale=guidance, out_dir=out_dir, name=name, height=height, width=width,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, extra_meta=extra,
    )
    return img, meta


def _save_grid(images, captions, ncols, path: Path, cell: int = 192) -> Path:
    make_image_grid(images, captions, ncols=ncols, cell_size=cell, caption_chars=26, font_size=12).save(
        path, "JPEG", quality=88)
    return path


def reproducibility_check(root: Path, prompts: list[dict]) -> dict:
    """The PNDM / 30-step / s=7.5 / CLIP image appears in three sweeps. Compare the copies pixel by pixel."""
    out = {}
    for p in prompts:
        files = [root / f"baseline/clip/{p['id']}.png", root / f"cfg/{p['id']}_s7.5.png",
                 root / f"scheduler/{p['id']}_pndm.png"]
        if not all(f.exists() for f in files):
            out[p["id"]] = None
            continue
        arrs = [np.asarray(Image.open(f).convert("RGB"), dtype=np.int16) for f in files]
        max_diff = max(int(np.abs(a - arrs[0]).max()) for a in arrs[1:])
        out[p["id"]] = {"identical": max_diff == 0, "max_abs_pixel_diff": max_diff}
    return out


def run_milestone2_sweeps(pipe, cfg: dict, paths: dict[str, Path], bert_encoder=None,
                          prompts: list[dict] | None = None) -> list[dict]:
    """Run all four sweeps and write grids, sweep_runs_log.csv/json and the reproducibility check."""
    prompts = prompts or select_prompts(cfg)
    g, sw, seed = cfg["generation"], cfg["milestone2_sweeps"], cfg["seed"]
    root = Path(paths["milestone2_samples"])
    logs = Path(paths["milestone2_logs"])
    root.mkdir(parents=True, exist_ok=True)
    common = dict(seed=seed, height=g["height"], width=g["width"])
    rows: list[dict] = []

    if bert_encoder is None:
        from .bert_adapter import ProjectedBertTextEncoder

        bert_encoder = ProjectedBertTextEncoder(cfg, paths["bert_adapter_ckpt"])
    uncond_bert = bert_encoder.encode_prompts([""])

    # 1. Baseline: native CLIP vs. projected BERT, at the default settings
    set_scheduler(pipe, g["scheduler"])
    clip_imgs, bert_imgs = [], []
    for p in prompts:
        img, meta = _run_one(pipe, root, "baseline/clip", p["id"], sweep="baseline", encoder="clip", prompt=p,
                             steps=g["steps"], guidance=g["guidance_scale"], **common)
        clip_imgs.append(img)
        rows.append(meta)
        img, meta = _run_one(pipe, root, "baseline/bert", p["id"], sweep="baseline", encoder="bert_projected",
                             prompt=p, steps=g["steps"], guidance=g["guidance_scale"],
                             prompt_embeds=bert_encoder.encode_prompts([p["prompt"]]),
                             negative_prompt_embeds=uncond_bert, **common)
        bert_imgs.append(img)
        rows.append(meta)
    _save_grid(clip_imgs + bert_imgs, [f"CLIP | {p['id']}" for p in prompts] + [f"BERT->CLIP | {p['id']}" for p in prompts],
               len(prompts), root / "grid_clip_vs_bert.jpg", cell=256)

    # 2. Classifier-free guidance scale (PNDM, default steps)
    imgs, caps = [], []
    for p in prompts:
        for s in sw["guidance_scales"]:
            img, meta = _run_one(pipe, root, "cfg", f"{p['id']}_s{s}", sweep="cfg_scale", encoder="clip", prompt=p,
                                 steps=g["steps"], guidance=s, **common)
            imgs.append(img)
            caps.append(f"{p['id']} | s={s}")
            rows.append(meta)
    _save_grid(imgs, caps, len(sw["guidance_scales"]), root / "grid_cfg_sweep.jpg")

    # 3. Noise scheduler (default steps and guidance)
    imgs, caps = [], []
    for p in prompts:
        for sch in sw["schedulers"]:
            set_scheduler(pipe, sch)
            img, meta = _run_one(pipe, root, "scheduler", f"{p['id']}_{sch}", sweep="scheduler", encoder="clip",
                                 prompt=p, steps=g["steps"], guidance=g["guidance_scale"], **common)
            imgs.append(img)
            caps.append(f"{p['id']} | {sch}")
            rows.append(meta)
    _save_grid(imgs, caps, len(sw["schedulers"]), root / "grid_scheduler_sweep.jpg")

    # 4. Sampling steps, for each scheduler in steps_schedulers
    imgs, caps = [], []
    for sch in sw["steps_schedulers"]:
        set_scheduler(pipe, sch)
        for p in prompts:
            for st in sw["steps_list"]:
                img, meta = _run_one(pipe, root, "steps", f"{p['id']}_{sch}_{st}", sweep="steps", encoder="clip",
                                     prompt=p, steps=st, guidance=g["guidance_scale"], **common)
                imgs.append(img)
                caps.append(f"{p['id']} | {sch} {st}")
                rows.append(meta)
    _save_grid(imgs, caps, len(sw["steps_list"]), root / "grid_steps_sweep.jpg")
    set_scheduler(pipe, g["scheduler"])

    repro = reproducibility_check(root, prompts)
    write_csv(rows, logs / "sweep_runs_log.csv", LOG_FIELDS)
    save_json({"total_runs": len(rows), "reproducibility_check": repro, "runs": rows},
              logs / "sweep_runs_log.json", add_env=True)
    print(f"{len(rows)} runs ({sum(r['resumed'] for r in rows)} reused) | reproducibility:", repro)
    return rows
