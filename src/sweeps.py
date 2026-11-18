"""Milestone 2 sweeps: CLIP vs projected-BERT baseline, CFG scale, noise scheduler and step count.

Every image is written to Drive next to its JSON sidecar. A rerun skips any
image whose PNG and JSON already exist, so a Colab disconnect only costs the
image that was in progress. A reused image must match the requested seed,
steps, guidance, size, scheduler, encoder and prompt. Otherwise it is
regenerated. Delete the samples folder to regenerate everything from scratch.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .generate import generate, set_scheduler
from .utils import load_json, make_image_grid, save_json

LOG_FIELDS = [
    "sweep", "encoder", "prompt_id", "prompt_text", "scheduler", "steps", "guidance_scale", "seed",
    "elapsed_sec", "peak_vram_gb", "nsfw_flagged", "resumed", "adapter_ckpt_sha256", "path",
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


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _run_one(pipe, root: Path, rel_dir: str, name: str, *, sweep: str, encoder: str, prompt: dict,
             seed: int, steps: int, guidance: float, height: int, width: int,
             embeds_fn=None, identity: dict | None = None):
    """Generate one image, or reuse it if its PNG and JSON exist with identical settings.

    embeds_fn() -> (prompt_embeds, negative_prompt_embeds) is only called when
    an image actually has to be generated. identity holds extra keys that must
    match for reuse (for example the adapter checkpoint hash), and they are
    stored in the sidecar.
    """
    out_dir = root / rel_dir
    png, js = out_dir / f"{name}.png", out_dir / f"{name}.json"
    rel_path = f"{rel_dir}/{name}.png"
    identity = identity or {}
    if png.exists() and js.exists():
        meta = load_json(js)
        wanted = {"seed": seed, "steps": steps, "guidance_scale": guidance, "height": height, "width": width,
                  "prompt_text": prompt["prompt"], "encoder": encoder,
                  "scheduler": getattr(pipe, "_ie7615_scheduler", None), **identity}
        stale = {k: (meta.get(k), v) for k, v in wanted.items() if meta.get(k) != v}
        if not stale:
            meta["resumed"] = True
            with Image.open(png) as im:
                return im.convert("RGB"), meta
        print(f"[regenerate] {rel_path}: settings changed {stale}")
    extra = {"sweep": sweep, "encoder": encoder, "prompt_id": prompt["id"],
             "prompt_text": prompt["prompt"], "path": rel_path, "resumed": False, **identity}
    cond, uncond = embeds_fn() if embeds_fn else (None, None)
    img, meta = generate(
        pipe, None if cond is not None else prompt["prompt"], seed=seed, steps=steps,
        guidance_scale=guidance, out_dir=out_dir, name=name, height=height, width=width,
        prompt_embeds=cond, negative_prompt_embeds=uncond, extra_meta=extra,
    )
    return img, meta


def _save_grid(images, captions, ncols, path: Path, cell: int = 192) -> Path:
    make_image_grid(images, captions, ncols=ncols, cell_size=cell, caption_chars=26, font_size=12).save(
        path, "JPEG", quality=88)
    return path


def duplicate_groups(cfg: dict, prompt_id: str) -> dict[str, list[str]]:
    """Images that different sweeps generate with identical settings, as lists of relative paths.

    'default': the default-scheduler / default-steps / default-guidance CLIP image
    (baseline, cfg sweep, scheduler sweep). '<sch>_default_steps': the steps-sweep
    image at the default step count vs. the scheduler-sweep image.
    """
    g, sw = cfg["generation"], cfg["milestone2_sweeps"]
    groups = {"default": [f"baseline/clip/{prompt_id}.png"]}
    if g["guidance_scale"] in sw["guidance_scales"]:
        groups["default"].append(f"cfg/{prompt_id}_s{g['guidance_scale']}.png")
    if g["scheduler"] in sw["schedulers"]:
        groups["default"].append(f"scheduler/{prompt_id}_{g['scheduler']}.png")
    if g["steps"] in sw["steps_list"]:
        for sch in sw["steps_schedulers"]:
            if sch in sw["schedulers"]:
                groups[f"{sch}_default_steps"] = [f"scheduler/{prompt_id}_{sch}.png",
                                                  f"steps/{prompt_id}_{sch}_{g['steps']}.png"]
    return {k: v for k, v in groups.items() if len(v) > 1}


def reproducibility_check(root: Path, cfg: dict, prompts: list[dict]) -> dict:
    """Compare same-settings images from different sweeps pixel by pixel. Identical means fully deterministic."""
    out = {}
    for p in prompts:
        for group, rels in duplicate_groups(cfg, p["id"]).items():
            files = [root / r for r in rels]
            if not all(f.exists() for f in files):
                out[f"{p['id']}/{group}"] = None
                continue
            arrs = [np.asarray(Image.open(f).convert("RGB"), dtype=np.int16) for f in files]
            max_diff = max(int(np.abs(a - arrs[0]).max()) for a in arrs[1:])
            out[f"{p['id']}/{group}"] = {"identical": max_diff == 0, "max_abs_pixel_diff": max_diff}
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

    # The projected-BERT encoder is loaded only if a BERT image actually has to be generated.
    # Reuse requires the same adapter checkpoint (by hash), so retraining in 04 regenerates these images.
    ckpt = Path(paths["bert_adapter_ckpt"])
    bert_identity = {"adapter_ckpt_sha256": file_sha256(ckpt) if ckpt.exists() else None}
    lazy: dict = {"enc": bert_encoder}

    def bert_embeds(text: str):
        def fn():
            if lazy["enc"] is None:
                from .bert_adapter import ProjectedBertTextEncoder

                lazy["enc"] = ProjectedBertTextEncoder(cfg, ckpt)
            if "uncond" not in lazy:
                lazy["uncond"] = lazy["enc"].encode_prompts([""])
            return lazy["enc"].encode_prompts([text]), lazy["uncond"]
        return fn

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
                             embeds_fn=bert_embeds(p["prompt"]), identity=bert_identity, **common)
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

    repro = reproducibility_check(root, cfg, prompts)
    write_csv(rows, logs / "sweep_runs_log.csv", LOG_FIELDS)
    save_json({"total_runs": len(rows), "reproducibility_check": repro, "runs": rows},
              logs / "sweep_runs_log.json", add_env=True)
    print(f"{len(rows)} runs ({sum(r['resumed'] for r in rows)} reused) | reproducibility:", repro)
    return rows
