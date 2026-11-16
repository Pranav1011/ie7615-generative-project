"""Milestone 2 sweeps: BERT vs CLIP baseline, CFG scale, noise scheduler, and step counts."""

from __future__ import annotations

import csv
from pathlib import Path

from .bert_adapter import ProjectedBertTextEncoder
from .generate import generate, set_scheduler
from .utils import make_image_grid, save_json


def _append_log_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_milestone2_sweeps(pipe, cfg: dict, paths: dict[str, Path], prompts: list[dict] | None = None):
    """Execute Milestone 2 baseline comparisons and CFG / scheduler / step sweeps."""
    prompts = prompts or cfg["milestone1_prompts"]
    g = cfg["generation"]
    sw = cfg["milestone2_sweeps"]
    seed = cfg["seed"]
    out_root = paths["milestone2_samples"]
    log_csv = paths["milestone2_logs"] / "sweep_runs_log.csv"

    bert_enc = ProjectedBertTextEncoder(cfg, paths["bert_adapter_ckpt"])
    uncond_bert = bert_enc.encode_prompts([""])

    all_rows = []

    # 1. Baseline Dual-Encoder Comparison (Native CLIP vs. BERT Projection)
    set_scheduler(pipe, g["scheduler"])
    clip_imgs, bert_imgs = [], []
    for p in prompts:
        img_c, meta_c = generate(
            pipe, p["prompt"], seed=seed, steps=g["steps"], guidance_scale=g["guidance_scale"],
            out_dir=out_root / "baseline_clip", name=p["id"],
            extra_meta={"sweep": "baseline_encoder", "encoder": "clip", "prompt_id": p["id"]},
        )
        cond_b = bert_enc.encode_prompts([p["prompt"]])
        img_b, meta_b = generate(
            pipe, prompt=None, seed=seed, steps=g["steps"], guidance_scale=g["guidance_scale"],
            out_dir=out_root / "baseline_bert", name=p["id"],
            prompt_embeds=cond_b, negative_prompt_embeds=uncond_bert,
            extra_meta={"sweep": "baseline_encoder", "encoder": "bert_projected", "prompt_id": p["id"], "prompt_text": p["prompt"]},
        )
        clip_imgs.append(img_c)
        bert_imgs.append(img_b)
        all_rows.extend([meta_c, meta_b])

    enc_grid = make_image_grid(
        clip_imgs + bert_imgs,
        [f"CLIP | {p['id']}" for p in prompts] + [f"BERT | {p['id']}" for p in prompts],
        ncols=len(prompts),
    )
    enc_grid.save(out_root / "grid_clip_vs_bert.png")

    # 2. Classifier-Free Guidance (CFG) Scale Sweep (on first 3 diagnostic prompts)
    set_scheduler(pipe, "pndm")
    cfg_prompts = prompts[:3]
    cfg_imgs, cfg_caps = [], []
    for p in cfg_prompts:
        for s in sw["guidance_scales"]:
            img, meta = generate(
                pipe, p["prompt"], seed=seed, steps=30, guidance_scale=s,
                out_dir=out_root / "cfg_sweep", name=f"{p['id']}_s{s}",
                extra_meta={"sweep": "cfg_scale", "encoder": "clip", "prompt_id": p["id"]},
            )
            cfg_imgs.append(img)
            cfg_caps.append(f"{p['id']} | s={s}")
            all_rows.append(meta)
    make_image_grid(cfg_imgs, cfg_caps, ncols=len(sw["guidance_scales"])).save(out_root / "grid_cfg_sweep.png")

    # 3. Noise Scheduler Sweep (30 steps, s=7.5)
    sch_imgs, sch_caps = [], []
    for p in cfg_prompts:
        for sch in sw["schedulers"]:
            set_scheduler(pipe, sch)
            img, meta = generate(
                pipe, p["prompt"], seed=seed, steps=30, guidance_scale=7.5,
                out_dir=out_root / "scheduler_sweep", name=f"{p['id']}_{sch}",
                extra_meta={"sweep": "scheduler", "encoder": "clip", "prompt_id": p["id"]},
            )
            sch_imgs.append(img)
            sch_caps.append(f"{p['id']} | {sch} ({meta['elapsed_sec']}s)")
            all_rows.append(meta)
    make_image_grid(sch_imgs, sch_caps, ncols=len(sw["schedulers"])).save(out_root / "grid_scheduler_sweep.png")

    # 4. Step Count Sweep (on DPM-Solver++ and DDIM, s=7.5)
    step_imgs, step_caps = [], []
    set_scheduler(pipe, "dpm")
    for p in cfg_prompts[:2]:
        for st in sw["steps_list"]:
            img, meta = generate(
                pipe, p["prompt"], seed=seed, steps=st, guidance_scale=7.5,
                out_dir=out_root / "steps_sweep", name=f"{p['id']}_dpm_{st}",
                extra_meta={"sweep": "steps", "encoder": "clip", "prompt_id": p["id"]},
            )
            step_imgs.append(img)
            step_caps.append(f"{p['id']} | dpm {st} steps")
            all_rows.append(meta)
    make_image_grid(step_imgs, step_caps, ncols=len(sw["steps_list"])).save(out_root / "grid_steps_sweep.png")

    save_json({"total_runs": len(all_rows), "runs": all_rows}, paths["milestone2_logs"] / "sweep_runs_log.json", add_env=True)
    return all_rows
