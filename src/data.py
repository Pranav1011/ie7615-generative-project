"""COCO 2017 val: download, caption parsing and cleaning, manifest building.

Re-running is safe: downloads, extraction and the manifest are skipped when
their outputs already exist (pass force=True to rebuild).
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import requests
from tqdm.auto import tqdm

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------- download / extract


def download(url: str, dest: Path, chunk: int = 1 << 20, timeout: int = 60) -> Path:
    """Stream url to dest. Skips if dest exists; writes to dest.part then renames,
    so an interrupted download never looks complete."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                bar.update(len(block))
    if total is not None and tmp.stat().st_size != total:
        raise IOError(f"Incomplete download of {url}: {tmp.stat().st_size} of {total} bytes")
    tmp.replace(dest)
    return dest


def safe_extract(zip_path: Path, out_dir: Path, members: Iterable[str] | None = None) -> list[Path]:
    """Extract a zip, refusing any member that would land outside out_dir (zip-slip)."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        names = list(members) if members is not None else zf.namelist()
        for name in names:
            target = (out_dir / name).resolve()
            if target != out_dir and out_dir not in target.parents:
                raise ValueError(f"Unsafe path in archive {zip_path.name}: {name!r}")
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
            written.append(target)
    return written


def download_coco_val(cfg: dict, paths: dict[str, Path], images: bool = True) -> dict[str, Path]:
    """Fetch captions_val2017.json and (optionally) the 5,000 val2017 images."""
    d = cfg["data"]
    base = d["base_url"].rstrip("/")

    captions_path = paths["annotations"] / d["captions_file"]
    if not captions_path.exists():
        ann_zip = download(f"{base}/{d['annotations_zip']}", paths["raw"] / Path(d["annotations_zip"]).name)
        # Only the val captions file is needed. The archive stores it under annotations/.
        safe_extract(ann_zip, paths["annotations"].parent, members=[f"annotations/{d['captions_file']}"])
    else:
        print(f"[skip] {captions_path.name} already extracted")

    if images:
        if paths["images"].is_dir() and sum(1 for _ in paths["images"].glob("*.jpg")) >= 5000:
            print("[skip] val2017 images already extracted")
        else:
            img_zip = download(f"{base}/{d['images_zip']}", paths["raw"] / Path(d["images_zip"]).name)
            safe_extract(img_zip, paths["images"].parent)
    return {"captions": captions_path, "images": paths["images"]}


# ---------------------------------------------------------------- caption cleaning


def clean_caption(text: str) -> str:
    """Strip and collapse internal whitespace (COCO captions contain stray newlines and spaces)."""
    return _WS.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def load_captions(captions_path: Path) -> tuple[dict, list]:
    with open(captions_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {img["id"]: img for img in raw["images"]}, raw["annotations"]


def build_manifest(
    captions_path: Path,
    cache_rel_dir: str = "coco/val2017_512",
    min_words: int = 4,
    max_words: int = 40,
) -> tuple[list[dict], dict]:
    """Group cleaned captions by image.

    Each record looks like this:
        {image_id, file_name, file_path, width, height,
         captions: [{caption_id, raw, clean, lower, n_words}]}
    file_name is the original COCO file. file_path is the preprocessed 512x512
    JPEG, relative to the artifact root, so the manifest stays valid on any
    machine or Drive mount. width and height are the original dimensions.
    Captions outside [min_words, max_words] are dropped. Images left with no
    caption are dropped too. Returns (records, stats).
    """
    images, anns = load_captions(captions_path)
    by_image: dict[int, list[dict]] = {}
    reasons: Counter = Counter()
    for a in anns:
        clean = clean_caption(a["caption"])
        n = word_count(clean)
        if n < min_words:
            reasons["too_short"] += 1
            continue
        if n > max_words:
            reasons["too_long"] += 1
            continue
        by_image.setdefault(a["image_id"], []).append(
            {"caption_id": a["id"], "raw": a["caption"], "clean": clean, "lower": clean.lower(), "n_words": n}
        )

    records = []
    for image_id in sorted(images):
        caps = by_image.get(image_id)
        if not caps:
            reasons["image_without_captions"] += 1
            continue
        img = images[image_id]
        records.append(
            {
                "image_id": image_id,
                "file_name": img["file_name"],
                "file_path": f"{cache_rel_dir.rstrip('/')}/{img['file_name']}",
                "width": img["width"],
                "height": img["height"],
                "captions": sorted(caps, key=lambda c: c["caption_id"]),
            }
        )
    stats = {
        "n_images_total": len(images),
        "n_captions_total": len(anns),
        "n_images_kept": len(records),
        "n_captions_kept": sum(len(r["captions"]) for r in records),
        "dropped": dict(reasons),
        "min_words": min_words,
        "max_words": max_words,
    }
    return records, stats


def save_manifest(records: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def manifest_to_frame(records: list[dict]):
    """One row per caption, which is the convenient shape for EDA."""
    import pandas as pd

    rows = [
        {"image_id": r["image_id"], "file_name": r["file_name"], **c}
        for r in records
        for c in r["captions"]
    ]
    return pd.DataFrame(rows)


def prepare_manifest(cfg: dict, paths: dict[str, Path], force: bool = False) -> tuple[list[dict], dict]:
    """Build and save the manifest (plus a stats JSON next to it). Reuses it if it already exists."""
    from .utils import load_json, save_json

    stats_path = paths["manifest"].with_suffix(".stats.json")
    if paths["manifest"].exists() and stats_path.exists() and not force:
        print(f"[skip] manifest exists at {paths['manifest']}")
        return load_manifest(paths["manifest"]), load_json(stats_path)
    records, stats = build_manifest(
        paths["annotations"] / cfg["data"]["captions_file"],
        cfg["paths"]["cache"],
        cfg["data"]["min_words"],
        cfg["data"]["max_words"],
    )
    save_manifest(records, paths["manifest"])
    save_json(stats, stats_path, add_env=True)
    return records, stats
