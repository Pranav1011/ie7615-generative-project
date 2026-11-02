"""Image preprocessing: resize the short side, center-crop to a square, normalize to [-1, 1].

Cropped 512x512 images are cached as JPEGs rather than tensors. A full fp16
tensor cache of 5k images would be about 7.9 GB. The JPEG cache is about
300 MB, and normalization is cheap enough to do at load time.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


def resize_center_crop(img: Image.Image, size: int = 512) -> Image.Image:
    """Scale so the short side equals size (bicubic), then center-crop to size x size."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_model_tensor(img: Image.Image):
    """PIL RGB -> float tensor [3, H, W] in [-1, 1], the range the SD VAE expects."""
    import numpy as np
    import torch

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def from_model_tensor(t) -> Image.Image:
    """Inverse of to_model_tensor, used for visual checks."""
    import numpy as np

    arr = ((t.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(arr))


def cached_path(record: dict, root: Path) -> Path:
    """Absolute path of a record's preprocessed image (manifest file_path is root-relative)."""
    return Path(root) / record["file_path"]


def count_uncached(records: list[dict], root: Path) -> int:
    return sum(not cached_path(r, root).exists() for r in records)


def cache_resized(
    records: list[dict],
    src_dir: Path,
    root: Path,
    size: int = 512,
    quality: int = 95,
    limit: int | None = None,
) -> dict:
    """Write a size x size JPEG for each manifest record to root/record['file_path'].

    Originals are read from src_dir/record['file_name']. Images that are
    already cached are skipped, and a JPEG is only renamed into place once it
    is fully written. Returns counts.
    """
    todo = records if limit is None else records[:limit]
    written = skipped = missing = 0
    for r in tqdm(todo, desc=f"resize->{size}"):
        out = cached_path(r, root)
        if out.exists():
            skipped += 1
            continue
        src = Path(src_dir) / r["file_name"]
        if not src.exists():
            missing += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.jpg")
        with Image.open(src) as img:
            resize_center_crop(img, size).save(tmp, "JPEG", quality=quality)
        tmp.replace(out)
        written += 1
    return {"written": written, "skipped": skipped, "missing_source": missing, "total": len(todo)}


class CocoImageCaptionDataset:
    """torch-style Dataset over the cached 512px images.

    Each item is {pixel_values: [3, S, S] in [-1, 1], caption, image_id}. With
    caption_index=None, a caption is picked at random per image. The pick is
    seeded by image_id, so it is reproducible.
    """

    def __init__(self, records: list[dict], root: Path, size: int = 512,
                 caption_field: str = "clean", caption_index: int | None = 0, seed: int = 42):
        self.root = Path(root)
        self.records = [r for r in records if cached_path(r, self.root).exists()]
        self.size = size
        self.caption_field = caption_field
        self.caption_index = caption_index
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        with Image.open(cached_path(r, self.root)) as img:
            img = img.convert("RGB")
            if img.size != (self.size, self.size):
                img = resize_center_crop(img, self.size)
            pixels = to_model_tensor(img)
        caps = r["captions"]
        if self.caption_index is None:
            cap = random.Random(self.seed + r["image_id"]).choice(caps)
        else:
            cap = caps[min(self.caption_index, len(caps) - 1)]
        return {"pixel_values": pixels, "caption": cap[self.caption_field], "image_id": r["image_id"]}
