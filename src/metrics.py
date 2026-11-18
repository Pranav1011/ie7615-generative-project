"""CLIP score: prompt-image alignment, computed as 100 * max(cos(image, text), 0).

The score uses CLIP ViT-B/32. That is a different checkpoint from the ViT-L/14
text encoder SD 1.5 conditions on, which limits self-scoring bias. The bias is
not zero, though, because both come from the same CLIP training recipe, and a
CLIP-family judge may slightly favor CLIP-conditioned images over projected-BERT ones.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .embeddings import ClipTextEncoder


def clip_score_from_embeds(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    """Row-wise CLIP score [N] for matched image/text embedding pairs."""
    cos = F.cosine_similarity(image_embeds.float(), text_embeds.float(), dim=-1)
    return 100.0 * cos.clamp(min=0)


class ClipScorer:
    def __init__(self, model_id: str, revision: str | None = None, encoder=None):
        self.model_id, self.revision = model_id, revision
        self.encoder = encoder or ClipTextEncoder(model_id, with_vision=True, revision=revision)

    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str], batch_size: int = 16) -> list[float]:
        if len(images) != len(texts):
            raise ValueError("images and texts must have the same length")
        img = self.encoder.encode_images(images, batch_size=batch_size)
        txt = self.encoder.encode(texts, batch_size=batch_size)["text_embeds"]
        return clip_score_from_embeds(img, txt).tolist()


def score_runs(rows: list[dict], samples_root: Path, scorer: ClipScorer, batch_size: int = 16) -> list[dict]:
    """Add 'clip_score' to every run row, scoring the image against its prompt text."""
    root = Path(samples_root)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        images = []
        for r in chunk:
            with Image.open(root / r["path"]) as im:
                images.append(im.convert("RGB"))
        scores = scorer.score(images, [r["prompt_text"] for r in chunk], batch_size=batch_size)
        for r, s in zip(chunk, scores):
            r["clip_score"] = round(s, 4)
            r["clip_score_model"] = scorer.model_id
    return rows


def summarize_runs(rows: list[dict]):
    """Mean/std CLIP score and mean time for each sweep setting, averaged over prompts."""
    import pandas as pd

    df = pd.DataFrame(rows)
    keys = ["sweep", "encoder", "scheduler", "steps", "guidance_scale"]
    out = (df.groupby(keys, dropna=False)
             .agg(n=("clip_score", "size"), clip_score_mean=("clip_score", "mean"),
                  clip_score_std=("clip_score", "std"), elapsed_sec_mean=("elapsed_sec", "mean"))
             .reset_index())
    return out.round(4)
