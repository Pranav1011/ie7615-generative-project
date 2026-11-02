"""Text encoders: CLIP ViT-L/14 (the encoder SD 1.5 conditions on) and BERT-base (comparison path).

Both expose encode(texts) -> {last_hidden_state [N, 77, 768], pooled [N, D], attention_mask}.

Note: BERT's hidden states are 768-d like CLIP-L's, but they live in a
different space. SD 1.5's U-Net was trained on CLIP-L hidden states only, so
BERT embeddings need a learned projection before they can condition the
U-Net (planned for Milestone 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _default_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


class _TextEncoder:
    name = "base"

    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None, max_length: int = 77):
        from .utils import get_device

        self.device = device or get_device()
        self.dtype = dtype or _default_dtype(self.device)
        self.max_length = max_length

    def _tokenize(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

    def _forward(self, tokens) -> dict:  # implemented by subclasses
        raise NotImplementedError

    @torch.no_grad()
    def encode_batch(self, texts: Sequence[str]) -> dict:
        """Encode one batch. Returns CPU tensors in the encoder's dtype."""
        tokens = self._tokenize(texts).to(self.device)
        return {k: v.detach().cpu() for k, v in self._forward(tokens).items()}

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> dict:
        """Encode texts in batches and concatenate the results."""
        outs: dict[str, list[torch.Tensor]] = {}
        batches = range(0, len(texts), batch_size)
        for start in tqdm(batches, desc=f"encode[{self.name}]", disable=not show_progress):
            for k, v in self.encode_batch(texts[start : start + batch_size]).items():
                outs.setdefault(k, []).append(v)
        return {k: torch.cat(v) for k, v in outs.items()}


class ClipTextEncoder(_TextEncoder):
    """CLIP ViT-L/14 text tower.

    The output keys are:
    - last_hidden_state: [N, 77, 768], the exact tensor SD 1.5 feeds to U-Net cross-attention
      (encoded without an attention mask, as the SD pipeline does).
    - pooled: [N, 768], the EOS-token embedding after final layer norm.
    - text_embeds: [N, 768], pooled projected into CLIP's joint text-image space.
    """

    name = "clip"

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", with_vision: bool = False,
                 revision: str | None = None, **kw):
        super().__init__(**kw)
        from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, CLIPTokenizer

        self.model_id = model_id
        self.revision = revision
        self.with_vision = with_vision
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, revision=revision)
        # Both classes expose .text_model and .text_projection. CLIPModel adds the vision tower.
        cls = CLIPModel if with_vision else CLIPTextModelWithProjection
        self.model = cls.from_pretrained(model_id, revision=revision, dtype=self.dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision) if with_vision else None

    def _forward(self, tokens) -> dict:
        # No attention_mask, deliberately. SD 1.5's pipeline calls its CLIP text encoder without one,
        # so this gives the exact hidden states the U-Net conditions on, padding positions included.
        # The EOS pooled output is unaffected, because CLIP's causal mask already hides later padding.
        out = self.model.text_model(input_ids=tokens.input_ids)
        return {
            "last_hidden_state": out.last_hidden_state,
            "pooled": out.pooler_output,
            "text_embeds": self.model.text_projection(out.pooler_output),
            "attention_mask": tokens.attention_mask,
        }

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 16) -> torch.Tensor:
        """CLIP image embeddings [N, 768] in the joint space (requires with_vision=True)."""
        if not self.with_vision:
            raise RuntimeError("Create ClipTextEncoder(with_vision=True) to encode images")
        feats = []
        for start in range(0, len(images), batch_size):
            px = self.processor(images=list(images[start : start + batch_size]), return_tensors="pt")
            px = px["pixel_values"].to(self.device, self.dtype)
            out = self.model.vision_model(pixel_values=px)
            feats.append(self.model.visual_projection(out.pooler_output).cpu())
        return torch.cat(feats)


class BertTextEncoder(_TextEncoder):
    """BERT-base-uncased.

    The output keys are:
    - last_hidden_state: [N, 77, 768].
    - pooled: [N, 768], the [CLS] token.
    - mean_pooled: [N, 768], the attention-masked mean over tokens.
    """

    name = "bert"

    def __init__(self, model_id: str = "google-bert/bert-base-uncased", revision: str | None = None, **kw):
        super().__init__(**kw)
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision, dtype=self.dtype).to(self.device).eval()

    def _forward(self, tokens) -> dict:
        out = self.model(**tokens)
        hidden = out.last_hidden_state
        mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return {"last_hidden_state": hidden, "pooled": hidden[:, 0], "mean_pooled": mean,
                "attention_mask": tokens.attention_mask}


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity [len(a), len(b)], computed in fp32."""
    return F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T


def flatten_captions(records: list[dict], field: str = "clean") -> tuple[list[str], list[tuple[int, int]]]:
    """Flatten the manifest into (texts, index). index[i] = (image_id, caption_position)."""
    texts, index = [], []
    for r in records:
        for j, c in enumerate(r["captions"]):
            texts.append(c[field])
            index.append((r["image_id"], j))
    return texts, index


def encode_and_save(encoder: _TextEncoder, records: list[dict], out_dir: Path, batch_size: int = 64,
                    field: str = "clean", force: bool = False) -> Path:
    """Encode every caption and save one .pt file per encoder. Skips if the file already exists.

    The file contains:
    - pooled embeddings for every caption.
    - the full last_hidden_state only for each image's first caption, which
      keeps the file around 590 MB in fp16 instead of about 3 GB.
    - an index mapping rows to (image_id, caption_position).
    """
    out_path = Path(out_dir) / f"{encoder.name}_coco_val2017.pt"
    if out_path.exists() and not force:
        print(f"[skip] {out_path.name} exists")
        return out_path
    texts, index = flatten_captions(records, field)
    pooled: dict[str, list[torch.Tensor]] = {}
    first_hidden: list[torch.Tensor] = []
    first_rows = [i for i, (_, j) in enumerate(index) if j == 0]
    first_set = set(first_rows)
    for start in tqdm(range(0, len(texts), batch_size), desc=f"encode[{encoder.name}]"):
        out = encoder.encode_batch(texts[start : start + batch_size])
        for k in ("pooled", "text_embeds", "mean_pooled"):
            if k in out:
                pooled.setdefault(k, []).append(out[k].half())
        keep = [i - start for i in range(start, min(start + batch_size, len(texts))) if i in first_set]
        if keep:
            first_hidden.append(out["last_hidden_state"][keep].half())
    payload = {
        "model_id": encoder.model_id,
        "revision": encoder.revision,
        "field": field,
        "index": index,
        "first_caption_rows": first_rows,
        **{k: torch.cat(v) for k, v in pooled.items()},
        "first_caption_hidden": torch.cat(first_hidden),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.part")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    return out_path
