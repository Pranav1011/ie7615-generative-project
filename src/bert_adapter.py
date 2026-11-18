"""Learned projection from BERT-base [B, 77, 768] into CLIP ViT-L/14 [B, 77, 768] space.

SD 1.5's U-Net was trained only on CLIP-L hidden states. To condition it on
BERT, BERT's token states are first regressed onto CLIP's, using paired
COCO captions.

Evaluation is kept honest in four ways:
- The split is made by image, so the captions of one image never sit on both sides.
- The prompt images (cfg.bert_adapter.exclude_image_ids) are left out of training and validation.
- Cosine is reported on content tokens (CLIP BOS ... EOS) as well as on all 77 positions.
  Padding positions are nearly identical across captions, so they inflate the all-token number.
- Two baselines are reported:
  - mean_clip_embedding predicts the per-position mean CLIP state. This is the meaningful bar, since it
    captures everything a model can get without reading the caption.
  - linear_per_position maps BERT position i to CLIP position i. It is a weak baseline, because the two
    tokenizers split words differently and so content positions don't line up. It only shows how much
    the adapter's attention-based realignment adds.
- The best epoch is chosen by validation loss on content tokens, not on the padding-dominated all-token loss.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .embeddings import BertTextEncoder, ClipTextEncoder, flatten_captions
from .utils import get_device, save_json


class BertToClipAdapter(nn.Module):
    """Maps BERT last_hidden_state [B, 77, 768] -> CLIP cross-attention space [B, 77, 768]."""

    def __init__(self, hidden_dim: int = 768, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors don't apply with norm_first=True, so turn them off explicitly.
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bert_hidden: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(bert_hidden)
        x = self.encoder(x)
        return self.out_proj(x)


def _build_adapter(acfg: dict) -> BertToClipAdapter:
    return BertToClipAdapter(hidden_dim=acfg["hidden_dim"], num_layers=acfg["num_layers"], dropout=acfg["dropout"])


class ProjectedBertTextEncoder:
    """Drop-in encoder returning CLIP-space last_hidden_state [N, 77, 768] for the SD 1.5 U-Net."""

    name = "bert_projected"

    def __init__(self, cfg: dict, ckpt_path: Path, device: str | None = None, dtype: torch.dtype | None = None,
                 bert_encoder=None):
        self.device = device or get_device()
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        # Same BERT dtype as in training (fp16 on GPU), so inference sees the inputs the adapter was fitted on.
        self.bert = bert_encoder or BertTextEncoder(
            model_id=cfg["models"]["bert"],
            revision=cfg["models"]["revisions"]["bert"],
            device=self.device,
            max_length=cfg["embeddings"]["max_length"],
        )
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        # Build from the architecture stored in the checkpoint, not the current config.
        self.adapter = _build_adapter(state.get("config") or cfg["bert_adapter"]).to(self.device)
        self.adapter.load_state_dict(state["model_state_dict"])
        self.adapter.eval()

    @torch.no_grad()
    def encode_prompts(self, texts: Sequence[str]) -> torch.Tensor:
        bert_out = self.bert.encode(list(texts), batch_size=len(texts))
        hidden = bert_out["last_hidden_state"].to(self.device, dtype=torch.float32)
        return self.adapter(hidden).to(dtype=self.dtype)


# ---------------------------------------------------------------- data


def excluded_image_ids(cfg: dict) -> list[int]:
    """Images kept out of adapter training: every COCO prompt's image_id, plus any listed in the config."""
    from_prompts = {p["image_id"] for p in cfg.get("milestone1_prompts", []) if p.get("image_id") is not None}
    return sorted(from_prompts | set(cfg["bert_adapter"].get("exclude_image_ids", [])))


@torch.no_grad()
def build_pairs(records: list[dict], clip_encoder, bert_encoder, batch_size: int = 128,
                field: str = "clean") -> dict:
    """Encode every caption with both encoders.

    Returns {x: BERT [N, 77, 768] fp16, y: CLIP [N, 77, 768] fp16,
    mask: content mask [N, 77], image_ids: [N]}. All tensors are on the CPU.
    The tensors are preallocated and filled in place, so peak RAM is about
    6 GB for 25k captions instead of double that.
    """
    texts, index = flatten_captions(records, field)
    n = len(texts)
    x = y = None
    mask = torch.zeros(n, clip_encoder.max_length, dtype=torch.bool)
    for start in tqdm(range(0, n, batch_size), desc="encode pairs"):
        chunk = texts[start : start + batch_size]
        clip_out = clip_encoder.encode_batch(chunk)
        cy = clip_out["last_hidden_state"]
        cx = bert_encoder.encode_batch(chunk)["last_hidden_state"]
        if x is None:
            x = torch.empty(n, *cx.shape[1:], dtype=torch.float16)
            y = torch.empty(n, *cy.shape[1:], dtype=torch.float16)
        end = start + len(chunk)
        x[start:end] = cx.half()
        y[start:end] = cy.half()
        # CLIP's tokenizer attention mask is 1 from BOS through the caption's EOS and 0 on the padding.
        mask[start:end] = clip_out["attention_mask"].bool()
    return {"x": x, "y": y, "mask": mask, "image_ids": torch.tensor([i for i, _ in index])}


def split_by_image(image_ids: torch.Tensor, val_frac: float, seed: int,
                   exclude: Sequence[int] = ()) -> tuple[torch.Tensor, torch.Tensor]:
    """Row indices for train/val. Rows from the same image share a side, and excluded images are dropped."""
    excluded = set(int(i) for i in exclude)
    unique = sorted(set(image_ids.tolist()) - excluded)
    random.Random(seed).shuffle(unique)
    n_val = max(1, round(len(unique) * val_frac))
    val_images = set(unique[:n_val])
    keep = torch.tensor([int(i) not in excluded for i in image_ids.tolist()])
    is_val = torch.tensor([int(i) in val_images for i in image_ids.tolist()])
    rows = torch.arange(len(image_ids))
    return rows[keep & ~is_val], rows[keep & is_val]


# ---------------------------------------------------------------- metrics and baselines


def token_metric_sums(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """Sums for MSE and cosine over all tokens and over content tokens. Divide by the counts."""
    pred, target = pred.float(), target.float()
    cos = F.cosine_similarity(pred, target, dim=-1)          # [B, L]
    sq = (pred - target).pow(2).mean(-1)                      # [B, L]
    m = mask.to(pred.device)
    return {
        "mse_all": sq.sum().item(), "cos_all": cos.sum().item(), "n_all": cos.numel(),
        "mse_content": sq[m].sum().item(), "cos_content": cos[m].sum().item(), "n_content": int(m.sum().item()),
    }


def _finalize(sums: dict) -> dict:
    return {
        "mse_all": sums["mse_all"] / sums["n_all"],
        "cos_all": sums["cos_all"] / sums["n_all"],
        "mse_content": sums["mse_content"] / max(sums["n_content"], 1),
        "cos_content": sums["cos_content"] / max(sums["n_content"], 1),
    }


def _accumulate(total: dict | None, part: dict) -> dict:
    return dict(part) if total is None else {k: total[k] + part[k] for k in total}


def _batches(rows: torch.Tensor, batch_size: int, shuffle: bool = False, seed: int = 0):
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        rows = rows[torch.randperm(len(rows), generator=g)]
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@torch.no_grad()
def evaluate(predict, pairs: dict, rows: torch.Tensor, batch_size: int, device: str) -> dict:
    """Run predict(x_batch) -> y_hat over the given rows and return finalized metrics."""
    total = None
    for b in _batches(rows, batch_size):
        x = pairs["x"][b].to(device).float()
        y = pairs["y"][b].to(device).float()
        total = _accumulate(total, token_metric_sums(predict(x), y, pairs["mask"][b]))
    return _finalize(total)


@torch.no_grad()
def fit_mean_baseline(pairs: dict, rows: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    """Per-position mean CLIP embedding over the training rows, [77, 768]."""
    acc = None
    for b in _batches(rows, batch_size):
        s = pairs["y"][b].to(device).float().sum(0)
        acc = s if acc is None else acc + s
    return acc / len(rows)


@torch.no_grad()
def fit_linear_baseline(pairs: dict, rows: torch.Tensor, batch_size: int, device: str,
                        ridge: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token linear map y = x W + b, fitted in closed form (ridge) from accumulated normal equations."""
    d = pairs["x"].shape[-1]
    xtx = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device)
    xty = torch.zeros(d + 1, pairs["y"].shape[-1], dtype=torch.float64, device=device)
    for b in _batches(rows, batch_size):
        x = pairs["x"][b].to(device).double().reshape(-1, d)
        x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype, device=device)], 1)
        y = pairs["y"][b].to(device).double().reshape(len(x), -1)
        xtx += x.T @ x
        xty += x.T @ y
    lam = ridge * xtx.diagonal().mean()
    w = torch.linalg.solve(xtx + lam * torch.eye(d + 1, dtype=xtx.dtype, device=device), xty)
    return w[:-1].float(), w[-1].float()


# ---------------------------------------------------------------- training


def _default_encoders(cfg: dict):
    m, max_len = cfg["models"], cfg["embeddings"]["max_length"]
    clip = ClipTextEncoder(m["clip"], revision=m["revisions"]["clip"], max_length=max_len)
    bert = BertTextEncoder(m["bert"], revision=m["revisions"]["bert"], max_length=max_len)
    return clip, bert


def train_bert_adapter(cfg: dict, paths: dict[str, Path], records: list[dict] | None = None,
                       clip_encoder=None, bert_encoder=None, device: str | None = None):
    """Train the adapter on all captions and write the checkpoint, the epoch log CSV and a metrics JSON.

    Returns (model, history, metrics).
    """
    from .data import load_manifest

    device = device or get_device()
    acfg = cfg["bert_adapter"]
    bs = acfg["batch_size"]
    records = records if records is not None else load_manifest(paths["manifest"])

    own_encoders = clip_encoder is None or bert_encoder is None
    if own_encoders:
        clip_encoder, bert_encoder = _default_encoders(cfg)
    pairs = build_pairs(records, clip_encoder, bert_encoder, batch_size=bs)
    if own_encoders:
        del clip_encoder, bert_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    excluded = excluded_image_ids(cfg)
    train_rows, val_rows = split_by_image(pairs["image_ids"], acfg["val_split"], cfg["seed"], excluded)
    print(f"pairs: {len(pairs['x'])} | train {len(train_rows)} | val {len(val_rows)} | "
          f"excluded images {excluded}")

    # Baselines, scored on the same validation rows
    mean_y = fit_mean_baseline(pairs, train_rows, bs, device)
    w, b = fit_linear_baseline(pairs, train_rows, bs, device)
    baselines = {
        "mean_clip_embedding": evaluate(lambda x: mean_y.expand(len(x), -1, -1), pairs, val_rows, bs, device),
        "linear_per_position": evaluate(lambda x: x @ w + b, pairs, val_rows, bs, device),
    }
    print("baselines (val):", {k: {m: round(v, 4) for m, v in r.items()} for k, r in baselines.items()})

    torch.manual_seed(cfg["seed"])
    model = _build_adapter(acfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=acfg["lr"], weight_decay=acfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=acfg["epochs"])
    ckpt_path = Path(paths["bert_adapter_ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best = None
    for epoch in range(1, acfg["epochs"] + 1):
        model.train()
        tr = None
        for batch in _batches(train_rows, bs, shuffle=True, seed=cfg["seed"] + epoch):
            x = pairs["x"][batch].to(device).float()
            y = pairs["y"][batch].to(device).float()
            pred = model(x)
            mse = F.mse_loss(pred, y)
            cos = F.cosine_similarity(pred, y, dim=-1).mean()
            loss = acfg["mse_weight"] * mse + acfg["cosine_weight"] * (1.0 - cos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr = _accumulate(tr, token_metric_sums(pred.detach(), y, pairs["mask"][batch]))
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        model.eval()
        val = evaluate(model, pairs, val_rows, bs, device)
        train = _finalize(tr)
        # Select on content tokens: all-token loss is dominated by padding that looks alike for every caption.
        val_loss = acfg["mse_weight"] * val["mse_content"] + acfg["cosine_weight"] * (1.0 - val["cos_content"])
        row = {"epoch": epoch, "lr": round(lr, 8),
               **{f"train_{k}": round(v, 6) for k, v in train.items()},
               **{f"val_{k}": round(v, 6) for k, v in val.items()},
               "val_loss_content": round(val_loss, 6)}
        history.append(row)
        print({k: row[k] for k in ("epoch", "train_mse_all", "val_mse_all", "val_cos_all", "val_cos_content")})
        if best is None or val_loss < best["val_loss_content"]:
            best = row
            tmp = ckpt_path.with_name(ckpt_path.name + ".part")
            torch.save({"model_state_dict": model.state_dict(), "config": acfg, "metrics": row}, tmp)
            tmp.replace(ckpt_path)

    logs = Path(paths["milestone2_logs"])
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "bert_training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "best_epoch": best,
        "baselines_val": baselines,
        "n_pairs_total": len(pairs["x"]),
        "n_train_pairs": len(train_rows),
        "n_val_pairs": len(val_rows),
        "n_train_images": len(set(pairs["image_ids"][train_rows].tolist())),
        "n_val_images": len(set(pairs["image_ids"][val_rows].tolist())),
        "excluded_image_ids": list(excluded),
        "split": "by image_id, seeded",
        "checkpoint": str(ckpt_path),
        "adapter_config": acfg,
    }
    save_json(metrics, logs / "bert_adapter_metrics.json", add_env=True)
    return model, history, metrics
