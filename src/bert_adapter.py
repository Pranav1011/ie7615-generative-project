"""Learned projection from BERT-base [B, 77, 768] into CLIP ViT-L/14 [B, 77, 768] space."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from .embeddings import BertTextEncoder
from .utils import get_device, make_generator


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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bert_hidden: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(bert_hidden)
        x = self.encoder(x)
        return self.out_proj(x)


class ProjectedBertTextEncoder:
    """Drop-in encoder returning CLIP-space last_hidden_state [N, 77, 768] for SD 1.5 U-Net."""

    name = "bert_projected"

    def __init__(self, cfg: dict, ckpt_path: Path, device: str | None = None, dtype: torch.dtype | None = None):
        self.device = device or get_device()
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.bert = BertTextEncoder(
            model_id=cfg["models"]["bert"],
            revision=cfg["models"]["revisions"]["bert"],
            device=self.device,
            dtype=torch.float32,
            max_length=cfg["embeddings"]["max_length"],
        )
        acfg = cfg["bert_adapter"]
        self.adapter = BertToClipAdapter(
            hidden_dim=acfg["hidden_dim"],
            num_layers=acfg["num_layers"],
            dropout=acfg["dropout"],
        ).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.adapter.load_state_dict(state["model_state_dict"])
        self.adapter.eval()

    @torch.no_grad()
    def encode_prompts(self, texts: Sequence[str]) -> torch.Tensor:
        bert_out = self.bert.encode(texts, batch_size=len(texts))
        hidden = bert_out["last_hidden_state"].to(self.device, dtype=torch.float32)
        projected = self.adapter(hidden)
        return projected.to(dtype=self.dtype)


def train_bert_adapter(cfg: dict, paths: dict[str, Path]) -> tuple[BertToClipAdapter, list[dict]]:
    """Train adapter on cached first_caption_hidden tensors from embeddings.py."""
    device = get_device()
    acfg = cfg["bert_adapter"]

    bert_pt = torch.load(paths["embeddings"] / "bert_coco_val2017.pt", map_location="cpu", weights_only=True)
    clip_pt = torch.load(paths["embeddings"] / "clip_coco_val2017.pt", map_location="cpu", weights_only=True)

    x_all = bert_pt["first_caption_hidden"].float()  # [5000, 77, 768]
    y_all = clip_pt["first_caption_hidden"].float()  # [5000, 77, 768]

    dataset = TensorDataset(x_all, y_all)
    n_val = int(len(dataset) * acfg["val_split"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=make_generator(cfg["seed"]))

    train_loader = DataLoader(train_ds, batch_size=acfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=acfg["batch_size"], shuffle=False)

    model = BertToClipAdapter(
        hidden_dim=acfg["hidden_dim"],
        num_layers=acfg["num_layers"],
        dropout=acfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=acfg["lr"], weight_decay=acfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=acfg["epochs"])

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, acfg["epochs"] + 1):
        model.train()
        tr_mse_sum = tr_cos_sum = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            mse = F.mse_loss(pred, by)
            cos = F.cosine_similarity(pred, by, dim=-1).mean()
            loss = acfg["mse_weight"] * mse + acfg["cosine_weight"] * (1.0 - cos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_mse_sum += mse.item() * len(bx)
            tr_cos_sum += cos.item() * len(bx)
        scheduler.step()

        model.eval()
        val_mse_sum = val_cos_sum = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_mse_sum += F.mse_loss(pred, by).item() * len(bx)
                val_cos_sum += F.cosine_similarity(pred, by, dim=-1).mean().item() * len(bx)

        row = {
            "epoch": epoch,
            "train_mse": round(tr_mse_sum / n_train, 6),
            "train_cos_sim": round(tr_cos_sum / n_train, 6),
            "val_mse": round(val_mse_sum / n_val, 6),
            "val_cos_sim": round(val_cos_sum / n_val, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(row)
        val_total = acfg["mse_weight"] * row["val_mse"] + acfg["cosine_weight"] * (1.0 - row["val_cos_sim"])
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({"model_state_dict": model.state_dict(), "config": acfg, "metrics": row}, paths["bert_adapter_ckpt"])

    log_csv = paths["milestone2_logs"] / "bert_training_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    return model, history
