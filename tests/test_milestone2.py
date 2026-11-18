"""CPU tests for the Milestone 2 adapter data and metrics, CLIP score and sweep logging. No downloads."""

import csv

import pytest
import torch

from src import bert_adapter as ba
from src import metrics, sweeps


def test_excluded_image_ids_follow_the_coco_prompts():
    from src.utils import load_config

    cfg = load_config()
    coco_ids = {p["image_id"] for p in cfg["milestone1_prompts"] if p.get("image_id")}
    assert coco_ids and coco_ids <= set(ba.excluded_image_ids(cfg))
    cfg["milestone1_prompts"].append({"id": "new", "prompt": "x", "image_id": 999})
    cfg["bert_adapter"]["exclude_image_ids"] = []
    assert ba.excluded_image_ids(cfg) == sorted(coco_ids | {999})


def test_duplicate_groups_follow_config():
    from src.utils import load_config

    cfg = load_config()
    groups = sweeps.duplicate_groups(cfg, "p1")
    assert groups["default"] == ["baseline/clip/p1.png", "cfg/p1_s7.5.png", "scheduler/p1_pndm.png"]
    assert groups["dpm_default_steps"] == ["scheduler/p1_dpm.png", "steps/p1_dpm_30.png"]
    cfg["generation"].update(guidance_scale=7.0, scheduler="ddim")
    assert sweeps.duplicate_groups(cfg, "p1")["default"] == ["baseline/clip/p1.png", "scheduler/p1_ddim.png"]


def test_split_by_image_no_overlap_and_excludes():
    image_ids = torch.tensor([i // 5 for i in range(100)])  # 20 images x 5 captions
    train, val = ba.split_by_image(image_ids, val_frac=0.2, seed=42, exclude=[3, 7])
    tr_imgs, va_imgs = set(image_ids[train].tolist()), set(image_ids[val].tolist())
    assert not tr_imgs & va_imgs
    assert not {3, 7} & (tr_imgs | va_imgs)
    assert len(va_imgs) == round(18 * 0.2) and len(train) + len(val) == 90
    again = ba.split_by_image(image_ids, 0.2, 42, [3, 7])
    assert torch.equal(train, again[0]) and torch.equal(val, again[1])


def _pairs(n=40, length=6, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, length, d, generator=g)
    w = torch.randn(d, d, generator=g)
    return {"x": x.half(), "y": (x @ w + 0.5).half(), "mask": torch.ones(n, length, dtype=torch.bool),
            "image_ids": torch.arange(n)}


def test_linear_baseline_recovers_linear_map():
    pairs = _pairs()
    rows = torch.arange(30)
    w, b = ba.fit_linear_baseline(pairs, rows, batch_size=8, device="cpu", ridge=1e-8)
    res = ba.evaluate(lambda x: x @ w + b, pairs, torch.arange(30, 40), batch_size=8, device="cpu")
    assert res["cos_all"] > 0.99 and res["cos_content"] > 0.99


def test_mean_baseline_and_metric_sums():
    pairs = _pairs()
    mean_y = ba.fit_mean_baseline(pairs, torch.arange(40), 16, "cpu")
    assert torch.allclose(mean_y, pairs["y"].float().mean(0), atol=1e-3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[:, 0] = True
    s = ba.token_metric_sums(torch.ones(2, 3, 4), torch.ones(2, 3, 4), mask)
    assert s["n_all"] == 6 and s["n_content"] == 2 and s["mse_all"] == 0
    assert ba._finalize(s)["cos_content"] == pytest.approx(1.0)


def test_clip_score_arithmetic():
    img = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    txt = torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert metrics.clip_score_from_embeds(img, txt).tolist() == pytest.approx([100.0, 0.0, 0.0])


def test_write_csv_overwrites_with_fixed_fields(tmp_path):
    path = tmp_path / "log.csv"
    sweeps.write_csv([{"sweep": "a", "extra": 1}], path, ["sweep", "steps"])
    sweeps.write_csv([{"sweep": "b"}, {"sweep": "c"}], path, ["sweep", "steps"])
    rows = list(csv.DictReader(open(path)))
    assert [r["sweep"] for r in rows] == ["b", "c"] and list(rows[0]) == ["sweep", "steps"]


def test_select_prompts_follows_config_order():
    from src.utils import load_config

    cfg = load_config()
    ids = [p["id"] for p in sweeps.select_prompts(cfg)]
    assert ids == cfg["milestone2_sweeps"]["sweep_prompts"] and len(ids) == 5


def test_config_milestone2_paths_resolve(tmp_path, monkeypatch):
    from src.utils import get_paths, load_config

    monkeypatch.setenv("IE7615_ARTIFACT_ROOT", str(tmp_path))
    paths = get_paths(load_config())
    for key in ("milestone2_samples", "milestone2_logs", "bert_adapter_ckpt"):
        assert str(paths[key]).startswith(str(tmp_path))
