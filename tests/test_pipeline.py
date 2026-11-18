"""Fast CPU tests for the data, preprocessing and utility code. No model downloads."""

import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from src import data, preprocess, utils


@pytest.fixture
def mini_coco(tmp_path):
    """Three images with mixed captions, laid out like COCO val2017."""
    img_dir = tmp_path / "val2017"
    img_dir.mkdir()
    images = []
    for i, (w, h) in enumerate([(640, 480), (480, 640), (500, 500)], start=1):
        name = f"{i:012d}.jpg"
        arr = np.random.default_rng(i).integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / name)
        images.append({"id": i, "file_name": name, "width": w, "height": h})
    anns = [
        {"id": 10, "image_id": 1, "caption": "  A dog  runs\non the beach. "},
        {"id": 11, "image_id": 1, "caption": "Dog."},
        {"id": 12, "image_id": 2, "caption": " ".join(["word"] * 41)},
        {"id": 13, "image_id": 2, "caption": "Two cats sit on a sofa together"},
        {"id": 14, "image_id": 3, "caption": "too short"},
    ]
    cap_path = tmp_path / "captions_val2017.json"
    cap_path.write_text(json.dumps({"images": images, "annotations": anns}))
    return cap_path, img_dir


def test_clean_caption_collapses_whitespace():
    assert data.clean_caption("  A dog  runs\non the beach. ") == "A dog runs on the beach."


def test_build_manifest_filters_and_groups(mini_coco):
    cap_path, _ = mini_coco
    records, stats = data.build_manifest(cap_path, "coco/val2017_512", min_words=4, max_words=40)
    assert [r["image_id"] for r in records] == [1, 2]
    assert records[0]["file_path"] == "coco/val2017_512/000000000001.jpg"
    assert records[0]["captions"][0]["clean"] == "A dog runs on the beach."
    assert records[0]["captions"][0]["lower"] == "a dog runs on the beach."
    assert stats["dropped"] == {"too_short": 2, "too_long": 1, "image_without_captions": 1}
    assert stats["n_captions_kept"] == 2


def test_manifest_roundtrip(mini_coco, tmp_path):
    records, _ = data.build_manifest(mini_coco[0])
    path = data.save_manifest(records, tmp_path / "m.jsonl")
    assert data.load_manifest(path) == records


def test_safe_extract_blocks_zip_slip(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="Unsafe path"):
        data.safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_skips_existing(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("annotations/a.json", "{}")
    assert len(data.safe_extract(z, tmp_path / "out")) == 1
    assert data.safe_extract(z, tmp_path / "out") == []


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (300, 200), (512, 512)])
def test_resize_center_crop_is_square(size):
    img = Image.new("RGB", size, "red")
    assert preprocess.resize_center_crop(img, 512).size == (512, 512)


def test_normalize_range_and_inverse():
    arr = np.array([[[0, 128, 255]]], dtype=np.uint8).repeat(4, 0).repeat(4, 1)
    t = preprocess.to_model_tensor(Image.fromarray(arr))
    assert t.shape == (3, 4, 4)
    assert t.min().item() == -1.0 and t.max().item() == 1.0
    assert np.array_equal(np.asarray(preprocess.from_model_tensor(t)), arr)


def test_cache_and_dataset(mini_coco, tmp_path):
    cap_path, img_dir = mini_coco
    records, _ = data.build_manifest(cap_path, "cache")
    root = tmp_path / "root"
    assert preprocess.count_uncached(records, root) == 2
    first = preprocess.cache_resized(records, img_dir, root, size=64)
    assert first["written"] == 2
    assert preprocess.count_uncached(records, root) == 0
    assert preprocess.cache_resized(records, img_dir, root, size=64)["skipped"] == 2
    with Image.open(root / records[0]["file_path"]) as im:
        assert im.size == (64, 64)
    ds = preprocess.CocoImageCaptionDataset(records, root, size=64)
    item = ds[0]
    assert item["pixel_values"].shape == (3, 64, 64)
    assert -1.0 <= item["pixel_values"].min() and item["pixel_values"].max() <= 1.0
    assert item["caption"] == "A dog runs on the beach."


def test_set_seed_is_deterministic():
    utils.set_seed(123)
    a = (torch.randn(4), np.random.rand(), utils.make_generator(5).initial_seed())
    utils.set_seed(123)
    b = (torch.randn(4), np.random.rand(), utils.make_generator(5).initial_seed())
    assert torch.equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2] == 5


def test_make_image_grid_size():
    imgs = [Image.new("RGB", (512, 512), c) for c in ("red", "green", "blue")]
    grid = utils.make_image_grid(imgs, ["a", "b", "c"], ncols=3, cell_size=64)
    assert grid.width == 3 * (64 + 6) + 6 and grid.height > 64


def test_save_json_records_environment(tmp_path):
    p = utils.save_json({"seed": 42}, tmp_path / "x.json", add_env=True)
    out = json.loads(p.read_text())
    assert out["seed"] == 42 and "timestamp_utc" in out
    assert "diffusers" in out["environment"]["packages"]


def test_config_prompts():
    cfg = utils.load_config()
    prompts = cfg["milestone1_prompts"]
    assert len(prompts) == 5
    assert sum(p["source"] == "coco_val2017" for p in prompts) == 2
    assert len({p["id"] for p in prompts}) == 5
    assert cfg["models"]["diffusion"] == "stable-diffusion-v1-5/stable-diffusion-v1-5"


def test_slugify():
    from src.generate import slugify

    assert slugify("A cat, left of a DOG!") == "a_cat_left_of_a_dog"


def test_verify_md5(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    data.verify_md5(f, "5d41402abc4b2a76b9719d911017c592")
    data.verify_md5(f, None)
    with pytest.raises(IOError, match="MD5 mismatch"):
        data.verify_md5(f, "0" * 32)


def test_build_manifest_rejects_path_in_file_name(mini_coco):
    cap_path, _ = mini_coco
    raw = json.loads(cap_path.read_text())
    raw["images"][0]["file_name"] = "../../evil.jpg"
    cap_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unexpected file_name"):
        data.build_manifest(cap_path)


def test_prepare_manifest_rebuilds_when_filters_change(mini_coco, tmp_path):
    cap_path, _ = mini_coco
    cfg = utils.load_config()
    cfg["data"]["captions_file"] = cap_path.name
    paths = {"manifest": tmp_path / "m.jsonl", "annotations": cap_path.parent}
    records, _ = data.prepare_manifest(cfg, paths)
    assert sum(len(r["captions"]) for r in records) == 2
    cfg["data"]["min_words"] = 8  # both kept captions have 6-7 words
    records, stats = data.prepare_manifest(cfg, paths)
    assert stats["min_words"] == 8 and records == []


def test_encode_and_save_roundtrip_weights_only(tmp_path):
    from src import embeddings

    class FakeEncoder(embeddings._TextEncoder):
        name, model_id, revision = "fake", "fake/model", "abc123"

        def __init__(self):
            self.device, self.dtype, self.max_length = "cpu", torch.float32, 77

        def encode_batch(self, texts):
            n = len(texts)
            return {"last_hidden_state": torch.randn(n, 77, 8), "pooled": torch.randn(n, 8)}

    records = [{"image_id": i, "captions": [{"clean": f"c{i}{j}"} for j in range(3)]} for i in range(5)]
    path = embeddings.encode_and_save(FakeEncoder(), records, tmp_path, batch_size=4)
    blob = torch.load(path, weights_only=True)
    assert blob["pooled"].shape == (15, 8) and blob["pooled"].dtype == torch.float16
    assert blob["first_caption_hidden"].shape == (5, 77, 8)
    assert blob["first_caption_rows"] == [0, 3, 6, 9, 12] and blob["revision"] == "abc123"


def test_bert_adapter_shape_and_dtype():
    import torch
    from src.bert_adapter import BertToClipAdapter

    adapter = BertToClipAdapter(hidden_dim=768, num_layers=2, dropout=0.1)
    dummy_bert = torch.randn(2, 77, 768)
    out = adapter(dummy_bert)
    assert out.shape == (2, 77, 768)
    assert out.half().dtype == torch.float16
