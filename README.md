# From Words to Worlds: Text-to-Image with Transformer Encoders + Diffusion

IE 7615 Deep Learning for AI group project. We condition **Stable Diffusion v1.5** on Transformer text embeddings (**CLIP ViT-L/14**, plus **BERT-base** for comparison). Captions and images come from **COCO 2017 val**. Then we study how guidance scale, sampling steps, noise schedulers and encoder choice affect image quality (FID, IS) and prompt alignment (CLIP score).

- **Proposal:** [`docs/proposal.md`](docs/proposal.md)
- **Milestone 1 samples:** [`outputs/milestone1_samples/`](outputs/milestone1_samples/)

## Repo layout
```
configs/config.yaml          every path, model ID, seed and generation setting
src/
  utils.py                   seeding, Drive-aware paths, JSON run logs, image grids
  data.py                    COCO download, caption cleaning, manifest
  preprocess.py              resize + center-crop to 512, [-1, 1] tensors, JPEG cache, Dataset
  embeddings.py              CLIP + BERT batch encoders, cosine similarity, save to .pt
  generate.py                SD 1.5 pipeline (fp16, attention slicing), seeded generate() + JSON sidecar
notebooks/
  00_colab_setup.ipynb       mount Drive, clone repo, install, GPU + version check
  01_data_prep.ipynb         download, clean, EDA, manifest + 512px cache to Drive
  02_text_embeddings.ipynb   CLIP/BERT embeddings, similarity checks, save to Drive
  03_baseline_generation.ipynb  5 prompts -> images, grid, prompts.json
docs/proposal.md             1-page Milestone 1 proposal
outputs/milestone1_samples/  committed Milestone 1 images + logs (small files only)
tests/                       fast CPU tests (no model downloads)
```

## Running in Google Colab
1. In Colab, go to **File → Open notebook → GitHub**, paste this repo's URL and open `notebooks/00_colab_setup.ipynb`.
2. Under **Runtime → Change runtime type**, choose **GPU** (a T4 is enough).
3. Run all cells. The first cell of every notebook mounts Drive, clones or pulls the repo to `/content/ie7615-generative-project` and installs `requirements.txt`, so each notebook also runs standalone.
4. Run the notebooks in order: **00 → 01 → 02 → 03**. Every step skips work whose output already exists, so it's safe to rerun cells or restart the runtime.

**Private repo?** Add a Colab secret named `GITHUB_TOKEN` (key icon in the left sidebar) holding a fine-grained, read-only token for this repo. The setup cell uses it for clone and pull, and it never saves the token to `.git/config`.

### Approximate runtimes on a T4
- **01:** the first run downloads the 778 MB image zip and writes 5,000 cached JPEGs to Drive. After that it's seconds.
- **02:** loads CLIP ViT-L/14 and BERT-base, then encodes all 25,012 captions with each.
- **03:** a few seconds per 512×512 image at 30 steps. Each image's JSON sidecar records the measured time and peak VRAM.

## Where artifacts live
Large files never go into git. On Colab they live on Google Drive:
```
/content/drive/MyDrive/ie7615-generative/
  coco/raw/                 annotations + val2017 zips (downloaded once)
  coco/annotations/         captions_val2017.json
  coco/val2017_512/         512x512 center-cropped JPEG cache (about 300 MB)
  coco/manifest.jsonl       one record per image: image_id, file_name, file_path, width, height, captions
  embeddings/               clip_coco_val2017.pt, bert_coco_val2017.pt, embeddings_summary.json
  outputs/                  EDA figures, env logs, Drive mirror of generated samples
```
The original images are extracted to the Colab VM's local disk (`/content/scratch`), which is faster than Drive. They're re-extracted from the zip on Drive whenever the 512px cache needs rebuilding. When you run locally, everything goes under `./data/`. Set `IE7615_ARTIFACT_ROOT` to override the location.

## Reproducibility
- `configs/config.yaml` holds every setting. Notebooks read it instead of hard-coding values.
- `set_seed()` seeds Python, NumPy and torch. Every image is drawn from its own `torch.Generator` seeded with the configured seed (42).
- Every generated image gets a JSON sidecar. It records the prompt, seed, steps, guidance, scheduler, model ID, dtype, elapsed time, peak VRAM, library versions and GPU.
- **Pinned versions:** `requirements.txt` pins diffusers 0.42.1, transformers 5.22.1, accelerate 1.15.0 and safetensors 0.8.0. Colab's own CUDA build of torch is used as-is.
- **Model IDs** (checked on the Hugging Face Hub 2026-11-02):
  - `stable-diffusion-v1-5/stable-diffusion-v1-5` (the `runwayml/` repo was taken down in 2024 and now redirects here)
  - `openai/clip-vit-large-patch14`
  - `google-bert/bert-base-uncased`

## Milestone 1 prompts
Seed 42, 30 steps, guidance 7.5, PNDM. Two prompts are real COCO captions and three are custom prompts of increasing difficulty. The second table previews the later analysis.

**Real COCO captions**

| id | prompt | source |
|---|---|---|
| p1_coco_zebra | Herd of zebra grazing on dried grasses in open field. | COCO image 508586 |
| p2_coco_train | A train traveling down tracks next to a brick building. | COCO image 201148 |

**Custom prompts**

| id | prompt | difficulty |
|---|---|---|
| p3_single_object | a red apple on a white wooden table | single object |
| p4_spatial | a cat sitting to the left of a dog on a park bench | multiple objects + spatial relation |
| p5_style | a lighthouse on a rocky cliff at sunset, watercolor painting | scene + style descriptor |

## Local development (optional, CPU only)
```bash
uv venv .venv && uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m pytest -q
```
