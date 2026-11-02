# From Words to Worlds: Text-to-Image Generation with Transformer Encoders and Diffusion

**IE 7615 Deep Learning for AI · Milestone 1 proposal · Team: _[team name]_**

## Objective
We will build and analyze a text-to-image pipeline that conditions a pretrained latent diffusion model on Transformer text embeddings. The study asks how the sampling settings (guidance scale, step count, noise scheduler) and the choice of text encoder (CLIP vs. BERT) trade off image fidelity, diversity, and prompt alignment.

## Dataset and why
**COCO Captions 2017 val:** 5,000 images, each with about 5 human captions (25,014 captions in all). We chose it for four reasons:
1. The captions describe everyday multi-object scenes, which tests compositional prompts.
2. It is the standard benchmark for text-to-image FID and CLIP score, so our numbers can be compared with published results.
3. Its 5k size fits a free Colab T4 and Google Drive.
4. The multiple captions per image let us measure caption-to-caption embedding consistency.

Preprocessing works as follows:
- **Captions:** whitespace-normalized, with a lowercased copy kept. Captions under 4 or over 40 words are dropped; on this split that removes only 2 captions. The kept captions are 7–38 words long, with a mean of 10.4.
- **Images:** the short side is resized to 512, then center-cropped to 512×512 and normalized to [-1, 1] for the VAE.

## Model architecture
We use Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16), a latent diffusion model, with these components:
1. **Text encoder:** CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) maps a prompt to 77×768 token embeddings.
2. **VAE:** encodes a 512×512×3 image to a 64×64×4 latent (8× downsampling) and decodes it back.
3. **U-Net:** predicts the noise in the latent at each timestep. Its cross-attention layers use the text embeddings as keys and values, which is how the text conditions the image.
4. **Scheduler:** PNDM (the default) or DDIM turns the noise predictions into a denoising trajectory over *T* steps.
5. **Classifier-free guidance** combines conditional and unconditional predictions: ε̂ = ε(∅) + s·(ε(c) − ε(∅)).

A secondary BERT-base (`google-bert/bert-base-uncased`) encoder gives an alternative 768-d text representation. The U-Net was trained only on CLIP-L embeddings, so BERT conditioning needs a learned projection into CLIP's embedding space. We will train that projection on the COCO caption pairs.

## Planned experiments
| Factor | Values | Fixed elsewhere |
|---|---|---|
| Guidance scale *s* | 1, 3, 5, 7.5, 10, 15 | 30 steps, PNDM |
| Sampling steps | 10, 20, 30, 50 | *s* = 7.5 |
| Noise scheduler | PNDM, DDIM, DPM-Solver++, Euler | 30 steps, *s* = 7.5 |
| Text conditioning | CLIP (native) vs. BERT → learned projection | best setting above |

Every run uses a fixed seed set on one shared prompt set, and every parameter is logged to JSON.

## Evaluation plan
- **FID:** generated images vs. the real COCO val images, using the same 512→299 Inception preprocessing.
- **Inception Score:** a second measure of image quality and diversity.
- **CLIP score:** cosine similarity between the prompt and the generated image, measuring alignment.
- **Qualitative grids:** grids per factor, plus failure analysis on counting, spatial relations and style.
- **Honest comparison:** CLIP-native conditioning is the reference baseline for the BERT variant.

## Team roles
| Member | Primary responsibility | Secondary |
|---|---|---|
| _[Name 1]_ | Data pipeline, EDA, preprocessing | Reproducibility / repo |
| _[Name 2]_ | Text encoders (CLIP, BERT projection) | Evaluation metrics |
| _[Name 3]_ | Diffusion experiments (guidance, steps, schedulers) | Results gallery |
| _[Name 4]_ | Evaluation (FID, IS, CLIP score) and analysis | Report and slides |

## Timeline (to Dec 13)
| Milestone | Target date | Deliverables |
|---|---|---|
| M1 Setup and data | _[Nov 9]_ | Proposal, 5 test generations, repo |
| M2 Integration and baselines | _[Nov 23]_ | Guidance/scheduler sweeps, BERT projection, 1-page observations |
| M3 Evaluation and analysis | _[Nov 30]_ | FID/IS/CLIP-score tables and plots, draft methods and results |
| M4 Report and presentation | Dec 13 | 6–8 page report, gallery, 10–15 min talk, final repo |
