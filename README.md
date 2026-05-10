# Ambigram Generator

Generates 180° rotational ambigrams from a text prompt using **Score Distillation Sampling (SDS)** with Stable Diffusion.

No training data needed. Each ambigram is produced by a single optimization run (~1500 steps, ~15 min on an A100).

---

## How it works

```
 ┌─────────────────────────────────────────────────────┐
 │  Optimizable image  (pixel space, sigmoid-constrained)│
 └────────────┬──────────────────────┬─────────────────┘
              │                      │ rotate 180°
              ▼                      ▼
        VAE encode              VAE encode
              │                      │
     SDS loss (word A)      SDS loss (word B)
              │                      │
              └──────── sum ─────────┘
                          │
                    Adam + cosine LR
```

Two frozen SDS losses guide a single image simultaneously:
- **Upright view** → reads as *word A*
- **Rotated 180°** → reads as *word B*

The image is parameterised as `sigmoid(raw)` so it stays in `[0, 1]`. Initialization blends a rendered rendering of word A with a 180°-rotated rendering of word B to give the optimizer a head start.

Timestep annealing gradually reduces the SDS noise level from coarse global structure to fine-grained legibility as optimization progresses.

---

## Setup

```bash
pip install -r requirements.txt
```

You need a CUDA GPU with at least **16 GB VRAM** for SD 1.5 in fp16. An A100 (40/80 GB) is recommended.

The first run downloads ~4 GB of model weights from HuggingFace.

---

## Usage

### Two-word ambigram
```bash
python generate.py --word-a love --word-b hate
```

### Self-ambigram (reads the same when flipped)
```bash
python generate.py --word-a SWIMS
```

### With custom settings
```bash
python generate.py \
  --word-a angel \
  --word-b devil \
  --steps 2000 \
  --lr 3e-3 \
  --lambda-b 1.0 \
  --guidance-scale 10.0 \
  --anneal-max-t 300 \
  --output-dir outputs/
```

---

## Outputs

Each run writes to `outputs/<word_a>__<word_b>/`:

| File | Description |
|------|-------------|
| `final.png` | Side-by-side: upright \| rotated 180° |
| `final_upright.png` | Upright image only |
| `final_rotated.png` | Rotated image only |
| `step_NNNNN.png` | Intermediate comparisons every `--log-every` steps |

---

## Key hyperparameters

| Flag | Default | Notes |
|------|---------|-------|
| `--steps` | 1500 | More steps → more refined; 1000–2000 usually sufficient |
| `--lr` | 5e-3 | Adam LR; reduce if the image becomes over-saturated |
| `--lambda-b` | 1.0 | Relative weight of the rotated-view loss; increase if word B is harder |
| `--guidance-scale` | 7.5 | CFG scale; 7–12 works well; higher = stronger adherence to prompt |
| `--min-t` / `--max-t` | 50 / 950 | SDS timestep range; wide range early, narrow late |
| `--anneal-max-t` | 400 | Final max timestep after annealing; lower = sharper but riskier |
| `--model` | SD 1.5 | `stabilityai/stable-diffusion-2-1` or `stabilityai/stable-diffusion-xl-base-1.0` also work |

---

## Tips

- **Short words (3–5 chars)** tend to produce cleaner results than long ones.
- **Words with rotational symmetry** (letters that map to other letters when flipped: n↔u, p↔d, b↔q, m↔w) are much easier.
- If one word is much longer than the other, increase `--lambda-b` for the shorter/harder one.
- For self-ambigrams, omit `--word-b`.
- Add `--use-wandb` to monitor loss curves in Weights & Biases.

---

## Method

Based on **DreamFusion** (Poole et al., 2022) — Score Distillation Sampling applied to 2D image optimization.

The key insight: the SDS gradient `w(t) * (ε̂_φ(z_t; y, t) − ε)` can be used as a direct pixel-space update without backpropagating through the UNet (phantom gradient trick), making it cheap to apply two independent SDS losses to the same image from different orientations.
# ambigram
