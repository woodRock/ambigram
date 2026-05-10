# Ambigram Generator

Generates 180° rotational **self-ambigrams** — words that read identically when rotated 180° — using CLIP-guided per-glyph optimisation. No training data or diffusion model required.

---

## How it works

The word is split into letter pairs. Each pair `(a, b)` at index `i` satisfies `a = word[i]` and `b = word[N-1-i]`, so a single glyph must read as `a` upright and `b` when rotated 180°. All glyphs are optimised simultaneously against CLIP text features, then composed into the final strip.

The composition identity guarantees the result is a self-ambigram regardless of glyph quality:

```
rotate_180([ g₀ | g₁ | g₂ | rot(g₁) | rot(g₀) ]) = [ g₀ | g₁ | g₂ | rot(g₁) | rot(g₀) ]
```

Two glyph representations are supported:

| Mode | Parameterisation | Output |
|------|-----------------|--------|
| `pixel` | Direct (1, H, W) tensor, projected gradient descent | PNG |
| `bezier` | Cubic Bézier stroke control points, differentiable soft rasterizer | PNG + SVG |

---

## Setup

```bash
pip install -r requirements.txt
```

Runs on CPU, MPS (Apple Silicon), or CUDA. A GPU with ≥8 GB VRAM is recommended; an A100 or H100 gives best speed. The first run downloads ~900 MB of CLIP weights.

---

## Usage

### Basic

```bash
python generate.py --word SWIMS
python generate.py --word NOON --steps 800 --glyph-size 384
```

### Bézier mode (smooth strokes + SVG export)

```bash
python generate.py --word SWIMS --mode bezier --n-strokes 12 --steps 800
```

Control points are initialized from the rendered letter's ink pixels, then optimised. SVG files are saved alongside the PNGs at the end.

### Perceptual loss

Adds VGG16 feature matching towards a clean rendered reference letter — encourages letterform aesthetics beyond what CLIP measures.

```bash
python generate.py --word SWIMS --use-perceptual --lambda-perc 0.1
```

### Character classifier loss

A small CNN (trained on synthetic font renders) acts as a differentiable readability signal. Train it once, then use it for all future runs.

```bash
# One-time setup (takes ~2 min on CPU, ~30s on GPU)
python tools/generate_font_dataset.py   # renders A-Z with augmentation → data/char_dataset/
python tools/train_classifier.py        # trains CNN → data/char_classifier.pth

# Use during optimisation
python generate.py --word SWIMS --use-classifier --lambda-char 0.5
```

### Faster inference (CUDA only)

```bash
python generate.py --word SWIMS --torch-compile
```

Applies `torch.compile` to the CLIP encoder. Adds ~30s warmup on the first run, then speeds up subsequent steps by ~20–30%.

### All improvements combined

```bash
python generate.py --word SWIMS \
  --mode bezier --n-strokes 12 \
  --use-perceptual \
  --use-classifier \
  --steps 800 --glyph-size 256
```

---

## Outputs

All files are written to `outputs/<WORD>/`.

| File | Description |
|------|-------------|
| `final.png` | Side-by-side comparison: upright \| rotated 180° |
| `final_upright.png` | Upright image only |
| `final_rotated.png` | Rotated image only |
| `glyph_i_A_B.png` | Final glyph for pair A↔B |
| `glyph_i_A_B_rot.png` | That glyph rotated 180° |
| `glyph_i_A_B.svg` | SVG export (Bézier mode only) |
| `glyph_i_A_B/step_NNNN.png` | Progress snapshots every `--log-every` steps |

---

## All flags

### Word and output
| Flag | Default | Description |
|------|---------|-------------|
| `--word` | *(required)* | Word to ambigramise |
| `--output-dir` | `outputs` | Root output directory |
| `--log-every` | `100` | Save progress snapshots every N steps |
| `--device` | auto | `cpu`, `cuda`, `mps`, or `cuda:1` etc. |

### CLIP
| Flag | Default | Description |
|------|---------|-------------|
| `--clip-model` | `ViT-L-14` | OpenCLIP model name |
| `--clip-pretrained` | `openai` | OpenCLIP pretrained weights tag |
| `--n-augments` | `16` | Random crops per orientation per step |
| `--torch-compile` | off | Compile CLIP with `torch.compile` (CUDA only) |

### Representation
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `pixel` | `pixel` or `bezier` |
| `--n-strokes` | `10` | Bézier strokes per glyph (bezier mode) |
| `--stroke-width` | `0.04` | Stroke half-width as fraction of canvas (bezier mode) |

### Optimisation
| Flag | Default | Description |
|------|---------|-------------|
| `--glyph-size` | `256` | Square canvas size per glyph in px |
| `--steps` | `500` | Optimisation steps |
| `--lr` | `2e-2` | Adam learning rate |

### Pixel regularisation
| Flag | Default | Description |
|------|---------|-------------|
| `--lambda-tv` | `2e-3` | Total variation weight (pixel mode) |
| `--lambda-bw` | `0.3` | Black-and-white push weight (pixel mode) |

### Character classifier
| Flag | Default | Description |
|------|---------|-------------|
| `--use-classifier` | off | Enable CNN readability loss |
| `--classifier-path` | `data/char_classifier.pth` | Path to trained checkpoint |
| `--lambda-char` | `0.5` | Classifier loss weight |

### Perceptual loss
| Flag | Default | Description |
|------|---------|-------------|
| `--use-perceptual` | off | Enable VGG16 perceptual loss |
| `--lambda-perc` | `0.1` | Perceptual loss weight |

---

## Tips

- **Words with natural rotational symmetry** are much easier: S, H, I, N, O, X, Z flip to themselves or each other; SWIMS and NOON are classic examples.
- **Bézier mode** produces cleaner, print-ready output. Pixel mode is faster to iterate.
- If letters are hard to read, increase `--n-augments` (32–48) or `--steps` (1000+).
- `--use-perceptual` is most effective when combined with `--use-classifier`.
- Reduce `--lambda-bw` if Bézier strokes are too thin or fragmented.
