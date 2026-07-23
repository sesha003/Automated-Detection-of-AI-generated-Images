# Automated Detection of AI-Generated Images

Telling real photographs apart from synthetic, AI-generated images using convolutional neural networks.

*Image Processing & Computer Vision — course project.*

---

## Overview

Modern generative models such as DALL·E produce images that are visually indistinguishable from real photographs. Given a single image and **no metadata**, the task is to decide whether a camera or a model created it.

This repository contains a complete, reproducible Jupyter notebook (`.ipynb`) that builds and compares two CNN-based classifiers for this binary task:

| | Approach 1 | Approach 2 |
|---|---|---|
| **Model** | SimpleCNN (3 conv blocks, from scratch) | Pretrained ResNet-50, fine-tuned |
| **Augmentation** | Flip + small rotation | + RandomResizedCrop + ColorJitter |
| **Optimizer** | Adam, lr 1e-3 | AdamW, lr 1e-4, weight decay 1e-4 |
| **LR schedule** | StepLR (halve every 5 epochs) | CosineAnnealingLR (smooth decay) |
| **Data split** | Train / Test | Train / Val / Test (proper held-out) |
| **Model selection** | Final epoch | Best validation accuracy (checkpointed) |
| **Test accuracy** | ~80% | **93.3%** |

## Why it matters

- **Misinformation** — fabricated photos of events, people and places spread faster than fact-checks.
- **Fraud & deception** — synthetic identities, forged documents and fake product/property images enable scams and account takeovers at scale.
- **Erosion of digital trust** — if any image could be fake, genuine evidence loses credibility too.

Human accuracy on the best AI-generated images sits near **50%**, barely better than a coin flip, while generation is effortless and instant. Automated detection is the only response that matches the pace of generation.

## Dataset

**DALL·E Recognition Dataset** (Kaggle) — 3,781 real vs 17,857 AI-generated images, a heavy ~1:4.7 class imbalance.

Handling the imbalance:
- **Balanced sampling at the source** — equal images per class, so training starts balanced (`real: 3,780 / 3,780`, `fake: 3,781 / 17,855`).
- **Class-weighted loss** as a safety net (weights computed from the train split, applied inside `CrossEntropyLoss`).
- **SMOTE deliberately avoided** — interpolating pixels yields blurry, meaningless images; it suits tabular data, not photographs.

Preprocessing pipeline: resize to 224 × 224 RGB → random horizontal flip (p = 0.5) → random rotation (±10°) → to tensor → normalize to ImageNet mean/std. Augmentation is applied to the training split only; validation and test are resized and normalized only.

## Approach 1 — Baseline CNN from scratch

A small convolutional network learning the real-vs-AI decision directly from pixels, as a clean baseline.

```
Input 224×224×3
  → Conv 3→32   · ReLU · MaxPool
  → Conv 32→64  · ReLU · MaxPool
  → Conv 64→128 · ReLU · MaxPool
  → Flatten → FC 256 · ReLU · Dropout 0.5
  → 2 logits (Real / AI-Generated)
```

Training: class-weighted `CrossEntropyLoss`, Adam (lr 1e-3), StepLR, batch size 32, stratified 80/20 split.

**Result: ~80% test accuracy.** Usable, but well short of dependable — and with clear headroom, motivating a pretrained backbone.

## Approach 2 — Transfer learning with ResNet-50

Rather than learning visual features from a cold start, the model reuses a network that already understands images and is taught only the final decision.

- **Pretrained on ImageNet** (1.2M images) — already detects edges, textures, materials and shapes.
- **Only the head is replaced** — final layer swapped from 2048→1000 to 2048→2, with dropout 0.3.
- **~23.5M parameters** across 50 layers carrying rich, transferable visual knowledge.

Data strategy: a balanced 50/50 sample with a proper stratified three-way split — **70% train / 15% validation / 15% test**. This fixes the earlier leak where the test set was used both to select the best epoch and to report the final number.

Fine-tuning details:
1. **Low learning rate (1e-4)** — small, careful updates so the pretrained weights aren't destroyed.
2. **AdamW + weight decay (1e-4)** — slightly better generalization than plain Adam.
3. **CosineAnnealingLR** — smooth decay instead of abrupt halving.
4. **Best-model checkpointing** — weights saved whenever validation accuracy improves; the best checkpoint is reloaded at the end.

## Results

| Metric | Score |
|---|---|
| Accuracy | **93.3%** |
| Precision | 92.7% |
| Recall | 94.0% |
| F1-score | 0.934 |

Per-class: Real 525/567 (92.6%) · AI-Generated 534/568 (94.0%).

Best validation accuracy of 0.930 was reached at epoch 3 of 4 and checkpointed. The test set was never seen during training or model selection, so this is a generalizable number.

**What drove the +13-point jump:**
- Pretrained features beat from-scratch learning on a small (~7.5k-image) dataset.
- Stronger augmentation reduced overfitting.
- AdamW + low LR + cosine schedule made stable, careful updates to already-good weights.
- A proper held-out test set makes the 93.3% trustworthy — no leakage from model selection.
- Best-epoch checkpointing kept the strongest model rather than the last one.

## Key takeaways

- **Automated detection is necessary.** Humans hover near chance on the best fakes; only a trained model scales to the volume of generated imagery.
- **Transfer learning is the bigger lever.** Swapping a from-scratch CNN for a pretrained ResNet-50 lifted accuracy from ~80% to 93.3% — more than any single hyperparameter tweak.
- **Methodology decides credibility.** Balanced sampling, a true held-out test split and best-epoch checkpointing are what make the final number honest, not just high.

## Limitations

- Trained mainly on DALL·E-style images; may not transfer to Midjourney, Stable Diffusion or newer generators.
- Subsampled to 3,781 fakes for balance — most of the 17,857 available fakes went unused.
- Only 4 fine-tuning epochs at 224×224 resolution.
- No insight yet into which image cues the model relies on.
- Adversarially edited or compressed fakes were not stress-tested.

## Future work

- Train on all 17,857 fakes using class weights instead of subsampling.
- Higher resolution (320–384) and stronger backbones — EfficientNet-B0, ConvNeXt-Tiny.
- Add Grad-CAM to visualize the regions driving each decision.
- Two-stage fine-tuning: freeze the backbone first, then unfreeze.
- Evaluate cross-generator generalization and robustness to compression.

## Tech stack

PyTorch · torchvision · scikit-learn · NumPy · Matplotlib · Jupyter Notebook
