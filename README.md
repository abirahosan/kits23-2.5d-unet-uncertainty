# Kidney, Tumour & Cyst Segmentation on KiTS23: 2.5D U-Net with MC Dropout Uncertainty

BSc thesis code, American International University-Bangladesh (AIUB), Fall 2025-26.

An image-only, reproducible baseline for segmenting kidney, tumour and cyst in contrast-enhanced CT (KiTS23), with a controlled 2D vs 2.5D ablation and post-hoc Monte Carlo Dropout uncertainty.

## Highlights

- **Model:** 2D U-Net, 3-slice axial stack as 3 input channels, 7,763,140 params
- **Training:** 40 epochs, 192x192 slices, HU windowing, per-volume z-score, foreground-biased slice sampling, seed 42
- **Split:** deterministic patient-level 342 / 73 / 74 (train / val / test)
- **Uncertainty:** MC Dropout (p = 0.3, 20 passes)
- **Hardware:** single NVIDIA A100 80GB (Google Colab)

## Results (74 held-out test patients)

| Class | Per-patient Dice [95% CI] | Global voxel Dice |
|---|---|---|
| Kidney | 0.879 [0.858, 0.897] | 0.879 |
| Tumour | 0.433 [0.359, 0.501] | 0.673 |
| Cyst | 0.219 [0.150, 0.297] | 0.434 |

**Depth ablation (2.5D vs 2D):** +0.003 kidney Dice, +0.046 tumour Dice; 2.5D better on 47/74 patients for tumour.

**Uncertainty:** RCC = 0.994 (well-calibrated confidence), error-detection AUC = 0.861 (predictive entropy). Thresholded uncertainty does not isolate errors (RIU < 0.05), so it is useful for *ranking* cases for review, not for hard thresholding.

**Cyst analysis:** Dice correlates with log cyst volume (Pearson r = 0.61, n = 42). Failure is mainly resolution and class imbalance, not purely architectural.

**Negative result:** the KiTS23 `aua_risk_score` metadata field leaks the malignancy label (96% accuracy); removing it drops a downstream classifier to chance (AUC 0.53). Avoid it in any metadata-based classification.

## Repo layout

```
kits23_segmentation_FINAL.ipynb   # full pipeline: preprocessing, training, ablation, MC Dropout, analysis
splits/                           # patient-level train/val/test split files
models/                           # trained checkpoints
results/                          # tables and figures
requirements.txt
```

`data/` and `cache/` are git-ignored. KiTS23 volumes are never committed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download KiTS23 from the official repository: https://github.com/neheller/kits23

Place it under `data/` (git-ignored). Check the notebook's config cell for the expected path.

## Usage

Open `kits23_segmentation_FINAL.ipynb` and run top to bottom. Training was done on a GPU (Colab A100); a smaller GPU will need a reduced batch size. The notebook builds the cached slices, trains the 2.5D model, retrains at depth 1 for the ablation, then runs MC Dropout and the cyst/metadata analyses.

## Limitations

- Image-only, single compact model; far smaller than the KiTS23 winning ensemble (15 models, ~5.5x10^9 params)
- Tumour and cyst per-patient Dice remain low; small lesions suffer at 192x192 resolution
- Single split, single seed

## Reference

Heller et al., KiTS23 dataset: https://github.com/neheller/kits23

## License

Code: add a license of your choice (e.g. MIT). KiTS23 data is governed by its own license.

## Figures

Per-class test Dice (74 held-out patients):

![Per-class test Dice](results/segmentation/Figure_4.2.jpeg)

2D vs 2.5D depth ablation:

![Depth ablation](results/ablation/Figure_4.6.jpeg)

Cyst Dice vs cyst volume (Pearson r = 0.61):

![Cyst Dice vs volume](results/segmentation/Figure_4.8.jpeg)

MC Dropout calibration vs uncertainty threshold:

![MC Dropout calibration](results/uncertainty/Figure_4.9.jpeg)

Uncertainty heatmaps (tumour probability variance):

![Uncertainty heatmaps](results/uncertainty/Figure_4.11.jpeg)

All figures and CSV tables are in [`results/`](results/): `segmentation/`, `ablation/`, `uncertainty/`.
