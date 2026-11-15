# Multimodal skin lesion classification: photo + patient information

Project 2.2 - Multimodal Data Analysis
Author: Swapnil Shetty (A1991188)

## What this project does
Does combining a smartphone photo of a skin lesion with patient information (age, body site, symptoms) classify the lesion better than either source alone, and what is the best way to combine them?

The task is 6-class classification on PAD-UFES-20: actinic keratosis (ACK), basal cell carcinoma (BCC), melanoma (MEL), nevus (NEV), squamous cell carcinoma (SCC) and seborrheic keratosis (SEK). BCC, MEL and SCC are cancers.

## Approach
- **Image features:** frozen pretrained EfficientNet-B0 (1,280 features) and ResNet-50 (2,048 features).
- **Patient features:** imputed, scaled and one-hot encoded (fitted on training data only). Two versions:
  - *conservative*: age, body region, itch, hurt, bleed, elevation (main result);
  - *full*: adds fields that are missing for about 35% of records (an ablation, because missingness itself leaks the label).
- **Fusion methods compared:**
  1. early fusion (PCA of image features + patient features, then one classifier),
  2. late fusion (weighted average of image and patient model probabilities),
  3. gated fusion (a small network where patient data gates the image features, inspired by MetaBlock),
  4. a concatenation network as a control.
- **Baselines:** image-only and patient-only models.
- **Splits:** patient-grouped and class-stratified (no patient appears in more than one of train / validation / test), seed 42.

## Main results (test set, macro-F1 with 95% patient-level bootstrap CI)
| Model | Macro-F1 |
|---|---|
| Photo only (EfficientNet-B0 + LR) | 0.534 |
| Patient data only (conservative, LR) | 0.465 |
| Late fusion (conservative) | 0.571 |
| Concat network (conservative) | 0.598 |
| Early fusion (conservative) | 0.610 |
| Gated fusion (conservative) | 0.629 |

Fusion beats either source alone, but the fusion methods are too close to rank statistically. Full tables, confidence intervals and figures are in `results/`.

## Repository layout
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── FinalCode.ipynb      # the full pipeline, run top to bottom
└── results/
    ├── figures/             # class distribution, learning curves, CI plot, confusion matrices, failure cases
    └── tables/              # test results, bootstrap CIs, per-class recall, subgroups, CV, efficiency
```

## Data
The dataset is not included due to its size. Download PAD-UFES-20 from Mendeley Data (DOI: 10.17632/zr7vgbcyr2.1) and unzip it so the folder contains `metadata.csv` and an `images/` folder.

Reference: Pacheco, A.G.C., et al. (2020) PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221.

## How to run
1. Install Python 3.10 or newer.
2. Create an environment and install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate          
   pip install -r requirements.txt
   ```
3. Open `notebook/codefinal.ipynb` in VS Code or Jupyter.
4. In the first code cell, change the `BASE` line to the folder that contains `metadata.csv` and `images/`.
5. Run all cells from top to bottom (Restart & Run All).

Outputs (figures, CSV tables and cached embeddings) are written to a `final_outputs` folder next to the dataset folder.

## Notes on reproducibility
- All random seeds are fixed (seed 42; networks use fixed per-run seeds).
- Image embeddings are extracted once and cached as `.npy` files. The first run takes a few minutes on CPU; later runs load the cache.
- Preprocessing (imputation, scaling, PCA) is always fitted on training rows only.
- Exact numbers can differ slightly across library versions or hardware. Timing in the efficiency table depends on the machine.
- Tested with the package versions listed in the report (Appendix E).

## Limitations
Small, single-centre dataset; melanoma is very rare (2.3%); no Fitzpatrick types 5-6, so fairness across skin tones cannot be assessed; frozen image features and no augmentation. This is a research prototype, not a diagnostic tool.
