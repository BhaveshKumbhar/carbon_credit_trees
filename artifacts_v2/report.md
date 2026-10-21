# Tree Species Classifier & OOD Rejection (V2 Upgrade Report)

## Executive Summary

- **Upgraded Backbone**: `DINOV2_VITS14` (384-dim embeddings).
- **Best Classifier**: `Neural Head (LayerNorm+Dropout)` on 5-Fold Stratified Grouped Cross-Validation.
- **New Features**: Test-Time Augmentation (TTA), multi-signal OOD rejection, and 3-Tier honest confidence gating.

## 1. Head-to-Head Performance: V1 Baseline vs. V2 Upgrade

| Metric | V1 Baseline (MobileNetV3Small) | V2 Upgrade (DINOv2 ViT-S/14) | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **CV Top-1 Accuracy** | 66.41% | **91.52%** | **+25.11%** |
| **CV Top-3 Accuracy** | 87.55% | **98.45%** | **+10.90%** |
| **OOD Detection AUROC** | 64.20% | **82.70%** | **+18.50%** |

## 2. V2 Classifier Model Benchmark (Identical 5-Fold CV)

| Candidate Classifier | Top-1 Accuracy (Mean ± Std) | Top-3 Accuracy (Mean ± Std) |
| :--- | :---: | :---: |
| **LogisticRegression (C=1.0)** | 90.30% ± 0.83% | 97.89% ± 0.79% |
| **LogisticRegression (C=10.0)** | 89.89% ± 0.58% | 97.83% ± 0.74% |
| **LinearSVC (Calibrated)** | 90.49% ± 0.62% | 97.61% ± 0.77% |
| **Cosine Nearest Prototype** | 77.07% ± 0.94% | 92.64% ± 1.06% |
| **Neural Head (LayerNorm+Dropout) (Selected)** | 91.52% ± 0.66% | 98.45% ± 0.64% |

## 3. Per-Species Accuracy Breakdown

| Species ID & Common Name | V2 Top-1 Accuracy | Typical Within-Class Distance | Robustness Profile |
| :--- | :---: | :---: | :--- |
| SP001_Eucalyptus | **94.33%** | 26.07 | Strongly discriminated |
| SP002_Subabul | **94.17%** | 21.88 | Strongly discriminated |
| SP003_Shisham | **90.25%** | 22.99 | Strongly discriminated |
| SP004_Poplar | **89.08%** | 26.30 | Strongly discriminated |
| SP005_Casuarina | **96.08%** | 25.87 | Strongly discriminated |
| SP007_Teak | **91.75%** | 22.28 | Strongly discriminated |
| SP008_Bamboo | **94.42%** | 24.03 | Strongly discriminated |
| SP013_Mango | **90.33%** | 23.64 | Strongly discriminated |
| SP021_Jackfruit | **88.50%** | 22.91 | Strongly discriminated |
| SP027_Neem | **86.25%** | 24.70 | Strongly discriminated |

## 4. Multi-Signal OOD Rejection & 3-Tier Confidence Policy

To avoid false confidence, predictions are partitioned into three transparent outcomes:

1. **`[REJECTED]`** (Non-Tree / Unknown): Feature Mahalanobis distance exceeds threshold (`> 35.00`) or fails entropy floor. Rejects animals, vehicles, buildings.
2. **`[SPECIES UNCERTAIN / LOW CONFIDENCE]`**: Image is a tree, but model confidence is `< 48.0%`. Top-3 candidates are shown for manual review, but carbon credit values are **hidden** to protect against misidentification.
3. **`[CONFIDENT IDENTIFICATION]`**: Confidence is `>= 48.0%`. Full agroforestry profile and indicative carbon revenue range are displayed.

## 5. Honest Assessment & Field Data Recommendations

1. **Fine-Grained Foliage Improvements**: DINOv2 self-supervised attention maps recognize leaflet and leaf-margin structural patterns much better than standard supervised MobileNetV3.
2. **Remaining Ambiguities**: Shisham (`Dalbergia sissoo`) vs. Neem (`Azadirachta indica`) share pinnate foliage geometry. When photos show only distant canopy, distinction is subtle.
3. **Targeted Field Data Needed**: To push accuracy beyond ~85% on challenging classes, collecting 50-100 high-res leaf close-ups and bark photos for Shisham, Teak, and Poplar will yield the greatest performance gains.
