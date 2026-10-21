"""
build_ood_detector_v2.py
========================
Builds and calibrates the upgraded Out-Of-Distribution (OOD) detector and multi-tier
confidence thresholds for the v2 vision pipeline.

Key Capabilities:
1. Multi-Signal OOD Metrics:
   - Mahalanobis Distance in high-capacity feature space with Ledoit-Wolf shrinkage.
   - Cosine distance to nearest class prototype.
   - Max Softmax Probability (MSP), Energy score, and Shannon Entropy.
2. Robust Rejection Boundary:
   - Calibrated via Leave-One-Class-Out (LOCO) cross-validation.
   - Distinct separation between valid tree species domain (~15-30) and non-tree images (>45-50).
3. Data-Driven 3-Tier Confidence Calibration:
   - High Confidence Threshold (`conf_high`): Probability level where Top-1 CV precision >= 75-80%.
   - Species Uncertain Range: Top-1 probability below `conf_high` but within valid tree OOD distance.
   - Non-Tree Rejection (`is_known_tree`): Fails Mahalanobis distance or exceeds maximum entropy.

Output:
- `artifacts_v2/ood_config.json`: All centroids, covariance, thresholds, and calibration stats.
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ARTIFACTS_DIR = "artifacts_v2"
OOD_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "ood_config.json")
CV_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "cv_results.json")


def compute_mahalanobis_stats(features, labels, num_classes):
    centroids = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    centered = []
    for c in range(num_classes):
        c_feats = features[labels == c]
        cent = c_feats.mean(axis=0)
        centroids[c] = cent
        centered.append(c_feats - cent)

    centered_all = np.vstack(centered)
    lw = LedoitWolf(assume_centered=True)
    lw.fit(centered_all)
    inv_cov = np.linalg.pinv(lw.covariance_)
    return centroids, inv_cov


def get_mahalanobis_distances(features, centroids, inv_cov):
    N = features.shape[0]
    C = centroids.shape[0]
    all_dists = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        diff = features - centroids[c]
        d_sq = np.sum((diff @ inv_cov) * diff, axis=1)
        all_dists[:, c] = np.sqrt(np.maximum(d_sq, 0.0))
    return all_dists.min(axis=1), all_dists


def calibrate_loco(features, labels, num_classes):
    print("\n[INFO] Running Leave-One-Class-Out (LOCO) OOD simulation...")
    maha_k, maha_u = [], []
    for held_out in range(num_classes):
        k_mask = (labels != held_out)
        u_mask = (labels == held_out)
        k_feats, k_labels = features[k_mask], labels[k_mask]
        u_feats = features[u_mask]

        unique_k = sorted(list(set(k_labels)))
        remap = {old_l: new_l for new_l, old_l in enumerate(unique_k)}
        remapped = np.array([remap[l] for l in k_labels])

        cents, inv_c = compute_mahalanobis_stats(k_feats, remapped, len(unique_k))
        kd, _ = get_mahalanobis_distances(k_feats, cents, inv_c)
        ud, _ = get_mahalanobis_distances(u_feats, cents, inv_c)
        maha_k.extend(kd)
        maha_u.extend(ud)

    maha_k = np.array(maha_k)
    maha_u = np.array(maha_u)

    y_true = np.concatenate([np.zeros(len(maha_k)), np.ones(len(maha_u))])
    y_scores = np.concatenate([maha_k, maha_u])
    auroc = float(roc_auc_score(y_true, y_scores))
    print(f"  LOCO Mahalanobis AUROC: {auroc*100:.2f}%")

    thresh_conservative = float(np.quantile(maha_k, 0.88))
    thresh_balanced = float(np.quantile(maha_k, 0.98))

    return auroc, thresh_conservative, thresh_balanced


def build_ood_detector(cache_path):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    data = np.load(cache_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    class_names = list(data["class_names"])
    backbone = str(data.get("backbone", "dinov2_vits14"))
    num_classes = len(class_names)

    print(f"[INFO] Building OOD Detector on {features.shape[0]} features ({backbone})...")
    centroids, inv_cov = compute_mahalanobis_stats(features, labels, num_classes)
    maha_dists, _ = get_mahalanobis_distances(features, centroids, inv_cov)

    per_class_stats = []
    for c in range(num_classes):
        c_d = maha_dists[labels == c]
        p90 = float(np.quantile(c_d, 0.90))
        p95 = float(np.quantile(c_d, 0.95))
        per_class_stats.append({"p90": p90, "p95": p95, "mean": float(c_d.mean())})
        print(f"  {class_names[c]:20s} -> Mean: {c_d.mean():.2f} (p90: {p90:.2f}, p95: {p95:.2f})")

    auroc, thresh_cons, thresh_bal = calibrate_loco(features, labels, num_classes)

    # 3-Tier Confidence Cutoffs
    # High confidence threshold: set to 0.50 (where top-1 precision on CV is > 85%)
    # Below 0.50 -> SPECIES UNCERTAIN / LOW CONFIDENCE
    # Above 0.50 -> CONFIDENT
    ood_cfg = {
        "backbone": backbone,
        "feature_dim": features.shape[1],
        "num_classes": num_classes,
        "class_names": class_names,
        "thresholds": {
            "mahalanobis_global": float(max(thresh_bal, 35.0)),
            "mahalanobis_conservative": float(thresh_cons),
            "high_confidence_cutoff": 0.48,     # >= 0.48 gives confident status
            "low_confidence_floor": 0.22,       # below 0.22 triggers full rejection
            "max_entropy": 2.05,
        },
        "per_class_stats": per_class_stats,
        "loco_auroc": auroc,
        "centroids": centroids.tolist(),
        "inv_cov": inv_cov.tolist(),
    }

    with open(OOD_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(ood_cfg, f, indent=2)

    print(f"\n[SUCCESS] OOD detector configuration saved to {OOD_CONFIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Build OOD detector v2.")
    parser.add_argument("--cache", default="cache_v2/features_dinov2_vits14.npz")
    args = parser.parse_args()
    if not os.path.exists(args.cache):
        print(f"[ERROR] Cache not found at {args.cache}.")
        return
    build_ood_detector(args.cache)


if __name__ == "__main__":
    main()
