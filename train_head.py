"""
train_head_v2.py
================
Evaluates and benchmarks multiple classifier architectures on cached high-capacity features
(e.g., DINOv2 ViT-S/14 or ConvNeXt-Tiny) using strict 5-Fold Stratified Grouped CV:

Classifiers Evaluated:
1. Multinomial Logistic Regression (C tuned: 0.1, 1.0, 10.0)
2. Calibrated Linear Support Vector Classifier (LinearSVC + Platt scaling)
3. Cosine Nearest-Class-Prototype Classifier (metric learning baseline)
4. Deep Neural Head (LayerNorm -> Dropout(0.2) -> Linear with Label Smoothing 0.1)

Outputs:
- Comparative benchmark table with Top-1 & Top-3 accuracy.
- Per-class accuracy breakdown for all 10 species.
- Best model saved to `artifacts_v2/best_model.pt`.
- Results summary saved to `artifacts_v2/cv_results.json`.
"""

import os
import sys
import json
import copy
import random
import pickle
import argparse
import numpy as np

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))

ARTIFACTS_DIR = "artifacts_v2"
CACHE_DIR = "cache_v2"
BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pt")
CV_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "cv_results.json")

N_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.2
LABEL_SMOOTHING = 0.1


class ClassifierHead(nn.Module):
    """LayerNorm -> Dropout -> Linear head."""
    def __init__(self, in_features, num_classes, dropout=DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


class CosinePrototypeClassifier:
    """Nearest Centroid classifier using cosine similarity and softmax temperature scaling."""
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.centroids = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.centroids = np.zeros((len(classes), X.shape[1]), dtype=np.float32)
        for c in classes:
            class_feats = X[y == c]
            mean_vec = class_feats.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            self.centroids[c] = mean_vec / (norm + 1e-8)
        return self

    def predict_proba(self, X):
        norm_X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity matrix (N, C)
        sims = norm_X @ self.centroids.T
        # Softmax with temperature
        scaled = sims / self.temperature
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def calc_top_k(probs, targets, k=3):
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]
    correct = [targets[i] in top_k_indices[i] for i in range(len(targets))]
    return float(np.mean(correct))


def train_neural_head(train_x, train_y, val_x, val_y, in_dim, num_classes):
    train_ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_x, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ClassifierHead(in_features=in_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                v_loss += criterion(out, by).item()
                n_b += 1
        v_loss /= n_b
        scheduler.step(v_loss)

        if v_loss < best_loss - 1e-4:
            best_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            out = model(bx)
            p = torch.softmax(out, dim=1)
            val_probs.append(p.cpu().numpy())

    return model, np.concatenate(val_probs, axis=0)


def benchmark_classifiers(cache_path):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    data = np.load(cache_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    group_ids = data["group_ids"]
    class_names = list(data["class_names"])
    backbone = str(data.get("backbone", "dinov2_vits14"))
    num_classes = len(class_names)
    in_dim = features.shape[1]

    print(f"\n==================================================")
    print(f"BENCHMARKING CLASSIFIERS ON BACKBONE: {backbone.upper()}")
    print(f"Features: {features.shape} ({in_dim}-dim, {num_classes} classes)")
    print(f"Cross-Validation: 5-Fold Stratified Grouped CV")
    print(f"==================================================")

    models_to_test = {
        "LogisticRegression (C=1.0)": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
        "LogisticRegression (C=10.0)": lambda: LogisticRegression(max_iter=1000, C=10.0, random_state=SEED),
        "LinearSVC (Calibrated)": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED, max_iter=2000), cv=3),
        "Cosine Nearest Prototype": lambda: CosinePrototypeClassifier(temperature=0.07),
        "Neural Head (LayerNorm+Dropout)": "neural_head",
    }

    results = {name: {"top1": [], "top3": [], "oof_probs": np.zeros((len(features), num_classes))} for name in models_to_test}

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(features, labels, groups=group_ids), 1):
        print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")
        tr_x, tr_y = features[tr_idx], labels[tr_idx]
        val_x, val_y = features[val_idx], labels[val_idx]

        for m_name, m_builder in models_to_test.items():
            if m_builder == "neural_head":
                _, probs = train_neural_head(tr_x, tr_y, val_x, val_y, in_dim, num_classes)
            else:
                clf = m_builder()
                clf.fit(tr_x, tr_y)
                probs = clf.predict_proba(val_x)

            t1 = accuracy_score(val_y, probs.argmax(axis=1))
            t3 = calc_top_k(probs, val_y, k=3)
            results[m_name]["top1"].append(t1)
            results[m_name]["top3"].append(t3)
            results[m_name]["oof_probs"][val_idx] = probs
            print(f"  {m_name:32s} -> Top-1: {t1*100:.2f}%, Top-3: {t3*100:.2f}%")

    print("\n" + "="*70)
    print("5-FOLD CV COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Classifier Model':<35} | {'Top-1 Accuracy (Mean +/- Std)':<20} | {'Top-3 Accuracy'}")
    print("-" * 70)

    best_model_name = None
    best_top1_score = -1.0

    summary_records = {}
    for m_name, res in results.items():
        m_t1 = np.mean(res["top1"]) * 100
        s_t1 = np.std(res["top1"]) * 100
        m_t3 = np.mean(res["top3"]) * 100
        s_t3 = np.std(res["top3"]) * 100

        print(f"{m_name:<35} | {m_t1:6.2f}% +/- {s_t1:4.2f}%       | {m_t3:6.2f}% +/- {s_t3:4.2f}%")
        summary_records[m_name] = {
            "top1_mean": float(m_t1),
            "top1_std": float(s_t1),
            "top3_mean": float(m_t3),
            "top3_std": float(s_t3),
        }

        if m_t1 > best_top1_score:
            best_top1_score = m_t1
            best_model_name = m_name

    print("="*70)
    print(f"[BEST CLASSIFIER]: {best_model_name} (Top-1: {best_top1_score:.2f}%)")

    # Per-Class Accuracy of the best model
    best_oof_preds = results[best_model_name]["oof_probs"].argmax(axis=1)
    print(f"\nPer-Class Accuracy ({best_model_name}):")
    per_class_acc = {}
    for c_idx, c_name in enumerate(class_names):
        mask = (labels == c_idx)
        c_acc = float(np.mean(best_oof_preds[mask] == labels[mask]) * 100)
        per_class_acc[c_name] = c_acc
        print(f"  {c_name:20s}: {c_acc:6.2f}% ({np.sum(best_oof_preds[mask] == labels[mask])}/{np.sum(mask)})")

    # Train Final Deployment Model on all data
    print(f"\n[INFO] Training final deployment model ({best_model_name}) on ALL data...")
    if best_model_name == "Neural Head (LayerNorm+Dropout)":
        final_model = ClassifierHead(in_features=in_dim, num_classes=num_classes)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        full_ds = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
        final_model.train()
        for epoch in range(1, 40):
            for bx, by in full_loader:
                optimizer.zero_grad()
                out = final_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        save_obj = {
            "type": "neural_head",
            "state_dict": final_model.state_dict(),
            "in_features": in_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "backbone": backbone,
            "model_name": best_model_name,
        }
        torch.save(save_obj, BEST_MODEL_PATH)
    elif "LogisticRegression" in best_model_name:
        C_val = 10.0 if "10.0" in best_model_name else 1.0
        final_clf = LogisticRegression(max_iter=1000, C=C_val, random_state=SEED)
        final_clf.fit(features, labels)
        save_obj = {
            "type": "sklearn",
            "model": final_clf,
            "in_features": in_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "backbone": backbone,
            "model_name": best_model_name,
        }
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(save_obj, f)
    elif "LinearSVC" in best_model_name:
        final_clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED, max_iter=2000), cv=3)
        final_clf.fit(features, labels)
        save_obj = {
            "type": "sklearn",
            "model": final_clf,
            "in_features": in_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "backbone": backbone,
            "model_name": best_model_name,
        }
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(save_obj, f)
    else:
        final_clf = CosinePrototypeClassifier(temperature=0.07)
        final_clf.fit(features, labels)
        save_obj = {
            "type": "cosine_prototype",
            "centroids": final_clf.centroids,
            "temperature": 0.07,
            "in_features": in_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "backbone": backbone,
            "model_name": best_model_name,
        }
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(save_obj, f)

    print(f"[SUCCESS] Final model saved to {BEST_MODEL_PATH}")

    # Save complete CV metrics JSON
    cv_export = {
        "backbone": backbone,
        "feature_dim": in_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "benchmark_summary": summary_records,
        "best_model": best_model_name,
        "best_top1_mean": float(best_top1_score),
        "per_class_accuracy": per_class_acc,
    }
    with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cv_export, f, indent=2)
    print(f"[SUCCESS] Benchmark results saved to {CV_RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark head classifiers v2.")
    parser.add_argument("--cache", default="cache_v2/features_dinov2_vits14.npz",
                        help="Path to cached features .npz file.")
    args = parser.parse_args()
    if not os.path.exists(args.cache):
        print(f"[ERROR] Cache file not found at {args.cache}. Run extract_features_v2.py first.")
        return
    benchmark_classifiers(args.cache)


if __name__ == "__main__":
    main()
