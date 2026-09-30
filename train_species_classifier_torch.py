# """
# PyTorch training pipeline for the 10-species tree classifier, using your
# NVIDIA RTX 3050's CUDA cores directly (no WSL2 needed — this is the main
# reason for migrating off TensorFlow, which dropped native Windows GPU
# support at 2.11+).

# WHY FINE-TUNING IS BACK ON:
# The TF version had to disable Phase 2 (unfreezing backbone layers) because
# it consistently destabilized training with only ~20 images/class across 40
# classes. With 10 classes and 250-300+ images/class, there's enough data to
# safely fine-tune — so DO_FINE_TUNE defaults to True here. If you see the
# same destabilization pattern (loss jumping up hard at the start of Phase 2
# and not recovering), that's your signal to set it back to False.

# Requires:
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#     pip install scikit-learn pillow

# Usage:
#     python train_species_classifier_torch.py
# """

# import os
# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
# from PIL import Image
# from sklearn.model_selection import StratifiedKFold

# DATA_DIR = "tree_images"
# IMG_SIZE = 224
# BATCH_SIZE = 32          # comfortable for 4GB VRAM with MobileNetV3Small — lower to 16 if you hit CUDA OOM
# N_FOLDS = 5
# HEAD_EPOCHS = 40
# FINE_TUNE_EPOCHS = 20
# HEAD_LR = 8e-4
# FINE_TUNE_LR = 3e-5
# UNFREEZE_LAST_N_BLOCKS = 4   # unfreeze the last N inverted-residual blocks of the backbone
# DO_FINE_TUNE = True          # see note above — flip to False if it destabilizes again
# EARLY_STOP_PATIENCE = 12
# MODEL_OUT_PATH = "tree_species_classifier_final.pt"
# CLASS_NAMES_OUT_PATH = "class_names.txt"

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


# def get_device():
#     if torch.cuda.is_available():
#         name = torch.cuda.get_device_name(0)
#         print(f"Using GPU: {name}")
#         return torch.device("cuda")
#     print("CUDA not available — falling back to CPU (check your PyTorch install if unexpected)")
#     return torch.device("cpu")


# def collect_filepaths_and_labels():
#     class_names = sorted([
#         d for d in os.listdir(DATA_DIR)
#         if os.path.isdir(os.path.join(DATA_DIR, d))
#     ])
#     filepaths, labels = [], []
#     for idx, class_name in enumerate(class_names):
#         class_dir = os.path.join(DATA_DIR, class_name)
#         for fname in os.listdir(class_dir):
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 filepaths.append(os.path.join(class_dir, fname))
#                 labels.append(idx)
#     return np.array(filepaths), np.array(labels), class_names


# # Lighter augmentation, matching what we proved works in the TF debugging —
# # no saturation shift (color can be discriminative between species), mild
# # jitter and crop only.
# train_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
#     transforms.RandomCrop(IMG_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.08, contrast=0.08),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])

# eval_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])


# class TreeDataset(Dataset):
#     def __init__(self, filepaths, labels, transform):
#         self.filepaths = filepaths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.filepaths)

#     def __getitem__(self, idx):
#         img = Image.open(self.filepaths[idx]).convert("RGB")
#         img = self.transform(img)
#         return img, self.labels[idx]


# def build_model(num_classes, device):
#     weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
#     backbone = mobilenet_v3_small(weights=weights)
#     features = backbone.features  # conv backbone, output 576 channels
#     for p in features.parameters():
#         p.requires_grad = False

#     class TreeClassifier(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.features = features
#             self.pool = nn.AdaptiveAvgPool2d(1)
#             self.norm = nn.LayerNorm(576)  # same fix as the TF version — normalizes
#                                             # pooled features before the head, which is
#                                             # what resolved the loss-explosion issue there
#             self.dropout = nn.Dropout(0.2)
#             self.classifier = nn.Linear(576, num_classes)

#         def forward(self, x):
#             x = self.features(x)
#             x = self.pool(x).flatten(1)
#             x = self.norm(x)
#             x = self.dropout(x)
#             return self.classifier(x)

#     model = TreeClassifier().to(device)
#     return model


# def top_k_accuracy(outputs, targets, k=3):
#     _, pred = outputs.topk(k, dim=1)
#     correct = pred.eq(targets.view(-1, 1).expand_as(pred))
#     return correct.any(dim=1).float().mean().item()


# def run_epoch(model, loader, criterion, optimizer, device, train=True):
#     model.train() if train else model.eval()
#     total_loss, total_acc, total_top3, n_batches = 0.0, 0.0, 0.0, 0
#     with torch.set_grad_enabled(train):
#         for imgs, labels in loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             if train:
#                 optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             if train:
#                 loss.backward()
#                 optimizer.step()
#             acc = (outputs.argmax(1) == labels).float().mean().item()
#             top3 = top_k_accuracy(outputs, labels, k=3)
#             total_loss += loss.item()
#             total_acc += acc
#             total_top3 += top3
#             n_batches += 1
#     return total_loss / n_batches, total_acc / n_batches, total_top3 / n_batches


# def train_phase(model, train_loader, val_loader, epochs, lr, device,
#                  label, has_validation=True):
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)
#     criterion = nn.CrossEntropyLoss()

#     best_loss = float("inf")
#     best_state = copy.deepcopy(model.state_dict())
#     patience_counter = 0

#     for epoch in range(1, epochs + 1):
#         train_loss, train_acc, train_top3 = run_epoch(
#             model, train_loader, criterion, optimizer, device, train=True)

#         if has_validation:
#             val_loss, val_acc, val_top3 = run_epoch(
#                 model, val_loader, criterion, optimizer, device, train=False)
#             monitor_loss = val_loss
#             print(f"  [{label}] epoch {epoch}/{epochs} - "
#                   f"loss {train_loss:.3f} acc {train_acc:.3f} top3 {train_top3:.3f} | "
#                   f"val_loss {val_loss:.3f} val_acc {val_acc:.3f} val_top3 {val_top3:.3f}")
#         else:
#             monitor_loss = train_loss
#             print(f"  [{label}] epoch {epoch}/{epochs} - "
#                   f"loss {train_loss:.3f} acc {train_acc:.3f} top3 {train_top3:.3f}")

#         scheduler.step(monitor_loss)

#         if monitor_loss < best_loss - 0.005:
#             best_loss = monitor_loss
#             best_state = copy.deepcopy(model.state_dict())
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= EARLY_STOP_PATIENCE:
#                 print(f"  [{label}] early stopping at epoch {epoch}")
#                 break

#     model.load_state_dict(best_state)
#     return model


# def train_one_fold(train_fp, train_lb, val_fp, val_lb, num_classes, fold_num, device):
#     train_ds = TreeDataset(train_fp, train_lb, train_transform)
#     val_ds = TreeDataset(val_fp, val_lb, eval_transform)
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#     model = build_model(num_classes, device)

#     print(f"  [fold {fold_num}] Phase 1: training head...")
#     model = train_phase(model, train_loader, val_loader, HEAD_EPOCHS, HEAD_LR,
#                          device, f"fold{fold_num}-phase1")

#     if DO_FINE_TUNE:
#         # Unfreeze the last N inverted-residual blocks of the backbone
#         blocks = list(model.features.children())
#         for block in blocks[-UNFREEZE_LAST_N_BLOCKS:]:
#             for p in block.parameters():
#                 p.requires_grad = True
#         print(f"  [fold {fold_num}] Phase 2: fine-tuning last {UNFREEZE_LAST_N_BLOCKS} blocks...")
#         model = train_phase(model, train_loader, val_loader, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
#                              device, f"fold{fold_num}-phase2")
#     else:
#         print(f"  [fold {fold_num}] Phase 2 skipped (DO_FINE_TUNE=False)")

#     criterion = nn.CrossEntropyLoss()
#     val_loss, val_acc, val_top3 = run_epoch(model, val_loader, criterion, None, device, train=False)
#     print(f"  [fold {fold_num}] FINAL val_accuracy={val_acc:.3f}  val_top3_acc={val_top3:.3f}")
#     return {"accuracy": val_acc, "top3_acc": val_top3}


# def run_cross_validation(filepaths, labels, num_classes, device):
#     skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
#     fold_metrics = []
#     for fold_num, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), 1):
#         print(f"\n=== Fold {fold_num}/{N_FOLDS} ===")
#         metrics = train_one_fold(filepaths[train_idx], labels[train_idx],
#                                   filepaths[val_idx], labels[val_idx],
#                                   num_classes, fold_num, device)
#         fold_metrics.append(metrics)

#     accs = [m["accuracy"] for m in fold_metrics]
#     top3s = [m["top3_acc"] for m in fold_metrics]
#     print("\n=== Cross-Validation Summary ===")
#     print(f"Top-1 accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
#     print(f"Top-3 accuracy: {np.mean(top3s):.3f} +/- {np.std(top3s):.3f}")


# def train_final_model(filepaths, labels, class_names, device):
#     print("\n=== Training final deployment model on all data ===")
#     full_ds = TreeDataset(filepaths, labels, train_transform)
#     full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#     model = build_model(len(class_names), device)
#     model = train_phase(model, full_loader, None, HEAD_EPOCHS, HEAD_LR,
#                          device, "final-phase1", has_validation=False)

#     if DO_FINE_TUNE:
#         blocks = list(model.features.children())
#         for block in blocks[-UNFREEZE_LAST_N_BLOCKS:]:
#             for p in block.parameters():
#                 p.requires_grad = True
#         model = train_phase(model, full_loader, None, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
#                              device, "final-phase2", has_validation=False)
#     else:
#         print("Phase 2 skipped (DO_FINE_TUNE=False)")

#     torch.save(model.state_dict(), MODEL_OUT_PATH)
#     with open(CLASS_NAMES_OUT_PATH, "w", encoding="utf-8") as f:
#         f.write("\n".join(class_names))
#     print(f"\nSaved final model to {MODEL_OUT_PATH}")
#     print(f"Saved class name order to {CLASS_NAMES_OUT_PATH}")


# def main():
#     device = get_device()
#     filepaths, labels, class_names = collect_filepaths_and_labels()
#     num_classes = len(class_names)
#     print(f"Found {len(filepaths)} images across {num_classes} species: {class_names}")

#     counts = np.bincount(labels)
#     if counts.min() < N_FOLDS:
#         print(f"[WARNING] Smallest class has only {counts.min()} images — "
#               f"reduce N_FOLDS or top up that class before running.")
#         return

#     run_cross_validation(filepaths, labels, num_classes, device)
#     train_final_model(filepaths, labels, class_names, device)


# if __name__ == "__main__":
#     main()

"""
PyTorch training pipeline for the 10-species tree classifier, using your
NVIDIA RTX 3050's CUDA cores directly (no WSL2 needed — this is the main
reason for migrating off TensorFlow, which dropped native Windows GPU
support at 2.11+).

WHY FINE-TUNING IS BACK ON:
The TF version had to disable Phase 2 (unfreezing backbone layers) because
it consistently destabilized training with only ~20 images/class across 40
classes. With 10 classes and 250-300+ images/class, there's enough data to
safely fine-tune — so DO_FINE_TUNE defaults to True here. If you see the
same destabilization pattern (loss jumping up hard at the start of Phase 2
and not recovering), that's your signal to set it back to False.

Requires:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn pillow

Usage:
    python train_species_classifier_torch.py
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "tree_images"
IMG_SIZE = 224
BATCH_SIZE = 32          # comfortable for 4GB VRAM with MobileNetV3Small — lower to 16 if you hit CUDA OOM
N_FOLDS = 5
HEAD_EPOCHS = 40
FINE_TUNE_EPOCHS = 20
HEAD_LR = 8e-4
FINE_TUNE_LR = 3e-5
UNFREEZE_LAST_N_BLOCKS = 4   # unfreeze the last N inverted-residual blocks of the backbone
DO_FINE_TUNE = True          # see note above — flip to False if it destabilizes again
EARLY_STOP_PATIENCE = 12
MODEL_OUT_PATH = "tree_species_classifier_final.pt"
CLASS_NAMES_OUT_PATH = "class_names.txt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return torch.device("cuda")
    print("CUDA not available — falling back to CPU (check your PyTorch install if unexpected)")
    return torch.device("cpu")


def collect_filepaths_and_labels():
    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    filepaths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return np.array(filepaths), np.array(labels), class_names


# Lighter augmentation, matching what we proved works in the TF debugging —
# no saturation shift (color can be discriminative between species), mild
# jitter and crop only.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TreeDataset(Dataset):
    def __init__(self, filepaths, labels, transform):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def build_model(num_classes, device):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = mobilenet_v3_small(weights=weights)
    features = backbone.features  # conv backbone, output 576 channels
    for p in features.parameters():
        p.requires_grad = False

    class TreeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.norm = nn.LayerNorm(576)  # same fix as the TF version — normalizes
                                            # pooled features before the head, which is
                                            # what resolved the loss-explosion issue there
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(576, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            x = self.norm(x)
            x = self.dropout(x)
            return self.classifier(x)

    model = TreeClassifier().to(device)
    return model


def top_k_accuracy(outputs, targets, k=3):
    _, pred = outputs.topk(k, dim=1)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_acc, total_top3, n_batches = 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            acc = (outputs.argmax(1) == labels).float().mean().item()
            top3 = top_k_accuracy(outputs, labels, k=3)
            total_loss += loss.item()
            total_acc += acc
            total_top3 += top3
            n_batches += 1
    return total_loss / n_batches, total_acc / n_batches, total_top3 / n_batches


def train_phase(model, train_loader, val_loader, epochs, lr, device,
                 label, has_validation=True, class_weights=None):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)
    # class_weights guards against imbalance if the new "not a tree" class
    # ends up with a different image count than the 10 species — without
    # this, a much larger or smaller negative class could bias the model
    # toward over- or under-predicting it.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_top3 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)

        if has_validation:
            val_loss, val_acc, val_top3 = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False)
            monitor_loss = val_loss
            print(f"  [{label}] epoch {epoch}/{epochs} - "
                  f"loss {train_loss:.3f} acc {train_acc:.3f} top3 {train_top3:.3f} | "
                  f"val_loss {val_loss:.3f} val_acc {val_acc:.3f} val_top3 {val_top3:.3f}")
        else:
            monitor_loss = train_loss
            print(f"  [{label}] epoch {epoch}/{epochs} - "
                  f"loss {train_loss:.3f} acc {train_acc:.3f} top3 {train_top3:.3f}")

        scheduler.step(monitor_loss)

        if monitor_loss < best_loss - 0.005:
            best_loss = monitor_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  [{label}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def train_one_fold(train_fp, train_lb, val_fp, val_lb, num_classes, fold_num, device, class_weights=None):
    train_ds = TreeDataset(train_fp, train_lb, train_transform)
    val_ds = TreeDataset(val_fp, val_lb, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes, device)

    print(f"  [fold {fold_num}] Phase 1: training head...")
    model = train_phase(model, train_loader, val_loader, HEAD_EPOCHS, HEAD_LR,
                         device, f"fold{fold_num}-phase1", class_weights=class_weights)

    if DO_FINE_TUNE:
        # Unfreeze the last N inverted-residual blocks of the backbone
        blocks = list(model.features.children())
        for block in blocks[-UNFREEZE_LAST_N_BLOCKS:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"  [fold {fold_num}] Phase 2: fine-tuning last {UNFREEZE_LAST_N_BLOCKS} blocks...")
        model = train_phase(model, train_loader, val_loader, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
                             device, f"fold{fold_num}-phase2", class_weights=class_weights)
    else:
        print(f"  [fold {fold_num}] Phase 2 skipped (DO_FINE_TUNE=False)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    val_loss, val_acc, val_top3 = run_epoch(model, val_loader, criterion, None, device, train=False)
    print(f"  [fold {fold_num}] FINAL val_accuracy={val_acc:.3f}  val_top3_acc={val_top3:.3f}")
    return {"accuracy": val_acc, "top3_acc": val_top3}


def run_cross_validation(filepaths, labels, num_classes, device, class_weights=None):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), 1):
        print(f"\n=== Fold {fold_num}/{N_FOLDS} ===")
        metrics = train_one_fold(filepaths[train_idx], labels[train_idx],
                                  filepaths[val_idx], labels[val_idx],
                                  num_classes, fold_num, device, class_weights=class_weights)
        fold_metrics.append(metrics)

    accs = [m["accuracy"] for m in fold_metrics]
    top3s = [m["top3_acc"] for m in fold_metrics]
    print("\n=== Cross-Validation Summary ===")
    print(f"Top-1 accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"Top-3 accuracy: {np.mean(top3s):.3f} +/- {np.std(top3s):.3f}")


def train_final_model(filepaths, labels, class_names, device, class_weights=None):
    print("\n=== Training final deployment model on all data ===")
    full_ds = TreeDataset(filepaths, labels, train_transform)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = build_model(len(class_names), device)
    model = train_phase(model, full_loader, None, HEAD_EPOCHS, HEAD_LR,
                         device, "final-phase1", has_validation=False, class_weights=class_weights)

    if DO_FINE_TUNE:
        blocks = list(model.features.children())
        for block in blocks[-UNFREEZE_LAST_N_BLOCKS:]:
            for p in block.parameters():
                p.requires_grad = True
        model = train_phase(model, full_loader, None, FINE_TUNE_EPOCHS, FINE_TUNE_LR,
                             device, "final-phase2", has_validation=False, class_weights=class_weights)
    else:
        print("Phase 2 skipped (DO_FINE_TUNE=False)")

    torch.save(model.state_dict(), MODEL_OUT_PATH)
    with open(CLASS_NAMES_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print(f"\nSaved final model to {MODEL_OUT_PATH}")
    print(f"Saved class name order to {CLASS_NAMES_OUT_PATH}")


def compute_class_weights(labels, num_classes, device):
    """Inverse-frequency weighting — classes with fewer images get a
    higher weight so the loss doesn't just favor whichever class has the
    most images. Only matters once classes are meaningfully imbalanced
    (e.g. the negative 'not a tree' class differing in size from the
    10 species) — with roughly balanced classes this is close to a no-op."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    device = get_device()
    filepaths, labels, class_names = collect_filepaths_and_labels()
    num_classes = len(class_names)
    print(f"Found {len(filepaths)} images across {num_classes} species: {class_names}")

    counts = np.bincount(labels)
    for name, count in zip(class_names, counts):
        print(f"  {name}: {count} images")
    if counts.min() < N_FOLDS:
        print(f"[WARNING] Smallest class has only {counts.min()} images — "
              f"reduce N_FOLDS or top up that class before running.")
        return

    class_weights = compute_class_weights(labels, num_classes, device)
    run_cross_validation(filepaths, labels, num_classes, device, class_weights)
    train_final_model(filepaths, labels, class_names, device, class_weights)


if __name__ == "__main__":
    main()