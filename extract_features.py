"""
extract_features_v2.py
======================
Extracts high-dimensional frozen embeddings for all images in `tree_images/`
using selected vision backbones with mild augmentations and image grouping.

Supported Backbones:
1. `dinov2_vits14` (DINOv2 ViT-S/14: 384-dim self-supervised transformer, best for fine-grained botany)
2. `convnext_tiny` (ConvNeXt-Tiny: 768-dim modern convolutional architecture)
3. `mobilenet_v3_small` (MobileNetV3Small: 576-dim mobile CNN baseline)

Augmentations per image:
- Original image
- Aug 1: Horizontal Flip
- Aug 2: Brightness +8%, Crop jitter (232 -> 224)
- Aug 3: Brightness -8%, Contrast +8%, Crop jitter

Output:
- `cache_v2/features_<backbone>.npz`: Contains features, labels, group_ids, source_paths, is_aug, class_names.
"""

import os
import sys
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))

DATA_DIR = "tree_images"
CACHE_DIR = "cache_v2"
CLASS_NAMES_FILE = "class_names.txt"
IMG_SIZE = 224
BATCH_SIZE = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Evaluation (original) transform
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_mild_augmentations():
    """Returns 3 mild augmentations (flip, brightness/contrast +/-8%, 8px crop jitter)."""
    t1 = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    t2 = transforms.Compose([
        transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ColorJitter(brightness=(1.08, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    t3 = transforms.Compose([
        transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ColorJitter(brightness=(0.92, 0.92), contrast=(1.08, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return [t1, t2, t3]


class BackboneWrapper(nn.Module):
    """Unified wrapper around different vision backbones to return flat (B, D) feature embeddings."""
    def __init__(self, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.device = torch.device("cpu")

        if self.backbone_name == "dinov2_vits14":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.feature_dim = 384
        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            base = models.convnext_tiny(weights=weights)
            self.features = base.features
            self.pool = base.avgpool
            self.feature_dim = 768
            self.model = None
        elif self.backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = 576
            self.model = None
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            if self.backbone_name == "dinov2_vits14":
                # DINOv2 returns the CLS token representation (B, 384)
                return self.model(x)
            elif self.backbone_name == "convnext_tiny":
                feat = self.features(x)
                return self.pool(feat).flatten(1)
            elif self.backbone_name == "mobilenet_v3_small":
                feat = self.features(x)
                return self.pool(feat).flatten(1)


def load_class_names():
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])


def collect_images(class_names):
    items = []
    group_counter = 0
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        valid_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        for fname in valid_files:
            items.append({
                "path": os.path.join(class_dir, fname),
                "class_idx": class_idx,
                "class_name": class_name,
                "group_id": group_counter,
            })
            group_counter += 1
    return items


def extract_features_for_backbone(backbone_name="dinov2_vits14"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_file = os.path.join(CACHE_DIR, f"features_{backbone_name}.npz")

    if os.path.exists(out_file):
        print(f"[INFO] Cache file already exists at {out_file}. Skipping extraction.")
        return out_file

    class_names = load_class_names()
    print(f"[INFO] Extracting features with backbone: {backbone_name}")
    print(f"[INFO] Total classes: {len(class_names)}")

    images = collect_images(class_names)
    print(f"[INFO] Found {len(images)} original images.")

    model = BackboneWrapper(backbone_name=backbone_name)
    model.eval()

    augs = get_mild_augmentations()
    total_vectors = len(images) * (1 + len(augs))
    print(f"[INFO] Total vectors to compute: {total_vectors} ({len(images)} original + {len(images)*len(augs)} augmented)")

    all_features = []
    all_labels = []
    all_group_ids = []
    all_source_paths = []
    all_is_aug = []

    batch_tensors = []
    batch_metadata = []

    def flush():
        nonlocal batch_tensors, batch_metadata
        if not batch_tensors:
            return
        t_batch = torch.stack(batch_tensors)
        feats = model(t_batch).cpu().numpy()
        all_features.append(feats)
        for meta in batch_metadata:
            all_labels.append(meta["class_idx"])
            all_group_ids.append(meta["group_id"])
            all_source_paths.append(meta["path"])
            all_is_aug.append(meta["is_aug"])
        batch_tensors.clear()
        batch_metadata.clear()

    pbar = tqdm(images, desc=f"Extracting ({backbone_name})", unit="img")
    for item in pbar:
        try:
            pil_img = Image.open(item["path"]).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Skipping corrupt file {item['path']}: {e}")
            continue

        # Original
        batch_tensors.append(eval_transform(pil_img))
        batch_metadata.append({"class_idx": item["class_idx"], "group_id": item["group_id"], "path": item["path"], "is_aug": 0})

        # Augmentations
        for aug_t in augs:
            batch_tensors.append(aug_t(pil_img))
            batch_metadata.append({"class_idx": item["class_idx"], "group_id": item["group_id"], "path": item["path"], "is_aug": 1})

        if len(batch_tensors) >= BATCH_SIZE:
            flush()

    flush()

    feats_arr = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    group_ids_arr = np.array(all_group_ids, dtype=np.int64)
    source_paths_arr = np.array(all_source_paths, dtype=object)
    is_aug_arr = np.array(all_is_aug, dtype=np.int32)
    class_names_arr = np.array(class_names, dtype=object)

    print(f"\n[INFO] Saving cache to {out_file}...")
    np.savez_compressed(
        out_file,
        features=feats_arr,
        labels=labels_arr,
        group_ids=group_ids_arr,
        source_paths=source_paths_arr,
        is_aug=is_aug_arr,
        class_names=class_names_arr,
        backbone=backbone_name,
    )
    print(f"[SUCCESS] Features saved! Shape: {feats_arr.shape} ({feats_arr.nbytes / (1024*1024):.2f} MB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Extract features v2 with configurable backbones.")
    parser.add_argument("--backbone", default="dinov2_vits14",
                        choices=["dinov2_vits14", "convnext_tiny", "mobilenet_v3_small"],
                        help="Backbone architecture to use.")
    args = parser.parse_args()
    extract_features_for_backbone(args.backbone)


if __name__ == "__main__":
    main()
