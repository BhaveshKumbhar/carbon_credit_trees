"""
Run this ONCE after training the model (and again if you ever retrain it).
It computes, for each of the 10 species, the "center" of that species'
images in the model's internal feature space, and how far real training
images of that species typically sit from that center. This calibrates
the out-of-distribution (OOD) rejection used by identify_tree.py.

Why this approach: softmax confidence alone can't reliably reject
unrelated images (a dog photo) because softmax is forced to normalize
across only the known classes — it has no way to express "none of these."
Distance in feature space DOES capture this: an unrelated image's features
won't resemble any tree species' training images, regardless of what the
final softmax layer guesses.

Usage:
    python compute_ood_reference.py

Output:
    ood_reference.pt — contains per-class centroids and distance thresholds,
    loaded automatically by identify_tree.py at inference time.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

DATA_DIR = "tree_images"
MODEL_PATH = "tree_species_classifier_final.pt"
OUT_PATH = "ood_reference.pt"
IMG_SIZE = 224
BATCH_SIZE = 32
# Percentile of within-class training distances used as the rejection
# threshold — 99th percentile means ~1% of real training images would sit
# beyond their own class's threshold (a deliberate, small false-rejection
# rate in exchange for reliably catching genuinely unrelated images).
THRESHOLD_PERCENTILE = 99

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TreeClassifier(nn.Module):
    """Must match train_species_classifier_torch.py / identify_tree.py exactly."""
    def __init__(self, num_classes):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(576)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(576, num_classes)

    def get_embedding(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        return x


class TreeDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return eval_transform(img), self.labels[idx]


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
    return filepaths, labels, class_names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    filepaths, labels, class_names = collect_filepaths_and_labels()
    print(f"Found {len(filepaths)} images across {len(class_names)} species")

    model = TreeClassifier(len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dataset = TreeDataset(filepaths, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            emb = model.get_embedding(imgs)
            all_embeddings.append(emb.cpu())
            all_labels.append(lbls)

    all_embeddings = torch.cat(all_embeddings)   # (N, 576)
    all_labels = torch.cat(all_labels)           # (N,)

    num_classes = len(class_names)
    centroids = torch.zeros(num_classes, 576)
    thresholds = torch.zeros(num_classes)

    for c in range(num_classes):
        class_embeddings = all_embeddings[all_labels == c]
        centroid = class_embeddings.mean(dim=0)
        distances = torch.norm(class_embeddings - centroid, dim=1)
        threshold = torch.quantile(distances, THRESHOLD_PERCENTILE / 100.0)
        centroids[c] = centroid
        thresholds[c] = threshold
        print(f"  {class_names[c]}: {len(class_embeddings)} images, "
              f"threshold={threshold:.2f} (mean dist={distances.mean():.2f})")

    torch.save({
        "centroids": centroids,
        "thresholds": thresholds,
        "class_names": class_names,
    }, OUT_PATH)
    print(f"\nSaved OOD reference data to {OUT_PATH}")


if __name__ == "__main__":
    main()