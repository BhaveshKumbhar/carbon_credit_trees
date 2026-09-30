"""
Run the trained model on one or more images and see its top-3 predictions
with confidence scores.

For a real test of generalization, use photos the model has NEVER seen —
ideally your own phone photos of local trees, not more scraped images from
the same sources the training data came from (those would be an easier
test than what a real farmer's photo represents).

Usage:
    python predict_species.py path/to/photo.jpg
    python predict_species.py path/to/folder_of_test_photos/
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = "tree_species_classifier_final.pt"
CLASS_NAMES_PATH = "class_names.txt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TreeClassifier(nn.Module):
    """Must match the architecture in train_species_classifier_torch.py exactly."""
    def __init__(self, num_classes):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)  # weights loaded from our checkpoint, not ImageNet
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(576)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


def load_model(device):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    model = TreeClassifier(len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, class_names


def predict(model, class_names, image_path, device):
    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    top3_probs, top3_idx = probs.topk(3)
    return [(class_names[i], p.item()) for p, i in zip(top3_probs, top3_idx)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_species.py <image_path_or_folder>")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(device)
    print(f"Loaded model with {len(class_names)} species, running on {device}\n")

    target = sys.argv[1]
    if os.path.isdir(target):
        image_paths = [os.path.join(target, f) for f in sorted(os.listdir(target))
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        image_paths = [target]

    for path in image_paths:
        try:
            results = predict(model, class_names, path, device)
        except Exception as e:
            print(f"{os.path.basename(path)}: could not process — {e}")
            continue
        print(f"{os.path.basename(path)}:")
        for rank, (species, prob) in enumerate(results, 1):
            print(f"  {rank}. {species} — {prob*100:.1f}%")
        print()


if __name__ == "__main__":
    main()