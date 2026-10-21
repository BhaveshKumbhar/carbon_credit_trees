"""
predict_v2.py
=============
Next-generation CLI inference tool with Test-Time Augmentation (TTA), multi-signal OOD rejection,
and honest 3-tier status reporting.

3-Tier Output Outcomes:
1. [REJECTED]:
   - Image fails OOD checks (Mahalanobis distance too high or extreme entropy).
   - Identifies non-tree images (cars, animals, buildings, everyday objects) or unsupported species.
2. [SPECIES UNCERTAIN / LOW CONFIDENCE]:
   - Recognized as a valid tree photo, but the model cannot distinguish between candidate species
     with sufficient certainty (confidence < cutoff).
   - Displays Top-3 candidates with probabilities, asks for a clearer photo, and intentionally
     SUPPRESSES economic carbon figures to avoid misleading recommendations.
3. [CONFIDENT IDENTIFICATION]:
   - High-confidence species classification (confidence >= cutoff).
   - Displays Top-3 candidates, full agroforestry / carbon credit profile from lookup CSV,
     and researcher-estimated disclaimer.

Usage:
    python predict_v2.py path/to/image.jpg
    python predict_v2.py --folder testing_trees/ --output v2_predictions.csv
"""

import os
import sys
import csv
import json
import pickle
import argparse
import numpy as np
from PIL import Image

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn
from torchvision import transforms

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts_v2")
BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pt")
OOD_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "ood_config.json")
LOOKUP_CSV_PATH = os.path.join(BASE_DIR, "species_lookup_10species.csv")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transforms for Test-Time Augmentation (TTA)
tta_transforms = [
    # 1. Standard resize
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    # 3. Center crop (slight zoom)
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]


class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


class TreePredictorV2:
    def __init__(self, use_tta=True):
        self.device = torch.device("cpu")
        self.use_tta = use_tta
        self._load_ood_config()
        self._load_backbone()
        self._load_classifier()
        self._load_lookup_table()

    def _load_ood_config(self):
        if not os.path.exists(OOD_CONFIG_PATH):
            raise FileNotFoundError(f"OOD config not found at {OOD_CONFIG_PATH}. Run build_ood_detector_v2.py first.")
        with open(OOD_CONFIG_PATH, "r", encoding="utf-8") as f:
            self.ood_cfg = json.load(f)

        self.backbone_name = self.ood_cfg.get("backbone", "dinov2_vits14")
        self.class_names = self.ood_cfg["class_names"]
        self.centroids = np.array(self.ood_cfg["centroids"], dtype=np.float32)
        self.inv_cov = np.array(self.ood_cfg["inv_cov"], dtype=np.float32)
        self.maha_thresh = self.ood_cfg["thresholds"]["mahalanobis_global"]
        self.conf_high_cutoff = self.ood_cfg["thresholds"]["high_confidence_cutoff"]
        self.conf_low_floor = self.ood_cfg["thresholds"]["low_confidence_floor"]
        self.max_entropy = self.ood_cfg["thresholds"]["max_entropy"]

    def _load_backbone(self):
        if self.backbone_name == "dinov2_vits14":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        elif self.backbone_name == "convnext_tiny":
            import torchvision.models as models
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            base = models.convnext_tiny(weights=weights)
            self.backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten(1)).to(self.device)
        else:
            import torchvision.models as models
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            base = models.mobilenet_v3_small(weights=weights)
            self.backbone = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1)).to(self.device)

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _load_classifier(self):
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}. Run train_head_v2.py first.")

        # Check if saved via pickle (sklearn) or torch
        try:
            with open(BEST_MODEL_PATH, "rb") as f:
                self.saved_model = pickle.load(f)
            self.model_type = "sklearn"
        except Exception:
            self.saved_model = torch.load(BEST_MODEL_PATH, map_location=self.device, weights_only=False)
            self.model_type = self.saved_model.get("type", "neural_head")

        if self.model_type == "neural_head":
            self.head = ClassifierHead(in_features=self.saved_model["in_features"],
                                       num_classes=self.saved_model["num_classes"]).to(self.device)
            self.head.load_state_dict(self.saved_model["state_dict"])
            self.head.eval()

    def _load_lookup_table(self):
        self.lookup = {}
        if os.path.exists(LOOKUP_CSV_PATH):
            with open(LOOKUP_CSV_PATH, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.lookup[row["species_id"]] = row

    def extract_embedding_tta(self, pil_img):
        """Extracts TTA-averaged embedding across multiple mild views."""
        transforms_to_use = tta_transforms if self.use_tta else [tta_transforms[0]]
        tensors = [t(pil_img) for t in transforms_to_use]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            embs = self.backbone(batch)  # (N_views, D)
            avg_emb = embs.mean(dim=0, keepdim=True)  # (1, D)
            # L2 normalize
            norm_emb = avg_emb / (torch.norm(avg_emb, dim=1, keepdim=True) + 1e-8)

        return avg_emb.cpu().numpy(), norm_emb.cpu().numpy()

    def predict_image(self, image_path):
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"File not found: {image_path}"}

        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"status": "error", "message": f"Could not load image: {e}"}

        raw_emb, norm_emb = self.extract_embedding_tta(pil_img)

        # 1. Compute Classifier Probabilities
        if self.model_type == "neural_head":
            with torch.no_grad():
                logits = self.head(torch.tensor(raw_emb, dtype=torch.float32).to(self.device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        elif self.model_type == "sklearn":
            probs = self.saved_model["model"].predict_proba(raw_emb)[0]
        else:
            # Cosine prototype
            sims = norm_emb @ self.saved_model["centroids"].T
            scaled = sims / self.saved_model["temperature"]
            exp_s = np.exp(scaled - np.max(scaled))
            probs = (exp_s / np.sum(exp_s))[0]

        # 2. OOD Evaluation
        diff = raw_emb - self.centroids
        dist_sq = np.sum((diff @ self.inv_cov) * diff, axis=1)
        maha_dists = np.sqrt(np.maximum(dist_sq, 0.0))
        min_maha = float(np.min(maha_dists))
        nearest_idx = int(np.argmin(maha_dists))

        max_prob = float(np.max(probs))
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))

        # Determine 3-Tier Outcome
        is_tree = bool((min_maha <= self.maha_thresh) and (entropy <= self.max_entropy) and (max_prob >= self.conf_low_floor))

        if not is_tree:
            status = "REJECTED"
            reason = f"Image feature distance ({min_maha:.2f} > {self.maha_thresh:.2f}) indicates non-tree object or unsupported species."
        elif max_prob < self.conf_high_cutoff:
            status = "SPECIES_UNCERTAIN"
            reason = f"Recognized as a tree, but species classification confidence ({max_prob*100:.1f}%) is below confident threshold ({self.conf_high_cutoff*100:.1f}%)."
        else:
            status = "CONFIDENT"
            reason = "High confidence identification."

        # Top 3 Candidates
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_list = []
        for idx in top3_indices:
            c_name = self.class_names[idx]
            sp_id = c_name.split("_")[0]
            info = self.lookup.get(sp_id, {})
            top3_list.append({
                "class_name": c_name,
                "common_name": info.get("common_name", c_name),
                "scientific_name": info.get("scientific_name", "N/A"),
                "hindi_name": info.get("hindi_name", ""),
                "probability": float(probs[idx]),
                "species_id": sp_id,
            })

        top1 = top3_list[0]
        top1_info = self.lookup.get(top1["species_id"], {})

        return {
            "status": status,
            "image_path": image_path,
            "min_mahalanobis": min_maha,
            "threshold_mahalanobis": self.maha_thresh,
            "max_probability": max_prob,
            "entropy": entropy,
            "rejection_reason": reason,
            "top_predictions": top3_list,
            "top_species": {
                "name": top1["common_name"],
                "scientific_name": top1["scientific_name"],
                "hindi_name": top1["hindi_name"],
                "confidence": top1["probability"],
                "program_relevance": top1_info.get("agroforestry_program_relevance", "N/A"),
                "indicative_value_range": top1_info.get("indicative_value_range_inr_per_tonne", "N/A"),
                "maturity_years": top1_info.get("maturity_years_for_carbon_relevance", "N/A"),
                "gov_program": top1_info.get("gov_resource_name", "Green Credit Programme (MoEFCC)"),
                "gov_url": top1_info.get("gov_resource_url", "https://www.moefcc-gcp.in/"),
            }
        }


def print_single_result_v2(res):
    print("\n" + "="*65)
    print(f"IMAGE: {res.get('image_path')}")
    print("="*65)

    if res["status"] == "error":
        print(f"[ERROR] {res['message']}")
        return

    status = res["status"]

    if status == "REJECTED":
        print("\n[REJECTED] NOT RECOGNIZED AS ONE OF THE SUPPORTED TREE SPECIES")
        print(f"   Reason               : {res['rejection_reason']}")
        print(f"   Mahalanobis Distance : {res['min_mahalanobis']:.2f} (Threshold: {res['threshold_mahalanobis']:.2f})")
        print(f"   Max Confidence       : {res['max_probability']*100:.1f}%")
        print("\nNote: Only images containing supported tree species can be processed.")
        return

    if status == "SPECIES_UNCERTAIN":
        print("\n[SPECIES UNCERTAIN / LOW CONFIDENCE]")
        print("   This appears to be a tree, but the species is uncertain.")
        print("   Please retake the photo in good lighting, showing leaves and bark up close.\n")
        print(f"   Leading candidate   : {res['top_species']['name']} ({res['max_probability']*100:.1f}% confidence)")
        print(f"   Runner-up candidate : {res['top_predictions'][1]['common_name']} ({res['top_predictions'][1]['probability']*100:.1f}% confidence)")
        print("\n[TOP 3 CANDIDATES]:")
        for rank, cand in enumerate(res["top_predictions"], 1):
            print(f"   {rank}. {cand['common_name']} ({cand['class_name']}): {cand['probability']*100:.1f}%")
        print("\n[NOTICE] Carbon credit and economic values are hidden because the species is uncertain.")
        return

    # Confident Identification
    top = res["top_species"]
    print("\n[CONFIDENT IDENTIFICATION] SUPPORTED TREE SPECIES FOUND")
    print(f"   Primary Prediction : {top['name']} ({top['scientific_name']})")
    if top['hindi_name']:
        print(f"   Hindi / Local Name : {top['hindi_name']}")
    print(f"   Model Confidence   : {top['confidence']*100:.1f}%")
    print(f"   Feature Distance   : {res['min_mahalanobis']:.2f} (Within boundary)")

    print("\n[INFO] CARBON CREDIT / AGROFORESTRY PROFILE:")
    print(f"   Program Relevance  : {top['program_relevance']}")
    print(f"   Indicative Value   : INR {top['indicative_value_range']} / tonne CO2")
    print(f"   Maturity Horizon   : {top['maturity_years']} years")
    print(f"   Govt Program Ref   : {top['gov_program']} ({top['gov_url']})")
    print("\n   [DISCLAIMER] Indicative carbon values and growth timelines are researcher-estimated")
    print("   estimates based on published agroforestry literature; actual market prices and")
    print("   carbon credit issuance require formal FPO verification and registry approval.")

    print("\n[TOP 3 CANDIDATES]:")
    for rank, cand in enumerate(res["top_predictions"], 1):
        print(f"   {rank}. {cand['common_name']} ({cand['class_name']}): {cand['probability']*100:.1f}%")


def process_folder_v2(predictor, folder_path, output_csv="v2_predictions.csv"):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    valid_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    print(f"[INFO] Found {len(valid_files)} images in '{folder_path}'. Running inference...")

    results = []
    for fname in valid_files:
        fpath = os.path.join(folder_path, fname)
        res = predictor.predict_image(fpath)
        results.append(res)

        if res.get("status") == "error":
            print(f"[ERROR]    {fname:30s} -> {res.get('message', 'Failed to process')}")
            continue

        status_tag = f"[{res['status'][:7]}]"
        top_name = res["top_species"]["name"] if res["status"] != "REJECTED" else "REJECTED (OOD)"
        conf = f"{res['max_probability']*100:.1f}%" if res["status"] != "REJECTED" else "-"
        print(f"{status_tag:10s} {fname:30s} -> {top_name:20s} (Conf: {conf:6s}, Dist: {res['min_mahalanobis']:.1f})")

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "status", "predicted_species", "confidence",
                "top2_species", "top2_conf", "top3_species", "top3_conf",
                "mahalanobis_dist", "reason"
            ])
            for res in results:
                if res.get("status") == "error":
                    continue
                fname = os.path.basename(res["image_path"])
                status = res["status"]
                preds = res.get("top_predictions", [])
                writer.writerow([
                    fname, status,
                    preds[0]["common_name"] if status != "REJECTED" and preds else "N/A",
                    f"{preds[0]['probability']:.4f}" if status != "REJECTED" and preds else "0.0",
                    preds[1]["common_name"] if status != "REJECTED" and len(preds) > 1 else "N/A",
                    f"{preds[1]['probability']:.4f}" if status != "REJECTED" and len(preds) > 1 else "0.0",
                    preds[2]["common_name"] if status != "REJECTED" and len(preds) > 2 else "N/A",
                    f"{preds[2]['probability']:.4f}" if status != "REJECTED" and len(preds) > 2 else "0.0",
                    f"{res.get('min_mahalanobis', 0.0):.2f}",
                    res.get("rejection_reason", "")
                ])
        print(f"\n[SUCCESS] Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Tree Species Classifier Inference v2.")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to a single image file.")
    parser.add_argument("--folder", default=None, help="Folder path for batch inference.")
    parser.add_argument("--output", default="v2_predictions.csv", help="CSV path for batch results.")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation.")
    args = parser.parse_args()

    predictor = TreePredictorV2(use_tta=not args.no_tta)

    if args.folder:
        process_folder_v2(predictor, args.folder, args.output)
    elif args.image_path:
        res = predictor.predict_image(args.image_path)
        print_single_result_v2(res)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
