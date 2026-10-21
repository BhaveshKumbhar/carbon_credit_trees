"""
inference_service.py
====================
Thin inference wrapper around the existing tree species classifier and OOD rejection model.
Provides a unified, cached interface for both the Streamlit web UI and batch prediction pipelines.

Features:
- Loads model weights, OOD configuration, class names, and agroforestry lookup table ONCE.
- Fast CPU-optimized execution with image downsampling for high-res inputs (max 1024px).
- Returns clean, normalized dictionary output adhering to the 3-tier status hierarchy:
    1. 'rejected': OOD non-tree / unsupported image
    2. 'low_confidence': Tree identified, but species differentiation is uncertain
    3. 'confident': High-confidence identification with full agroforestry/carbon profile
"""

import os
import sys
import json
import csv
import numpy as np
from PIL import Image

# Ensure the root carbon_credit_trees directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch

try:
    from predict import TreePredictorV2, ARTIFACTS_DIR, BEST_MODEL_PATH, OOD_CONFIG_PATH, LOOKUP_CSV_PATH, CLASS_NAMES_PATH
except ImportError:
    # Fallback paths if imported in different context
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts_v2")
    BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pt")
    OOD_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "ood_config.json")
    LOOKUP_CSV_PATH = os.path.join(PROJECT_ROOT, "species_lookup_10species.csv")
    CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.txt")
    from predict import TreePredictorV2


class TreeInferenceService:
    """
    Singleton-style inference service wrapper for CPU deployment.
    """
    _instance = None

    def __init__(self, use_tta=True):
        # Resolve absolute paths to ensure reliable loading from any working directory
        self.artifacts_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "artifacts_v2"))
        self.best_model_path = os.path.join(self.artifacts_dir, "best_model.pt")
        self.ood_config_path = os.path.join(self.artifacts_dir, "ood_config.json")
        self.lookup_csv_path = os.path.abspath(os.path.join(PROJECT_ROOT, "species_lookup_10species.csv"))
        self.class_names_path = os.path.abspath(os.path.join(PROJECT_ROOT, "class_names.txt"))

        # Check required files
        self._validate_artifacts()

        # Initialize the underlying predictor
        self.predictor = TreePredictorV2(use_tta=use_tta)
        self.model_version = f"DINOv2 ViT-S/14 + Multi-signal OOD (Backbone: {self.predictor.backbone_name})"

        # Load lookup table with absolute path
        self.lookup = self._load_lookup_table()
        self.supported_species_list = self._get_supported_species_list()

    def _validate_artifacts(self):
        """Validates that all necessary artifact files exist before starting."""
        missing = []
        if not os.path.exists(self.best_model_path):
            missing.append(f"Model weights ({self.best_model_path})")
        if not os.path.exists(self.ood_config_path):
            missing.append(f"OOD detector config ({self.ood_config_path})")
        if not os.path.exists(self.lookup_csv_path):
            missing.append(f"Agroforestry lookup table ({self.lookup_csv_path})")
        
        if missing:
            raise FileNotFoundError(
                "Missing required model artifact(s):\n" + "\n".join(f"- {m}" for m in missing)
            )

    def _load_lookup_table(self):
        """Loads the species agroforestry lookup CSV into a dictionary keyed by species_id."""
        lookup = {}
        if os.path.exists(self.lookup_csv_path):
            with open(self.lookup_csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lookup[row["species_id"]] = row
        return lookup

    def _get_supported_species_list(self):
        """Returns a formatted list of all supported species from class_names or lookup."""
        species = []
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sp_id = line.split("_")[0]
                        info = self.lookup.get(sp_id, {})
                        species.append({
                            "class_id": line,
                            "species_id": sp_id,
                            "common_name": info.get("common_name", line.split("_")[-1]),
                            "scientific_name": info.get("scientific_name", "N/A"),
                            "hindi_name": info.get("hindi_name", ""),
                            "category": info.get("category", "agroforestry")
                        })
        return species

    def predict_image(self, image_input):
        """
        Runs full OOD rejection and species classification on a PIL Image, path, or file-like object.
        
        Args:
            image_input: PIL.Image.Image instance, file path (str), or bytes/BytesIO
            
        Returns:
            dict: Normalized results adhering to the required schema:
                - 'status': 'rejected' | 'low_confidence' | 'confident' | 'error'
                - 'top3': list of candidate dictionaries
                - 'ood_score': float (minimum Mahalanobis distance)
                - 'ood_threshold': float (Mahalanobis cutoff boundary)
                - 'confidence': float (Top-1 probability [0.0 - 1.0])
                - 'entropy': float (Shannon entropy)
                - 'rejection_reason': str
                - 'species_name': str
                - 'hindi_name': str
                - 'scientific_name': str
                - 'tier2': dict of agroforestry and carbon credit information
                - 'model_version': str
        """
        # 1. Load and validate PIL Image
        try:
            if isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            elif isinstance(image_input, (str, bytes, os.PathLike)) or hasattr(image_input, "read"):
                pil_img = Image.open(image_input).convert("RGB")
            else:
                return {
                    "status": "error",
                    "error_message": f"Unsupported image input type: {type(image_input)}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Could not decode image file: {str(e)}"
            }

        # Validate minimum image dimensions
        width, height = pil_img.size
        if width < 32 or height < 32:
            return {
                "status": "error",
                "error_message": f"Image is too small ({width}x{height}px). Minimum required resolution is 32x32px."
            }

        # 2. Downsample if very large (e.g. longest side > 1024px) for fast CPU processing
        max_dim = 1024
        if max(width, height) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        # 3. Extract embedding with Test-Time Augmentation (TTA)
        try:
            raw_emb, norm_emb = self.predictor.extract_embedding_tta(pil_img)
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Embedding extraction failed: {str(e)}"
            }

        # 4. Compute Classifier Probabilities
        if self.predictor.model_type == "neural_head":
            with torch.no_grad():
                logits = self.predictor.head(torch.tensor(raw_emb, dtype=torch.float32).to(self.predictor.device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        elif self.predictor.model_type == "sklearn":
            probs = self.predictor.saved_model["model"].predict_proba(raw_emb)[0]
        else:
            # Cosine Prototype
            sims = norm_emb @ self.predictor.saved_model["centroids"].T
            scaled = sims / self.predictor.saved_model["temperature"]
            exp_s = np.exp(scaled - np.max(scaled))
            probs = (exp_s / np.sum(exp_s))[0]

        # 5. Multi-Signal OOD Metrics
        diff = raw_emb - self.predictor.centroids
        dist_sq = np.sum((diff @ self.predictor.inv_cov) * diff, axis=1)
        maha_dists = np.sqrt(np.maximum(dist_sq, 0.0))
        min_maha = float(np.min(maha_dists))

        max_prob = float(np.max(probs))
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))

        # 6. Evaluate 3-Tier Status
        is_tree = bool(
            (min_maha <= self.predictor.maha_thresh) and
            (entropy <= self.predictor.max_entropy) and
            (max_prob >= self.predictor.conf_low_floor)
        )

        if not is_tree:
            status = "rejected"
            reason = (
                f"Feature distance to supported tree species ({min_maha:.2f} > {self.predictor.maha_thresh:.2f}) "
                "indicates this is a non-tree object or an unsupported species."
            )
        elif max_prob < self.predictor.conf_high_cutoff:
            status = "low_confidence"
            reason = (
                f"Recognized as a tree photo, but classification confidence ({max_prob*100:.1f}%) "
                f"is below the confident identification threshold ({self.predictor.conf_high_cutoff*100:.1f}%)."
            )
        else:
            status = "confident"
            reason = "High confidence species identification within valid distribution boundary."

        # 7. Construct Top-3 Candidate List
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_list = []
        for idx in top3_indices:
            c_name = self.predictor.class_names[idx]
            sp_id = c_name.split("_")[0]
            info = self.lookup.get(sp_id, {})
            top3_list.append({
                "class_name": c_name,
                "species_id": sp_id,
                "species_name": info.get("common_name", c_name.split("_")[-1]),
                "scientific_name": info.get("scientific_name", "N/A"),
                "hindi_name": info.get("hindi_name", ""),
                "probability": float(probs[idx]),
            })

        top1 = top3_list[0]
        top1_info = self.lookup.get(top1["species_id"], {})

        tier2_info = {
            "program_relevance": top1_info.get("agroforestry_program_relevance", "N/A"),
            "indicative_value": top1_info.get("indicative_value_range_inr_per_tonne", "N/A"),
            "maturity_years": top1_info.get("maturity_years_for_carbon_relevance", "N/A"),
            "gov_program": top1_info.get("gov_resource_name", "Green Credit Programme (MoEFCC)"),
            "gov_url": top1_info.get("gov_resource_url", "https://www.moefcc-gcp.in/"),
            "native_status": top1_info.get("native_status", "N/A"),
            "notes": top1_info.get("notes", "")
        }

        return {
            "status": status,
            "species_name": top1["species_name"],
            "hindi_name": top1["hindi_name"],
            "scientific_name": top1["scientific_name"],
            "confidence": top1["probability"],
            "top3": top3_list,
            "ood_score": min_maha,
            "ood_threshold": self.predictor.maha_thresh,
            "entropy": entropy,
            "rejection_reason": reason,
            "tier2": tier2_info,
            "model_version": self.model_version
        }


# Cached singleton helper
_service_instance = None

def get_inference_service():
    """Returns the cached TreeInferenceService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = TreeInferenceService()
    return _service_instance
