# """
# The core product logic: photo -> species prediction -> carbon-program
# lookup -> farmer-facing result. This is what the UI layer (web app,
# WhatsApp bot, whatever we build next) should call — it should never need
# to touch the raw model directly.

# CONFIDENCE HANDLING (important — ties back to the original project framing):
# We never want to confidently tell a farmer "this is eligible" when we're
# not sure, or confidently name the wrong species. So:
# - High confidence (top prediction >= HIGH_CONFIDENCE_THRESHOLD): give a
#   direct answer.
# - Medium confidence: give the top prediction but flag it as uncertain,
#   and show the runner-up.
# - Low confidence (below LOW_CONFIDENCE_THRESHOLD): don't name a species
#   at all — ask for a clearer/different photo instead. A wrong guess here
#   is worse than no guess.

# Usage as a script:
#     python identify_tree.py path/to/photo.jpg

# Usage as a module (for the UI layer):
#     from identify_tree import identify
#     result = identify("path/to/photo.jpg")
# """

# import os
# import csv
# import sys
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import mobilenet_v3_small
# from PIL import Image

# IMG_SIZE = 224
# MODEL_PATH = "tree_species_classifier_final.pt"
# CLASS_NAMES_PATH = "class_names.txt"
# LOOKUP_CSV_PATH = "species_lookup_10species.csv"
# INDIAN_CARBON_MARKET_URL = "https://www.beeindia.gov.in/carbon-market.php"  # broader ICM/CCTS context, BEE (Ministry of Power)
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# HIGH_CONFIDENCE_THRESHOLD = 0.55
# LOW_CONFIDENCE_THRESHOLD = 0.25

# eval_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])


# class TreeClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         backbone = mobilenet_v3_small(weights=None)
#         self.features = backbone.features
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.norm = nn.LayerNorm(576)
#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Linear(576, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x).flatten(1)
#         x = self.norm(x)
#         x = self.dropout(x)
#         return self.classifier(x)


# _model = None
# _class_names = None
# _lookup = None
# _device = None


# def _load_lookup_table():
#     lookup = {}
#     with open(LOOKUP_CSV_PATH, "r", newline="", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             lookup[row["species_id"]] = row
#     return lookup


# def _class_label_to_display(class_label):
#     """Class labels come from folder names like 'SP001_Eucalyptus' (species_id
#     + common_name, joined by the download script). This was the actual bug:
#     the lookup CSV is keyed by plain common_name ('Eucalyptus'), so looking
#     it up directly by the raw class label ('SP001_Eucalyptus') always missed.
#     species_id is the reliable join key — it's the first token before '_'."""
#     species_id = class_label.split("_")[0]
#     info = _lookup.get(species_id, {})
#     display_name = info.get("common_name", class_label)
#     return display_name, species_id, info


# def _ensure_loaded():
#     """Lazy-loads the model/lookup once, so importing this module is cheap
#     and repeated calls to identify() don't reload from disk each time."""
#     global _model, _class_names, _lookup, _device
#     if _model is not None:
#         return
#     _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         _class_names = [line.strip() for line in f if line.strip()]
#     _model = TreeClassifier(len(_class_names)).to(_device)
#     _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
#     _model.eval()
#     _lookup = _load_lookup_table()


# def identify(image_path):
#     """Returns a dict describing the result — always safe to display
#     directly to a farmer-facing UI without further logic needed there."""
#     _ensure_loaded()

#     img = Image.open(image_path).convert("RGB")
#     tensor = eval_transform(img).unsqueeze(0).to(_device)
#     with torch.no_grad():
#         outputs = _model(tensor)
#         probs = torch.softmax(outputs, dim=1)[0]
#     top3_probs, top3_idx = probs.topk(3)
#     raw_predictions = [(_class_names[i], p.item()) for p, i in zip(top3_probs, top3_idx)]
#     # Convert raw class labels ('SP001_Eucalyptus') to clean display names
#     # ('Eucalyptus') via the lookup table for everything shown to the user.
#     predictions = [(_class_label_to_display(label)[0], prob) for label, prob in raw_predictions]

#     top_class_label, top_conf = raw_predictions[0]
#     top_species, species_id, species_info = _class_label_to_display(top_class_label)

#     if top_conf < LOW_CONFIDENCE_THRESHOLD:
#         return {
#             "status": "uncertain",
#             "message": "We couldn't confidently identify this tree. Please try a "
#                         "clearer photo — ideally showing the leaves or bark up close, "
#                         "in good light.",
#             "predictions": predictions,
#         }

#     result = {
#         "status": "confident" if top_conf >= HIGH_CONFIDENCE_THRESHOLD else "medium_confidence",
#         "predicted_species": top_species,
#         "confidence": top_conf,
#         "predictions": predictions,
#         "hindi_name": species_info.get("hindi_name", ""),
#         "program_relevance": species_info.get("agroforestry_program_relevance", "unknown"),
#         "indicative_value_range_inr_per_tonne": species_info.get("indicative_value_range_inr_per_tonne", ""),
#         "maturity_years": species_info.get("maturity_years_for_carbon_relevance", ""),
#         "notes": species_info.get("notes", ""),
#         "gov_resource_name": species_info.get("gov_resource_name", ""),
#         "gov_resource_url": species_info.get("gov_resource_url", ""),
#         "indian_carbon_market_url": INDIAN_CARBON_MARKET_URL,
#     }

#     if result["status"] == "medium_confidence":
#         result["message"] = (
#             f"This looks like {top_species}, but we're not fully certain — "
#             f"it could also be {predictions[1][0]}."
#         )
#     else:
#         result["message"] = f"This looks like {top_species} ({top_conf*100:.0f}% confident)."

#     # Standing disclaimer, always shown regardless of confidence — this is
#     # the core ethical framing from the project design: we identify species,
#     # we never claim to determine actual carbon-credit eligibility. All the
#     # earning/timeline/registration facts live in the structured fields
#     # above (pulled from species_lookup_10species.csv), not hardcoded text —
#     # this line is just the one thing that's always true regardless of species.
#     result["disclaimer"] = (
#         "This tool identifies the tree species only. To find out if YOUR specific "
#         "tree qualifies for carbon credits, you'll need to register through a local "
#         "FPO or carbon-credit aggregator — this app can't determine eligibility on its own."
#     )

#     return result


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python identify_tree.py <image_path>")
#         return
#     result = identify(sys.argv[1])
#     print(f"\nStatus: {result['status']}")
#     print(f"Message: {result['message']}\n")
#     if result["status"] != "uncertain":
#         print(f"Program relevance: {result['program_relevance']}")
#         print(f"Indicative value: ₹{result['indicative_value_range_inr_per_tonne']}/tonne CO2")
#         print(f"Maturity: {result['maturity_years']} years")
#         print(f"Reference: {result['gov_resource_name']} — {result['gov_resource_url']}")
#         print(f"\n{result['disclaimer']}\n")
#     print("Top 3 candidates:")
#     for species, prob in result["predictions"]:
#         print(f"  {species}: {prob*100:.1f}%")


# if __name__ == "__main__":
#     main()

"""
The core product logic: photo -> species prediction -> carbon-program
lookup -> farmer-facing result. This is what the UI layer (web app,
WhatsApp bot, whatever we build next) should call — it should never need
to touch the raw model directly.

CONFIDENCE HANDLING (important — ties back to the original project framing):
We never want to confidently tell a farmer "this is eligible" when we're
not sure, or confidently name the wrong species. So:
- High confidence (top prediction >= HIGH_CONFIDENCE_THRESHOLD): give a
  direct answer.
- Medium confidence: give the top prediction but flag it as uncertain,
  and show the runner-up.
- Low confidence (below LOW_CONFIDENCE_THRESHOLD): don't name a species
  at all — ask for a clearer/different photo instead. A wrong guess here
  is worse than no guess.

Usage as a script:
    python identify_tree.py path/to/photo.jpg

Usage as a module (for the UI layer):
    from identify_tree import identify
    result = identify("path/to/photo.jpg")
"""

import os
import csv
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = "tree_species_classifier_final.pt"
CLASS_NAMES_PATH = "class_names.txt"
LOOKUP_CSV_PATH = "species_lookup_10species.csv"
OOD_REFERENCE_PATH = "ood_reference.pt"
INDIAN_CARBON_MARKET_URL = "https://www.beeindia.gov.in/carbon-market.php"  # broader ICM/CCTS context, BEE (Ministry of Power)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HIGH_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.25
# How much further than the calibrated per-class threshold an image's
# embedding is allowed to be before we reject it outright as "not a tree
# we recognize." 1.0 = exactly at the calibrated threshold; raise this if
# real tree photos start getting rejected too often, lower it if unrelated
# images are still slipping through.
OOD_DISTANCE_MARGIN = 1.15
# Must match NEGATIVE_CLASS_ID in download_negative_images.py — once you
# retrain with the 11th "not a tree" class, this is how we recognize its
# predictions and short-circuit to a clean rejection instead of trying to
# look up (nonexistent) carbon-program info for it.
NEGATIVE_CLASS_ID = "OTHER"

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TreeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(576)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(576, num_classes)

    def get_embedding(self, x):
        """The 576-dim feature vector just before the classifier — used
        for out-of-distribution detection via distance to class centroids,
        not just the final softmax prediction."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


_model = None
_class_names = None
_lookup = None
_device = None
_ood_centroids = None
_ood_thresholds = None
_ood_available = False


def _load_lookup_table():
    lookup = {}
    with open(LOOKUP_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["species_id"]] = row
    return lookup


def _class_label_to_display(class_label):
    """Class labels come from folder names like 'SP001_Eucalyptus' (species_id
    + common_name, joined by the download script). This was the actual bug:
    the lookup CSV is keyed by plain common_name ('Eucalyptus'), so looking
    it up directly by the raw class label ('SP001_Eucalyptus') always missed.
    species_id is the reliable join key — it's the first token before '_'."""
    species_id = class_label.split("_")[0]
    info = _lookup.get(species_id, {})
    display_name = info.get("common_name", class_label)
    return display_name, species_id, info


def _ensure_loaded():
    """Lazy-loads the model/lookup once, so importing this module is cheap
    and repeated calls to identify() don't reload from disk each time."""
    global _model, _class_names, _lookup, _device, _ood_centroids, _ood_thresholds, _ood_available
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        _class_names = [line.strip() for line in f if line.strip()]
    _model = TreeClassifier(len(_class_names)).to(_device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    _model.eval()
    _lookup = _load_lookup_table()

    if os.path.exists(OOD_REFERENCE_PATH):
        ood_data = torch.load(OOD_REFERENCE_PATH, map_location=_device)
        _ood_centroids = ood_data["centroids"].to(_device)
        _ood_thresholds = ood_data["thresholds"].to(_device)
        _ood_available = True
    else:
        # Not a crash — but the model will accept anything without this,
        # so make it loud rather than silently skipping OOD rejection.
        print(f"[WARNING] {OOD_REFERENCE_PATH} not found — out-of-distribution "
              f"rejection is DISABLED. Run compute_ood_reference.py to enable it.")
        _ood_available = False


def identify(image_path):
    """Returns a dict describing the result — always safe to display
    directly to a farmer-facing UI without further logic needed there."""
    _ensure_loaded()

    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        if _ood_available:
            embedding = _model.get_embedding(tensor)  # (1, 576)
            # Distance to every class centroid, then how far that is
            # relative to that class's own calibrated threshold — this
            # normalizes for classes whose images are naturally more
            # spread out in feature space than others.
            distances = torch.norm(embedding - _ood_centroids, dim=1)  # (num_classes,)
            relative_distances = distances / _ood_thresholds
            nearest_idx = relative_distances.argmin().item()
            nearest_relative_distance = relative_distances[nearest_idx].item()

            if nearest_relative_distance > OOD_DISTANCE_MARGIN:
                return {
                    "status": "invalid_input",
                    "message": "This doesn't look like any of the 10 tree species we "
                                "recognize. Please upload a clear photo of a tree "
                                "(leaves, bark, or the whole tree).",
                    "predictions": [],
                }

        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    top3_probs, top3_idx = probs.topk(3)
    raw_predictions = [(_class_names[i], p.item()) for p, i in zip(top3_probs, top3_idx)]

    top_class_label, top_conf = raw_predictions[0]
    top_species_id = top_class_label.split("_")[0]

    # Once retrained with the 11th "not a tree" class: if the model's top
    # prediction IS that class, trust it directly — this is a real learned
    # decision, not a heuristic, so we don't second-guess it with confidence
    # thresholds the way we do for ambiguity between real species.
    if top_species_id == NEGATIVE_CLASS_ID:
        return {
            "status": "invalid_input",
            "message": "This doesn't look like any of the 10 tree species we "
                        "recognize. Please upload a clear photo of a tree "
                        "(leaves, bark, or the whole tree).",
            "predictions": [],
        }

    # Convert raw class labels ('SP001_Eucalyptus') to clean display names
    # ('Eucalyptus') via the lookup table for everything shown to the user.
    # Filter out the negative class from the displayed candidate list too —
    # showing "OTHER: 12%" alongside real species names would be confusing.
    predictions = [(_class_label_to_display(label)[0], prob) for label, prob in raw_predictions
                   if label.split("_")[0] != NEGATIVE_CLASS_ID]

    top_species, species_id, species_info = _class_label_to_display(top_class_label)

    if top_conf < LOW_CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "message": "We couldn't confidently identify this tree. Please try a "
                        "clearer photo — ideally showing the leaves or bark up close, "
                        "in good light.",
            "predictions": predictions,
        }

    result = {
        "status": "confident" if top_conf >= HIGH_CONFIDENCE_THRESHOLD else "medium_confidence",
        "predicted_species": top_species,
        "confidence": top_conf,
        "predictions": predictions,
        "hindi_name": species_info.get("hindi_name", ""),
        "program_relevance": species_info.get("agroforestry_program_relevance", "unknown"),
        "indicative_value_range_inr_per_tonne": species_info.get("indicative_value_range_inr_per_tonne", ""),
        "maturity_years": species_info.get("maturity_years_for_carbon_relevance", ""),
        "notes": species_info.get("notes", ""),
        "gov_resource_name": species_info.get("gov_resource_name", ""),
        "gov_resource_url": species_info.get("gov_resource_url", ""),
        "indian_carbon_market_url": INDIAN_CARBON_MARKET_URL,
    }

    if result["status"] == "medium_confidence":
        result["message"] = (
            f"This looks like {top_species}, but we're not fully certain — "
            f"it could also be {predictions[1][0]}."
        )
    else:
        result["message"] = f"This looks like {top_species} ({top_conf*100:.0f}% confident)."

    # Standing disclaimer, always shown regardless of confidence — this is
    # the core ethical framing from the project design: we identify species,
    # we never claim to determine actual carbon-credit eligibility. All the
    # earning/timeline/registration facts live in the structured fields
    # above (pulled from species_lookup_10species.csv), not hardcoded text —
    # this line is just the one thing that's always true regardless of species.
    result["disclaimer"] = (
        "This tool identifies the tree species only. To find out if YOUR specific "
        "tree qualifies for carbon credits, you'll need to register through a local "
        "FPO or carbon-credit aggregator — this app can't determine eligibility on its own."
    )

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python identify_tree.py <image_path>")
        return
    result = identify(sys.argv[1])
    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}\n")
    if result["status"] == "invalid_input":
        return  # no species data, no predictions to show
    if result["status"] != "uncertain":
        print(f"Program relevance: {result['program_relevance']}")
        print(f"Indicative value: ₹{result['indicative_value_range_inr_per_tonne']}/tonne CO2")
        print(f"Maturity: {result['maturity_years']} years")
        print(f"Reference: {result['gov_resource_name']} — {result['gov_resource_url']}")
        print(f"\n{result['disclaimer']}\n")
    print("Top 3 candidates:")
    for species, prob in result["predictions"]:
        print(f"  {species}: {prob*100:.1f}%")


if __name__ == "__main__":
    main()