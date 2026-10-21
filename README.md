# 🌳 Carbon Credit Trees — Indian Tree Species Identification & Carbon Credit Advisory Platform

> **AI-powered tree species identification from photographs with Explainable AI reports for Indian agroforestry and carbon credit programs.**

---

## 📋 Project Overview

This project helps **Indian farmers, field officers, and agroforestry practitioners** identify tree species from photographs and understand their value in India's carbon credit ecosystem. The system combines:

1. **On-device Deep Learning Classification** — Uses a DINOv2 ViT-S/14 vision transformer backbone with a trained neural classifier head to identify 10 key Indian agroforestry tree species from photos.
2. **Out-of-Distribution (OOD) Rejection** — Uses Mahalanobis distance, Shannon entropy, and confidence thresholds to reliably reject non-tree images (animals, vehicles, buildings, etc.) and unsupported species.
3. **Carbon Credit & Agroforestry Lookup** — Maps identified species to their carbon credit value, government program eligibility, growth timelines, and agroforestry relevance from a curated CSV database.
4. **🤖 Gemini AI Explainable Reports (NEW)** — When a tree is confidently identified, uses Google Gemini API to generate a comprehensive, farmer-friendly report covering medicinal value, carbon credit registration process, minimum land requirements, government subsidies, and much more.

---

## 🌿 Supported Tree Species (10)

| # | Common Name | Scientific Name | Hindi Name | Category |
|---|-------------|-----------------|------------|----------|
| 1 | Eucalyptus | *Eucalyptus tereticornis* | Nilgiri (नीलगिरी) | Fast Growing Timber |
| 2 | Subabul | *Leucaena leucocephala* | Subabul (सुबबूल) | Fast Growing Timber |
| 3 | Shisham | *Dalbergia sissoo* | Shisham (शीशम) | Fast Growing Timber |
| 4 | Poplar | *Populus deltoides* | Poplar (पॉपलर) | Fast Growing Timber |
| 5 | Casuarina | *Casuarina equisetifolia* | Casuarina (कैसुरीना) | Fast Growing Timber |
| 6 | Teak | *Tectona grandis* | Sagwan (सागवान) | Fast Growing Timber |
| 7 | Bamboo | *Bambusa bambos* | Bans (बांस) | Fast Growing Timber |
| 8 | Mango | *Mangifera indica* | Aam (आम) | Fruit |
| 9 | Jackfruit | *Artocarpus heterophyllus* | Kathal (कटहल) | Fruit |
| 10 | Neem | *Azadirachta indica* | Neem (नीम) | Native Multipurpose |

---

## 🏗️ How The Project Works (End-to-End Pipeline)

### Phase 1: Data Collection (One-time, Already Completed)
```
download_tree_images_v6.py  →  Downloads 250-300+ images per species from iNaturalist, GBIF,
                                Wikimedia Commons, and Flickr (CC-licensed)
download_negative_images.py →  Downloads diverse "not a tree" images (dogs, cars, buildings, etc.)
                                for OOD rejection training
clean_corrupt_images.py     →  Removes corrupt/invalid downloaded images
reconcile_metadata.py       →  Syncs metadata CSV after manual cleanup
```
**Output:** `tree_images/` folder with ~3000+ labeled images across 10 species + negatives.

### Phase 2: Feature Extraction (One-time, Already Completed)
```
extract_features.py  →  Loads each image through DINOv2 ViT-S/14 backbone (frozen),
                         applies 4 augmentations per image, extracts 384-dim embeddings,
                         and saves to cache_v2/features_dinov2_vits14.npz
```
**Output:** `cache_v2/features_dinov2_vits14.npz` — cached feature vectors.

### Phase 3: Model Training (One-time, Already Completed)
```
train_head.py  →  Evaluates 4 classifier architectures (Logistic Regression, LinearSVC,
                   Cosine Prototype, Neural Head) using 5-Fold Stratified Grouped CV.
                   Saves the best performer to artifacts_v2/best_model.pt
```
**Output:** `artifacts_v2/best_model.pt` — trained classifier head weights.

### Phase 4: OOD Detector Calibration (One-time, Already Completed)
```
build_ood_detector.py  →  Computes per-class centroids, shared covariance matrix (Ledoit-Wolf),
                           and calibrates 3-tier confidence thresholds via Leave-One-Class-Out CV.
                           Saves to artifacts_v2/ood_config.json
```
**Output:** `artifacts_v2/ood_config.json` — OOD detection configuration (~4MB).

### Phase 5: Inference & Web App (Runtime — What Users Interact With)
```
predict.py              →  CLI inference tool (single image or batch folder)
app/inference_service.py →  Backend wrapper (loads models once, provides predict_image() API)
app/gemini_report_service.py → Gemini API integration for AI-powered tree reports (NEW)
app/streamlit_app.py    →  Streamlit web UI with bilingual interface
```

### Runtime Flow (What Happens When You Upload a Photo):
```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  User       │───▶│ Streamlit UI     │───▶│ Inference       │───▶│ DINOv2 Backbone  │
│  Uploads    │    │ (streamlit_app)  │    │ Service         │    │ (frozen ViT-S/14)│
│  Photo      │    │                  │    │ (inference_svc) │    │ Extract 384-d emb│
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                    │
                                                    ▼
                          ┌────────────────────────────────────────┐
                          │  3-Tier Decision Engine                │
                          │                                        │
                          │  1. Mahalanobis Distance → OOD check   │
                          │  2. Shannon Entropy → uncertainty check │
                          │  3. Max Softmax Prob → confidence check │
                          └─────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             🚫 REJECTED        ⚠️ UNCERTAIN        ✅ CONFIDENT
             (Not a tree)       (Low confidence)    (Species ID + 
                                                     Carbon Profile)
                                                          │
                                                          ▼
                                              ┌─────────────────────┐
                                              │ 🤖 Gemini AI Report │
                                              │ (Optional, on-demand│
                                              │  via API key)       │
                                              └─────────────────────┘
```

---

## 🚀 How to Run the App

### Prerequisites
- Python 3.9+ 
- Internet connection (for first-time DINOv2 backbone download + Gemini reports)

### Step 1: Install Dependencies
```powershell
cd carbon_credit_trees
pip install -r app/requirements_app.txt
```

### Step 2: Start the Streamlit App
```powershell
streamlit run app/streamlit_app.py
```
The app opens at `http://localhost:8501`.

### Step 3: (Optional) Enable AI Reports
1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Enter it in the **sidebar** under "🔑 Gemini API Key"
3. After identifying a tree, click **"📝 Generate Detailed AI Report"**

### Access on Phone (Same Wi-Fi)
```powershell
streamlit run app/streamlit_app.py --server.address 0.0.0.0
```
Open the Network URL shown in terminal on your phone's browser.

---

## 🤖 Gemini AI Report — What's Included

When a tree is confidently identified and the user clicks "Generate Detailed AI Report," the Gemini API produces a comprehensive markdown report covering:

| Section | Details |
|---------|---------|
| 🌳 Tree Identity & Overview | Full botanical classification, all common names, physical description |
| 📈 Growth Period & Lifecycle | Germination timeline, maturity period, max height, best planting season |
| 💊 Medicinal & Traditional Value | Ayurvedic uses, AYUSH-recognized properties, cultural significance |
| 🌍 Carbon Credit Value in India | Annual CO2 sequestration, INR/tonne value, eligible registries |
| 📐 Minimum Land Area | Minimum area for carbon credit qualification, trees/hectare, spacing |
| 🌱 Soil & Climate Requirements | Soil type, pH, rainfall, temperature, drought resistance |
| 💰 Economic Value | Timber, fruit, by-products market prices, employment potential |
| 🌿 Biodiversity & Ecology | Wildlife support, soil improvement, water table impact |
| 🚜 Agroforestry Guide | Compatible crops, planting models, companion species |
| 🏛️ Government Schemes | MGNREGA, Green India Mission, state subsidies, FPO benefits |
| ⚠️ Risks & Considerations | Diseases, pests, invasive potential, legal restrictions |
| 📋 Quick Reference Summary | Concise farmer-friendly table with key decision metrics |

The report is available in **English** and **Hindi (हिंदी)** based on the selected language. It can also be **downloaded as a Markdown file**.

---

## 📁 Project Structure & File Roles

### ✅ Files REQUIRED for Running the App (Essential)

| File/Folder | Purpose |
|-------------|---------|
| `app/streamlit_app.py` | Main Streamlit web UI with bilingual interface, 3-tier result cards, Gemini report integration |
| `app/inference_service.py` | Backend inference wrapper — loads models, provides `predict_image()` API |
| `app/gemini_report_service.py` | **NEW** — Generates comprehensive AI reports via Google Gemini API |
| `app/requirements_app.txt` | Python dependencies for the web app |
| `app/__init__.py` | Package marker for the `app` module |
| `predict.py` | Core predictor class (`TreePredictorV2`) used by both CLI and web app |
| `artifacts_v2/best_model.pt` | Trained neural classifier head weights (~20KB) |
| `artifacts_v2/ood_config.json` | OOD detector centroids, covariance matrix, thresholds (~4MB) |
| `species_lookup_10species.csv` | Agroforestry & carbon credit metadata for all 10 species |
| `class_names.txt` | Ordered list of 10 supported species class IDs |
| `requirements.txt` | Full project dependencies (including training tools) |

### 🧪 Files Used for Training/Development Only (NOT needed at runtime)

| File/Folder | Purpose | Needed at Runtime? |
|-------------|---------|-------------------|
| `download_tree_images_v6.py` | Downloads training images from iNaturalist, GBIF, Wikimedia, Flickr | ❌ No |
| `download_negative_images.py` | Downloads OOD negative class images (dogs, cars, etc.) | ❌ No |
| `clean_corrupt_images.py` | Removes corrupt/invalid images from training set | ❌ No |
| `reconcile_metadata.py` | Syncs metadata CSV after manual image cleanup | ❌ No |
| `extract_features.py` | Extracts DINOv2 embeddings for all training images | ❌ No |
| `train_head.py` | Trains and evaluates classifier heads (5-fold CV) | ❌ No |
| `build_ood_detector.py` | Calibrates OOD rejection thresholds | ❌ No |
| `evaluate.py` | Generates v1 vs v2 comparison report | ❌ No |
| `image_metadata_tier1.csv` | Downloaded image metadata (source URLs, licenses) | ❌ No |
| `tree_images/` | Training image dataset (~3000+ images) | ❌ No |
| `cache/` | V1 cached feature vectors (MobileNetV3) | ❌ No |
| `cache_v2/` | V2 cached feature vectors (DINOv2) | ❌ No |

### ⚠️ Legacy / Deprecated Files (Unnecessary for Execution)

| File/Folder | Why It's Unnecessary |
|-------------|---------------------|
| `app.py` (root) | **Old v1 Streamlit app** — replaced by `app/streamlit_app.py`. Uses the deprecated `identify_tree.py` module. Not needed. |
| `identify_tree.py` | **Entirely commented out** — old v1 inference logic, superseded by `predict.py` and `app/inference_service.py`. Not needed. |
| `predict_species.py` | **Old v1 CLI predictor** — uses MobileNetV3Small backbone. Superseded by `predict.py` (which uses DINOv2). Not needed. |
| `train_species_classifier_torch.py` | **Entirely commented out** — old v1 PyTorch training script. Superseded by `train_head.py`. Not needed. |
| `compute_ood_reference.py` | **Old v1 OOD calibration** — for the MobileNetV3 pipeline. Superseded by `build_ood_detector.py`. Not needed. |
| `artifacts/` (v1 folder) | **Old v1 model artifacts** — `head_model.pt` and `ood_config.json` for MobileNetV3 pipeline. Not needed. |
| `test_results.csv` | V1 batch test results. Not needed. |
| `v2_test_results.csv` | V2 batch test results output. Not needed at runtime. |
| `artifacts_v2/confusion_pairs.csv` | Training analysis artifact. Not needed at runtime. |
| `artifacts_v2/cv_results.json` | Cross-validation results. Not needed at runtime. |
| `artifacts_v2/report.md` | Model evaluation report. Not needed at runtime. |
| `testing_trees/` | Test images for manual verification. Not needed at runtime (but useful for demo). |
| `__pycache__/` | Python bytecode cache. Auto-generated, can be deleted. |
| `app/__pycache__/` | Python bytecode cache. Auto-generated, can be deleted. |

---

## 🔧 Technical Architecture

### Model Pipeline
- **Backbone:** DINOv2 ViT-S/14 (384-dimensional self-supervised vision transformer from Meta)
- **Classifier Head:** Neural head (LayerNorm → Dropout(0.2) → Linear) trained with Label Smoothing (0.1)
- **Test-Time Augmentation (TTA):** 3 views per image (original, horizontal flip, center crop) averaged for robust embeddings
- **OOD Rejection:** Multi-signal approach combining:
  - Mahalanobis distance to per-class centroids (Ledoit-Wolf covariance)
  - Shannon entropy of prediction distribution
  - Minimum softmax probability floor

### 3-Tier Decision System
1. **🚫 REJECTED** — Fails OOD boundary (Mahalanobis distance too high, entropy too high, or confidence too low). Result: "This is not a recognized tree species."
2. **⚠️ LOW CONFIDENCE** — Passes OOD check but species confidence below threshold. Result: Shows top candidates, hides carbon values to prevent misguidance.
3. **✅ CONFIDENT** — High confidence identification. Result: Full species profile + carbon credit info + optional Gemini AI report.

### Gemini Integration
- Uses `google-generativeai` SDK with `gemini-2.0-flash` model
- API key entered at runtime via sidebar (not stored anywhere)
- Report generated on-demand per identified species
- Supports bilingual output (English / Hindi)
- Report cached in Streamlit session state to avoid re-generation on page reruns

---

## 🌐 Bilingual Support

The entire interface supports **English** and **Hindi (हिंदी)** toggled via a sidebar radio button. This includes:
- All UI labels, buttons, and status messages
- Species names displayed in both languages
- Gemini AI reports generated in the selected language
- Error messages and help text

---

## ⚠️ Disclaimer

> **This tool identifies tree species only.** It cannot determine whether a specific tree or land parcel is eligible for carbon credits, which requires formal scheme registration, third-party verification, and compliance with specific program rules (e.g., Verra VCS, Gold Standard, MoEFCC Green Credit Programme). All carbon values shown are researcher-estimated ranges from published agroforestry literature. The Gemini AI report is advisory and should be validated with local forest department / Krishi Vigyan Kendra (KVK) guidance.

---

## 📜 License & Credits

- **Training Data:** Creative Commons licensed images from iNaturalist, GBIF, Wikimedia Commons, and Flickr
- **DINOv2 Backbone:** Meta AI Research ([Apache 2.0 License](https://github.com/facebookresearch/dinov2))
- **Gemini AI:** Google DeepMind ([Terms of Service](https://ai.google.dev/terms))
- **Carbon Credit Data:** Researcher-estimated from published Indian agroforestry literature and MoEFCC Green Credit Programme documentation

---

*Built for the Indian farmer. 🇮🇳*
