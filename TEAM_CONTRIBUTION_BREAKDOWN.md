# 👥 Team Work Division & Presentation Blueprint (3 Members)

> **Document Type:** Team Contribution Breakdown, Code File Ownership & Presentation Script  
> **Project Title:** *VrikshaCarbon: Indian Tree Species Identification, Statistical OOD Rejection & Carbon Credit Advisory*  
> **Audience:** Academic Presentation, Viva Voce Panel, and Project Documentation  

---

## 📊 Overview of 3-Way Project Division

To present this project seamlessly as a team of three, the architecture is split into three logical, distinct components:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PERSON 1: DATA ENGINEERING, CURATION & VISION EMBEDDINGS (THE FOUNDATION) │
├────────────────────────────────────────────────────────────────────────┤
│ • Automated Data Scraping & Botanical Quality Control                  │
│ • Self-Supervised Vision Transformer (DINOv2 ViT-S/14) Feature Pipeline │
│ • Test-Time Augmentation (TTA) Data Pipeline                           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PERSON 2: CLASSIFIER MODELING, OOD REJECTION & EVALUATION (THE BRAIN)  │
├────────────────────────────────────────────────────────────────────────┤
│ • 5-Fold Stratified Grouped Cross-Validation & Neural Head Training   │
│ • High-Dimensional Covariance (Ledoit-Wolf) & Mahalanobis Distance     │
│ • Shannon Entropy Epistemic Uncertainty & 3-Tier Decision Engine       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PERSON 3: APPLICATION INTEGRATION, CARBON ADVISORY & EXPLAINABLE AI    │
├────────────────────────────────────────────────────────────────────────┤
│ • Core Runtime Predictor & Inference Service Caching                   │
│ • Carbon Credit Economics (₹/tCO2e) & Agroforestry Database Lookup    │
│ • Generative AI Advisory (Gemini 3.8 Flash) & Streamlit UI             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 👤 PERSON 1: Data Engineering, Data Curation & Foundation Vision Embeddings

### 🎯 Primary Responsibilities & Role
Person 1 is responsible for the entire **Data Pipeline and Foundation Vision Representation**. They handle how botanical and negative imagery was collected, curated, cleaned, and transformed into 384-dimensional dense semantic vectors using self-supervised learning.

### 💻 Code Files Owned & Explained
1. **`download_tree_images_v6.py`**:
   - Automated multi-source scraper pulling 250–300+ field images per species from iNaturalist, GBIF, Wikimedia Commons, and Flickr under Creative Commons licenses.
   - Filters by species taxonomy and botanical keywords.
2. **`download_negative_images.py`**:
   - Curates the Out-Of-Distribution (OOD) negative dataset (~500 images of tractors, cattle, domestic animals, soil, buildings, and vehicles) essential for training and calibrating the rejection engine.
3. **`clean_corrupt_images.py` & `reconcile_metadata.py`**:
   - Integrity checking, EXIF rotation handling, corrupt image removal, and synchronization with `image_metadata_tier1.csv`.
4. **`extract_features.py`**:
   - Loads the frozen Meta AI DINOv2 (`dinov2_vits14`) Vision Transformer backbone.
   - Applies Test-Time Augmentation (TTA: standard resize, horizontal flip, center crop).
   - Extracts 384-dimensional L2-normalized feature embeddings for the entire dataset and serializes them to `cache_v2/features_dinov2_vits14.npz`.

### 🎤 What Person 1 Should Speak in the Presentation
- *"I was responsible for Phase 1 and Phase 2: Data Engineering and Feature Extraction."*
- Explain why standard supervised transfer learning (ImageNet MobileNet) failed: supervised networks focus on broad object outlines, whereas trees need fine-grained botanical features (leaf margins, venation, bark textures).
- Explain DINOv2 self-supervised learning: how it learns dense spatial attention without human labels.
- Explain Test-Time Augmentation (TTA): why averaging three views (original, flipped, zoomed) stabilizes embeddings against field lighting and camera angles.

---

## 👤 PERSON 2: Model Architecture, Statistical OOD Rejection & Cross-Validation

### 🎯 Primary Responsibilities & Role
Person 2 is responsible for **Machine Learning Modeling, Statistical Uncertainty, and the 3-Tier Decision Engine**. They explain how different candidate classifiers were benchmarked, how data leakage was avoided, and how the system mathematically rejects non-trees.

### 💻 Code Files Owned & Explained
1. **`train_head.py`**:
   - Evaluates 4 candidate classifier architectures on identical 5-fold cross-validation splits:
     - Logistic Regression ($C=1.0, 10.0$)
     - Platt-scaled Linear Support Vector Classifier (LinearSVC)
     - Cosine Nearest Prototype / Centroid
     - Neural Head (`LayerNorm → Dropout(0.2) → Linear(384 → 10)`)
   - Implements Cross-Entropy with Label Smoothing ($\alpha = 0.1$).
   - Implements Stratified Grouped K-Fold splitting to prevent geographical/photographer data leakage.
   - Saves the champion model weights to `artifacts_v2/best_model.pt`.
2. **`build_ood_detector.py`**:
   - Computes 384-dimensional per-class centroids ($\boldsymbol{\mu}_c$).
   - Calculates the pooled covariance matrix using Ledoit-Wolf empirical shrinkage ($\mathbf{\Sigma}_{\text{LW}}$) to ensure invertibility in high dimensions.
   - Computes global Mahalanobis distance thresholds ($D_M > 35.0$) and Shannon entropy thresholds ($H > 2.05$).
   - Serializes calibration parameters to `artifacts_v2/ood_config.json`.
3. **`artifacts_v2/report.md` & `evaluate.py`**:
   - Benchmark evaluation tracking: V1 vs. V2 comparison showing the jump from **66.41% to 91.52% Top-1 Accuracy** and **+18.5% AUROC in OOD discrimination**.

### 🎤 What Person 2 Should Speak in the Presentation
- *"I handled Phase 3 and Phase 4: Model Training, Out-of-Distribution Rejection, and Statistical Calibration."*
- Explain the **Overconfidence Hazard**: Why standard softmax classifiers cannot be trusted in financial/carbon applications because they give high probabilities to random objects.
- Present the mathematics of Mahalanobis distance:
  $$D_M(\mathbf{e}) = \min_{c} \sqrt{(\mathbf{e} - \boldsymbol{\mu}_c)^\top \mathbf{\Sigma}_{\text{LW}}^{-1} (\mathbf{e} - \boldsymbol{\mu}_c)}$$
- Explain Ledoit-Wolf shrinkage: Why sample covariance matrices fail in 384 dimensions and how shrinkage guarantees mathematical stability.
- Explain the **3-Tier Decision Policy**:
  - `[REJECTED]`: Non-tree or unsupported species.
  - `[LOW CONFIDENCE]`: Tree detected, but ambiguous species; suppresses carbon figures to protect the farmer.
  - `[CONFIDENT]`: High confidence; unlocks carbon valuation and AI report.

---

## 👤 PERSON 3: Inference Engine, Carbon Valuation & Explainable AI (LLM) Integration

### 🎯 Primary Responsibilities & Role
Person 3 is responsible for **System Integration, Economic Carbon Modeling, Generative AI Advisory, and the User Experience**. They explain how the ML models run on-device on CPU, how carbon economics are calculated, and how Gemini 3.8 Flash delivers explainable advisory to farmers.

### 💻 Code Files Owned & Explained
1. **`predict.py`**:
   - Core inference engine (`TreePredictorV2`) capable of single-image prediction and bulk batch evaluation (`--folder`).
   - Runs inference on standard CPU in ~140ms with 3-view TTA.
2. **`species_lookup_10species.csv`**:
   - Curated agronomic database linking botanical species to indicative carbon credit values (₹1,400–2,200/tCO2e), maturity horizons (3–20 years), and MoEFCC Green Credit registry links.
3. **`app/inference_service.py`**:
   - Singleton service layer for the web application with resource caching (`@st.cache_resource`), thread-safe CPU execution, and batch processing.
4. **`app/gemini_report_service.py`**:
   - Integration with Google Gemini 3.8 Flash LLM.
   - Constructs a structured agronomic prompt based on model confidence and Tier-2 carbon metrics.
   - Generates a 12-section explainable farmer advisory (Ayurvedic/medicinal uses, minimum land area eligibility, companion intercropping crops, government subsidies, and risk factors).
   - Supports bilingual generation in English and Hindi (हिंदी) and enables Markdown report download.
5. **`app/streamlit_app.py`**:
   - Modern, responsive web interface designed with a clean split-column layout (no endless vertical scrolling).
   - Custom CSS gradients, color-coded carbon metric cards, Altair candidate distribution charts, live camera input, and batch CSV export.

### 🎤 What Person 3 Should Speak in the Presentation
- *"I was responsible for Phase 5: Inference Pipeline, Carbon Credit Modeling, Explainable AI Advisory, and Application Deployment."*
- Demonstrate the real-time inference speed: running fully on-device on CPU in ~140ms without expensive cloud GPU servers.
- Explain the Carbon Credit Economics: How indicative values (₹/tCO2e) and rotation timelines are derived from Indian allometric equations and the Green Credit Programme (MoEFCC).
- Explain the Explainable AI (XAI) layer with Gemini 3.8 Flash: Why raw taxonomy is insufficient for farmers and how LLM-generated reports provide actionable field advice on land qualification, AYUSH medicinal properties, and intercropping.
- Walk the panel through the live Streamlit dashboard (both Single Image and Batch modes).

---

## 📋 Summary Table for Slides / Submission

| Team Member | Project Phase | Core Functional Modules | Key Code Files | Key Metrics / Contributions |
|---|---|---|---|---|
| **Person 1** | Data Engineering & Representation | Data Scraping, Cleaning, Augmentation & DINOv2 Feature Extraction | `download_tree_images_v6.py`<br>`download_negative_images.py`<br>`clean_corrupt_images.py`<br>`extract_features.py` | • ~3,000 botanical images + 500 OOD negatives<br>• DINOv2 ViT-S/14 384-dim embedding pipeline<br>• 3-view Test-Time Augmentation (TTA) |
| **Person 2** | ML Modeling & OOD Gating | Neural Head Training, Ledoit-Wolf Covariance & 3-Tier Policy | `train_head.py`<br>`build_ood_detector.py`<br>`evaluate.py`<br>`artifacts_v2/report.md` | • **91.52% Top-1 Accuracy** (+25.1% over baseline)<br>• **82.70% OOD AUROC** (+18.5%)<br>• Mahalanobis & Shannon Entropy 3-Tier Gating |
| **Person 3** | Integration, Economics & XAI | Runtime Predictor, Carbon Lookup, Gemini 3.8 Advisory & Web App | `predict.py`<br>`species_lookup_10species.csv`<br>`app/inference_service.py`<br>`app/gemini_report_service.py`<br>`app/streamlit_app.py` | • ~140ms on-device CPU latency<br>• ₹1,400–2,200/tCO2e Carbon Matrix<br>• Gemini 3.8 Flash 12-section XAI advisory<br>• Modern bilingual (EN/HI) Streamlit UI |
