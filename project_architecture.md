# 🌳 Project Architecture & Technical Specification: VrikshaCarbon (Carbon Credit Trees)

> **Document Type:** Technical & System Architecture Specification for Academic Research  
> **Target Audience:** Co-authors, ML Researchers, Agroforestry Scientists, and Peer Reviewers  
> **Project Title:** *VrikshaCarbon: An Explainable Agroforestry Vision System for Tree Species Classification, Out-of-Distribution Gating, and Carbon Credit Estimation in India*

---

## 1. Executive Summary & Problem Formulation

### 1.1 Context & Motivation
India's agroforestry sector represents one of the most scalable pathways for voluntary and compliance carbon sequestration under schemes such as the **Green Credit Programme (MoEFCC)**, the **National Agroforestry Policy**, and the **Bureau of Energy Efficiency (BEE) Carbon Credit Trading Scheme (CCTS)**. However, rural smallholder farmers and agricultural extension officers face major barriers in accessing carbon markets:
- **Taxonomic barrier:** Visual misidentification of agroforestry species leads to inaccurate allometric biomass calculations.
- **Overconfidence hazard in Computer Vision:** Standard deep learning classifiers assign high-confidence predictions even to non-tree inputs (e.g., livestock, vehicles, indoor objects, weeds) or unsupported wild species, generating fraudulent or erroneous carbon credit projections.
- **Explainability & Advisory Gap:** Raw species classifications do not translate into actionable agronomic guidance, such as intercropping recommendations, rotation cycles, soil viability, medicinal value, and regulatory registry requirements.

### 1.2 System Solution
**VrikshaCarbon** is an end-to-end, edge-deployable (CPU-friendly) machine learning framework that bridges computer vision, statistical out-of-distribution (OOD) detection, allometric carbon modeling, and Generative AI (LLM) advisory:
1. **Self-Supervised Vision Transformer (DINOv2 ViT-S/14)**: Extracts 384-dimensional dense semantic representations invariant to illumination and viewpoint variations.
2. **Test-Time Augmentation (TTA)**: Multi-view aggregation (Original, Horizontal Flip, Center Zoom) for test-time robustness.
3. **Multi-Signal OOD Decision Engine**: A 3-tier confidence policy integrating Mahalanobis distance to class centroids (with Ledoit-Wolf shrinkage covariance), Shannon entropy, and maximum softmax probability.
4. **Allometric Carbon Mapping**: Linking species predictions to carbon credit values (INR 1,400–2,200 / tCO2e), maturity timelines, and MoEFCC registry data.
5. **Generative Agronomic Advisory (Gemini 3.8 Flash)**: An on-demand explainable AI engine providing multilingual (English/Hindi) farmer advisory reports.

---

## 2. End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA & INGESTION LAYER                         │
├────────────────────────────────────────────────────────────────────────┤
│  Field Images (Camera / Web Upload / Batch CSV)                        │
│  Dataset: 10 Agroforestry Species (~3,000 imgs) + OOD Negatives        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION & EMBEDDING                      │
├────────────────────────────────────────────────────────────────────────┤
│  Backbone: DINOv2 ViT-S/14 (Frozen Meta AI Weights, Patch Size 14x14)  │
│  TTA Pipeline: [Resize 224x224, Horizontal Flip, Center Zoom Crop]     │
│  Feature Embedding Vector: e ∈ ℝ³⁸⁴ (L2 Normalized)                    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   MULTI-SIGNAL OOD REJECTION & GATING                  │
├────────────────────────────────────────────────────────────────────────┤
│  1. Mahalanobis Distance: D_M(e, μ_c) = √((e - μ_c)ᵀ Σ⁻¹ (e - μ_c))     │
│     Covariance Estimation: Ledoit-Wolf Empirical Shrinkage (Σ ∈ ℝ³⁸⁴ˣ³⁸⁴)│
│  2. Shannon Entropy: H(p) = -∑ p_i log(p_i)                            │
│  3. Softmax Confidence: max(p_i)                                       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   [REJECTED]     │      │ [LOW CONFIDENCE] │      │   [CONFIDENT]    │
│  Non-Tree / OOD  │      │ Species Ambiguous│      │ Species ID >= 48%│
│  D_M > 35.0 or   │      │ OOD Passed,      │      │ Low Entropy,     │
│  H(p) > 2.05     │      │ Conf < Cutoff    │      │ D_M <= 35.0      │
│  Action: Abort   │      │ Action: Top-3 ID,│      └────────┬─────────┘
│  carbon valuation│      │ Suppress Carbon  │               │
└──────────────────┘      └──────────────────┘               ▼
                                                    ┌──────────────────┐
                                                    │ CARBON & TIER-2  │
                                                    │ LOOKUP ENGINE    │
                                                    ├──────────────────┤
                                                    │ • INR / tCO2e    │
                                                    │ • Maturity Horizon│
                                                    │ • MoEFCC Registry│
                                                    └────────┬─────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │ EXPLAINABLE AI   │
                                                    │ GEMINI 3.8 FLASH │
                                                    ├──────────────────┤
                                                    │ • Medicinal/Ayush│
                                                    │ • Min Land Area  │
                                                    │ • Intercropping  │
                                                    │ • EN / HI Export │
                                                    └──────────────────┘
```

---

## 3. Machine Learning Pipeline & Mathematics

### 3.1 Dataset Specification
The dataset comprises 10 prioritized agroforestry species relevant to Indian state forest policies, afforestation subsidies, and carbon credits:

| Species ID | Common Name | Botanical Name | Hindi / Regional Name | Agroforestry Category | Primary Region in India |
|---|---|---|---|---|---|
| `SP001` | Eucalyptus | *Eucalyptus tereticornis* | Nilgiri (नीलगिरी) | Fast-Growing Timber | Pan-India (Punjab, AP, TN) |
| `SP002` | Subabul | *Leucaena leucocephala* | Subabul (सुबबूल) | Fodder / Green Manure | Central & Western India |
| `SP003` | Shisham | *Dalbergia sissoo* | Shisham (शीशम) | High-Value Timber | Indo-Gangetic Plains |
| `SP004` | Poplar | *Populus deltoides* | Poplar (पॉपलर) | Agro-Industrial Timber | Punjab, Haryana, Western UP |
| `SP005` | Casuarina | *Casuarina equisetifolia* | Casuarina (कैसुरीना) | Coastal Shelterbelt | Coastal TN, Odisha, AP |
| `SP007` | Teak | *Tectona grandis* | Sagwan (सागवान) | Premium Hardwood Timber | MP, Maharashtra, Kerala |
| `SP008` | Bamboo | *Bambusa bambos* | Bans (बांस) | Fast-Biomass Grass/Timber | Northeast, MP, Western Ghats |
| `SP013` | Mango | *Mangifera indica* | Aam (आम) | Horti-Agroforestry Fruit | Pan-India (UP, AP, Bihar) |
| `SP021` | Jackfruit | *Artocarpus heterophyllus*| Kathal (कटहल) | Horti-Agroforestry Fruit | South India, East India |
| `SP027` | Neem | *Azadirachta indica* | Neem (नीम) | Multipurpose Native | Pan-India (Dry/Arid Zones) |

- **In-Distribution (ID) Images:** ~3,000+ validated field and herbarium images collected via iNaturalist, GBIF, Wikimedia Commons, and Flickr under Creative Commons licenses.
- **Out-of-Distribution (OOD) Negatives:** ~500 non-tree and common rural distractors (tractors, cattle, domestic animals, soil, buildings, indoor equipment).

### 3.2 Vision Transformer Backbone
- **Model:** `DINOv2 ViT-S/14` (Vision Transformer Small, 14×14 patch size, 22M parameters).
- **Rationale:** Self-supervised training with DINOv2 learns structural boundaries, leaf venation, bark textures, and crown architecture without reliance on human annotations, yielding strong linear separation for botanical taxonomy.
- **Embedding Dimensionality:** $d = 384$.
- **Test-Time Augmentation (TTA):**
  Given input image $I$, three transformations are applied:
  $$I_1 = \text{Resize}_{224 \times 224}(I)$$
  $$I_2 = \text{HorizontalFlip}(\text{Resize}_{224 \times 224}(I))$$
  $$I_3 = \text{CenterCrop}_{224 \times 224}(\text{Resize}_{248 \times 248}(I))$$
  The final embedding $e$ is:
  $$\mathbf{e} = \frac{1}{3} \sum_{k=1}^3 \frac{f_\theta(I_k)}{\|f_\theta(I_k)\|_2}$$

### 3.3 Classification Head
- **Architecture:**
  $$\mathbf{h} = \text{Dropout}_{0.2}(\text{LayerNorm}(\mathbf{e}))$$
  $$\mathbf{z} = \mathbf{W}\mathbf{h} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{10 \times 384}, \mathbf{b} \in \mathbb{R}^{10}$$
  $$\mathbf{p} = \text{Softmax}(\mathbf{z})$$
- **Loss Function:** Cross-Entropy with Label Smoothing ($\alpha = 0.1$) to prevent overconfident softmax saturation:
  $$y^{\text{smooth}}_{i} = (1 - \alpha) y_i + \frac{\alpha}{K}$$

### 3.4 Multi-Signal Out-Of-Distribution (OOD) Formulation
To prevent false-positive classifications on non-trees, the model calculates three distinct metrics:

1. **Mahalanobis Distance with Ledoit-Wolf Covariance:**
   For class centroids $\boldsymbol{\mu}_c = \frac{1}{N_c} \sum_{i \in C_c} \mathbf{e}_i$, and pooled covariance $\mathbf{\Sigma} \in \mathbb{R}^{384 \times 384}$:
   $$\mathbf{\Sigma}_{\text{LW}} = (1 - \beta)\mathbf{\Sigma}_{\text{sample}} + \beta \left(\frac{\text{Tr}(\mathbf{\Sigma}_{\text{sample}})}{d}\right) \mathbf{I}$$
   $$D_M(\mathbf{e}) = \min_{c \in \{1,\dots,10\}} \sqrt{(\mathbf{e} - \boldsymbol{\mu}_c)^\top \mathbf{\Sigma}_{\text{LW}}^{-1} (\mathbf{e} - \boldsymbol{\mu}_c)}$$
   - *Threshold:* $D_M > 35.0 \implies \text{Reject as Out-of-Distribution}$.

2. **Shannon Entropy of Output Distribution:**
   $$H(\mathbf{p}) = -\sum_{c=1}^{10} p_c \ln(p_c)$$
   - *Threshold:* $H(\mathbf{p}) > 2.05 \implies \text{Reject due to excessive epistemic uncertainty}$.

3. **3-Tier Decision Policy:**
   - **Tier 1 — `[REJECTED]`**: $D_M > 35.0$ OR $H(\mathbf{p}) > 2.05$ OR $\max(\mathbf{p}) < 0.22$. Outcome: Flagged as non-tree / unsupported species. Economic carbon models are bypassed.
   - **Tier 2 — `[LOW CONFIDENCE]`**: $D_M \le 35.0$ but $0.22 \le \max(\mathbf{p}) < 0.48$. Outcome: Valid tree detected, but ambiguous class. Returns Top-3 candidate species without financial claims.
   - **Tier 3 — `[CONFIDENT]`**: $D_M \le 35.0$ AND $H(\mathbf{p}) \le 2.05$ AND $\max(\mathbf{p}) \ge 0.48$. Outcome: Full species resolution, carbon revenue projection, and eligibility for Gemini AI advisory.

---

## 4. Empirical Evaluation & Benchmark Results

### 4.1 Comparative Model Progression: V1 vs. V2
During 5-Fold Stratified Grouped Cross-Validation (grouped by source collection to prevent data leakage):

| Performance Metric | V1 Baseline (MobileNetV3-Small) | V2 System (DINOv2 ViT-S/14) | Absolute Delta |
|---|---|---|---|
| **Top-1 CV Accuracy** | 66.41% | **91.52% ± 0.66%** | **+25.11%** |
| **Top-3 CV Accuracy** | 87.55% | **98.45% ± 0.64%** | **+10.90%** |
| **OOD AUROC (Negative Discrimination)** | 64.20% | **82.70%** | **+18.50%** |
| **Feature Representation** | 1024-dim Supervised ConvNet | 384-dim Self-Supervised ViT | Dense semantic spatial attention |
| **Inference Latency (Standard Intel i5 CPU)** | ~35 ms | **~140 ms (with 3-view TTA)** | Fully Real-Time & Edge Capable |

### 4.2 Classifier Head Architecture Exploration on DINOv2 Embeddings
All candidate heads evaluated under identical 5-fold cross-validation splits:

| Candidate Classifier Architecture | Top-1 Accuracy (Mean ± Std) | Top-3 Accuracy (Mean ± Std) | Status |
|---|---|---|---|
| Logistic Regression ($C=1.0$) | 90.30% ± 0.83% | 97.89% ± 0.79% | Baseline |
| Logistic Regression ($C=10.0$) | 89.89% ± 0.58% | 97.83% ± 0.74% | Evaluated |
| Linear Support Vector Classifier (Platt Scaled) | 90.49% ± 0.62% | 97.61% ± 0.77% | Evaluated |
| Cosine Nearest Centroid / Prototype | 77.07% ± 0.94% | 92.64% ± 1.06% | Underperformed |
| **Neural Head (LayerNorm + Dropout 0.2 + Linear)** | **91.52% ± 0.66%** | **98.45% ± 0.64%** | **Selected Best** |

### 4.3 Per-Species Empirical Accuracy & Mahalanobis Distance Profiles

| Species ID | Common Name | Top-1 Accuracy | Within-Class Mean $D_M$ | 90th Percentile $D_M$ | Taxonomic Confusion Profile |
|---|---|---|---|---|---|
| `SP001` | Eucalyptus | **94.33%** | 19.87 | 26.07 | Distinct lancet leaves & peeling bark |
| `SP002` | Subabul | **94.17%** | 18.08 | 21.88 | Bipinnate leaflets clearly resolved |
| `SP003` | Shisham | **90.25%** | 17.90 | 22.99 | Slight confusion with *Azadirachta* (Neem) |
| `SP004` | Poplar | **89.08%** | 20.02 | 26.30 | Ovate leaf margins distinct from Teak |
| `SP005` | Casuarina | **96.08%** | 19.66 | 25.87 | Needle-like cladodes highly distinct |
| `SP007` | Teak | **91.75%** | 18.40 | 22.28 | Large ovate leaves and fibrous trunk |
| `SP008` | Bamboo | **94.42%** | 19.40 | 24.03 | Culm structure & linear foliage |
| `SP013` | Mango | **90.33%** | 23.64 | 27.50 | Dense evergreen canopy |
| `SP021` | Jackfruit | **88.50%** | 22.91 | 27.10 | Glossy coriaceous leaves |
| `SP027` | Neem | **86.25%** | 24.70 | 28.50 | Pinnate serrate leaves; confused with Shisham |

---

## 5. Carbon Credit Modeling & Policy Integration

### 5.1 Carbon Valuation Matrix (Tier-2 Agronomic Database)
Values benchmarked against voluntary registry methodologies (Verra VM0047, Gold Standard Afforestation/Reforestation) and India's Green Credit Programme guidelines:

| Species ID | Common Name | Indicative Value (INR / tCO2e) | Carbon Maturity Horizon | Indian Regulatory Status | Agroforestry Deployment Model |
|---|---|---|---|---|---|
| `SP001` | Eucalyptus | ₹1,400 – ₹2,000 | 5–7 Years | Introduced / Naturalized | Bund / Perimeter plantation |
| `SP002` | Subabul | ₹1,400 – ₹2,000 | 3–5 Years | Introduced | High-density short rotation coppice |
| `SP003` | Shisham | ₹1,500 – ₹2,000 | 8–10 Years | Native | Intercropped with Rabi/Kharif crops |
| `SP004` | Poplar | ₹1,500 – ₹2,000 | 6–8 Years | Introduced | Agrisilviculture (wheat/mustard) |
| `SP005` | Casuarina | ₹1,400 – ₹1,900 | 4–6 Years | Introduced | Windbreak & coastal sand fixation |
| `SP007` | Teak | ₹1,500 – ₹2,200 | 15–20 Years | Native | Long-term block timber plantation |
| `SP008` | Bamboo | ₹1,600 – ₹2,200 | 3–5 Years | Native | Fast sequestration culms / river banks |
| `SP013` | Mango | ₹1,400 – ₹2,000 | 8–10 Years | Native | Horti-silvopastoral / Union Budget 2026 |
| `SP021` | Jackfruit | ₹1,400 – ₹1,900 | 8–10 Years | Native | Multi-tier orchard agroforestry |
| `SP027` | Neem | ₹1,400 – ₹1,900 | 8–10 Years | Native | Silvopastoral / Dryland boundary shelter |

### 5.2 Regulatory Compliance & Researcher Disclaimer Logic
To maintain academic integrity and consumer protection, the system embeds a strict disclaimer pipeline:
- Raw indicative values represent **above-ground and below-ground biomass models** under standard regional wood density and allometric tables.
- Official carbon credit monetization strictly requires:
  1. Baseline land eligibility audit (land must not have experienced prior deforestation).
  2. Additionality proof and permanence guarantee.
  3. Digital Monitoring, Reporting, and Verification (MRV) with 3rd-party audit under MoEFCC GCP / Verra / Gold Standard.

---

## 6. Generative Explainable AI (Gemini 3.8 Flash Advisory)

When a tree achieves `[CONFIDENT]` classification, an automated prompt is compiled and dispatched to Google Gemini 3.8 Flash to synthesize agronomic intelligence.

### 6.1 Report Knowledge Schema
The LLM response is structured across 12 standardized sections:
1. **Botanical Taxonomy & Morphology:** Family, genus, leaf morphology, crown architecture.
2. **Growth Lifecycle:** Germination period, juvenile phase, canopy diameter at maturity.
3. **Ayurvedic & Medicinal Heritage:** AYUSH-recognized properties, pharmacological compounds.
4. **Carbon Sequestration Metrics:** Annual tCO2e sequestration rate per tree and per hectare.
5. **Land Eligibility & Plantation Geometry:** Minimum land area (e.g., 0.5 – 1.0 ha for registry aggregation), spacing grid ($3m \times 3m$, $4m \times 4m$), stocking density.
6. **Edaphoclimatic Requirements:** Soil pH tolerance, annual rainfall window, drought/frost resilience.
7. **Farmer Revenue & By-Product Economics:** Secondary yields (fodder, fruit, seed oil, resin, timer).
8. **Ecological Synergy & Biodiversity:** Nitrogen fixation, pollinator support, soil organic carbon (SOC) enhancement.
9. **Companion Planting & Intercropping Models:** Compatible seasonal pulses, oilseeds, and shade crops.
10. **State & Central Government Subsidies:** National Agroforestry Mission, Green India Mission, NABARD refinancing.
11. **Pathological & Agronomic Hazards:** Susceptibility to borer attacks, root rot, invasive weed characteristics.
12. **Farmer Decision Matrix:** High-density summary table for field decision-making.

---

## 7. Software Architecture & Implementation Details

### 7.1 Tech Stack Specification
- **Core Runtime:** Python 3.9+
- **Deep Learning Framework:** PyTorch 2.x, Torchvision
- **Vision Foundation Model:** `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
- **Covariance & Statistics:** Scikit-Learn (Ledoit-Wolf shrinkage), SciPy, NumPy
- **Interactive UI:** Streamlit (configured in wide responsive mode with bilingual English/Hindi localization)
- **Data Visualization:** Altair (declarative candidate probability distribution charts)
- **Generative AI SDK:** `google-generativeai` (Gemini 3.8 Flash)

### 7.2 Directory & Artifact Topology
```
carbon_credit_trees/
├── app/
│   ├── streamlit_app.py         # Responsive bilingual web application
│   ├── inference_service.py     # Singleton inference wrapper & session caching
│   ├── gemini_report_service.py # Gemini 3.8 Flash explainable AI integration
│   └── requirements_app.txt     # Runtime application dependencies
├── artifacts_v2/
│   ├── best_model.pt            # Serialized PyTorch neural head weights (20 KB)
│   ├── ood_config.json          # Centroids, Ledoit-Wolf inv-cov, thresholds (4.2 MB)
│   └── report.md                # Empirical evaluation benchmark records
├── predict.py                   # Unified CLI inference & batch evaluation engine
├── species_lookup_10species.csv # Curated carbon credit & agroforestry lookup table
├── class_names.txt              # Class index mapping (10 species)
├── README.md                    # Operational developer documentation
└── project_architecture.md      # Comprehensive research & system specification (This File)
```

---

## 8. Limitations & Future Research Directions

1. **Seasonal Phenology & Deciduous Foliage:** Performance during winter leaf shedding (e.g., *Populus deltoides*) remains dependent on bark and silhouette patterns. Integrating multi-modal bark and twig feature classifiers is a recommended next step.
2. **LiDAR & Satellite Fusion:** Ground-level camera identification can be coupled with Sentinel-2 NDVI time-series or drone LiDAR for automated canopy volume and biomass calculation.
3. **Hyperlocal Geospatial Soil Matching:** Integrating GPS metadata with ICAR / Soil Health Card databases to validate species suitability before planting.
