# Indian Tree Species Classifier & Carbon Credit Web App

A clean, responsive, CPU-friendly Streamlit web application for identifying Indian tree species from photos and looking up agroforestry & carbon credit program relevance under Indian government frameworks (e.g., MoEFCC Green Credit Programme).

---

## Features
- **100% On-Device CPU Inference**: Runs locally on Intel Core i5 laptops without requiring a GPU.
- **Fast Response**: Employs image pre-scaling and cached DINOv2 ViT-S/14 embeddings with Test-Time Augmentation (TTA). Average prediction time is ~1.0–1.5 seconds on CPU.
- **3-Tier Outcome States**:
  1. 🚫 **Rejected**: Rejects non-tree objects (cars, dogs, everyday objects) or unsupported species using Mahalanobis distance & Shannon entropy out-of-distribution (OOD) checks.
  2. ⚠️ **Species Uncertain / Low Confidence**: Recognizes valid tree photos but warns if confidence is below cutoff threshold. Intentionally suppresses carbon values to avoid misleading farmers.
  3. ✅ **Confident Identification**: Displays identified tree species (English + Hindi name), Top-3 candidate probability bar chart, and agroforestry/carbon credit profile.
- **Photo Upload & Live Camera Capture**: Upload `.jpg`, `.jpeg`, `.png` or take photos directly with your phone/webcam.
- **Batch Evaluation Tab**: Upload multiple images simultaneously, view the results table, and download results as CSV.
- **Bilingual Interface**: Toggle between English and Hindi (हिंदी).
- **Researcher Disclaimer**: Visible disclaimer on all economic and carbon credit estimates.

---

## How to Run

### Step 1: Install Dependencies (if not already installed)
From the project folder, run:
```powershell
pip install -r app/requirements_app.txt
```

### Step 2: Start the Web App
From the `carbon_credit_trees` directory:
```powershell
streamlit run app/streamlit_app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## How to Access the App on Your Phone (Local Wi-Fi Network)

To test the camera input and UI on a mobile phone without uploading anything to the internet:
1. Ensure your laptop and phone are connected to the **same Wi-Fi network**.
2. Run Streamlit bound to your network address:
   ```powershell
   streamlit run app/streamlit_app.py --server.address 0.0.0.0
   ```
3. Look at the terminal output for the **Network URL** (for example: `http://192.168.1.15:8501`).
4. Open that URL in your phone's mobile browser (Chrome / Safari). You can now take live tree photos directly in the field!

---

## Project Structure
```
carbon_credit_trees/
├── app/
│   ├── __init__.py
│   ├── inference_service.py     # Backend model wrapper (cached, fast CPU inference)
│   ├── streamlit_app.py         # Streamlit frontend with bilingual UI & 3-tier cards
│   ├── requirements_app.txt     # Minimal UI requirements
│   └── README.md                # Usage instructions & network sharing guide
├── artifacts_v2/
│   ├── best_model.pt            # Classifier head weights
│   └── ood_config.json          # OOD detector centroids & covariance
├── species_lookup_10species.csv # Agroforestry & carbon credit metadata
├── class_names.txt              # Supported 10 tree species IDs
└── testing_trees/               # Test images (trees and non-tree OOD samples)
```
