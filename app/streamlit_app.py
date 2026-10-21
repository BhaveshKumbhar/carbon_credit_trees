"""
streamlit_app.py
================
Clean, modern, and lively Streamlit Web UI for Indian Tree Species Identification,
Agroforestry & Carbon Credit Advisory, and Gemini-powered Explainable AI Reports.

Theme: Agroforestry, Indian Forestry, Carbon Credits, and Farmer Empowerment.
Layout: Modern multi-column dashboard with rich gradients, badges, and responsive cards.

Usage:
    streamlit run app/streamlit_app.py
    Gemini API key = AQ.Ab8RN6KO4i6CxkXFdst6m3Nb_4lo-ABGQnv6C8KXbk9pEbuNIQ
"""

import os
import sys
import io
import time
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
import streamlit as st

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from inference_service import TreeInferenceService, get_inference_service
    from gemini_report_service import configure_gemini, generate_tree_report
except ImportError:
    from app.inference_service import TreeInferenceService, get_inference_service
    from app.gemini_report_service import configure_gemini, generate_tree_report

# -----------------------------------------------------------------------------
# Page Configuration: Wide mode for modern balanced layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VrikshaCarbon | Indian Agroforestry & Carbon Credit AI",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling: Modern Agroforestry & Carbon Credit Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top container spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #063c2c 0%, #0d5c41 45%, #137553 80%, #1e824c 100%);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(6, 60, 44, 0.22);
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }

    .hero-container::after {
        content: "🌿";
        position: absolute;
        right: 1.5rem;
        bottom: -0.8rem;
        font-size: 6.5rem;
        opacity: 0.12;
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #d1fae5;
        margin-top: 0.4rem;
        margin-bottom: 1.1rem;
        max-width: 850px;
        line-height: 1.5;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.85rem;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #ffffff;
        backdrop-filter: blur(8px);
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        margin: 1rem 0;
    }

    .carbon-card {
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        border: 1px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .carbon-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }

    .card-emerald {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.04));
        border-color: rgba(16, 185, 129, 0.35);
    }

    .card-amber {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.12), rgba(217, 119, 6, 0.04));
        border-color: rgba(245, 158, 11, 0.35);
    }

    .card-blue {
        background: linear-gradient(145deg, rgba(14, 165, 233, 0.12), rgba(2, 132, 199, 0.04));
        border-color: rgba(14, 165, 233, 0.35);
    }

    .card-forest {
        background: linear-gradient(145deg, rgba(34, 197, 94, 0.12), rgba(21, 128, 61, 0.04));
        border-color: rgba(34, 197, 94, 0.35);
    }

    .card-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #4b5563;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .card-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .card-sub {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Result Header Card */
    .species-header-card {
        background: linear-gradient(135deg, #064e3b 0%, #047857 60%, #10b981 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 1.3rem 1.6rem;
        box-shadow: 0 8px 24px rgba(4, 120, 87, 0.2);
        margin-bottom: 1.1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .species-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }

    .species-subtitle {
        font-size: 1.05rem;
        color: #a7f3d0;
        font-style: italic;
        margin-top: 0.2rem;
    }

    .conf-badge {
        background: rgba(255, 255, 255, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        text-align: center;
        backdrop-filter: blur(6px);
    }

    .conf-badge-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
    }

    .conf-badge-lbl {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #d1fae5;
    }

    /* Welcome / Guide Card */
    .guide-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1.5px dashed #86efac;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        color: #14532d;
    }

    .guide-step {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        margin-bottom: 0.85rem;
    }

    .guide-num {
        background: #15803d;
        color: white;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    /* Disclaimer Card */
    .disclaimer-banner {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-size: 0.86rem;
        color: #78350f;
        margin-top: 1rem;
        line-height: 1.45;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
    }

    /* AI Report Box */
    .ai-report-card {
        background: linear-gradient(135deg, #f0fdfa 0%, #e6fffa 100%);
        border: 1.5px solid #5eead4;
        border-radius: 16px;
        padding: 1.6rem;
        margin-top: 1.4rem;
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.08);
    }

    /* Button Enhancements */
    .stButton>button {
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Sidebar species item */
    .sidebar-species-chip {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #d1fae5;
        border-radius: 8px;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Bilingual Text Dictionary (English / Hindi)
# -----------------------------------------------------------------------------
TEXTS = {
    "en": {
        "title": "VrikshaCarbon • Indian Agroforestry AI",
        "subtitle": "Empowering Indian farmers & landowners with real-time tree identification, carbon credit valuation, and explainable AI agroforestry advisory.",
        "badge_species": "10 Key Species",
        "badge_ood": "Multi-Signal OOD Guard",
        "badge_carbon": "MoEFCC Green Credit Aligned",
        "badge_ai": "Gemini 3.8 Advisory",
        "tab_single": "📸 Tree Identification & Valuation",
        "tab_batch": "📁 Batch Evaluation (Multi-Image)",
        "upload_mode_file": "📁 Upload Photo",
        "upload_mode_camera": "📷 Live Camera",
        "choose_file": "Choose a clear tree photo (leaves, bark, or canopy)...",
        "take_photo": "Snap a clear photo using your device camera",
        "identify_btn": "🔍 Identify Tree & Calculate Carbon Credits",
        "preview_title": "Uploaded Tree Image",
        "status_rejected_title": "🚫 Image Rejected (Out of Distribution)",
        "status_rejected_msg": "This image does not match any of the 10 supported Indian agroforestry tree species, or is not a tree.",
        "tips_title": "💡 Field Tips for Accurate Identification:",
        "tip_1": "• Capture the tree in natural daytime lighting.",
        "tip_2": "• Focus closely on distinctive leaves, bark texture, or crown structure.",
        "tip_3": "• Avoid non-tree subjects, indoor lighting, animals, or blurry captures.",
        "status_uncertain_title": "⚠️ Low Confidence / Ambiguous Species",
        "status_uncertain_msg": "Our vision transformer detected a tree, but confidence falls below our safe threshold.",
        "uncertain_recommendation": "Please capture a sharper photo with closer foliage or trunk details in brighter light. Carbon valuations are withheld to prevent misguidance.",
        "status_confident_title": "✅ Species Verified & Valued",
        "primary_species": "Primary Species Identified",
        "confidence_label": "Confidence",
        "top3_title": "📊 Candidate Species Probability Distribution",
        "carbon_profile_title": "🌱 Agroforestry & Carbon Credit Profile",
        "program_relevance": "Agroforestry Relevance",
        "indicative_value": "Indicative Carbon Value",
        "maturity_years": "Maturity Horizon",
        "gov_program": "Government Framework / Registry",
        "native_status": "Ecological Status",
        "notes": "Field & Cultivation Notes",
        "disclaimer_text": "⚠️ **Farmer & Researcher Disclaimer**: Indicative carbon values are researcher-benchmarked estimates (based on standard Indian allometric models & voluntary carbon markets). This tool identifies species and provides reference values. Formal carbon credit issuance requires scheme registration, baseline documentation, and third-party audit under MoEFCC / voluntary registries (Verra / Gold Standard).",
        "tech_details": "🛠️ Technical Vision & OOD Diagnostics",
        "ood_distance": "Mahalanobis Distance (OOD)",
        "ood_boundary": "Rejection Threshold",
        "entropy": "Prediction Entropy",
        "model_version": "Active Neural Backbone",
        "gemini_report_title": "🤖 Comprehensive Gemini AI Agroforestry Report",
        "gemini_report_desc": "Generates a complete multilingual advisory covering medicinal uses, land eligibility, intercropping, and Indian carbon registry guidelines.",
        "gemini_report_btn": "📝 Generate Detailed AI Farmer Advisory",
        "gemini_report_generating": "Consulting Gemini AI for comprehensive agroforestry & carbon advisory...",
        "gemini_api_key_label": "🔑 Gemini API Key",
        "gemini_api_key_help": "Enter your Google Gemini API key to activate explainable AI farmer advisory. Get one free at https://aistudio.google.com/apikey",
        "gemini_api_key_missing": "💡 Enter your Gemini API key in the left sidebar to unlock the in-depth AI Carbon & Agroforestry Report.",
        "gemini_report_download": "📥 Download Farmer Advisory as Markdown",
        "sidebar_about_title": "About This Project",
        "sidebar_about_body": (
            "VrikshaCarbon is an applied AI system engineered for Indian farmers, field officers, and agroforestry "
            "practitioners. It combines a fine-tuned Vision Transformer with Mahalanobis Out-Of-Distribution (OOD) "
            "rejection to safeguard against incorrect credit valuations."
        ),
        "sidebar_species_title": "Supported Agroforestry Trees (10)",
        "sidebar_registry_title": "🏛️ Carbon Registries Reference",
        "lang_select": "🌐 Interface Language / भाषा",
        "batch_upload_label": "Upload multiple tree photos for bulk classification",
        "batch_run_btn": "⚡ Run Bulk Analysis",
        "batch_progress": "Processing tree images...",
        "batch_results_title": "Bulk Classification & Valuation Summary",
        "download_csv": "📥 Export Batch Results to CSV",
        "error_file_type": "Please upload a valid image file (JPG, JPEG, or PNG).",
        "error_loading_model": "Failed to load model artifacts. Please verify that 'artifacts_v2/' exists.",
        "welcome_guide_title": "🌾 Ready to Analyze Your Tree",
        "welcome_guide_sub": "Follow these quick steps to identify your tree and evaluate its carbon potential:",
        "guide_step1": "Upload a clear photo or snap a photo of the tree leaves, bark, or canopy.",
        "guide_step2": "Our on-device Vision Transformer instantly verifies the species and ensures Out-Of-Distribution rejection.",
        "guide_step3": "View indicative carbon credit value (₹/tCO2), maturity horizon, and MoEFCC Green Credit alignment.",
        "guide_step4": "Generate an AI advisory report via Gemini for land eligibility, intercropping, and registry tips.",
    },
    "hi": {
        "title": "वृक्ष कार्बन • भारतीय कृषि वानिकी AI",
        "subtitle": "भारतीय किसानों और भूमि मालिकों के लिए त्वरित वृक्ष पहचान, कार्बन क्रेडिट मूल्यांकन एवं AI कृषि वानिकी सलाह।",
        "badge_species": "१० मुख्य प्रजातियां",
        "badge_ood": "मल्टी-सिग्नल OOD सुरक्षा",
        "badge_carbon": "पर्यावरण मंत्रालय ग्रीन क्रेडिट संरेखित",
        "badge_ai": "Gemini ३.८ AI रिपोर्ट",
        "tab_single": "📸 वृक्ष पहचान एवं कार्बन मूल्य",
        "tab_batch": "📁 थोक विश्लेषण (एकाधिक चित्र)",
        "upload_mode_file": "📁 फोटो अपलोड करें",
        "upload_mode_camera": "📷 सीधा कैमरा लें",
        "choose_file": "पेड़ की स्पष्ट तस्वीर चुनें (पत्तियां, छाल या पूरा पेड़)...",
        "take_photo": "अपने कैमरे से पेड़ की तस्वीर खींचें",
        "identify_btn": "🔍 वृक्ष पहचानें और कार्बन क्रेडिट जांचें",
        "preview_title": "अपलोड की गई तस्वीर",
        "status_rejected_title": "🚫 छवि अस्वीकृत (असमर्थित वस्तु / प्रजाति)",
        "status_rejected_msg": "यह तस्वीर समर्थित १० भारतीय कृषि वानिकी प्रजातियों में से किसी से मेल नहीं खाती।",
        "tips_title": "💡 सटीक पहचान के लिए किसान सुझाव:",
        "tip_1": "• दिन के प्राकृतिक उजाले में पेड़ की स्पष्ट फोटो लें।",
        "tip_2": "• पत्तियों के आकार, तने की छाल या पेड़ के मुकुट को नजदीक से दिखाएं।",
        "tip_3": "• वाहन, जानवर, धुंधली या घर के अंदर की तस्वीरों से बचें।",
        "status_uncertain_title": "⚠️ प्रजाति अनिश्चित / कम विश्वास",
        "status_uncertain_msg": "यह एक पेड़ जैसा दिखता है, लेकिन प्रजाति पूरी तरह निश्चित नहीं है।",
        "uncertain_recommendation": "कृपया बेहतर रोशनी में पत्तियों या तने की स्पष्ट फोटो दोबारा लें। गलत मार्गदर्शन से बचने के लिए कार्बन मूल्य छिपाए गए हैं।",
        "status_confident_title": "✅ प्रजाति की पुष्टि एवं मूल्यांकन सफल",
        "primary_species": "पहचानी गई प्रजाति",
        "confidence_label": "सटीकता / विश्वास",
        "top3_title": "📊 शीर्ष ३ संभावित प्रजातियों का वितरण",
        "carbon_profile_title": "🌱 कृषि-वानिकी और कार्बन क्रेडिट विवरण",
        "program_relevance": "कृषि वानिकी प्रासंगिकता",
        "indicative_value": "अनुमानित कार्बन मूल्य",
        "maturity_years": "परिपक्वता अवधि",
        "gov_program": "सरकारी कार्यक्रम / रजिस्ट्री संदर्भ",
        "native_status": "पारिस्थितिकी स्थिति",
        "notes": "खेत एवं रोपण सुझाव",
        "disclaimer_text": "⚠️ **किसान एवं शोधकर्ता अस्वीकरण**: सांकेतिक कार्बन मूल्य शोधकर्ताओं द्वारा भारतीय मानक बायोमास सूत्रों एवं स्वैच्छिक कार्बन बाजार के आधार पर अनुमानित हैं। वास्तविक कार्बन क्रेडिट प्राप्ति हेतु पर्यावरण मंत्रालय (MoEFCC) अथवा अंतर्राष्ट्रीय संस्थाओं (Verra / Gold Standard) में विधिवत पंजीकरण एवं ऑडिट आवश्यक है।",
        "tech_details": "🛠️ तकनीकी मॉडल एवं OOD परीक्षण विवरण",
        "ood_distance": "महलानोबिस दूरी (OOD)",
        "ood_boundary": "अस्वीकृति सीमा (थ्रेशोल्ड)",
        "entropy": "अनिश्चितता (एंट्रॉपी)",
        "model_version": "सक्रिय मॉडल बैकबोन",
        "gemini_report_title": "🤖 व्यापक Gemini AI कृषि वानिकी रिपोर्ट",
        "gemini_report_desc": "पेड़ के औषधीय गुण, न्यूनतम भूमि आवश्यकता, अंतर-फसल एवं कार्बन रजिस्ट्री नियमों पर संपूर्ण रिपोर्ट।",
        "gemini_report_btn": "📝 विस्तृत AI किसान सलाहकार रिपोर्ट बनाएं",
        "gemini_report_generating": "Gemini AI से संपूर्ण कृषि वानिकी एवं कार्बन रिपोर्ट तैयार हो रही है...",
        "gemini_api_key_label": "🔑 Gemini API कुंजी",
        "gemini_api_key_help": "विस्तृत किसान रिपोर्ट हेतु अपनी Google Gemini API कुंजी दर्ज करें। निःशुल्क प्राप्त करें: https://aistudio.google.com/apikey",
        "gemini_api_key_missing": "💡 विस्तृत AI रिपोर्ट अनलॉक करने के लिए बाईं ओर साइडबार में अपनी Gemini API कुंजी दर्ज करें।",
        "gemini_report_download": "📥 किसान सलाहकार रिपोर्ट मार्कडाउन में डाउनलोड करें",
        "sidebar_about_title": "परियोजना के बारे में",
        "sidebar_about_body": (
            "वृक्ष कार्बन एक व्यावहारिक AI प्रणाली है जो भारतीय किसानों और वानिकी कार्यकर्ताओं की सहायता के लिए बनाई गई है। "
            "यह सीपीयू पर तेज़ विज़न मॉडल और OOD सुरक्षा तंत्र के साथ गलत पहचान को रोकती है।"
        ),
        "sidebar_species_title": "समर्थित कृषि वानिकी वृक्ष (१०)",
        "sidebar_registry_title": "🏛️ कार्बन रजिस्ट्री संदर्भ",
        "lang_select": "🌐 Language / भाषा",
        "batch_upload_label": "थोक विश्लेषण के लिए कई फोटो अपलोड करें",
        "batch_run_btn": "⚡ थोक विश्लेषण प्रारंभ करें",
        "batch_progress": "तस्वीरों का विश्लेषण हो रहा है...",
        "batch_results_title": "थोक पहचान एवं मूल्यांकन तालिका",
        "download_csv": "📥 परिणाम CSV के रूप में डाउनलोड करें",
        "error_file_type": "कृपया वैध छवि फ़ाइल (JPG, JPEG, PNG) अपलोड करें।",
        "error_loading_model": "मॉडल लोड करने में त्रुटि। कृपया सुनिश्चित करें कि 'artifacts_v2/' मौजूद है।",
        "welcome_guide_title": "🌾 पेड़ का विश्लेषण करने के लिए तैयार",
        "welcome_guide_sub": "अपने पेड़ की पहचान और कार्बन क्षमता का मूल्यांकन करने के आसान चरण:",
        "guide_step1": "पेड़ की पत्तियों, छाल या पूरे पेड़ की स्पष्ट तस्वीर अपलोड करें या कैमरे से खींचें।",
        "guide_step2": "हमारा ऑन-डिवाइस विज़न मॉडल तुरंत प्रजाति की पुष्टि करेगा और असमर्थित वस्तुओं को अस्वीकार करेगा।",
        "guide_step3": "अनुमानित कार्बन मूल्य (₹/tCO2), परिपक्वता अवधि और सरकारी ग्रीन क्रेडिट संरेखण देखें।",
        "guide_step4": "Gemini AI की सहायता से औषधीय मूल्य, भूमि सीमा और अंतर-फसल पर विस्तृत रिपोर्ट प्राप्त करें।",
    }
}

# -----------------------------------------------------------------------------
# Cached Inference Service Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading agroforestry models onto CPU...")
def load_service():
    """Initializes and caches the tree inference service."""
    try:
        service = get_inference_service()
        return service, None
    except Exception as e:
        return None, str(e)


def main():
    service, load_error = load_service()

    # -------------------------------------------------------------------------
    # Sidebar: Clean, Earthy, and Informative
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🌐 भाषा / Language")
        lang_choice = st.radio(
            label="Select Language",
            options=["English", "हिंदी (Hindi)"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        lang = "hi" if "Hindi" in lang_choice or "हिंदी" in lang_choice else "en"
        T = TEXTS[lang]

        st.markdown("---")
        st.markdown(f"### {T['gemini_api_key_label']}")
        gemini_api_key = st.text_input(
            T["gemini_api_key_label"],
            type="password",
            help=T["gemini_api_key_help"],
            label_visibility="collapsed",
            placeholder="AIzaSy...",
        )
        if gemini_api_key:
            configure_gemini(gemini_api_key)
            st.success("✅ Gemini AI Key Connected")
        else:
            st.caption(f"💡 {T['gemini_api_key_help']}")

        st.markdown("---")
        st.markdown(f"### {T['sidebar_species_title']}")
        if service is not None and service.supported_species_list:
            for idx, sp in enumerate(service.supported_species_list, 1):
                h_name = f" • {sp['hindi_name']}" if sp.get("hindi_name") else ""
                st.markdown(
                    f"<div class='sidebar-species-chip'><b>{idx}. {sp['common_name']}</b>{h_name}<br>"
                    f"<span style='color: #059669; font-style: italic; font-size: 0.78rem;'>{sp['scientific_name']}</span></div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "1. Eucalyptus (Nilgiri)\n2. Subabul\n3. Shisham\n4. Poplar\n"
                "5. Casuarina\n6. Teak (Sagwan)\n7. Bamboo (Bans)\n8. Mango (Aam)\n"
                "9. Jackfruit (Kathal)\n10. Neem"
            )

        st.markdown("---")
        st.markdown(f"### {T['sidebar_registry_title']}")
        st.markdown("""
        - [MoEFCC Green Credit Portal](https://www.moefcc-gcp.in/)
        - [National Agroforestry Mission](https://agricoop.nic.in/)
        - [BEE Carbon Credit Trading Scheme](https://beeindia.gov.in/)
        """)

        st.markdown("---")
        st.caption("🔒 **100% Private On-Device CPU Inference** • Made for Indian Agroforestry")

    # -------------------------------------------------------------------------
    # Top Hero Banner with rich green gradient & badge tags
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">
            <span>🌳</span> {T['title']}
        </div>
        <div class="hero-subtitle">
            {T['subtitle']}
        </div>
        <div class="hero-badges">
            <span class="badge-pill">🌿 {T['badge_species']}</span>
            <span class="badge-pill">🛡️ {T['badge_ood']}</span>
            <span class="badge-pill">🏛️ {T['badge_carbon']}</span>
            <span class="badge-pill">✨ {T['badge_ai']}</span>
            <span class="badge-pill">⚡ Fast CPU Inference</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if load_error or service is None:
        st.error(f"{T['error_loading_model']}\n\n**Details:** `{load_error}`")
        st.stop()

    # -------------------------------------------------------------------------
    # Main Navigation Tabs
    # -------------------------------------------------------------------------
    tab_single, tab_batch = st.tabs([T["tab_single"], T["tab_batch"]])

    # =========================================================================
    # TAB 1: Single Tree Identification & Carbon Valuation (Balanced Grid Layout)
    # =========================================================================
    with tab_single:
        col_input, col_results = st.columns([1, 1.25], gap="large")

        # ---------------------------------------------------------------------
        # LEFT COLUMN: Input Methods, Photo Preview & Photo Tips
        # ---------------------------------------------------------------------
        with col_input:
            st.markdown("#### 📸 1. Input Tree Photo")
            input_method = st.radio(
                label="Image Input Method",
                options=[T["upload_mode_file"], T["upload_mode_camera"]],
                horizontal=True,
                label_visibility="collapsed"
            )

            image_data = None
            if input_method == T["upload_mode_file"]:
                uploaded_file = st.file_uploader(
                    T["choose_file"],
                    type=["jpg", "jpeg", "png"],
                    help="Upload a clear image of a tree, leaves, or bark."
                )
                if uploaded_file is not None:
                    image_data = uploaded_file
            else:
                camera_file = st.camera_input(T["take_photo"])
                if camera_file is not None:
                    image_data = camera_file

            pil_image = None
            if image_data is not None:
                try:
                    pil_image = Image.open(image_data)
                    st.image(pil_image, caption=T["preview_title"], use_container_width=True)
                except Exception as e:
                    st.error(f"{T['error_file_type']} ({str(e)})")
                    pil_image = None

            # Action Button
            if pil_image is not None:
                if st.button(T["identify_btn"], type="primary", use_container_width=True):
                    with st.spinner("Analyzing tree features on CPU (TTA + Mahalanobis OOD)..."):
                        start_time = time.time()
                        res = service.predict_image(pil_image)
                        elapsed = time.time() - start_time
                    st.session_state["last_prediction"] = res
                    st.session_state["last_elapsed"] = elapsed

            # Farmer Photo Tips Card
            with st.expander(f"💡 {T['tips_title']}", expanded=(image_data is None)):
                st.markdown(f"""
                {T['tip_1']}  
                {T['tip_2']}  
                {T['tip_3']}
                """)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: Real-Time Results & Valuation Dashboard
        # ---------------------------------------------------------------------
        with col_results:
            res = st.session_state.get("last_prediction", None)
            elapsed = st.session_state.get("last_elapsed", 0.0)

            # If no prediction has run yet, show the welcoming farmer guide card
            if res is None:
                st.markdown(f"""
                <div class="guide-card">
                    <h3 style="margin-top: 0; color: #14532d; font-family: 'Outfit', sans-serif;">
                        {T['welcome_guide_title']}
                    </h3>
                    <p style="font-size: 0.95rem; color: #166534; margin-bottom: 1.2rem;">
                        {T['welcome_guide_sub']}
                    </p>
                    <div class="guide-step">
                        <div class="guide-num">1</div>
                        <div><b>{T['upload_mode_file']}:</b> {T['guide_step1']}</div>
                    </div>
                    <div class="guide-step">
                        <div class="guide-num">2</div>
                        <div><b>AI Verification:</b> {T['guide_step2']}</div>
                    </div>
                    <div class="guide-step">
                        <div class="guide-num">3</div>
                        <div><b>Carbon Valuation:</b> {T['guide_step3']}</div>
                    </div>
                    <div class="guide-step">
                        <div class="guide-num">4</div>
                        <div><b>Gemini Advisory:</b> {T['guide_step4']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # Prediction exists! Render the outcome
                if res.get("status") == "error":
                    st.error(f"❌ Error: {res.get('error_message', 'Unknown prediction error.')}")

                else:
                    status = res["status"]

                    # ---------------------------------------------------------
                    # 1. OUTCOME: REJECTED (Non-tree or Unsupported)
                    # ---------------------------------------------------------
                    if status == "rejected":
                        st.error(f"### {T['status_rejected_title']}\n\n**{T['status_rejected_msg']}**")
                        st.markdown(f"**Rejection Diagnostic:** `{res['rejection_reason']}`")
                        st.warning(f"**Distance from tree species boundary:** `{res.get('ood_score', 0):.2f}` (Threshold: `{res.get('ood_threshold', 0):.2f}`)")

                    # ---------------------------------------------------------
                    # 2. OUTCOME: SPECIES UNCERTAIN / LOW CONFIDENCE
                    # ---------------------------------------------------------
                    elif status == "low_confidence":
                        st.warning(f"### {T['status_uncertain_title']}\n\n**{T['status_uncertain_msg']}**")
                        st.markdown(T["uncertain_recommendation"])

                        # Top 3 Candidates Bar Chart
                        st.markdown(f"#### {T['top3_title']}")
                        chart_data = pd.DataFrame([
                            {
                                "Species": f"{item['species_name']} ({item['hindi_name']})" if item.get('hindi_name') else item['species_name'],
                                "Probability (%)": round(item['probability'] * 100, 1)
                            }
                            for item in res["top3"]
                        ])

                        bar_chart = alt.Chart(chart_data).mark_bar(
                            color="#f59e0b",
                            cornerRadiusTopRight=6,
                            cornerRadiusBottomRight=6
                        ).encode(
                            x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title="Confidence (%)"),
                            y=alt.Y("Species:N", sort="-x", title="Candidate Species"),
                            tooltip=["Species", "Probability (%)"]
                        ).properties(height=180)
                        st.altair_chart(bar_chart, use_container_width=True)

                    # ---------------------------------------------------------
                    # 3. OUTCOME: CONFIDENT IDENTIFICATION & FULL PROFILE
                    # ---------------------------------------------------------
                    elif status == "confident":
                        hindi_disp = f" • {res['hindi_name']}" if res.get("hindi_name") else ""
                        conf_pct = res['confidence'] * 100

                        # Vibrant Species Header Banner
                        st.markdown(f"""
                        <div class="species-header-card">
                            <div>
                                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #a7f3d0; font-weight: 700;">
                                    {T['status_confident_title']}
                                </div>
                                <div class="species-title">
                                    🌿 {res['species_name']}{hindi_disp}
                                </div>
                                <div class="species-subtitle">
                                    {res['scientific_name']}
                                </div>
                            </div>
                            <div class="conf-badge">
                                <div class="conf-badge-val">{conf_pct:.1f}%</div>
                                <div class="conf-badge-lbl">{T['confidence_label']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Carbon & Agroforestry Metric Cards (2x2 Grid)
                        t2 = res.get("tier2", {})
                        relevance = t2.get("program_relevance", "N/A").replace("_", " ").title()
                        val_num = t2.get("indicative_value", "N/A")
                        mat_years = t2.get("maturity_years", "N/A")
                        native = t2.get("native_status", "N/A").capitalize()

                        st.markdown(f"""
                        <div class="metric-grid">
                            <div class="carbon-card card-emerald">
                                <div>
                                    <div class="card-label">💰 {T['indicative_value']}</div>
                                    <div class="card-value" style="color: #065f46;">₹ {val_num}</div>
                                </div>
                                <div class="card-sub">Estimated per tCO2 equivalent</div>
                            </div>
                            <div class="carbon-card card-amber">
                                <div>
                                    <div class="card-label">⏳ {T['maturity_years']}</div>
                                    <div class="card-value" style="color: #92400e;">{mat_years} Years</div>
                                </div>
                                <div class="card-sub">Optimal harvest / biomass peak</div>
                            </div>
                            <div class="carbon-card card-blue">
                                <div>
                                    <div class="card-label">📜 {T['program_relevance']}</div>
                                    <div class="card-value" style="font-size: 1.15rem; color: #075985;">{relevance}</div>
                                </div>
                                <div class="card-sub">Agroforestry & boundary planting</div>
                            </div>
                            <div class="carbon-card card-forest">
                                <div>
                                    <div class="card-label">🇮🇳 {T['native_status']}</div>
                                    <div class="card-value" style="font-size: 1.15rem; color: #166534;">{native}</div>
                                </div>
                                <div class="card-sub">Ecological adaptation in India</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Top-3 Candidates Chart
                        st.markdown(f"##### {T['top3_title']}")
                        chart_data = pd.DataFrame([
                            {
                                "Species": f"{item['species_name']} ({item['hindi_name']})" if item.get('hindi_name') else item['species_name'],
                                "Probability (%)": round(item['probability'] * 100, 1)
                            }
                            for item in res["top3"]
                        ])

                        bar_chart = alt.Chart(chart_data).mark_bar(
                            color="#10b981",
                            cornerRadiusTopRight=5,
                            cornerRadiusBottomRight=5
                        ).encode(
                            x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title="Probability (%)"),
                            y=alt.Y("Species:N", sort="-x", title="Candidate"),
                            tooltip=["Species", "Probability (%)"]
                        ).properties(height=140)
                        st.altair_chart(bar_chart, use_container_width=True)

                        # Registry Framework & Field Notes
                        gov_url = t2.get("gov_url", "https://www.moefcc-gcp.in/")
                        gov_prog = t2.get("gov_program", "Green Credit Programme (MoEFCC)")
                        st.markdown(f"🏛️ **{T['gov_program']}:** [{gov_prog}]({gov_url})")

                        if t2.get("notes"):
                            st.info(f"🌾 **{T['notes']}:** {t2['notes']}")

                        # Mandatory Disclaimer Banner
                        st.markdown(f"<div class='disclaimer-banner'>{T['disclaimer_text']}</div>", unsafe_allow_html=True)

            # Technical Diagnostics Expander (Available once prediction exists)
            if res is not None:
                with st.expander(f"🛠️ {T['tech_details']}", expanded=False):
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.write(f"**{T['ood_distance']}:** `{res.get('ood_score', 0):.2f}`")
                        st.write(f"**{T['ood_boundary']}:** `{res.get('ood_threshold', 0):.2f}`")
                    with tc2:
                        st.write(f"**{T['entropy']}:** `{res.get('entropy', 0):.3f}`")
                        st.write(f"**Inference Time:** `{elapsed:.2f}s` (CPU)")
                    st.write(f"**{T['model_version']}:** `{res.get('model_version', 'v2')}`")

        # ---------------------------------------------------------------------
        # FULL WIDTH SECTION: Gemini AI Explainable Agroforestry Report
        # ---------------------------------------------------------------------
        res = st.session_state.get("last_prediction", None)
        if res is not None and res.get("status") == "confident":
            st.markdown("---")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #064e3b 0%, #0d9488 100%); padding: 1.2rem 1.6rem; border-radius: 14px; color: white; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: white; font-family: 'Outfit', sans-serif;">
                    {T['gemini_report_title']}
                </h3>
                <p style="margin: 0.3rem 0 0 0; color: #ccfbf1; font-size: 0.95rem;">
                    {T['gemini_report_desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            if not gemini_api_key:
                st.info(T["gemini_api_key_missing"])
            else:
                report_key = f"gemini_report_{res['species_name']}_{res['confidence']:.4f}_{lang}"

                t2 = res.get("tier2", {})
                if st.button(T["gemini_report_btn"], type="primary", use_container_width=True, key="gemini_btn"):
                    with st.spinner(T["gemini_report_generating"]):
                        report_md = generate_tree_report(
                            species_name=res["species_name"],
                            scientific_name=res["scientific_name"],
                            hindi_name=res.get("hindi_name", ""),
                            confidence=res["confidence"],
                            tier2_info=t2,
                            top3=res["top3"],
                            lang=lang,
                        )
                        st.session_state[report_key] = report_md

                if report_key in st.session_state:
                    report_md = st.session_state[report_key]
                    st.markdown(f"<div class='ai-report-card'>", unsafe_allow_html=True)
                    st.markdown(report_md)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.download_button(
                        label=T["gemini_report_download"],
                        data=report_md,
                        file_name=f"{res['species_name'].replace(' ', '_')}_Farmer_Advisory.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

    # =========================================================================
    # TAB 2: Batch Evaluation (Bulk Processing)
    # =========================================================================
    with tab_batch:
        st.markdown(f"### {T['tab_batch']}")
        st.markdown("Upload a set of field photos to rapidly classify species, flag out-of-distribution images, and estimate portfolio carbon potential.")

        batch_files = st.file_uploader(
            T["batch_upload_label"],
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if batch_files and len(batch_files) > 0:
            st.success(f"📁 **{len(batch_files)} image(s)** ready for batch processing.")
            if st.button(T["batch_run_btn"], type="primary", use_container_width=True):
                results_list = []
                progress_bar = st.progress(0, text=T["batch_progress"])
                
                start_batch = time.time()
                for i, b_file in enumerate(batch_files):
                    try:
                        b_img = Image.open(b_file)
                        pred = service.predict_image(b_img)
                        if pred.get("status") == "error":
                            status_val = "ERROR"
                            top_spec = "N/A"
                            top_conf = 0.0
                            ood_val = 0.0
                            carbon_est = "N/A"
                        else:
                            status_val = pred["status"].upper()
                            top_spec = f"{pred['species_name']} ({pred.get('hindi_name', '')})"
                            top_conf = round(pred["confidence"] * 100, 1)
                            ood_val = round(pred["ood_score"], 2)
                            carbon_est = pred.get("tier2", {}).get("indicative_value", "N/A")
                    except Exception as ex:
                        status_val = "CORRUPTED"
                        top_spec = "N/A"
                        top_conf = 0.0
                        ood_val = 0.0
                        carbon_est = "N/A"

                    results_list.append({
                        "Filename": b_file.name,
                        "Status": status_val,
                        "Species Prediction": top_spec,
                        "Confidence (%)": top_conf,
                        "OOD Distance": ood_val,
                        "Carbon Value (INR/tCO2)": carbon_est
                    })
                    progress_bar.progress((i + 1) / len(batch_files), text=f"Processing {i+1}/{len(batch_files)} images...")

                total_time = time.time() - start_batch
                df_results = pd.DataFrame(results_list)
                
                st.markdown(f"### {T['batch_results_title']}")
                
                # Batch quick stats
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    st.metric("Total Images Processed", len(batch_files))
                with b_col2:
                    conf_count = sum(1 for r in results_list if r["Status"] == "CONFIDENT")
                    st.metric("Confident Tree Matches", f"{conf_count} / {len(batch_files)}")
                with b_col3:
                    st.metric("Processing Speed", f"{total_time/len(batch_files):.2f}s / image")

                st.dataframe(df_results, use_container_width=True)

                # Download CSV
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=T["download_csv"],
                    data=csv_buffer.getvalue(),
                    file_name="vrikshacarbon_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
