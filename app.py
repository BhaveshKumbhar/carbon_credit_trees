"""
Streamlit demo app for the Carbon-Credit Tree Identification project —
forest/agriculture themed.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

All model/lookup/confidence logic lives in identify_tree.py — this file
is purely presentation.
"""

import streamlit as st
from PIL import Image
from identify_tree import identify

st.set_page_config(
    page_title="Tree Species & Carbon Credit Identifier",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# THEME — forest / agriculture palette, injected once
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

:root {
    --forest-deep: #1b4332;
    --forest-mid: #2d6a4f;
    --forest-leaf: #40916c;
    --forest-light: #74c69d;
    --forest-pale: #d8f3dc;
    --earth-brown: #7f5539;
    --earth-tan: #b08968;
    --harvest-gold: #e9b44c;
    --cream: #faf6ee;
    --ink: #1c2b22;
}

html, body, [class*="css"]  {
    font-family: 'Nunito Sans', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f4f1e8 0%, #faf6ee 100%);
}

/* Safety net: every custom card/content block gets an EXPLICIT text color
   rather than inheriting Streamlit's ambient theme color (which is near-
   white in dark mode and was making text invisible on our light cards). */
.result-card, .result-card *,
.stat-box, .stat-box *,
.disclaimer-box, .disclaimer-box *,
.notes-text,
.hero-subtitle {
    color: var(--ink);
}

/* ---------- Animations ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gentleSway {
    0%, 100% { transform: rotate(-1.5deg); }
    50%      { transform: rotate(1.5deg); }
}
.hero-banner, .result-card, .stat-row, .disclaimer-box, .reg-badge, .stExpander {
    animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1);
}
.leaf-divider {
    text-align: center;
    font-size: 1.3rem;
    margin: 0.6rem 0 1.1rem 0;
    opacity: 0.7;
    animation: gentleSway 3.5s ease-in-out infinite;
}

/* ---------- Hero banner ---------- */
.hero-banner {
    background: linear-gradient(120deg, var(--forest-deep) 0%, var(--forest-mid) 45%, var(--forest-leaf) 85%, var(--harvest-gold) 140%);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(27, 67, 50, 0.3);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "🌿🌾🌳🍃🌱";
    position: absolute;
    right: -10px; top: -10px;
    font-size: 5rem;
    opacity: 0.14;
    letter-spacing: 8px;
    transform: rotate(8deg);
}
.hero-title {
    font-family: 'Fraunces', serif;
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
.hero-subtitle {
    font-size: 1.02rem;
    margin-top: 0.4rem;
    max-width: 640px;
    color: #eaf6ee !important;
}

/* ---------- Info expander card ---------- */
.stExpander {
    border: 1px solid var(--forest-pale) !important;
    border-radius: 12px !important;
    background: #B65949;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--forest-leaf);
    border-radius: 14px;
    padding: 1rem;
    background: #247F62;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--harvest-gold);
    background: #2EA34D;
}

/* ---------- Result card ---------- */
.result-card {
    background: linear-gradient(160deg, #ffffff 0%, #f7fbf6 100%);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 6px 18px rgba(27, 67, 50, 0.12);
    border-left: 7px solid var(--forest-leaf);
    margin-bottom: 1rem;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.result-card:hover {
    box-shadow: 0 10px 26px rgba(27, 67, 50, 0.18);
    transform: translateY(-2px);
}
.result-card.medium { border-left-color: var(--harvest-gold); }
.result-card.uncertain { border-left-color: #a08d78; }

.species-name {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--forest-deep) !important;
    margin: 0;
}
.species-hindi {
    color: var(--earth-brown) !important;
    font-size: 1rem;
    font-style: italic;
}
.result-message {
    color: var(--ink) !important;
    font-size: 0.96rem;
    margin-top: 0.6rem !important;
    line-height: 1.5;
}
.notes-text {
    color: #5c5346 !important;
    font-size: 0.87rem;
    font-style: italic;
    margin-top: 0.4rem !important;
}
.confidence-pill {
    display: inline-block;
    padding: 0.28rem 0.8rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #ffffff !important;
    background: linear-gradient(90deg, var(--forest-leaf), var(--forest-mid));
    margin-top: 0.4rem;
    box-shadow: 0 2px 8px rgba(64, 145, 108, 0.35);
}
.confidence-pill.medium {
    background: linear-gradient(90deg, var(--harvest-gold), #f4a261);
    color: var(--ink) !important;
    box-shadow: 0 2px 8px rgba(233, 180, 76, 0.4);
}

/* ---------- Stat boxes ---------- */
.stat-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-box {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(160deg, var(--forest-pale) 0%, #c3e8cf 100%);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 3px 10px rgba(27, 67, 50, 0.08);
}
.stat-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 18px rgba(27, 67, 50, 0.18);
}
.stat-box .label {
    font-size: 0.78rem;
    color: var(--forest-deep) !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700;
}
.stat-box .value {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    color: var(--forest-deep) !important;
    font-weight: 700;
}

/* ---------- Registration badge ---------- */
.reg-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.9rem;
    margin: 0.4rem 0;
}
.reg-common { background: linear-gradient(90deg, #d8f3dc, #b7e4c7); color: var(--forest-deep) !important; }
.reg-sometimes { background: linear-gradient(90deg, #fdebd0, #fbdba0); color: #8a5a00 !important; }
.reg-rare { background: linear-gradient(90deg, #ece7e1, #ddd5c9); color: #5c5346 !important; }

/* ---------- Gov link buttons ---------- */
.gov-link-btn {
    display: inline-block;
    background: linear-gradient(120deg, var(--earth-brown), var(--earth-tan));
    color: #ffffff !important;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    text-decoration: none !important;
    font-weight: 700;
    font-size: 0.88rem;
    margin: 0.3rem 0.5rem 0.3rem 0;
    box-shadow: 0 3px 10px rgba(127, 85, 57, 0.3);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.gov-link-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(127, 85, 57, 0.4);
}

/* ---------- Disclaimer box ---------- */
.disclaimer-box {
    background: linear-gradient(120deg, #fff7e6, #fdf0d0);
    border: 1px solid var(--harvest-gold);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: #6b4c00 !important;
    margin-top: 0.6rem;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--forest-deep) 0%, var(--forest-mid) 100%);
}
[data-testid="stSidebar"] * { color: #f1faee !important; }
.sidebar-species {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.92rem;
    transition: background 0.2s ease, transform 0.2s ease;
}
.sidebar-species:hover {
    background: rgba(255,255,255,0.18);
    transform: translateX(4px);
}

/* ---------- Footer ---------- */
.footer-note {
    text-align: center;
    color: var(--earth-brown) !important;
    font-size: 0.85rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--forest-pale);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_html(html):
    """Streamlit's markdown renderer treats any line indented 4+ spaces as
    a preformatted code block — which is what caused the stray '</div>'
    to render as visible text instead of being interpreted as HTML. This
    strips leading whitespace from every line before rendering, so
    multi-line f-strings can stay readably indented in the source without
    breaking at render time."""
    st.markdown("\n".join(line.strip() for line in html.strip().splitlines()), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
SPECIES_ICONS = {
    "Eucalyptus": "🌲", "Subabul": "🌿", "Shisham": "🌳", "Poplar": "🌲",
    "Casuarina": "🌴", "Teak": "🪵", "Bamboo": "🎋", "Mango": "🥭",
    "Jackfruit": "🟢", "Neem": "🍃",
}

with st.sidebar:
    st.markdown("### 🌱 About this tool")
    st.write(
        "Upload a photo of a tree and this tool identifies its species, then shows "
        "whether it's commonly used in Indian carbon-credit agroforestry programs."
    )
    st.markdown("### 🌳 Species it recognizes")
    for name, icon in SPECIES_ICONS.items():
        st.markdown(f"<div class='sidebar-species'>{icon} &nbsp; {name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(
        "⚠️ Identifies species only — does not determine actual carbon-credit "
        "eligibility for your land. Contact a local FPO or aggregator for that."
    )

# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
render_html(
    """
    <div class="hero-banner">
        <p class="hero-title">🌳 Tree Species & Carbon Credit Identifier</p>
        <p class="hero-subtitle">Snap a photo of a tree — find out what it is, and
        whether it could be a source of carbon-credit income under India's
        agroforestry programs.</p>
    </div>
    """
)

render_html('<div class="leaf-divider">🍃 • 🌱 • 🍃</div>')

with st.expander("ℹ️ What this tool does and doesn't do"):
    st.write(
        "This tool identifies the **species** of a tree from a photo, and tells you whether "
        "that species is commonly used in Indian agroforestry/carbon programs, with an "
        "indicative earning range.\n\n"
        "**It does NOT determine whether your specific tree is eligible for carbon credits.** "
        "Eligibility depends on scheme registration, land documentation, plantation records, "
        "and verification by an accredited agency — none of which a photo can show. "
        "To find out about eligibility for your land, contact a local Farmer Producer "
        "Organisation (FPO) or a carbon-credit aggregator."
    )

# ---------------------------------------------------------------------
# UPLOAD + RESULTS
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("📷 Upload a photo of a tree", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded photo", width='stretch')

    with col2:
        with st.spinner("🔍 Examining leaves, bark, and canopy shape..."):
            uploaded_file.seek(0)
            result = identify(uploaded_file)

        if result["status"] == "uncertain":
            render_html(
                f"""<div class="result-card uncertain">
                        <p class="species-name">🤔 Not sure yet</p>
                        <p class="result-message">{result['message']}</p>
                    </div>"""
            )
            st.write("Closest guesses (low confidence):")
            for species, prob in result["predictions"]:
                st.write(f"- {species}: {prob*100:.0f}%")

        else:
            card_class = "" if result["status"] == "confident" else "medium"
            pill_class = "" if result["status"] == "confident" else "medium"
            icon = SPECIES_ICONS.get(result["predicted_species"], "🌳")

            hindi = f"<span class='species-hindi'>({result['hindi_name']})</span>" if result["hindi_name"] else ""
            extra_line = ""
            if result["status"] == "medium_confidence":
                extra_line = f"<p class='result-message'>{result['message']}</p>"
            render_html(f"""
                <div class="result-card {card_class}">
                    <p class="species-name">{icon} {result['predicted_species']} {hindi}</p>
                    <span class="confidence-pill {pill_class}">{result['confidence']*100:.0f}% confident</span>
                    {extra_line}
                </div>
            """)

            render_html(f"""
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="label">💰 Earning</div>
                        <div class="value">₹{result['indicative_value_range_inr_per_tonne']}</div>
                        <div class="label">per tonne CO₂</div>
                    </div>
                    <div class="stat-box">
                        <div class="label">⏳ Maturity</div>
                        <div class="value">{result['maturity_years']} yrs</div>
                    </div>
                </div>
            """)

            relevance = result["program_relevance"]
            badge_class, badge_label = {
                "commonly_registered": ("reg-common", "✅ Commonly registered in carbon programs"),
                "sometimes_registered": ("reg-sometimes", "🟡 Sometimes registered in carbon programs"),
                "rarely_registered": ("reg-rare", "⚪ Rarely part of formal carbon programs"),
            }.get(relevance, ("reg-rare", relevance))
            st.markdown(f"<span class='reg-badge {badge_class}'>{badge_label}</span>", unsafe_allow_html=True)

            if result["notes"]:
                render_html(f"<p class='notes-text'>🍂 {result['notes']}</p>")

            links_html = ""
            if result["gov_resource_url"]:
                links_html += f"<a class='gov-link-btn' href='{result['gov_resource_url']}' target='_blank'>🔗 {result['gov_resource_name']}</a>"
            links_html += f"<a class='gov-link-btn' href='{result['indian_carbon_market_url']}' target='_blank'>🔗 Indian Carbon Market (BEE)</a>"
            st.markdown(f"<div style='margin-top:0.6rem;'>{links_html}</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='disclaimer-box'>⚠️ {result['disclaimer']}</div>", unsafe_allow_html=True)

            with st.expander("See other possibilities"):
                for species, prob in result["predictions"][1:]:
                    icon2 = SPECIES_ICONS.get(species, "🌳")
                    st.write(f"{icon2} {species}: {prob*100:.0f}%")

render_html("""
    <div class="footer-note">🌾 Student data science project — species identification
    model trained on 10 common Indian carbon-credit-relevant tree species. 🌾</div>
""")





# """
# Streamlit demo app for the Carbon-Credit Tree Identification project —
# forest/agriculture themed.

# Run with:
#     pip install -r requirements.txt
#     streamlit run app.py

# All model/lookup/confidence logic lives in identify_tree.py — this file
# is purely presentation.
# """

# import streamlit as st
# from PIL import Image
# from identify_tree import identify

# st.set_page_config(
#     page_title="Tree Species & Carbon Credit Identifier",
#     page_icon="🌳",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------------------------------------------------------------------
# # THEME — forest / agriculture palette, injected once
# # ---------------------------------------------------------------------
# CUSTOM_CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

# :root {
#     --forest-deep: #1b4332;
#     --forest-mid: #2d6a4f;
#     --forest-leaf: #40916c;
#     --forest-light: #74c69d;
#     --forest-pale: #d8f3dc;
#     --earth-brown: #7f5539;
#     --earth-tan: #b08968;
#     --harvest-gold: #e9b44c;
#     --cream: #faf6ee;
#     --ink: #1c2b22;
# }

# html, body, [class*="css"]  {
#     font-family: 'Nunito Sans', sans-serif;
# }

# .stApp {
#     background: linear-gradient(180deg, #f4f1e8 0%, #faf6ee 100%);
# }

# /* Safety net: every custom card/content block gets an EXPLICIT text color
#    rather than inheriting Streamlit's ambient theme color (which is near-
#    white in dark mode and was making text invisible on our light cards). */
# .result-card, .result-card *,
# .stat-box, .stat-box *,
# .disclaimer-box, .disclaimer-box *,
# .notes-text,
# .hero-subtitle {
#     color: var(--ink);
# }

# /* ---------- Animations ---------- */
# @keyframes fadeInUp {
#     from { opacity: 0; transform: translateY(16px); }
#     to   { opacity: 1; transform: translateY(0); }
# }
# @keyframes gentleSway {
#     0%, 100% { transform: rotate(-1.5deg); }
#     50%      { transform: rotate(1.5deg); }
# }
# .hero-banner, .result-card, .stat-row, .disclaimer-box, .reg-badge, .stExpander {
#     animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1);
# }
# .leaf-divider {
#     text-align: center;
#     font-size: 1.3rem;
#     margin: 0.6rem 0 1.1rem 0;
#     opacity: 0.7;
#     animation: gentleSway 3.5s ease-in-out infinite;
# }

# /* ---------- Hero banner ---------- */
# .hero-banner {
#     background: linear-gradient(120deg, var(--forest-deep) 0%, var(--forest-mid) 45%, var(--forest-leaf) 85%, var(--harvest-gold) 140%);
#     border-radius: 18px;
#     padding: 2.2rem 2.4rem;
#     margin-bottom: 1.6rem;
#     box-shadow: 0 10px 30px rgba(27, 67, 50, 0.3);
#     position: relative;
#     overflow: hidden;
# }
# .hero-banner::before {
#     content: "🌿🌾🌳🍃🌱";
#     position: absolute;
#     right: -10px; top: -10px;
#     font-size: 5rem;
#     opacity: 0.14;
#     letter-spacing: 8px;
#     transform: rotate(8deg);
# }
# .hero-title {
#     font-family: 'Fraunces', serif;
#     color: #ffffff;
#     font-size: 2.1rem;
#     font-weight: 700;
#     margin: 0;
#     text-shadow: 0 2px 10px rgba(0,0,0,0.25);
# }
# .hero-subtitle {
#     font-size: 1.02rem;
#     margin-top: 0.4rem;
#     max-width: 640px;
#     color: #eaf6ee !important;
# }

# /* ---------- Info expander card ---------- */
# .stExpander {
#     border: 1px solid var(--forest-pale) !important;
#     border-radius: 12px !important;
#     background: #ffffffcc;
# }

# /* ---------- Uploader ---------- */
# [data-testid="stFileUploader"] {
#     border: 2px dashed var(--forest-leaf);
#     border-radius: 14px;
#     padding: 1rem;
#     background: #ffffffb0;
#     transition: border-color 0.2s ease, background 0.2s ease;
# }
# [data-testid="stFileUploader"]:hover {
#     border-color: var(--harvest-gold);
#     background: #ffffffe0;
# }

# /* ---------- Result card ---------- */
# .result-card {
#     background: linear-gradient(160deg, #ffffff 0%, #f7fbf6 100%);
#     border-radius: 16px;
#     padding: 1.4rem 1.6rem;
#     box-shadow: 0 6px 18px rgba(27, 67, 50, 0.12);
#     border-left: 7px solid var(--forest-leaf);
#     margin-bottom: 1rem;
#     transition: box-shadow 0.25s ease, transform 0.25s ease;
# }
# .result-card:hover {
#     box-shadow: 0 10px 26px rgba(27, 67, 50, 0.18);
#     transform: translateY(-2px);
# }
# .result-card.medium { border-left-color: var(--harvest-gold); }
# .result-card.uncertain { border-left-color: #a08d78; }
# .result-card.invalid { border-left-color: #c1443f; background: linear-gradient(160deg, #ffffff 0%, #fdf3f2 100%); }

# .species-name {
#     font-family: 'Fraunces', serif;
#     font-size: 1.6rem;
#     font-weight: 700;
#     color: var(--forest-deep) !important;
#     margin: 0;
# }
# .species-hindi {
#     color: var(--earth-brown) !important;
#     font-size: 1rem;
#     font-style: italic;
# }
# .result-message {
#     color: var(--ink) !important;
#     font-size: 0.96rem;
#     margin-top: 0.6rem !important;
#     line-height: 1.5;
# }
# .notes-text {
#     color: #5c5346 !important;
#     font-size: 0.87rem;
#     font-style: italic;
#     margin-top: 0.4rem !important;
# }
# .confidence-pill {
#     display: inline-block;
#     padding: 0.28rem 0.8rem;
#     border-radius: 999px;
#     font-weight: 700;
#     font-size: 0.85rem;
#     color: #ffffff !important;
#     background: linear-gradient(90deg, var(--forest-leaf), var(--forest-mid));
#     margin-top: 0.4rem;
#     box-shadow: 0 2px 8px rgba(64, 145, 108, 0.35);
# }
# .confidence-pill.medium {
#     background: linear-gradient(90deg, var(--harvest-gold), #f4a261);
#     color: var(--ink) !important;
#     box-shadow: 0 2px 8px rgba(233, 180, 76, 0.4);
# }

# /* ---------- Stat boxes ---------- */
# .stat-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
# .stat-box {
#     flex: 1;
#     min-width: 140px;
#     background: linear-gradient(160deg, var(--forest-pale) 0%, #c3e8cf 100%);
#     border-radius: 12px;
#     padding: 0.9rem 1rem;
#     text-align: center;
#     transition: transform 0.2s ease, box-shadow 0.2s ease;
#     box-shadow: 0 3px 10px rgba(27, 67, 50, 0.08);
# }
# .stat-box:hover {
#     transform: translateY(-4px);
#     box-shadow: 0 8px 18px rgba(27, 67, 50, 0.18);
# }
# .stat-box .label {
#     font-size: 0.78rem;
#     color: var(--forest-deep) !important;
#     text-transform: uppercase;
#     letter-spacing: 0.04em;
#     font-weight: 700;
# }
# .stat-box .value {
#     font-family: 'Fraunces', serif;
#     font-size: 1.35rem;
#     color: var(--forest-deep) !important;
#     font-weight: 700;
# }

# /* ---------- Registration badge ---------- */
# .reg-badge {
#     display: inline-block;
#     padding: 0.35rem 0.9rem;
#     border-radius: 8px;
#     font-weight: 700;
#     font-size: 0.9rem;
#     margin: 0.4rem 0;
# }
# .reg-common { background: linear-gradient(90deg, #d8f3dc, #b7e4c7); color: var(--forest-deep) !important; }
# .reg-sometimes { background: linear-gradient(90deg, #fdebd0, #fbdba0); color: #8a5a00 !important; }
# .reg-rare { background: linear-gradient(90deg, #ece7e1, #ddd5c9); color: #5c5346 !important; }

# /* ---------- Gov link buttons ---------- */
# .gov-link-btn {
#     display: inline-block;
#     background: linear-gradient(120deg, var(--earth-brown), var(--earth-tan));
#     color: #ffffff !important;
#     padding: 0.5rem 1rem;
#     border-radius: 8px;
#     text-decoration: none !important;
#     font-weight: 700;
#     font-size: 0.88rem;
#     margin: 0.3rem 0.5rem 0.3rem 0;
#     box-shadow: 0 3px 10px rgba(127, 85, 57, 0.3);
#     transition: transform 0.18s ease, box-shadow 0.18s ease;
# }
# .gov-link-btn:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 6px 16px rgba(127, 85, 57, 0.4);
# }

# /* ---------- Disclaimer box ---------- */
# .disclaimer-box {
#     background: linear-gradient(120deg, #fff7e6, #fdf0d0);
#     border: 1px solid var(--harvest-gold);
#     border-radius: 10px;
#     padding: 0.8rem 1rem;
#     font-size: 0.9rem;
#     color: #6b4c00 !important;
#     margin-top: 0.6rem;
# }

# /* ---------- Sidebar ---------- */
# [data-testid="stSidebar"] {
#     background: linear-gradient(180deg, var(--forest-deep) 0%, var(--forest-mid) 100%);
# }
# [data-testid="stSidebar"] * { color: #f1faee !important; }
# .sidebar-species {
#     background: rgba(255,255,255,0.08);
#     border-radius: 8px;
#     padding: 0.45rem 0.7rem;
#     margin-bottom: 0.35rem;
#     font-size: 0.92rem;
#     transition: background 0.2s ease, transform 0.2s ease;
# }
# .sidebar-species:hover {
#     background: rgba(255,255,255,0.18);
#     transform: translateX(4px);
# }

# /* ---------- Footer ---------- */
# .footer-note {
#     text-align: center;
#     color: var(--earth-brown) !important;
#     font-size: 0.85rem;
#     margin-top: 2rem;
#     padding-top: 1rem;
#     border-top: 1px solid var(--forest-pale);
# }
# </style>
# """
# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# def render_html(html):
#     """Streamlit's markdown renderer treats any line indented 4+ spaces as
#     a preformatted code block — which is what caused the stray '</div>'
#     to render as visible text instead of being interpreted as HTML. This
#     strips leading whitespace from every line before rendering, so
#     multi-line f-strings can stay readably indented in the source without
#     breaking at render time."""
#     st.markdown("\n".join(line.strip() for line in html.strip().splitlines()), unsafe_allow_html=True)

# # ---------------------------------------------------------------------
# # SIDEBAR
# # ---------------------------------------------------------------------
# SPECIES_ICONS = {
#     "Eucalyptus": "🌲", "Subabul": "🌿", "Shisham": "🌳", "Poplar": "🌲",
#     "Casuarina": "🌴", "Teak": "🪵", "Bamboo": "🎋", "Mango": "🥭",
#     "Jackfruit": "🟢", "Neem": "🍃",
# }

# with st.sidebar:
#     st.markdown("### 🌱 About this tool")
#     st.write(
#         "Upload a photo of a tree and this tool identifies its species, then shows "
#         "whether it's commonly used in Indian carbon-credit agroforestry programs."
#     )
#     st.markdown("### 🌳 Species it recognizes")
#     for name, icon in SPECIES_ICONS.items():
#         st.markdown(f"<div class='sidebar-species'>{icon} &nbsp; {name}</div>", unsafe_allow_html=True)
#     st.markdown("---")
#     st.caption(
#         "⚠️ Identifies species only — does not determine actual carbon-credit "
#         "eligibility for your land. Contact a local FPO or aggregator for that."
#     )

# # ---------------------------------------------------------------------
# # HERO
# # ---------------------------------------------------------------------
# render_html(
#     """
#     <div class="hero-banner">
#         <p class="hero-title">🌳 Tree Species & Carbon Credit Identifier</p>
#         <p class="hero-subtitle">Snap a photo of a tree — find out what it is, and
#         whether it could be a source of carbon-credit income under India's
#         agroforestry programs.</p>
#     </div>
#     """
# )

# render_html('<div class="leaf-divider">🍃 • 🌱 • 🍃</div>')

# with st.expander("ℹ️ What this tool does and doesn't do"):
#     st.write(
#         "This tool identifies the **species** of a tree from a photo, and tells you whether "
#         "that species is commonly used in Indian agroforestry/carbon programs, with an "
#         "indicative earning range.\n\n"
#         "**It does NOT determine whether your specific tree is eligible for carbon credits.** "
#         "Eligibility depends on scheme registration, land documentation, plantation records, "
#         "and verification by an accredited agency — none of which a photo can show. "
#         "To find out about eligibility for your land, contact a local Farmer Producer "
#         "Organisation (FPO) or a carbon-credit aggregator."
#     )

# # ---------------------------------------------------------------------
# # UPLOAD + RESULTS
# # ---------------------------------------------------------------------
# uploaded_file = st.file_uploader("📷 Upload a photo of a tree", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     col1, col2 = st.columns([1, 1.3])

#     with col1:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded photo", use_container_width=True)

#     with col2:
#         with st.spinner("🔍 Examining leaves, bark, and canopy shape..."):
#             uploaded_file.seek(0)
#             result = identify(uploaded_file)

#         if result["status"] == "invalid_input":
#             render_html(
#                 f"""<div class="result-card invalid">
#                         <p class="species-name">🚫 Not a recognized tree</p>
#                         <p class="result-message">{result['message']}</p>
#                     </div>"""
#             )

#         elif result["status"] == "uncertain":
#             render_html(
#                 f"""<div class="result-card uncertain">
#                         <p class="species-name">🤔 Not sure yet</p>
#                         <p class="result-message">{result['message']}</p>
#                     </div>"""
#             )
#             st.write("Closest guesses (low confidence):")
#             for species, prob in result["predictions"]:
#                 st.write(f"- {species}: {prob*100:.0f}%")

#         else:
#             card_class = "" if result["status"] == "confident" else "medium"
#             pill_class = "" if result["status"] == "confident" else "medium"
#             icon = SPECIES_ICONS.get(result["predicted_species"], "🌳")

#             hindi = f"<span class='species-hindi'>({result['hindi_name']})</span>" if result["hindi_name"] else ""
#             extra_line = ""
#             if result["status"] == "medium_confidence":
#                 extra_line = f"<p class='result-message'>{result['message']}</p>"
#             render_html(f"""
#                 <div class="result-card {card_class}">
#                     <p class="species-name">{icon} {result['predicted_species']} {hindi}</p>
#                     <span class="confidence-pill {pill_class}">{result['confidence']*100:.0f}% confident</span>
#                     {extra_line}
#                 </div>
#             """)

#             render_html(f"""
#                 <div class="stat-row">
#                     <div class="stat-box">
#                         <div class="label">💰 Earning</div>
#                         <div class="value">₹{result['indicative_value_range_inr_per_tonne']}</div>
#                         <div class="label">per tonne CO₂</div>
#                     </div>
#                     <div class="stat-box">
#                         <div class="label">⏳ Maturity</div>
#                         <div class="value">{result['maturity_years']} yrs</div>
#                     </div>
#                 </div>
#             """)

#             relevance = result["program_relevance"]
#             badge_class, badge_label = {
#                 "commonly_registered": ("reg-common", "✅ Commonly registered in carbon programs"),
#                 "sometimes_registered": ("reg-sometimes", "🟡 Sometimes registered in carbon programs"),
#                 "rarely_registered": ("reg-rare", "⚪ Rarely part of formal carbon programs"),
#             }.get(relevance, ("reg-rare", relevance))
#             st.markdown(f"<span class='reg-badge {badge_class}'>{badge_label}</span>", unsafe_allow_html=True)

#             if result["notes"]:
#                 render_html(f"<p class='notes-text'>🍂 {result['notes']}</p>")

#             links_html = ""
#             if result["gov_resource_url"]:
#                 links_html += f"<a class='gov-link-btn' href='{result['gov_resource_url']}' target='_blank'>🔗 {result['gov_resource_name']}</a>"
#             links_html += f"<a class='gov-link-btn' href='{result['indian_carbon_market_url']}' target='_blank'>🔗 Indian Carbon Market (BEE)</a>"
#             st.markdown(f"<div style='margin-top:0.6rem;'>{links_html}</div>", unsafe_allow_html=True)

#             st.markdown(f"<div class='disclaimer-box'>⚠️ {result['disclaimer']}</div>", unsafe_allow_html=True)

#             with st.expander("See other possibilities"):
#                 for species, prob in result["predictions"][1:]:
#                     icon2 = SPECIES_ICONS.get(species, "🌳")
#                     st.write(f"{icon2} {species}: {prob*100:.0f}%")

# render_html("""
#     <div class="footer-note">🌾 Student data science project — species identification
#     model trained on 10 common Indian carbon-credit-relevant tree species. 🌾</div>
# """)