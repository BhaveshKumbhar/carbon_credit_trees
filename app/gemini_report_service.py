"""
gemini_report_service.py
========================
Generates comprehensive, farmer-friendly Explainable AI (XAI) reports for
identified tree species using the Google Gemini Generative AI API.

When the tree classifier confidently identifies a species, this module sends
a structured prompt to Gemini containing the species information, model
confidence, and carbon credit context.  Gemini returns a rich report covering:

    - Common & Scientific Name, Hindi Name
    - Growth Period & Lifecycle Stages
    - Medicinal / Traditional Value
    - Carbon Credit Registry Value in India
    - Minimum Land Area for Carbon Credit Qualification
    - Soil & Climate Suitability
    - Economic Value (Timber, Fruit, Resin, etc.)
    - Biodiversity & Ecological Benefits
    - Agroforestry Integration Guidance
    - Government Schemes & Subsidies

The report is formatted in Markdown and can be rendered directly in Streamlit.
"""

import google.generativeai as genai


def configure_gemini(api_key: str) -> None:
    """Configures the Gemini SDK with the provided API key."""
    genai.configure(api_key=api_key)


def generate_tree_report(
    species_name: str,
    scientific_name: str,
    hindi_name: str,
    confidence: float,
    tier2_info: dict,
    top3: list,
    lang: str = "en",
) -> str:
    """
    Generates a comprehensive Explainable AI report for the identified tree species.

    Args:
        species_name:    Common name of the identified tree (e.g., "Neem").
        scientific_name: Scientific binomial (e.g., "Azadirachta indica").
        hindi_name:      Hindi / local name (e.g., "नीम").
        confidence:      Model confidence as a float 0.0 – 1.0.
        tier2_info:      Dict with keys like program_relevance, indicative_value,
                         maturity_years, gov_program, native_status, notes.
        top3:            List of top-3 candidate dicts from the classifier.
        lang:            "en" for English or "hi" for Hindi report.

    Returns:
        Markdown-formatted string with the complete report.
    """

    # Build context string from model predictions
    top3_text = "\n".join(
        f"  {i+1}. {c['species_name']} ({c['scientific_name']}) — "
        f"{c['probability']*100:.1f}% confidence"
        for i, c in enumerate(top3)
    )

    language_instruction = (
        "Write the entire report in Hindi (हिंदी), using Devanagari script. "
        "Use simple Hindi that an Indian farmer can easily understand."
        if lang == "hi"
        else "Write the report in clear, simple English that an Indian farmer "
        "with basic literacy can understand. Avoid overly technical jargon."
    )

    prompt = f"""You are an expert botanist, agroforestry advisor, and carbon credit consultant 
specializing in Indian tree species. An AI tree identification model has confidently 
identified a tree from a photograph. Generate a comprehensive, farmer-friendly report 
about this tree species.

## Identified Tree Information
- **Common Name:** {species_name}
- **Scientific Name:** {scientific_name}
- **Hindi / Local Name:** {hindi_name}
- **AI Model Confidence:** {confidence*100:.1f}%
- **Top 3 Candidates from Model:**
{top3_text}

## Known Carbon Credit / Agroforestry Profile (from our database)
- **Program Relevance:** {tier2_info.get('program_relevance', 'N/A')}
- **Indicative Carbon Value:** INR {tier2_info.get('indicative_value', 'N/A')} per tonne CO2
- **Maturity Period:** {tier2_info.get('maturity_years', 'N/A')} years
- **Government Program:** {tier2_info.get('gov_program', 'N/A')}
- **Native Status:** {tier2_info.get('native_status', 'N/A')}
- **Field Notes:** {tier2_info.get('notes', 'N/A')}

## Instructions
{language_instruction}

Generate a detailed report covering ALL of the following sections. Use markdown 
formatting with headers (##), bullet points, and bold text for readability.

### Required Sections:

1. **🌳 Tree Identity & Overview**
   - Full botanical classification (Family, Genus, Species)
   - All common names (English, Hindi, Regional Indian languages)
   - Brief description of appearance (height, bark, leaves, flowers, fruit)

2. **📈 Growth Period & Lifecycle**
   - Germination to sapling timeline
   - Sapling to mature tree timeline
   - Expected maximum height and canopy spread
   - Lifespan of the tree
   - Best planting season in India

3. **💊 Medicinal & Traditional Value**
   - Known medicinal uses (Ayurvedic, traditional, modern research)
   - Parts of tree used (leaves, bark, fruit, seeds, gum, etc.)
   - Any FDA / AYUSH recognized medicinal properties
   - Traditional / cultural significance in India

4. **🌍 Carbon Credit Value in India**
   - Annual CO2 sequestration capacity (kg CO2/year per tree, approximate)
   - Value in Indian Carbon Credit Registry (INR per tonne CO2)
   - Eligible carbon credit programs (Verra VCS, Gold Standard, Indian Carbon Market, Green Credit Programme)
   - How farmers can register for carbon credits with this tree species
   - Current carbon credit market trends in India

5. **📐 Minimum Land Area for Carbon Credit Qualification**
   - Minimum plantation area required to qualify as a carbon credit reserve in India
   - Minimum number of trees per hectare for carbon credit registration
   - Spacing recommendations between trees
   - Land eligibility criteria (wasteland, farmland, community land, etc.)

6. **🌱 Soil & Climate Requirements**
   - Ideal soil type and pH range
   - Rainfall requirements (mm/year)
   - Temperature tolerance range
   - Drought resistance capability
   - Indian states / regions where this tree thrives best

7. **💰 Economic Value Beyond Carbon Credits**
   - Timber value and uses
   - Fruit / seed / flower commercial value
   - By-products (gum, resin, oil, fodder, etc.)
   - Market price ranges in India (approximate)
   - Employment generation potential

8. **🌿 Biodiversity & Ecological Benefits**
   - Wildlife supported (birds, insects, pollinators)
   - Soil improvement capabilities (nitrogen fixation, erosion control)
   - Water table impact
   - Microclimate benefits

9. **🚜 Agroforestry Integration Guide**
   - Compatible crops for intercropping
   - Recommended agroforestry models (boundary planting, block planting, etc.)
   - Companion tree species
   - Integration with livestock farming

10. **🏛️ Government Schemes & Subsidies**
    - Central government schemes (MGNREGA, Green India Mission, Sub-Mission on Agroforestry)
    - State-specific subsidies for planting this species
    - How to apply for subsidies
    - FPO (Farmer Producer Organization) benefits

11. **⚠️ Risks & Considerations**
    - Common diseases and pests
    - Invasive potential (if applicable)
    - Any environmental concerns
    - Legal restrictions on harvesting (if any)

12. **📋 Quick Reference Summary for Farmers**
    - A concise table summarizing: Growth period, Carbon value, Min land, Investment needed, 
      Expected returns, Best states to plant

End the report with a clear disclaimer that this is an AI-generated advisory report and 
farmers should consult local forest department / Krishi Vigyan Kendra (KVK) for 
region-specific guidance.
"""

    try:
        model = genai.GenerativeModel("gemini-3.8-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
            return (
                "❌ **Gemini API Error:** Invalid API key. Please check your "
                "Gemini API key in the sidebar and try again.\n\n"
                f"*Technical details:* `{error_msg}`"
            )
        return (
            f"❌ **Gemini Report Generation Failed**\n\n"
            f"Could not generate the AI report. Please check your internet "
            f"connection and API key.\n\n*Error:* `{error_msg}`"
        )
