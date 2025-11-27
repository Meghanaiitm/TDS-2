# llm_agent.py
import os
import logging
import json
import google.generativeai as genai
from google.api_core.exceptions import NotFound

logger = logging.getLogger("llm_agent")

# Load and clean API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("⚠️ GEMINI_API_KEY is missing! Quiz solving will fail.")

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def ask_llm_for_action(page_text: str, pre_text: str = None) -> dict | None:
    """
    Sends instruction + page text to Gemini and expects EXACT JSON output.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY missing.")
        return None

    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        )

        prompt = f"""
You are an autonomous agent solving an instruction-based data quiz.
Your job is to output a JSON object describing the next computational action.

VALID ACTIONS:
- "return_text"
- "sum"
- "mean"
- "max"
- "min"
- "count"
- "chart"
- "download_return_file"
- "pdf_read"

VALID OPTIONAL FIELDS:
- "column": column name to operate on
- "cutoff": numeric cutoff (e.g. > 50)
- "page": page number for PDFs

Return ONLY JSON. No explanation.

INSTRUCTION:
{pre_text or "None"}

PAGE TEXT:
{page_text[:8000]}
"""

        # Must wrap prompt in a list
        response = model.generate_content([prompt])

        # Extract clean JSON text
        raw = response.candidates[0].content.parts[0].text.strip()

        # Parse JSON
        result = json.loads(raw)

        logger.info(f"LLM decision → {result}")
        return result

    except NotFound:
        logger.error("❌ Gemini model not found. Check model name 'gemini-1.5-flash'.")
        return None

    except json.JSONDecodeError:
        logger.error(f"❌ LLM returned non-JSON: {raw}")
        return None

    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return None
