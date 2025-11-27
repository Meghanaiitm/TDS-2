import os
import logging
import json
import google.generativeai as genai

logger = logging.getLogger("llm_agent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def ask_llm_for_action(page_text: str, pre_text: str = None):
    if not GEMINI_API_KEY:
        logger.error("Gemini API key missing")
        return None

    prompt = f"""
You analyze quiz questions and return ONLY a JSON object describing the next action.

JSON FORMAT:
{{
  "action": "sum | max | min | mean | count | chart | return_text | pdf_read",
  "column": "optional column name",
  "cutoff": number,
  "page": number
}}

INSTRUCTION:
{pre_text or ""}

PAGE_TEXT:
{page_text[:4000]}
"""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"temperature": 0, "response_mime_type": "application/json"}
        )

        response = model.generate_content(prompt)
        return json.loads(response.text)

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None
