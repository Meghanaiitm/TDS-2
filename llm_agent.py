# llm_agent.py
import os
import json
import logging
import requests

logger = logging.getLogger("llm_agent")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY is not set")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def ask_llm_for_action(page_text: str, pre_text: str = None):
    """
    Calls Gemini API and expects ONLY JSON output.
    """
    if not API_KEY:
        logger.error("No GEMINI_API_KEY set.")
        return None

    prompt = (
        "You are a helper that converts instructions into a structured JSON action.\n"
        "Return ONLY a JSON object.\n"
        "Possible keys: action, column, page, cutoff.\n\n"
        f"Instruction:\n{pre_text or ''}\n{page_text}\n\n"
        "Output JSON only."
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={API_KEY}",
            json=payload,
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]

        return json.loads(ai_text)  # parse JSON output

    except Exception as e:
        logger.exception(f"Gemini call failed: {e}")
        return None
