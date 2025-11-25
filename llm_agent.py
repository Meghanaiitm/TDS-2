# llm_agent.py
import os
import json
import logging
import requests

logger = logging.getLogger("llm_agent")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not set")

# WORKING FREE MODEL
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def ask_llm_for_action(page_text: str, pre_text: str = None):
    if not API_KEY:
        logger.error("GEMINI_API_KEY missing")
        return None

    prompt = (
        "You are a helper that converts instructions into a JSON object. "
        "Return ONLY valid JSON. Keys: action, column, page, cutoff.\n\n"
        f"{pre_text or ''}\n{page_text}\n\nJSON:"
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
            timeout=20
        )
        response.raise_for_status()

        data = response.json()

        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]

        return json.loads(ai_text)

    except Exception as e:
        logger.exception(f"Gemini call failed: {e}")
        return None
