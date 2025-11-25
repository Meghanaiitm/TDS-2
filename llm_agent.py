# llm_agent.py
import os
import logging
from typing import Optional
import requests
import json

logger = logging.getLogger("llm_agent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

def ask_llm_for_action(page_text: str, pre_text: Optional[str] = None) -> Optional[dict]:
    """
    Call Gemini to interpret instructions and return JSON action.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY missing in env")
        return None

    prompt = f"""
You are a helper that returns ONLY a JSON object describing the next action.
Possible fields: action, column, page, cutoff.
Instruction + page text:
{pre_text or ""}
{page_text}
Respond with JSON only.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            f"{API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        model_reply = data["candidates"][0]["content"]["parts"][0]["text"]

        parsed = json.loads(model_reply)
        return parsed

    except Exception as e:
        logger.exception("Gemini LLM call failed: %s", e)
        return None
