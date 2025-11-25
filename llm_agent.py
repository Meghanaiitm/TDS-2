# llm_agent.py
import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
import json

logger = logging.getLogger("llm_agent")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def ask_llm_for_action(page_text: str, pre_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Calls the LLM using the new OpenAI API (>=1.0).
    Returns a structured JSON as a Python dict.
    """

    if not OPENAI_KEY:
        logger.warning("OPENAI_API_KEY not set: cannot call LLM.")
        return None

    prompt = (
        "You are a helper that converts a human instruction into a structured JSON action.\n"
        "Fields allowed: action, column, page, cutoff.\n"
        "Respond ONLY with valid JSON.\n\n"
        f"Instruction:\n{pre_text or ''}\n{page_text}\n\n"
        "Output JSON:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You respond ONLY with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0,
        )

        text = response.choices[0].message.content.strip()

        # Ensure JSON decode
        return json.loads(text)

    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return None
