# llm_agent.py
import os
import logging
from typing import Optional
from openai import OpenAI
import json

logger = logging.getLogger("llm_agent")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

def ask_llm_for_action(page_text: str, pre_text: Optional[str] = None) -> Optional[dict]:
    """
    Send instructions + page text to OpenAI
    and receive a JSON action.
    """

    if not OPENAI_KEY:
        logger.error("OPENAI_API_KEY not set")
        return None

    prompt = (
        "You are a helper that converts a human instruction into a structured action.\n"
        "Respond ONLY with valid JSON.\n"
        "Possible fields: action, column, page, cutoff.\n\n"
        f"Instruction:\n{pre_text or ''}\n\nPage Text:\n{page_text}\n\nOutput JSON:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )

        txt = response.choices[0].message["content"].strip()
        return json.loads(txt)

    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return None
