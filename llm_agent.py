# llm_agent.py
import os
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger("llm_agent")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# New OpenAI client (works for 2024+)
client = OpenAI(api_key=OPENAI_KEY)


def ask_llm_for_action(page_text: str, pre_text: Optional[str] = None) -> Optional[dict]:
    """
    Converts a quiz instruction to a JSON action dict using OpenAI Chat Completions API.
    """

    if not OPENAI_KEY:
        logger.warning("OPENAI_API_KEY not set")
        return None

    prompt = (
        "You are a helper that converts a human instruction into a structured JSON action.\n"
        "Respond ONLY with a JSON object.\n\n"
        f"Page text:\n{page_text}\n\n"
        f"Instruction:\n{pre_text or ''}\n\n"
        "Output JSON:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        text = response.choices[0].message.content.strip()

        import json
        return json.loads(text)

    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return None
