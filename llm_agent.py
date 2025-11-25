# llm_agent.py
import os
import logging
from typing import Optional
import requests
import json

logger = logging.getLogger("llm_agent")

AI_PIPE_KEY = os.getenv("OPENAI_API_KEY")  # AI Pipe key is stored here

API_URL = "https://api.aipipe.ai/v1/chat/completions"

def ask_llm_for_action(page_text: str, pre_text: Optional[str] = None) -> Optional[dict]:
    """
    Calls AI Pipe (NOT OpenAI) to interpret instructions into a JSON action.
    """

    if not AI_PIPE_KEY:
        logger.warning("AI PIPE KEY not set.")
        return None

    prompt = (
        "You are a helper that converts a human instruction into a structured action.\n"
        "Respond only with a JSON object.\n"
        "Possible fields: action, column, page, cutoff.\n\n"
        f"Instruction:\n{pre_text or ''}\n{page_text}\n\nOutput JSON:"
    )

    try:
        headers = {
            "Authorization": f"Bearer {AI_PIPE_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",   # AI Pipe supports this
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        # AI Pipe response structure mirrors OpenAI 1.x
        content = data["choices"][0]["message"]["content"]

        return json.loads(content)

    except Exception as e:
        logger.exception("AI Pipe call failed: %s", e)
        return None
