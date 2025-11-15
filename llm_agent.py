# llm_agent.py
import os
import logging
from typing import Optional
import openai

logger = logging.getLogger("llm_agent")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def ask_llm_for_action(page_text: str, pre_text: Optional[str] = None) -> Optional[dict]:
    """
    Call LLM to interpret a complex instruction and return a JSON spec describing the action.
    Returns dict like {'action':'sum', 'column':'value', 'page':2, 'cutoff':38636}
    """
    if not OPENAI_KEY:
        logger.warning("OPENAI_API_KEY not set: cannot call LLM.")
        return None

    prompt = (
        "You are a helper that converts a human instruction into a structured action.\n"
        "Input page text (the quiz question + any instructions). Respond ONLY with a JSON object.\n"
        "Possible fields: action (sum/count/max/min/mean/chart/pdf_read/download_return_file/return_text), "
        "column (optional), page (optional), cutoff (optional).\n\n"
        f"Instruction:\n{pre_text or ''}\n{page_text}\n\n"
        "Output JSON:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # replace with available model
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        txt = resp.choices[0].message['content'].strip()
        # Expect JSON only
        import json
        parsed = json.loads(txt)
        return parsed
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return None
