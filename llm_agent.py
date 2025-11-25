# llm_agent.py
import os
import logging
import json
import google.generativeai as genai
from google.api_core.exceptions import NotFound

logger = logging.getLogger("llm_agent")

# 1. Get Key and Clean it (fixes copy-paste whitespace issues)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# 2. Configure SDK
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def ask_llm_for_action(page_text: str, pre_text: str = None) -> dict | None:
    """
    Uses Gemini 1.5 Flash to determine the next action.
    Returns a Python dictionary (parsed from JSON).
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is missing/empty.")
        return None

    try:
        # Use JSON mode for reliable output
        generation_config = {
            "temperature": 0.0,
            "response_mime_type": "application/json"
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )

        prompt = f"""
        You are an autonomous agent solving a data quiz.
        Analyze the instruction and page text to decide the next action.
        
        Return ONLY a JSON object with these keys:
        - "action": One of ["return_text", "sum", "mean", "max", "min", "count", "chart", "download_return_file", "pdf_read"]
        - "column": (string) The specific column name to operate on, if applicable.
        - "cutoff": (number) Filter value, if instruction says "greater than X".
        - "page": (number) Page number if reading a PDF.

        INSTRUCTION:
        {pre_text or "No specific instruction."}

        PAGE CONTENT:
        {page_text[:5000]} 
        """
        # Note: Truncated page_text to 5000 chars to save tokens/speed

        response = model.generate_content(prompt)
        
        # Parse the response
        result = json.loads(response.text)
        logger.info(f"LLM Decision: {result}")
        return result

    except Exception as e:
        logger.error(f"Gemini LLM call failed: {e}")
        return None
