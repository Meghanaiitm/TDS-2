# solver.py
import asyncio
import time
import logging
import re
import json
import requests
import pandas as pd
from urllib.parse import urljoin
from typing import Optional
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
import os

from utils import (
    parse_question_text,
    compute_answer_from_csv_bytes,
    compute_answer_from_excel_bytes,
    compute_answer_from_pdf_bytes,
    file_bytes_to_data_uri,
    df_to_chart_data_uri,
    safe_json_parse,
)
from llm_agent import ask_llm_for_action

# audio
import openai

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solver")

def solve_quiz_with_deadline(url: str, email: str, secret: str, start_time: float, max_seconds: int):
    deadline = start_time + max_seconds
    remaining = deadline - time.time()
    if remaining <= 3:
        logger.warning("Not enough time left to start solver.")
        return
    try:
        asyncio.run(_solve_quiz_full(url, email, secret, deadline))
    except Exception as e:
        logger.exception("Solver error: %s", e)

async def _solve_quiz_full(initial_url: str, email: str, secret: str, deadline: float):
    time_left = lambda: max(0, int(deadline - time.time()))
    current_url = initial_url
    visited = set()
    logger.info("Starting solver chain at %s", initial_url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        while current_url and time_left() > 6:
            if current_url in visited:
                logger.warning("Loop detected; stopping.")
                break
            visited.add(current_url)

            logger.info("Loading %s (time left %d)", current_url, time_left())
            try:
                await page.goto(current_url, wait_until="networkidle", timeout=45000)
            except PWTimeout:
                try:
                    await page.goto(current_url, wait_until="load", timeout=75000)
                except Exception as e:
                    logger.error("Page load failed: %s", e)
                    break
            except Exception as e:
                logger.error("Navigate error: %s", e)
                break

            await page.wait_for_timeout(300)

            try:
                page_text = await page.evaluate("() => document.body.innerText") or ""
            except Exception:
                page_text = ""

            try:
                pre_text = await page.eval_on_selector_all("pre", "nodes => nodes.map(n=>n.innerText).join('\\n\\n')")
            except Exception:
                pre_text = ""

            logger.info("Page snippet: %s", page_text[:300].replace("\n"," "))

            # detect submit (absolute / relative)
            submit_url = detect_submit_url(page_text, pre_text, current_url)
            if not submit_url:
                logger.error("Submit URL not found. Stopping.")
                break
            logger.info("Submit URL: %s", submit_url)

            # detect data file
            file_url = detect_file_url(page_text, pre_text)
            file_bytes = None
            file_ext = None
            if file_url:
                try:
                    r = requests.get(file_url, timeout=25)
                    if r.ok:
                        file_bytes = r.content
                        file_ext = file_url.rsplit(".",1)[-1].lower()
                        logger.info("Downloaded file: %s (%d bytes)", file_ext, len(file_bytes))
                except Exception as e:
                    logger.error("File download error: %s", e)

            # detect scrape instruction
            scrape_url = detect_scrape_url(page_text, current_url)
            secret_code = None
            if scrape_url:
                logger.info("Found scrape instruction -> visiting %s", scrape_url)
                secret_code = await scrape_secondary_page(scrape_url, page)

            # detect audio link on page (look for common audio file extensions)
            audio_url = detect_audio_url(page_text, pre_text)
            audio_transcript = None
            if audio_url:
                logger.info("Detected audio URL: %s", audio_url)
                audio_transcript = transcribe_audio(audio_url)

            # decide action via heuristics; if not clear, consult LLM
            question_spec = parse_question_text(page_text, pre_text)
            if question_spec.get("action") == "return_text" or question_spec.get("action") is None:
                # ask LLM for clearer spec if available
                llm_spec = None
                try:
                    llm_spec = ask_llm_for_action(page_text, pre_text)
                except Exception:
                    llm_spec = None
                if llm_spec:
                    question_spec.update(llm_spec)

            # compute answer
            answer = None
            try:
                if secret_code:
                    answer = secret_code
                elif audio_transcript:
                    answer = audio_transcript.strip()
                elif file_bytes and file_ext:
                    if file_ext == "csv":
                        answer = compute_answer_from_csv_bytes(file_bytes, question_spec)
                    elif file_ext in ("xls","xlsx"):
                        answer = compute_answer_from_excel_bytes(file_bytes, question_spec)
                    elif file_ext == "pdf":
                        val = compute_answer_from_pdf_bytes(file_bytes, question_spec)
                        answer = val if val is not None else file_bytes_to_data_uri(file_bytes, "pdf")
                    elif file_ext == "json":
                        parsed = safe_json_parse(file_bytes.decode("utf-8", errors="ignore"))
                        answer = parsed if parsed else file_bytes.decode("utf-8", errors="ignore")[:1000]
                    else:
                        answer = file_bytes_to_data_uri(file_bytes, file_ext)
                else:
                    # text-based question
                    act = question_spec.get("action")
                    if act == "count":
                        answer = len([ln for ln in page_text.splitlines() if ln.strip()])
                    elif act == "chart":
                        try:
                            dfs = pd.read_html(page_text)
                            if dfs:
                                answer = df_to_chart_data_uri(dfs[0])
                            else:
                                answer = "no-table"
                        except Exception:
                            answer = "no-table"
                    else:
                        # fallback to snippet
                        answer = page_text.strip()[:800]
            except Exception as e:
                logger.exception("Error computing answer: %s", e)
                answer = page_text.strip()[:600]

            # post payload
            payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
            payload = enforce_payload_limit(payload)

            # submit with retry (single attempt here)
            next_url = None
            try:
                resp = requests.post(submit_url, json=payload, timeout=25)
                logger.info("Submit HTTP %s", resp.status_code)
                logger.info("Submit text: %s", resp.text[:1000])
                if resp.ok:
                    try:
                        jr = resp.json()
                        next_url = jr.get("url")
                    except:
                        pass
                else:
                    logger.warning("Submit returned non-200.")
            except Exception as e:
                logger.error("Submit error: %s", e)
                break

            if next_url:
                logger.info("Next URL -> %s", next_url)
                current_url = next_url
            else:
                logger.info("No next; finishing.")
                break

        await browser.close()
    logger.info("Solver completed; time left: %d", time_left())

# ----------------- helpers -----------------
def detect_submit_url(page_text: str, pre_text: str, current_url: str) -> Optional[str]:
    content = (pre_text or "") + "\n" + (page_text or "")
    m = re.search(r"https?://[^\s'\"<>]+/submit[^\s'\"<>]*", content, flags=re.I)
    if m:
        return m.group(0)
    m = re.search(r"https?://[^\s'\"<>]+/(submit|post|answer)[^\s'\"<>]*", content, flags=re.I)
    if m:
        return m.group(0)
    m = re.search(r"(^|[^A-Za-z])(\/submit[^\s'\"<>]*)", content, flags=re.I)
    if m:
        return urljoin(current_url, m.group(2))
    m = re.search(r"post\s+back\s+to\s+(\/submit[^\s'\"<>]*)", content, flags=re.I)
    if m:
        return urljoin(current_url, m.group(1))
    return None

def detect_file_url(page_text: str, pre_text: str) -> Optional[str]:
    content = (pre_text or "") + "\n" + (page_text or "")
    m = re.search(r"https?://[^\s'\"<>]+\.(csv|pdf|xlsx|xls|json|wav|mp3)", content, flags=re.I)
    return m.group(0) if m else None

def detect_scrape_url(page_text: str, current_url: str) -> Optional[str]:
    m = re.search(r"scrape\s+([\/][^\s'\"<>]+)", page_text, flags=re.I)
    if m:
        return urljoin(current_url, m.group(1))
    return None

async def scrape_secondary_page(scrape_url: str, page) -> Optional[str]:
    try:
        await page.goto(scrape_url, wait_until="networkidle", timeout=30000)
    except:
        try:
            await page.goto(scrape_url, wait_until="load", timeout=45000)
        except:
            return None
    await page.wait_for_timeout(200)
    try:
        text = await page.evaluate("() => document.body.innerText") or ""
    except:
        return None
    m = re.search(r"secret\s*code\s*[:\-]?\s*([A-Za-z0-9_-]+)", text, flags=re.I)
    if m:
        return m.group(1)
    return text.strip()[:300]

def detect_audio_url(page_text: str, pre_text: str) -> Optional[str]:
    content = (pre_text or "") + "\n" + (page_text or "")
    m = re.search(r"https?://[^\s'\"<>]+\.(mp3|wav|m4a|ogg)", content, flags=re.I)
    return m.group(0) if m else None

def transcribe_audio(audio_url: str) -> Optional[str]:
    try:
        r = requests.get(audio_url, timeout=30)
        if not r.ok:
            logger.error("Audio download failed: %s", r.status_code)
            return None
        audio_bytes = r.content
    except Exception as e:
        logger.error("Audio fetch error: %s", e)
        return None

    # Prefer OpenAI speech-to-text if API key present
    if OPENAI_KEY:
        try:
            # write to temp file
            import tempfile
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tf.write(audio_bytes)
            tf.flush()
            tf.close()
            with open(tf.name, "rb") as f:
                resp = openai.Audio.transcribe("gpt-4o-transcribe", f)  # or "whisper-1" if available
                # resp structure depends on openai package; handle generically
                txt = None
                if isinstance(resp, dict):
                    txt = resp.get("text") or resp.get("transcription") or None
                else:
                    # try attribute
                    txt = getattr(resp, "text", None)
                return txt
        except Exception as e:
            logger.exception("OpenAI audio transcribe failed: %s", e)
            # fallthrough to local whisper if available

    # Fallback: local whisper (if installed)
    try:
        import whisper
        model = whisper.load_model("small")
        import tempfile, os
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tf.write(audio_bytes); tf.flush(); tf.close()
        res = model.transcribe(tf.name)
        return res.get("text")
    except Exception as e:
        logger.exception("Local whisper transcription failed: %s", e)
        return None

def enforce_payload_limit(payload: dict) -> dict:
    try:
        b = json.dumps(payload).encode("utf-8")
        if len(b) > 900_000:
            if isinstance(payload.get("answer"), str):
                payload["answer"] = payload["answer"][:200000]
    except:
        pass
    return payload
