# utils.py
import re
import json
from io import BytesIO
import base64
import logging

import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber

logger = logging.getLogger("utils")
logging.basicConfig(level=logging.INFO)


# ---------------- parse simple questions heuristics ----------------

def parse_question_text(page_text: str, pre_text: str = None) -> dict:
    """
    Heuristic parsing for common instructions.
    Returns dict with keys: action, column, cutoff, page
    """
    content = (pre_text or "") + "\n" + (page_text or "")
    out = {"action": None, "column": None, "cutoff": None, "page": None}

    txt = content.lower()

    # common: "count the rows" / "how many rows"
    if re.search(r"\b(count|how many)\b.*\b(rows|entries|lines)\b", txt):
        out["action"] = "count"
        return out

    # sum / mean / max / min
    m = re.search(r"(sum|total|mean|average|max|min|median)\s+of\s+([A-Za-z0-9_ \-]+)", txt)
    if m:
        verb = m.group(1)
        col = m.group(2).strip().replace(" ", "_")
        if "sum" in verb or "total" in verb:
            out["action"] = "sum"
        elif "mean" in verb or "average" in verb:
            out["action"] = "mean"
        elif "max" in verb:
            out["action"] = "max"
        elif "min" in verb:
            out["action"] = "min"
        elif "median" in verb:
            out["action"] = "median"
        out["column"] = col
        return out

    # filter like 'greater than 100' or '> 100' for cutoff
    m2 = re.search(r"(?:greater than|>|\bmore than\b)\s*([0-9,\.]+)", txt)
    if m2:
        out["cutoff"] = float(m2.group(1).replace(",", ""))

    # chart
    if re.search(r"\b(chart|plot|graph)\b", txt):
        out["action"] = "chart"
        return out

    # pdf page
    m3 = re.search(r"page\s+(\d+)", txt)
    if m3:
        out["page"] = int(m3.group(1))

    # fallback: return_text
    out["action"] = out["action"] or "return_text"
    return out


# ---------------- CSV / Excel / PDF computation helpers ----------------

def compute_answer_from_csv_bytes(b: bytes, spec: dict):
    buf = BytesIO(b)
    try:
        df = pd.read_csv(buf)
    except Exception:
        # try excel-ish reading
        buf.seek(0)
        df = pd.read_csv(buf, engine="python", error_bad_lines=False)
    return _compute_from_dataframe(df, spec)


def compute_answer_from_excel_bytes(b: bytes, spec: dict):
    buf = BytesIO(b)
    try:
        df = pd.read_excel(buf)
    except Exception:
        # try reading all sheets and pick first
        buf.seek(0)
        xls = pd.ExcelFile(buf)
        df = xls.parse(xls.sheet_names[0])
    return _compute_from_dataframe(df, spec)


def compute_answer_from_pdf_bytes(b: bytes, spec: dict):
    """
    Extract text from PDF. If the spec asks for numeric aggregation and a table is present,
    we attempt to parse tables with pandas read_html (rare) or simple regex numbers.
    Otherwise return extracted text.
    """
    try:
        with pdfplumber.open(BytesIO(b)) as pdf:
            page_no = spec.get("page", 1)
            page_no = max(1, page_no)
            if page_no <= len(pdf.pages):
                page = pdf.pages[page_no - 1]
                text = page.extract_text() or ""
            else:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        text = b.decode("utf-8", errors="ignore")[:2000]

    # if spec wants sum/mean/... try simple numeric extraction on page text
    action = spec.get("action")
    if action in ("sum", "mean", "max", "min", "count", "median"):
        nums = [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d{1,3}(?:[,]\d{3})*(?:\.\d+)?|\d+\.\d+", text)]
        if not nums:
            return text.strip()[:1000]
        if action == "sum":
            return sum(nums)
        if action == "mean":
            return sum(nums) / len(nums)
        if action == "max":
            return max(nums)
        if action == "min":
            return min(nums)
        if action == "count":
            return len(nums)
        if action == "median":
            nums.sort()
            n = len(nums)
            mid = n // 2
            return (nums[mid] if n % 2 else (nums[mid - 1] + nums[mid]) / 2)
    return text.strip()[:2000]


def _compute_from_dataframe(df: pd.DataFrame, spec: dict):
    """
    Unified operations for DataFrame:
    - actions: sum/mean/max/min/count/chart/return_text
    - columns: accept common variants (case-insensitive)
    """
    if df is None or df.empty:
        return "empty"

    action = spec.get("action")
    col = spec.get("column")
    cutoff = spec.get("cutoff")

    # Normalize and allow some fuzzy column matches for booknow/cinepos
    colname = None
    if col:
        # try direct match first
        if col in df.columns:
            colname = col
        else:
            # case-insensitive match
            for c in df.columns:
                if c.lower() == col.lower() or c.lower().replace(" ", "_") == col.lower().replace(" ", "_"):
                    colname = c
                    break
    else:
        # if only one numeric column, pick it for simple ops
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) == 1:
            colname = numeric_cols[0]

    # handle known dataset column patterns (BOOKNOW and CINEPOS)
    # Use user-specified names (from problem): 
    # BOOKNOW: ['book_theater_id','show_datetime','booking_datetime','tickets_booked']
    # CINEPOS: ['cine_theater_id','show_datetime','booking_datetime','tickets_sold']
    # Accept both 'tickets_booked' and 'tickets_sold' as numeric column candidates
    for candidate in ("tickets_booked", "tickets_sold", "tickets"):
        if candidate in df.columns and not colname:
            colname = candidate
            break

    try:
        if action in ("sum", "mean", "max", "min", "count", "median"):
            if not colname:
                return "no-column"
            series = pd.to_numeric(df[colname], errors="coerce").dropna()
            if cutoff is not None:
                series = series[series > float(cutoff)]
            if series.empty:
                return 0 if action == "sum" else None
            if action == "sum":
                return float(series.sum())
            if action == "mean":
                return float(series.mean())
            if action == "max":
                return float(series.max())
            if action == "min":
                return float(series.min())
            if action == "count":
                return int(series.count())
            if action == "median":
                return float(series.median())
        elif action == "chart":
            # produce chart data URI of first two columns if numeric available
            return df_to_chart_data_uri(df)
        else:
            # default: return a short JSON/text summary
            return df.head(10).to_json(orient="records")
    except Exception as e:
        logger.exception("Dataframe compute failed: %s", e)
        return df.head(5).to_json(orient="records")


# ---------------- helpers: file -> data URI and small utils ----------------

def file_bytes_to_data_uri(b: bytes, ext: str) -> str:
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:application/{ext};base64,{b64}"


def df_to_chart_data_uri(df):
    # Simple chart: first numeric column vs index
    try:
        s = None
        numcols = df.select_dtypes(include="number").columns.tolist()
        if numcols:
            s = df[numcols[0]].dropna()
            plt.figure(figsize=(6, 3))
            plt.plot(s.index.values, s.values)
            plt.title(str(numcols[0]))
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode("ascii")
            plt.close()
            return f"data:image/png;base64,{data}"
        else:
            # fallback: table snapshot as text image
            txt = df.head(10).to_string()
            plt.figure(figsize=(6, 3))
            plt.text(0, 1, txt, fontsize=8, va="top")
            plt.axis("off")
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode("ascii")
            plt.close()
            return f"data:image/png;base64,{data}"
    except Exception as e:
        logger.exception("Chart creation failed: %s", e)
        return "chart-failed"


def safe_json_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None
