import os
import time
import threading
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from solver import solve_quiz_with_deadline

# load local .env if present (safe for local dev only)
load_dotenv()

APP = Flask(__name__)

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)

QUIZ_SECRET = os.getenv("QUIZ_SECRET", "").strip()
if not QUIZ_SECRET:
    logging.warning("QUIZ_SECRET not set in environment - set it in Render or .env for local dev")

MAX_TOTAL_SECONDS = int(os.getenv("QUIZ_TIMEOUT", "170"))

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@APP.route("/api/quiz", methods=["POST"])
def api_quiz():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if not email or not secret or not url:
        return jsonify({"error": "Missing fields: email, secret, url required"}), 400

    if secret != QUIZ_SECRET:
        return jsonify({"error": "Invalid secret"}), 403

    # Accept quickly, then solve in background
    resp = jsonify({"status": "accepted"})
    resp.status_code = 200

    start_time = time.time()
    thread = threading.Thread(
        target=solve_quiz_with_deadline,
        args=(url, email, secret, start_time, MAX_TOTAL_SECONDS),
        daemon=True,
    )
    thread.start()

    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    APP.run(host="0.0.0.0", port=port, debug=(os.getenv("FLASK_ENV") == "development"))
