# app.py
import time
import threading
import logging
from flask import Flask, request, jsonify

# Import validated configuration
from config import SECRET as QUIZ_SECRET, PORT
from solver import solve_quiz_with_deadline

APP = Flask(__name__)

# -------------------------
# LOGGING CONFIGURATION
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Silence noisy libraries
for lib in ["urllib3", "pdfminer", "pdfplumber", "matplotlib"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


MAX_TOTAL_SECONDS = 170   # Hard limit: 2 minutes 50 seconds


# -------------------------
# ROUTES
# -------------------------

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

    # Validate the secret
    if secret != QUIZ_SECRET:
        return jsonify({"error": "Invalid secret"}), 403

    # Accept request immediately
    response = jsonify({"status": "accepted"})
    response.status_code = 200
