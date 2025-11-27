import os
import time
import threading
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from solver import solve_quiz_with_deadline

load_dotenv()
APP = Flask(__name__)

logging.basicConfig(level=logging.INFO)

QUIZ_SECRET = os.getenv("QUIZ_SECRET")
MAX_TOTAL_SECONDS = 170

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@APP.route("/api/quiz", methods=["POST"])
def api_quiz():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400

    data = request.get_json()
    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if not email or not secret or not url:
        return jsonify({"error": "email, secret, url required"}), 400

    if secret != QUIZ_SECRET:
        return jsonify({"error": "Invalid secret"}), 403

    resp = jsonify({"status": "accepted"})

    start = time.time()
    t = threading.Thread(
        target=solve_quiz_with_deadline,
        args=(url, email, secret, start, MAX_TOTAL_SECONDS),
        daemon=True,
    )
    t.start()

    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    APP.run(host="0.0.0.0", port=port)

