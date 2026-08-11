"""Confirm the per-IP rate limit fires, and that free routes are unaffected.

Run the server with a small cap so this costs no model calls:
    $env:RATE_LIMIT_PER_HOUR=3; $env:PORT=8011; python server.py
    python test_ratelimit.py
"""
import requests

BASE = "http://127.0.0.1:8011"

print("-- a rate-limited route (/api/voice/turn), cap should be 3 --")
for attempt in range(1, 6):
    response = requests.post(f"{BASE}/api/voice/turn", json={"utterance": ""}, timeout=30)
    note = ""
    if response.status_code == 429:
        note = f"  Retry-After={response.headers.get('Retry-After')}s"
    print(f"  request {attempt}: HTTP {response.status_code}{note}")

print("\n-- a free route (/api/model/predict) must NOT be capped --")
for attempt in range(1, 4):
    response = requests.post(
        f"{BASE}/api/model/predict", json={"text": "free prize claim now"}, timeout=30
    )
    body = response.json()
    print(f"  request {attempt}: HTTP {response.status_code} -> {body.get('label', body.get('error'))}")

print("\n-- GET routes must NOT be capped --")
for attempt in range(1, 3):
    response = requests.get(f"{BASE}/api/dashboard", timeout=30)
    print(f"  request {attempt}: HTTP {response.status_code}")
