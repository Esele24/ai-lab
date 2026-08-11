"""One Gemini client, used by all seven projects.

Raw `requests` against the REST API rather than an SDK. Two reasons:
  1. No extra dependency to break on a Python upgrade.
  2. You can see the actual HTTP call, which is the thing worth understanding.

Three public functions:
  generate(prompt)            -> str            free-text answer
  generate_json(prompt, schema) -> dict|list    answer forced into a shape
  embed(texts)                -> list[list[float]]
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from core import config

BASE = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 120


class GeminiError(RuntimeError):
    """Raised with the API's own message, not a generic 'request failed'."""


def _headers() -> dict[str, str]:
    if not config.GEMINI_API_KEY:
        raise GeminiError(
            "No GEMINI_API_KEY. Copy .env.example to .env and paste the key from "
            "tender-tool/.env.local."
        )
    return {
        "x-goog-api-key": config.GEMINI_API_KEY,
        "Content-Type": "application/json",
    }


RETRY_AFTER = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
MAX_BACKOFF = 65.0


def _post(url: str, payload: dict[str, Any], retries: int = 2) -> dict[str, Any]:
    """POST, honouring the API's own retry hint on 429.

    TWO DIFFERENT 429s, and telling them apart matters:

      'limit: 0'   -> this key has NO quota for that model name. The model is wrong.
                      Retrying will never fix it, so fail immediately with the body.
      'limit: 20'  -> a real rate limit. The body carries "Please retry in 50.9s",
                      which is the only honest wait -- a fixed 2s backoff just burns
                      another request and gets refused again.

    The free tier is genuinely small. A test run that makes twenty model calls will
    exhaust it, which is not a bug in this code.
    """
    last = ""
    for attempt in range(retries + 1):
        response = requests.post(url, headers=_headers(), json=payload, timeout=TIMEOUT)
        if response.status_code == 200:
            return response.json()
        last = f"HTTP {response.status_code}: {response.text[:600]}"
        if "limit: 0" in response.text or response.status_code not in (429, 500, 503):
            break
        if attempt == retries:
            break
        hint = RETRY_AFTER.search(response.text)
        wait = min(float(hint.group(1)) + 1.0, MAX_BACKOFF) if hint else 2.0 * (attempt + 1)
        time.sleep(wait)
    raise GeminiError(last)


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the text out of a generateContent response.

    Defensive on purpose: a response blocked by a safety filter has candidates
    but no parts, and naive data['candidates'][0]['content']['parts'][0]['text']
    raises KeyError instead of telling you why.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise GeminiError(f"Model returned nothing ({reason}).")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        raise GeminiError(
            f"Empty response (finishReason={candidates[0].get('finishReason')})."
        )
    return text


def generate(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    model: str | None = None,
    parts: list[dict[str, Any]] | None = None,
) -> str:
    """Free-text generation. `parts` lets callers attach inline data (e.g. audio)."""
    content_parts: list[dict[str, Any]] = [{"text": prompt}]
    if parts:
        content_parts.extend(parts)
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": content_parts}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"{BASE}/models/{model or config.GEMINI_MODEL}:generateContent"
    return _extract_text(_post(url, payload))


def generate_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    system: str | None = None,
    temperature: float = 0.0,
    model: str | None = None,
) -> Any:
    """Generation forced into a shape.

    temperature=0 and a responseSchema together are what make model output safe to
    hand to code. Without the schema you are string-parsing prose and hoping.
    """
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"{BASE}/models/{model or config.GEMINI_MODEL}:generateContent"
    text = _extract_text(_post(url, payload))
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Schema-constrained call returned non-JSON: {text[:300]}") from exc


def embed(texts: list[str], *, task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of strings.

    Sent one request per text rather than batchEmbedContents: the batch endpoint
    fails the whole batch on one bad input, and a 60-page PDF is exactly where
    one bad chunk is likely. Slower, but a single bad chunk cannot cost the
    whole upload.
    """
    url = f"{BASE}/models/{config.EMBED_MODEL}:embedContent"
    vectors: list[list[float]] = []
    for text in texts:
        payload = {
            "model": f"models/{config.EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": task,
        }
        data = _post(url, payload)
        values = (data.get("embedding") or {}).get("values")
        if not values:
            raise GeminiError(f"Embedding response had no values: {str(data)[:300]}")
        vectors.append(values)
    return vectors
