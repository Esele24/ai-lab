"""AI Lab — one HTTP server for all seven projects. Standard library only.

    python server.py          then open http://localhost:8000

No Flask, no FastAPI, no Streamlit. Not out of purism -- this machine's connection
could not finish downloading scipy, and `http.server` was already installed. It also
means the whole backend is one file you can read.

Uploads arrive as base64 inside JSON rather than as multipart/form-data, because
Python 3.13 removed the `cgi` module that used to parse multipart. Writing a
multipart parser by hand to save a 33% base64 overhead on a local upload would be
the wrong trade.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from core import analysis, config, docs, gemini, mlmodel, prompts, review, store, voice
from core.vectors import VectorIndex

WEB_DIR = Path(__file__).resolve().parent / "web"
INDEX_DIR = config.DATA_DIR / "doc_index"
MAX_BODY = 40 * 1024 * 1024  # 40 MB of base64 ~= a 30 MB file

MIME = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8", ".json": "application/json",
    ".svg": "image/svg+xml", ".ico": "image/x-icon",
}

# Server-side state. This is a single-user local tool, so an in-process cache is
# correct here -- but it IS the reason this app is not multi-user as written.
# The document index is also written to disk so a restart does not re-embed.
STATE: dict[str, Any] = {"index": None, "frame": None, "frame_name": None, "model": None}


def load_doc_index() -> VectorIndex | None:
    if STATE["index"] is None and (INDEX_DIR / "vectors.npy").exists():
        STATE["index"] = VectorIndex.load(INDEX_DIR)
    return STATE["index"]


def load_classifier():
    if STATE["model"] is None:
        if not (config.MODEL_DIR / "spam_model.npz").exists():
            return None
        STATE["model"] = mlmodel.load(config.MODEL_DIR)
    return STATE["model"]


def decode_upload(item: dict[str, str]) -> tuple[str, bytes]:
    name = item.get("name", "upload")
    raw = item.get("b64", "")
    if "," in raw[:100]:          # strips a data: URL prefix if the browser sent one
        raw = raw.split(",", 1)[1]
    return name, base64.b64decode(raw)


# --- route handlers --------------------------------------------------------

def api_dashboard(_: dict) -> dict:
    model = load_classifier()
    index = load_doc_index()
    return {
        "total_runs": store.total_runs(),
        "failures": store.failure_count(),
        "by_project": store.counts_by_project(),
        "recent": [dict(row) for row in store.recent_runs(30)],
        "prompts_saved": len(store.list_prompts()),
        "bookings": len(store.list_bookings()),
        "model_ready": model is not None,
        "model_accuracy": model[2]["metrics"]["accuracy"] if model else None,
        "doc_chunks": len(index) if index else 0,
        "dataset_loaded": STATE["frame_name"],
        "chat_model": config.GEMINI_MODEL,
        "embed_model": config.EMBED_MODEL,
        "key_present": config.has_key(),
    }


def api_doc_index(payload: dict) -> dict:
    files = payload.get("files") or []
    if not files:
        raise ValueError("No files were sent.")
    chunks: list[docs.Chunk] = []
    skipped: list[str] = []
    for item in files:
        name, data = decode_upload(item)
        try:
            chunks.extend(docs.chunk_file(name, data))
        except Exception as exc:
            skipped.append(f"{name}: {exc}")
    if not chunks:
        store.log("01 Doc Assistant", "index", "; ".join(skipped)[:400], ok=False)
        raise ValueError("Nothing could be indexed. " + " | ".join(skipped))

    vectors = gemini.embed([chunk.text for chunk in chunks])
    index = VectorIndex(chunks, vectors)
    index.save(INDEX_DIR)
    STATE["index"] = index
    sources = sorted({chunk.source for chunk in chunks})
    store.log("01 Doc Assistant", "index", f"{len(chunks)} chunks from {len(sources)} file(s)")
    return {"chunks": len(chunks), "sources": sources, "skipped": skipped}


def api_doc_ask(payload: dict) -> dict:
    index = load_doc_index()
    if index is None:
        raise ValueError("No documents indexed yet.")
    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("Ask a question first.")
    k = max(3, min(int(payload.get("k", 5)), 10))

    # Query and document embeddings use different task types on purpose; using the
    # same one for both measurably degrades retrieval quality.
    query_vector = gemini.embed([question], task="RETRIEVAL_QUERY")[0]
    hits = index.search(query_vector, k=k)
    excerpts = "\n\n".join(
        f"[{position}] ({chunk.label})\n{chunk.text}"
        for position, (chunk, _) in enumerate(hits, start=1)
    )
    answer = gemini.generate(
        f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}",
        system=(
            "You answer using ONLY the numbered excerpts supplied.\n"
            "1. If the excerpts do not contain the answer, say exactly: "
            "\"That isn't in the documents you uploaded.\" Never fall back on general knowledge.\n"
            "2. Cite excerpt numbers like [1] or [2][3] right after the claim they support.\n"
            "3. Quote figures, dates, names and amounts exactly as written. Never round or convert.\n"
            "4. If two excerpts disagree, say so and cite both.\n"
            "Be concise. No preamble."
        ),
        temperature=0.1,
    )
    store.log("01 Doc Assistant", "question", question[:200])
    return {
        "answer": answer,
        "sources": [
            {"n": position, "label": chunk.label, "score": round(score, 4),
             "text": chunk.text[:600]}
            for position, (chunk, score) in enumerate(hits, start=1)
        ],
    }


def api_doc_status(_: dict) -> dict:
    index = load_doc_index()
    if index is None:
        return {"chunks": 0, "sources": []}
    return {"chunks": len(index), "sources": sorted({c.source for c in index.chunks})}


def api_data_load(payload: dict) -> dict:
    name, data = decode_upload(payload)
    if name.lower().endswith((".xlsx", ".xls")):
        raise ValueError("Excel needs the openpyxl package, which is not installed. "
                         "Save the sheet as CSV and upload that.")
    frame = pd.read_csv(io.BytesIO(data))
    if frame.empty:
        raise ValueError("That CSV has no rows.")
    STATE["frame"] = frame
    STATE["frame_name"] = name
    store.log("02 Data Analysis", "load", f"{name} · {len(frame)} rows × {len(frame.columns)} cols")
    return {
        "name": name,
        "profile": analysis.profile(frame),
        "preview": json.loads(frame.head(8).to_json(orient="records", date_format="iso")),
        "columns": [str(c) for c in frame.columns],
    }


def api_data_ask(payload: dict) -> dict:
    frame = STATE.get("frame")
    if frame is None:
        raise ValueError("Load a CSV first.")
    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("Ask a question first.")
    outcome = analysis.ask(frame, question)
    store.log("02 Data Analysis", "question", question[:200])
    return outcome


def api_model_report(_: dict) -> dict:
    model = load_classifier()
    if model is None:
        return {"trained": False,
                "hint": "Run `python train_model.py` once. It takes about a minute."}
    _, _, report = model
    return {"trained": True, "report": report}


def api_model_predict(payload: dict) -> dict:
    model = load_classifier()
    if model is None:
        raise ValueError("The model has not been trained. Run `python train_model.py` first.")
    vectorizer, classifier, _ = model
    text = (payload.get("text") or "").strip()
    if not text:
        raise ValueError("Enter a message to classify.")
    threshold = float(payload.get("threshold", 0.5))
    X = vectorizer.transform([text])
    probability = float(classifier.predict_proba(X)[0])

    # Show which tokens actually moved the decision. This is what makes the
    # prediction auditable instead of an oracle.
    contributions: list[dict] = []
    for token in dict.fromkeys(mlmodel.tokenize(text)):
        column = vectorizer.vocabulary_.get(token)
        if column is None:
            continue
        contributions.append({
            "token": token,
            "weight": round(float(classifier.weights[column]), 3),
            "contribution": round(float(X[0, column] * classifier.weights[column]), 4),
        })
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    unknown = [t for t in dict.fromkeys(mlmodel.tokenize(text))
               if t not in vectorizer.vocabulary_]

    store.log("03 Custom Model", "predict", f"p(spam)={probability:.3f}")
    return {
        "probability": round(probability, 4),
        "label": "spam" if probability >= threshold else "ham",
        "threshold": threshold,
        "drivers": contributions[:10],
        "unknown_tokens": unknown[:12],
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def api_prompt_build(payload: dict) -> dict:
    return prompts.build(
        payload.get("task", "Other"), payload.get("details", ""),
        payload.get("tone", "Professional"), payload.get("format", "Prose"),
    )


def api_prompt_save(payload: dict) -> dict:
    name = (payload.get("name") or "").strip()
    body = (payload.get("body") or "").strip()
    if not name or not body:
        raise ValueError("A saved prompt needs a name and a body.")
    store.save_prompt(name, payload.get("category", "Other"), payload.get("tone", ""),
                      body, prompts.find_variables(body))
    store.log("05 Prompt System", "save", name)
    return {"saved": name}


def api_prompt_list(_: dict) -> dict:
    return {"prompts": store.list_prompts(), "categories": prompts.CATEGORIES,
            "tones": prompts.TONES, "formats": prompts.FORMATS}


def api_prompt_delete(payload: dict) -> dict:
    store.delete_prompt(int(payload["id"]))
    return {"deleted": payload["id"]}


def api_prompt_favourite(payload: dict) -> dict:
    store.toggle_favourite(int(payload["id"]))
    return {"toggled": payload["id"]}


def api_prompt_fill(payload: dict) -> dict:
    filled, missing = prompts.fill(payload.get("template", ""), payload.get("values", {}))
    return {"filled": filled, "missing": missing}


def api_prompt_compare(payload: dict) -> dict:
    a = (payload.get("prompt_a") or "").strip()
    b = (payload.get("prompt_b") or "").strip()
    if not a or not b:
        raise ValueError("Both prompts are needed to compare.")
    return prompts.compare(a, b, payload.get("test_input", ""))


def api_industry_list(_: dict) -> dict:
    return {"industries": {name: profile["checklist"]
                           for name, profile in review.INDUSTRIES.items()}}


def api_industry_review(payload: dict) -> dict:
    industry = payload.get("industry", "")
    text = payload.get("text", "")
    if payload.get("file"):
        name, data = decode_upload(payload["file"])
        text = "\n\n".join(part for _, part in docs.extract(name, data))
    result = review.review_document(industry, text)
    store.log("06 Industry Helper", "review",
              f"{industry} · {len(result.get('findings', []))} findings")
    return result


def api_voice_turn(payload: dict) -> dict:
    utterance = (payload.get("utterance") or "").strip()
    if not utterance:
        raise ValueError("Nothing was heard.")
    return voice.respond(payload.get("history") or [], utterance)


def api_voice_bookings(_: dict) -> dict:
    return {"bookings": [dict(row) for row in store.list_bookings()],
            "business": voice.BUSINESS}


def api_voice_handled(payload: dict) -> dict:
    store.mark_handled(int(payload["id"]))
    return {"handled": payload["id"]}


ROUTES: dict[str, Callable[[dict], dict]] = {
    "/api/dashboard": api_dashboard,
    "/api/doc/index": api_doc_index,
    "/api/doc/ask": api_doc_ask,
    "/api/doc/status": api_doc_status,
    "/api/data/load": api_data_load,
    "/api/data/ask": api_data_ask,
    "/api/model/report": api_model_report,
    "/api/model/predict": api_model_predict,
    "/api/prompt/build": api_prompt_build,
    "/api/prompt/save": api_prompt_save,
    "/api/prompt/list": api_prompt_list,
    "/api/prompt/delete": api_prompt_delete,
    "/api/prompt/favourite": api_prompt_favourite,
    "/api/prompt/fill": api_prompt_fill,
    "/api/prompt/compare": api_prompt_compare,
    "/api/industry/list": api_industry_list,
    "/api/industry/review": api_industry_review,
    "/api/voice/turn": api_voice_turn,
    "/api/voice/bookings": api_voice_bookings,
    "/api/voice/handled": api_voice_handled,
}

PAGES = {
    "/": "index.html", "/doc": "doc.html", "/data": "data.html",
    "/model": "model.html", "/prompt": "prompt.html",
    "/industry": "industry.html", "/voice": "voice.html",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "AILab/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Quieter than the default, but errors still surface.
        if not str(args[0] if args else "").startswith(("GET /api", "POST /api")):
            return
        super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ROUTES:
            self._dispatch(path, {})
            return
        filename = PAGES.get(path, path.lstrip("/"))
        target = (WEB_DIR / filename).resolve()
        # Path containment check: without it, /../../.env is served to anyone.
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        self._send(200, target.read_bytes(),
                   MIME.get(target.suffix, "application/octet-stream"))

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path not in ROUTES:
            self._json(404, {"error": f"No such endpoint: {path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(413, {"error": f"Body too large ({length / 1e6:.1f} MB). Cap is 30 MB per upload."})
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Body was not valid JSON."})
            return
        self._dispatch(path, payload)

    def _dispatch(self, path: str, payload: dict) -> None:
        try:
            self._json(200, ROUTES[path](payload))
        except (ValueError, gemini.GeminiError) as exc:
            # Expected, explainable failures: the user gets the real message.
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            # Unexpected: full traceback to the terminal, short message to the browser.
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(port: int = 8000) -> None:
    if not config.has_key():
        print("WARNING: no GEMINI_API_KEY in .env — six of the seven tools will fail.")
    if not (config.MODEL_DIR / "spam_model.npz").exists():
        print("NOTE: classifier not trained yet. Run `python train_model.py` for project 03.")
    print(f"AI Lab on http://localhost:{port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
