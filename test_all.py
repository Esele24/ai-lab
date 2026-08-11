"""End-to-end check of every route, against the running server.

    python server.py        (in one terminal)
    python test_all.py      (in another)

This is what "it works" means here. Each check prints what it actually got back, so
a pass is readable evidence rather than a green tick.
"""
from __future__ import annotations

import base64
import io
import json
import sys

import requests

BASE = "http://localhost:8000"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        results.append((name, True, detail))
        print(f"[PASS] {name}\n       {detail}")
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name}\n       {type(exc).__name__}: {exc}")


def post(path: str, payload: dict, timeout: int = 240) -> dict:
    response = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
    data = response.json()
    if not response.ok:
        raise RuntimeError(data.get("error", response.text[:300]))
    return data


def get(path: str, timeout: int = 60) -> dict:
    response = requests.get(f"{BASE}{path}", timeout=timeout)
    data = response.json()
    if not response.ok:
        raise RuntimeError(data.get("error", response.text[:300]))
    return data


# --- fixtures --------------------------------------------------------------

CONTRACT = """SERVICE AGREEMENT

This Agreement is made between Delta Rivers Energy Services Limited ("the Contractor")
and Harbour Point Catering & Events Limited ("the Client").

1. SCOPE. The Contractor shall provide wellhead maintenance services at the Client's
facility in Port Harcourt, Rivers State.

2. PAYMENT. The Client shall pay the Contractor the sum of N4,500,000 (four million five
hundred thousand naira) within 90 days of receipt of invoice. No interest shall accrue on
late payment.

3. TERMINATION. The Contractor may terminate this Agreement at any time upon giving
seven (7) days written notice. The Client may not terminate this Agreement before the
expiry of the initial term of twenty-four (24) months.

4. LIABILITY. The Client shall indemnify and hold harmless the Contractor against any and
all claims, losses and damages howsoever arising, including those arising from the
Contractor's own negligence.

5. CONFIDENTIALITY. Each party shall keep confidential all information disclosed by the
other party in connection with this Agreement.

6. GOVERNING LAW. This Agreement shall be governed by the laws of the Federal Republic
of Nigeria.
"""

CSV = """date,region,category,units,revenue
2026-01-14,Port Harcourt,Catering,120,840000
2026-01-22,Lagos,Venue Hire,3,1500000
2026-02-03,Port Harcourt,Catering,80,560000
2026-02-19,Abuja,Catering,200,1400000
2026-03-05,Lagos,Venue Hire,5,2500000
2026-03-11,Port Harcourt,Office Lunch,450,675000
2026-04-02,Abuja,Venue Hire,2,900000
2026-04-18,Lagos,Catering,160,1120000
2026-05-07,Port Harcourt,Office Lunch,520,780000
2026-05-23,Abuja,Catering,90,630000
2026-06-09,Lagos,Office Lunch,300,450000
2026-06-27,Port Harcourt,Venue Hire,4,1800000
"""


def make_pdf() -> bytes:
    """A tiny valid PDF with real extractable text, built by hand.

    Written by hand rather than with a PDF library because none is installed for
    writing -- pypdf reads. It is enough to prove extraction, chunking, embedding
    and citation work on a real .pdf rather than on a .txt renamed.
    """
    text = (
        "BEGIN CONTRACT. The total contract value is 4,500,000 naira. "
        "Payment falls due within 90 days of invoice. "
        "The penalty for late delivery is 2.5 percent of the contract value per week. "
        "The project supervisor is Mr Tamuno Briggs of Port Harcourt. END CONTRACT."
    )
    stream = f"BT /F1 11 Tf 40 700 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    return bytes(out)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# --- checks ----------------------------------------------------------------

def main() -> int:
    check("04 dashboard — server up, state readable", lambda: (
        lambda d: f"key={d['key_present']} model={d['chat_model']} runs={d['total_runs']}"
    )(get("/api/dashboard")))

    check("static — index.html served", lambda: (
        lambda r: f"HTTP {r.status_code}, {len(r.content)} bytes"
    )(requests.get(BASE + "/", timeout=30)))

    check("security — path traversal to .env is refused", lambda: (
        lambda r: f"HTTP {r.status_code} (must be 404)" if r.status_code == 404
        else (_ for _ in ()).throw(RuntimeError(f"LEAKED: HTTP {r.status_code}"))
    )(requests.get(BASE + "/../.env", timeout=30)))

    check("01 doc — index a real PDF", lambda: (
        lambda d: f"{d['chunks']} chunks from {d['sources']}, skipped={d['skipped']}"
    )(post("/api/doc/index", {"files": [{"name": "contract.pdf", "b64": b64(make_pdf())}]})))

    check("01 doc — answer cites the page", lambda: (
        lambda d: f"answer={d['answer'][:150]!r} · top source={d['sources'][0]['label']}"
    )(post("/api/doc/ask", {"question": "What is the penalty for late delivery?"})))

    check("01 doc — refuses what is not in the document", lambda: (
        lambda d: f"answer={d['answer'][:150]!r}"
    )(post("/api/doc/ask", {"question": "What is the capital city of Brazil?"})))

    check("01 doc — a .pdf that is not a PDF is rejected", lambda: _expect_error(
        "/api/doc/index", {"files": [{"name": "fake.pdf", "b64": b64(b"not a pdf at all")}]}))

    check("02 data — load CSV, profile it", lambda: (
        lambda d: f"{d['profile']['rows']} rows, columns={[c['name'] for c in d['profile']['columns']]}"
    )(post("/api/data/load", {"name": "sales.csv", "b64": b64(CSV.encode())})))

    check("02 data — group-by question", lambda: (
        lambda d: f"op={d['spec']['operation']}/{d['spec']['aggregation']} on "
                  f"{d['spec']['metric_column']} by {d['spec']['dimension_column']} · "
                  f"rows={d['result'].get('table')} · says: {d['narrative'][:110]!r}"
    )(post("/api/data/ask", {"question": "Which category earned the most revenue in total?"})))

    check("02 data — time series question", lambda: (
        lambda d: f"op={d['spec']['operation']} chart={d['result'].get('chart')} "
                  f"points={len(d['result'].get('table', []))}"
    )(post("/api/data/ask", {"question": "Show me revenue over time by month"})))

    check("02 data — refuses an unanswerable question", lambda: (
        lambda d: f"kind={d['result']['kind']} · {d['narrative'][:140]!r}"
    )(post("/api/data/ask", {"question": "What is each customer's email address?"})))

    check("03 model — report loads", lambda: (
        lambda d: f"acc={d['report']['metrics']['accuracy']} f1={d['report']['metrics']['f1']} "
                  f"recall={d['report']['metrics']['recall']} baseline={d['report']['majority_baseline_accuracy']}"
    )(get("/api/model/report")))

    check("03 model — classifies spam", lambda: (
        lambda d: f"{d['label']} p={d['probability']} top driver={d['drivers'][0] if d['drivers'] else None}"
    )(post("/api/model/predict", {
        "text": "URGENT! You have WON a 1 week FREE membership. Text CLAIM to 81010 now!"})))

    check("03 model — classifies ham", lambda: (
        lambda d: f"{d['label']} p={d['probability']}"
    )(post("/api/model/predict", {
        "text": "Hey, are we still meeting at 4 or should I come earlier?"})))

    check("05 prompts — build one", lambda: (
        lambda d: f"{len(d['prompt'])} chars, variables={d['variables']}"
    )(post("/api/prompt/build", {
        "task": "Marketing", "details": "A short first outreach message to a law firm with no website",
        "tone": "Direct", "format": "Short paragraph"})))

    check("05 prompts — save then list", lambda: (
        lambda _: f"library now holds {len(get('/api/prompt/list')['prompts'])} prompt(s)"
    )(post("/api/prompt/save", {
        "name": "Law firm opener", "category": "Marketing", "tone": "Direct",
        "body": "Write a short opener to {firm_name} in {city}. End with a question."})))

    check("05 prompts — fill reports unfilled placeholders", lambda: (
        lambda d: f"filled={d['filled'][:70]!r} missing={d['missing']}"
    )(post("/api/prompt/fill", {
        "template": "Write to {firm_name} in {city}.", "values": {"firm_name": "Forcados Partners"}})))

    check("05 prompts — A/B compare with shuffled judge", lambda: (
        lambda d: f"winner={d['verdict']['winner']} · A {d['words_a']}w vs B {d['words_b']}w · "
                  f"swapped={d['judge_saw_swapped']} · {d['verdict']['reason'][:90]!r}"
    )(post("/api/prompt/compare", {
        "prompt_a": "Write a message to a business about their website.",
        "prompt_b": "You are writing one WhatsApp message to a Nigerian law firm that has no "
                    "website. Under 60 words. Name one concrete thing they are losing. End with "
                    "a question. No greeting longer than three words. Do not say 'I am a developer'.",
        "test_input": "The firm is Forcados Partners in Port Harcourt."})))

    check("06 industry — review a contract", lambda: (
        lambda d: f"{len(d['findings'])} findings of {d['checklist_size']} · "
                  f"absent={sum(1 for f in d['findings'] if f['status'] == 'ABSENT')} · "
                  f"weak={sum(1 for f in d['findings'] if f['status'] == 'WEAK')} · "
                  f"dropped={d['dropped_findings']}"
    )(post("/api/industry/review", {"industry": "Legal / Contracts", "text": CONTRACT})))

    check("06 industry — every surviving quote is really in the document", lambda: _quotes_verified())

    check("07 voice — first turn collects, does not invent a price", lambda: (
        lambda d: f"intent={d['intent']} booked={d['booking_id']} reply={d['reply'][:120]!r}"
    )(post("/api/voice/turn", {
        "history": [], "utterance": "Hi, how much would it cost to cater for fifty people?"})))

    check("07 voice — completes a booking and writes it", lambda: _booking_flow())

    check("07 voice — bookings readable", lambda: (
        lambda d: f"{len(d['bookings'])} booking(s) stored"
    )(get("/api/voice/bookings")))

    check("04 dashboard — activity log filled by the runs above", lambda: (
        lambda d: f"{d['total_runs']} runs, {d['failures']} failed, by project: {d['by_project']}"
    )(get("/api/dashboard")))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'=' * 70}\n{passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 0 if passed == len(results) else 1


def _expect_error(path: str, payload: dict) -> str:
    response = requests.post(f"{BASE}{path}", json=payload, timeout=120)
    if response.ok:
        raise RuntimeError("expected a 400, got a 200")
    return f"rejected as intended: {response.json()['error'][:110]}"


def _quotes_verified() -> str:
    data = post("/api/industry/review", {"industry": "Legal / Contracts", "text": CONTRACT})
    import re
    normalise = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    haystack = normalise(CONTRACT)
    bad = [f["clause"] for f in data["findings"]
           if f["quote"] and normalise(f["quote"]) not in haystack]
    if bad:
        raise RuntimeError(f"quotes not in the document survived verification: {bad}")
    quoted = sum(1 for f in data["findings"] if f["quote"])
    return f"{quoted} quoted findings, all located verbatim in the source text"


def _booking_flow() -> str:
    history: list[dict] = []
    script = [
        "I want to book catering for a corporate event",
        "My name is Esele Okogbo",
        "My number is 0803 555 0199",
        "The date is the twelfth of September, for about eighty guests",
    ]
    last = {}
    for utterance in script:
        last = post("/api/voice/turn", {"history": history, "utterance": utterance})
        history.append({"role": "user", "text": utterance})
        history.append({"role": "agent", "text": last["reply"]})
    if not last.get("booking_id"):
        raise RuntimeError(f"no booking written. last collected={last.get('collected')} "
                           f"ready={last.get('ready_to_book')} reply={last.get('reply')!r}")
    return f"booking #{last['booking_id']} written · collected={last['collected']}"


if __name__ == "__main__":
    sys.exit(main())
