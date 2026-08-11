"""One single model call, to find out whether the free-tier quota has recovered.

Deliberately the smallest possible request. Use this instead of re-running
test_all.py when you suspect a rate limit -- test_all.py makes about twenty calls
and will exhaust the free tier by itself.
"""
from core import gemini

try:
    print("OK:", gemini.generate("Reply with exactly: OK", temperature=0.0).strip())
except Exception as exc:
    text = str(exc)
    if "limit: 0" in text:
        print("WRONG MODEL for this key — not a rate limit. Check GEMINI_MODEL in .env.")
    elif "429" in text:
        import re
        hint = re.search(r"retry in ([\d.]+)s", text)
        print(f"RATE LIMITED. API says retry in {hint.group(1) if hint else '?'}s.")
    else:
        print("FAILED:", text[:300])
