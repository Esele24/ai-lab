"""Prove the key, the chat model, the schema mode and the embedding model all work.

Run this FIRST, before blaming any of the seven apps:
    python check_setup.py
"""
from core import config, gemini


def main() -> int:
    print(f"key present : {config.has_key()}")
    print(f"chat model  : {config.GEMINI_MODEL}")
    print(f"embed model : {config.EMBED_MODEL}")
    failures = 0

    try:
        text = gemini.generate("Reply with exactly: OK", temperature=0.0)
        print(f"[pass] generate      -> {text.strip()[:40]!r}")
    except Exception as exc:
        print(f"[FAIL] generate      -> {exc}")
        failures += 1

    try:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
        }
        data = gemini.generate_json("Port Harcourt is in which country?", schema)
        print(f"[pass] generate_json -> {data}")
    except Exception as exc:
        print(f"[FAIL] generate_json -> {exc}")
        failures += 1

    try:
        vectors = gemini.embed(["wellhead maintenance contract"])
        print(f"[pass] embed         -> {len(vectors[0])} dimensions")
    except Exception as exc:
        print(f"[FAIL] embed         -> {exc}")
        failures += 1

    print("\nALL PASS" if not failures else f"\n{failures} CHECK(S) FAILED")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
