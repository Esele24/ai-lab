"""Configuration + a tiny .env reader.

Deliberately no python-dotenv dependency. This file is ~30 lines and you can read
all of it, which matters more here than saving 30 lines.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ.

    Existing environment variables win, so a real shell export always beats the
    file. Blank lines and '#' comments are skipped. Values are stripped of
    surrounding quotes because 'KEY="value"' is a common way to write these.
    """
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def has_key() -> bool:
    return bool(GEMINI_API_KEY)
