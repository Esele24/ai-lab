"""SQLite persistence shared by all seven projects.

Why SQLite and not localStorage or a dict in session state:
  - Streamlit reruns your whole script on every interaction, so anything held in a
    plain Python variable is gone. `st.session_state` survives reruns but dies with
    the browser tab.
  - The tender tool learned this the expensive way with localStorage: per-browser,
    no sharing, wiped by "clear browsing data".

One file on disk, `data/ai_lab.db`. The seam to swap for Postgres/Supabase later is
this module only -- keep the function names, replace the bodies.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from core.config import DATA_DIR

DB_PATH = DATA_DIR / "ai_lab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    project TEXT NOT NULL,
    action  TEXT NOT NULL,
    detail  TEXT,
    ok      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS runs_project_ts ON runs(project, ts DESC);

CREATE TABLE IF NOT EXISTS prompts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    name     TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    tone     TEXT,
    body     TEXT NOT NULL,
    variables TEXT NOT NULL DEFAULT '[]',
    favourite INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bookings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    intent     TEXT NOT NULL,
    caller     TEXT,
    detail     TEXT,
    transcript TEXT,
    handled    INTEGER NOT NULL DEFAULT 0
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


# --- activity log (powers the dashboard in project 04) ---------------------

def log(project: str, action: str, detail: str = "", ok: bool = True) -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO runs (ts, project, action, detail, ok) VALUES (?,?,?,?,?)",
            (_now(), project, action, detail[:2000], int(ok)),
        )


def recent_runs(limit: int = 25) -> list[sqlite3.Row]:
    with connect() as connection:
        return connection.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def counts_by_project() -> dict[str, int]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT project, COUNT(*) AS n FROM runs GROUP BY project"
        ).fetchall()
    return {row["project"]: row["n"] for row in rows}


def total_runs() -> int:
    with connect() as connection:
        return connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def failure_count() -> int:
    with connect() as connection:
        return connection.execute("SELECT COUNT(*) FROM runs WHERE ok = 0").fetchone()[0]


# --- prompt library (project 05) -------------------------------------------

def save_prompt(
    name: str, category: str, tone: str, body: str, variables: list[str]
) -> None:
    """Upsert by name, so editing a prompt replaces it instead of duplicating it."""
    with connect() as connection:
        connection.execute(
            """INSERT INTO prompts (ts, name, category, tone, body, variables)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                   ts=excluded.ts, category=excluded.category,
                   tone=excluded.tone, body=excluded.body,
                   variables=excluded.variables""",
            (_now(), name.strip(), category, tone, body, json.dumps(variables)),
        )


def list_prompts(category: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM prompts"
    args: tuple[Any, ...] = ()
    if category and category != "All":
        query += " WHERE category = ?"
        args = (category,)
    query += " ORDER BY favourite DESC, name ASC"
    with connect() as connection:
        rows = connection.execute(query, args).fetchall()
    return [{**dict(row), "variables": json.loads(row["variables"])} for row in rows]


def toggle_favourite(prompt_id: int) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE prompts SET favourite = 1 - favourite WHERE id = ?", (prompt_id,)
        )


def delete_prompt(prompt_id: int) -> None:
    with connect() as connection:
        connection.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))


# --- voice agent bookings (project 07) -------------------------------------

def save_booking(intent: str, caller: str, detail: str, transcript: str) -> int:
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO bookings (ts, intent, caller, detail, transcript) VALUES (?,?,?,?,?)",
            (_now(), intent, caller, detail, transcript),
        )
        return int(cursor.lastrowid)


def list_bookings(limit: int = 50) -> list[sqlite3.Row]:
    with connect() as connection:
        return connection.execute(
            "SELECT * FROM bookings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def mark_handled(booking_id: int) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE bookings SET handled = 1 WHERE id = ?", (booking_id,)
        )
