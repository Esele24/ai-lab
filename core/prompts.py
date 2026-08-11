"""05 — prompt builder, library, and A/B comparison.

The comparison is the only part of this that is hard, and the reason is worth knowing:
an LLM asked to judge two outputs has two measurable biases. It prefers the longer
answer, and it prefers whichever it was shown first (position bias).

So the compare below does three things a naive version does not:
  - shuffles which output is labelled A, so position bias is not systematic
  - scores against named criteria rather than "which is better"
  - reports each output's length beside its score, so you can see for yourself when
    a win is really just verbosity

A judge is still a weaker signal than your own eyes on the output. It is a filter,
not a verdict.
"""
from __future__ import annotations

import random
import re
from typing import Any

from core import gemini, store

CATEGORIES = ["Marketing", "Content", "Social Media", "Email", "Research", "Code", "Other"]
TONES = ["Professional", "Friendly", "Direct", "Persuasive", "Academic", "Plain English"]
FORMATS = ["Prose", "Bullet points", "Numbered steps", "Table", "JSON", "Short paragraph"]

BUILD_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "The finished prompt, ready to paste."},
        "variables": {
            "type": "array", "items": {"type": "string"},
            "description": "Placeholder names used in the prompt, without braces.",
        },
        "notes": {"type": "string", "description": "One or two sentences on why it is built this way."},
    },
    "required": ["prompt", "variables", "notes"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["A", "B"]},
                    "instruction_following": {"type": "integer", "description": "0-25"},
                    "specificity": {"type": "integer", "description": "0-25"},
                    "usefulness": {"type": "integer", "description": "0-25"},
                    "conciseness": {"type": "integer", "description": "0-25"},
                    "comment": {
                        "type": "string",
                        "description": "One sentence about THIS output only. Never mention the other one, and never write the letters A or B.",
                    },
                },
                "required": ["label", "instruction_following", "specificity",
                             "usefulness", "conciseness", "comment"],
            },
        },
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {
            "type": "string",
            "description": ("One sentence naming the deciding difference, phrased as a property "
                            "of the winner, e.g. 'the winner produced the message itself rather "
                            "than describing one'. NEVER write the letters A or B -- the labels "
                            "are re-mapped after you answer, so any label you write will be wrong."),
        },
    },
    "required": ["scores", "winner", "reason"],
}

VARIABLE = re.compile(r"\{([a-zA-Z0-9_ ]+)\}")


def find_variables(template: str) -> list[str]:
    seen: list[str] = []
    for name in VARIABLE.findall(template):
        if name not in seen:
            seen.append(name)
    return seen


def fill(template: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """Substitute {placeholders}. Returns the filled text and any left unfilled.

    Unfilled placeholders are reported rather than silently left in the text -- a
    prompt sent with a literal '{product_name}' in it is the most common way this
    kind of tool wastes a call.
    """
    missing: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group(1)
        value = values.get(name, "").strip()
        if not value:
            missing.append(name)
            return match.group(0)
        return value

    return VARIABLE.sub(replace, template), missing


def build(task: str, details: str, tone: str, output_format: str) -> dict[str, Any]:
    result = gemini.generate_json(
        f"Task type: {task}\nWhat the user wants: {details}\n"
        f"Tone: {tone}\nOutput format: {output_format}",
        BUILD_SCHEMA,
        system=(
            "You write prompts for other people to use with an LLM.\n"
            "A good prompt states the role, the task, the constraints, and the output "
            "format explicitly, and says what NOT to do where that is the likely failure.\n"
            "Use {placeholder_name} braces for anything the user should swap per use. "
            "Prefer three or four well-chosen placeholders over ten.\n"
            "Do not write a preamble like 'Here is your prompt'. Return the prompt itself."
        ),
        temperature=0.4,
    )
    result["variables"] = result.get("variables") or find_variables(result.get("prompt", ""))
    store.log("05 Prompt System", "build", task)
    return result


def compare(prompt_a: str, prompt_b: str, test_input: str = "") -> dict[str, Any]:
    """Run both prompts, then score them against criteria with the order shuffled."""
    def run(prompt: str) -> str:
        full = f"{prompt}\n\n{test_input}".strip() if test_input else prompt
        return gemini.generate(full, temperature=0.4)

    output_a, output_b = run(prompt_a), run(prompt_b)

    # Shuffle which real output gets shown as "A" to the judge.
    flipped = random.random() < 0.5
    shown_a, shown_b = (output_b, output_a) if flipped else (output_a, output_b)

    verdict = gemini.generate_json(
        f"The instruction both were given:\n{test_input or '(the prompt itself was the instruction)'}\n\n"
        f"--- Output A ---\n{shown_a[:6000]}\n\n--- Output B ---\n{shown_b[:6000]}",
        JUDGE_SCHEMA,
        system=(
            "You score two candidate outputs against four criteria, 0-25 each, 100 total.\n"
            "instruction_following: did it do what was asked, including format and length.\n"
            "specificity: concrete detail over generic filler.\n"
            "usefulness: could the reader act on it.\n"
            "conciseness: reward saying it in fewer words. Length is NOT quality -- "
            "penalise padding, repetition and restating the question.\n"
            "Score both. Then pick a winner, or 'tie' if the totals are within 4 points.\n"
            "CRITICAL: the labels A and B are shuffled before you see them and re-mapped after "
            "you answer. Set `winner` to the label as shown to you, but NEVER write the letters "
            "A or B inside `reason` or any `comment` -- describe the output's property instead."
        ),
    )

    # Translate the judge's labels back to the real prompts.
    def real_label(label: str) -> str:
        if not flipped:
            return label
        return "B" if label == "A" else "A"

    for score in verdict.get("scores", []):
        score["label"] = real_label(score["label"])
        score["total"] = sum(
            int(score.get(key, 0)) for key in
            ("instruction_following", "specificity", "usefulness", "conciseness")
        )
    if verdict.get("winner") in ("A", "B"):
        verdict["winner"] = real_label(verdict["winner"])

    store.log("05 Prompt System", "compare", f"winner={verdict.get('winner')}")
    return {
        "output_a": output_a,
        "output_b": output_b,
        "words_a": len(output_a.split()),
        "words_b": len(output_b.split()),
        "verdict": verdict,
        "judge_saw_swapped": flipped,
    }
