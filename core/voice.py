"""07 — the voice agent's brain.

WHAT THIS IS, STATED PLAINLY: a browser voice agent, not a phone agent.

The carousel shows a telephony dashboard -- inbound calls, call routing, transfers.
That needs Twilio or Vapi, a purchased number, and per-minute billing. None of that
exists here, so none of it is claimed here.

What does work, end to end, for free: speech in and speech out through the browser's
own Web Speech API, this module for the conversation and intent, and a real booking
written to SQLite. The turn logic below is the part that would carry over unchanged
if a phone number were ever attached -- swapping the transport does not change the
agent.
"""
from __future__ import annotations

from typing import Any

from core import gemini, store

BUSINESS = {
    "name": "Harbour Point Catering & Events",
    "location": "Port Harcourt, Rivers State",
    "hours": "Monday to Saturday, 9am to 7pm",
    "services": [
        "Corporate catering (from 20 to 500 guests)",
        "Private event hire of the rooftop terrace",
        "Wedding and birthday packages",
        "Daily office lunch delivery within Port Harcourt",
    ],
    "lead_time": "48 hours for catering orders, 7 days for full event hire",
    "deposit": "50% deposit confirms a booking",
}

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "What to say out loud. Under 40 words, spoken register, no bullet points or markdown.",
        },
        "intent": {
            "type": "string",
            "enum": ["booking", "pricing", "hours", "menu", "complaint",
                     "general_question", "human_handoff", "goodbye"],
        },
        "collected": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "phone": {"type": "string"},
                "date": {"type": "string"},
                "guests": {"type": "string"},
                "service": {"type": "string"},
            },
            "required": ["name", "phone", "date", "guests", "service"],
        },
        "ready_to_book": {
            "type": "boolean",
            "description": "True ONLY when name, phone, date and service are all known.",
        },
        "needs_human": {"type": "boolean"},
    },
    "required": ["reply", "intent", "collected", "ready_to_book", "needs_human"],
}


def _system_prompt() -> str:
    services = "\n".join(f"- {s}" for s in BUSINESS["services"])
    return f"""You are the phone assistant for {BUSINESS['name']} in {BUSINESS['location']}.

Opening hours: {BUSINESS['hours']}
Services:
{services}
Lead time: {BUSINESS['lead_time']}
Deposit: {BUSINESS['deposit']}

HOW TO SPEAK. Your reply is read aloud by a speech synthesiser, so:
- Under 40 words. Someone on a call cannot follow a paragraph.
- No markdown, no lists, no emoji, no asterisks. They get pronounced.
- Say figures the way a person says them: "fifty thousand naira", not "N50,000".
- One question per turn. Two questions on a call means the caller answers one.

WHAT YOU MUST NOT DO:
- Never invent a price. You have not been given a price list. If asked what something
  costs, say a coordinator will confirm the quote, and collect the details instead.
- Never confirm a date as available. You cannot see a calendar. Say it will be
  confirmed.
- Never claim to have sent an email, a text or a calendar invite.

BOOKING. To book you need: name, phone number, date, and which service. Collect the
missing ones one at a time. Set ready_to_book true only when you have all four.
Set needs_human true for complaints, refunds, or anything you were told not to answer."""


def respond(history: list[dict[str, str]], utterance: str) -> dict[str, Any]:
    """One conversational turn.

    History is passed as text rather than as multi-turn `contents` on purpose: the
    schema-constrained call is a single-shot classification plus reply, and keeping it
    single-shot means the strict schema applies to every turn identically.
    """
    transcript = "\n".join(
        f"{'Caller' if turn['role'] == 'user' else 'Agent'}: {turn['text']}"
        for turn in history[-12:]
    )
    turn = gemini.generate_json(
        f"Conversation so far:\n{transcript or '(this is the first thing they said)'}\n\n"
        f"Caller just said: {utterance}",
        TURN_SCHEMA,
        system=_system_prompt(),
        temperature=0.3,
    )

    collected = turn.get("collected", {})
    booking_id = None
    if turn.get("ready_to_book") and collected.get("name") and collected.get("phone"):
        # The booking is written by Python after checking the fields, not by the model
        # deciding it happened. This is why the agent cannot "confirm" a booking that
        # does not exist in the database.
        detail = ", ".join(
            f"{key}: {value}" for key, value in collected.items() if value
        )
        booking_id = store.save_booking(
            intent=turn.get("intent", "booking"),
            caller=collected.get("name", ""),
            detail=detail,
            transcript=transcript + f"\nCaller: {utterance}\nAgent: {turn.get('reply')}",
        )
        store.log("07 Voice Agent", "booking", detail[:200])

    turn["booking_id"] = booking_id
    return turn
