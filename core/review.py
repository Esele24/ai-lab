"""06 — industry-specific document review.

The generic version of this ("AI legal assistant") is a chatbot with a different
placeholder in the input box. What makes it industry-specific is that each industry
gets its own checklist of clauses that SHOULD be present, so the tool can report an
absence -- and an absence is the finding a general chatbot structurally cannot make,
because it only reacts to text that is there.

Same three anti-hallucination guarantees as the tender tool:
  1. temperature 0
  2. a strict responseSchema, so output is a shape and not prose
  3. a post-parse strip: any finding whose quote is not literally in the source text
     is dropped before it reaches the screen
"""
from __future__ import annotations

import re
from typing import Any

from core import gemini

INDUSTRIES: dict[str, dict[str, Any]] = {
    "Legal / Contracts": {
        "role": "a Nigerian commercial contracts lawyer reviewing a draft for your client",
        "checklist": [
            "Governing law and jurisdiction", "Termination rights and notice period",
            "Limitation of liability", "Indemnity", "Confidentiality",
            "Payment terms and late-payment interest", "Dispute resolution / arbitration",
            "Force majeure", "Assignment and change of control", "Intellectual property ownership",
        ],
    },
    "Oil & Gas Services": {
        "role": "a prequalification officer assessing a service company's tender response",
        "checklist": [
            "NCDMB / NOGIC certificate and expiry", "DPR / NUPRC permit",
            "ISO 9001 quality certification", "HSE policy and incident statistics",
            "Local content plan", "Equipment register and ownership proof",
            "Audited financial statements", "Comparable project references",
            "Insurance cover and limits", "CAC incorporation documents",
        ],
    },
    "Real Estate": {
        "role": "a property lawyer verifying a Nigerian land or lease transaction",
        "checklist": [
            "Certificate of Occupancy or title document", "Governor's consent",
            "Survey plan and boundary description", "Existing encumbrances or mortgage",
            "Rent, service charge and review mechanism", "Repair and maintenance obligations",
            "Permitted use", "Renewal and quiet-enjoyment covenant",
        ],
    },
    "Healthcare": {
        "role": "a clinic administrator reviewing a service or supplier agreement",
        "checklist": [
            "Patient data protection (NDPA compliance)", "Professional indemnity insurance",
            "Practitioner licensing and credentials", "Standard of care obligations",
            "Record retention period", "Consent and disclosure terms",
            "Termination and continuity of care", "Equipment servicing responsibility",
        ],
    },
    "E-commerce": {
        "role": "an operations lead reviewing a vendor, logistics or payment agreement",
        "checklist": [
            "Delivery timelines and liability for loss", "Returns and refund policy",
            "Payment settlement period and charges", "Chargeback responsibility",
            "Product warranty and defect handling", "Data protection for customer records",
            "Exclusivity or territory limits", "Price change mechanism",
        ],
    },
    "Education": {
        "role": "a school administrator reviewing a supplier, staff or partnership agreement",
        "checklist": [
            "Safeguarding and child protection", "Staff qualification requirements",
            "Term dates and delivery schedule", "Fee structure and payment schedule",
            "Liability for pupil injury", "Data protection for pupil records",
            "Termination mid-session", "Intellectual property in teaching materials",
        ],
    },
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "parties": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "description": "3 sentences maximum."},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause": {"type": "string", "description": "Which checklist item this concerns."},
                    "status": {"type": "string", "enum": ["PRESENT", "WEAK", "ABSENT"]},
                    "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "quote": {
                        "type": "string",
                        "description": "VERBATIM text copied from the document. Empty string if status is ABSENT.",
                    },
                    "concern": {"type": "string", "description": "What the risk is, one or two sentences."},
                    "recommendation": {"type": "string", "description": "The concrete change to ask for."},
                },
                "required": ["clause", "status", "severity", "quote", "concern", "recommendation"],
            },
        },
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "party": {"type": "string"},
                    "obligation": {"type": "string"},
                    "deadline": {"type": "string", "description": "As written, or empty if none stated."},
                },
                "required": ["party", "obligation", "deadline"],
            },
        },
    },
    "required": ["document_type", "parties", "summary", "findings", "obligations"],
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def verify_quotes(review: dict[str, Any], source: str) -> tuple[dict[str, Any], list[str]]:
    """Drop any PRESENT/WEAK finding whose quote is not actually in the document.

    This is the check that turns 'probably grounded' into 'grounded'. A finding that
    cites a clause the document does not contain is worse than no finding at all --
    it is the one a user would act on and be embarrassed by.

    Compared on normalised text because extraction mangles whitespace, quote marks
    and hyphenation, and a real quote should not be discarded over a curly apostrophe.
    """
    haystack = _normalise(source)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for finding in review.get("findings", []):
        quote = (finding.get("quote") or "").strip()
        if finding.get("status") == "ABSENT":
            finding["quote"] = ""
            kept.append(finding)
            continue
        if not quote:
            # Claims the clause is present but cannot show it. Demote rather than delete,
            # so the user still sees the checklist item was considered.
            finding["status"] = "ABSENT"
            finding["concern"] = ("The model reported this clause but could not quote it, "
                                 "so it is reported as absent. " + finding.get("concern", ""))
            kept.append(finding)
            continue
        needle = _normalise(quote)
        if needle and needle in haystack:
            kept.append(finding)
        else:
            dropped.append(f"{finding.get('clause')}: quote not found in document")
    review["findings"] = kept
    return review, dropped


def review_document(industry: str, text: str) -> dict[str, Any]:
    if industry not in INDUSTRIES:
        raise ValueError(f"Unknown industry: {industry}")
    if len(text.strip()) < 200:
        raise ValueError("That document is too short to review (under 200 characters).")

    profile = INDUSTRIES[industry]
    checklist = "\n".join(f"- {item}" for item in profile["checklist"])
    # A long contract is truncated rather than silently half-read, and the user is told.
    truncated = text[:120_000]

    review = gemini.generate_json(
        f"Checklist for this industry:\n{checklist}\n\nDocument:\n\"\"\"\n{truncated}\n\"\"\"",
        REVIEW_SCHEMA,
        system=(
            f"You are {profile['role']}.\n"
            "Assess the document against EVERY item on the checklist -- one finding per "
            "checklist item, including the items that are missing.\n"
            "status=PRESENT means adequately covered; WEAK means present but one-sided, "
            "vague or unenforceable; ABSENT means not in the document at all.\n"
            "For PRESENT and WEAK you MUST copy the exact wording from the document into "
            "`quote`. Copy it character for character. Never paraphrase into `quote`, and "
            "never write a quote for an ABSENT item.\n"
            "Severity is about consequence to the reader, not about how interesting the clause is.\n"
            "Do not give legal advice framed as certainty; state the risk and the ask."
        ),
    )
    review, dropped = verify_quotes(review, truncated)
    review["dropped_findings"] = dropped
    review["truncated"] = len(text) > 120_000
    review["industry"] = industry
    review["checklist_size"] = len(profile["checklist"])
    return review
