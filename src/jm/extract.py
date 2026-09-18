"""Validation for host-extracted cards. The host is the extractor."""

from __future__ import annotations

from jm.types import CardIn, TurnIn

EXTRACTION_CONTRACT = """\
Extract durable, answer-bearing information from the supplied conversation as atomic memory cards, then pass them in `cards` on this same call.

Each card is one independently retrievable claim: a fact, event, preference, state, measurement, relation, list, or interaction, with an explicit subject. Do not merge unrelated claims.

Rules:
- Write subject-explicit compact text. The `fact` field should still make sense next to `subject`.
- Preserve exact entities, dates, values, durations, and relations when they matter.
- Resolve local references from the surrounding turns. Do not invent details or use outside knowledge.
- Event cards keep subject and time (`event_date`) so later recall can join them.
- Distinguish completed events from plans, ongoing states, and stable facts via `lifecycle`.
- When important interaction content cannot be represented compactly, keep a short card that points at the original turns.

Card fields:
- subject: who or what the card is about
- fact: the compact note (one claim)
- kind: fact | event | preference | state | measurement | relation | list | interaction
- lifecycle: stable | completed | planned | ongoing
- event_date: YYYY-MM-DD, or "unknown"
- turn_start, turn_end: inclusive session turn indices for provenance
"""


def parse_turns(raw: list[dict]) -> list[TurnIn]:
    if not raw:
        raise ValueError("turns must not be empty")
    return [TurnIn.model_validate(item) for item in raw]


def parse_cards(raw: list[dict] | None) -> list[CardIn]:
    if not raw:
        return []
    return [CardIn.model_validate(item) for item in raw]
