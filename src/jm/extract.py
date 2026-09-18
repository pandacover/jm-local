"""Validation for host-extracted cards. The host is the extractor."""

from __future__ import annotations

from jm.types import CardIn, TurnIn

EXTRACTION_CONTRACT = """\
Extract a few durable, answer-bearing cards from this slice, then pass them in `cards` on this same call. Each card is one independently retrievable claim with an explicit subject. Do not merge unrelated claims. Prefer a short list over a complete changelog.

Keep:
- preferences, standing facts, dated events, and decisions the user would ask about later
- subject-explicit compact text; `fact` should still make sense next to `subject`
- exact entities, dates, values, durations, and relations when they matter
- event cards with subject + `event_date` so later recall can join them
- lifecycle: completed events vs plans vs ongoing states vs stable facts

Skip:
- process, tooling, and changelog details (how a server launches, tool lists, pull requests, test counts) unless the user asked to remember them
- restating what the assistant just did
- confirmations or interpretations the user did not make
- a card per implementation detail of the current project

Same session + subject + kind + turn span updates the existing card; do not mint duplicates. To fix a bad extraction, amend via memory_correct (new fact) instead of writing a sibling card.

Card fields:
- subject: who or what the card is about
- fact: the compact note (one claim)
- kind: fact | event | preference | state | measurement | relation | list | interaction
- lifecycle: stable | completed | planned | ongoing (default stable)
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
