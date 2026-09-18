import pytest
from pydantic import ValidationError

from jm.extract import parse_cards, parse_turns


def test_parse_turns_requires_content():
    with pytest.raises(ValidationError):
        parse_turns([{"role": "user", "content": "   "}])
    with pytest.raises(ValueError, match="empty"):
        parse_turns([])


def test_parse_cards_span_and_kind_alias():
    cards = parse_cards(
        [
            {
                "subject": "User",
                "fact": "likes tea",
                "kind": "factual",
                "status": "stable",
                "event_date": "unknown",
                "turn_start": 0,
                "turn_end": 0,
            }
        ]
    )
    assert cards[0].kind.value == "fact"
    assert cards[0].lifecycle.value == "stable"
    with pytest.raises(ValidationError):
        parse_cards(
            [
                {
                    "subject": "User",
                    "fact": "likes tea",
                    "kind": "fact",
                    "turn_start": 2,
                    "turn_end": 1,
                }
            ]
        )
