import pytest

from jm.embed import FakeEmbedder
from jm.memory import Memory


def mem(tmp_path) -> Memory:
    return Memory(tmp_path / "memory.db", embedder=FakeEmbedder())


SESSION_TURNS = [
    {
        "role": "user",
        "content": "I prefer small, quiet hotels. Crowds make me anxious.",
        "index": 0,
    },
    {
        "role": "assistant",
        "content": "Noted. I will keep that in mind for lodging.",
        "index": 1,
    },
    {
        "role": "user",
        "content": "I visited Kyoto, Japan in April 2025.",
        "index": 2,
    },
    {
        "role": "assistant",
        "content": "8:00 breakfast near the hotel; 9:30 walk along the Kamo River; 12:30 leave for Kyoto Station.",
        "index": 3,
    },
    {
        "role": "user",
        "content": "I visited Lisbon, Portugal in October 2025.",
        "index": 4,
    },
]

SESSION_CARDS = [
    {
        "subject": "User",
        "fact": "prefers small, quiet hotels and feels anxious in crowded places",
        "kind": "preference",
        "lifecycle": "stable",
        "event_date": "unknown",
        "turn_start": 0,
        "turn_end": 1,
    },
    {
        "subject": "User",
        "fact": "visited Kyoto, Japan",
        "kind": "event",
        "lifecycle": "completed",
        "event_date": "2025-04-01",
        "turn_start": 2,
        "turn_end": 2,
    },
    {
        "subject": "Assistant",
        "fact": "created a timed itinerary for the user's last morning in Kyoto",
        "kind": "interaction",
        "lifecycle": "completed",
        "event_date": "2025-04-01",
        "turn_start": 3,
        "turn_end": 3,
    },
    {
        "subject": "User",
        "fact": "visited Lisbon, Portugal",
        "kind": "event",
        "lifecycle": "completed",
        "event_date": "2025-10-01",
        "turn_start": 4,
        "turn_end": 4,
    },
]


@pytest.fixture
def loaded(tmp_path) -> Memory:
    memory = mem(tmp_path)
    memory.remember("s1", SESSION_TURNS, SESSION_CARDS, session_time="2025-04-01")
    return memory


def test_remember_raw_without_cards(tmp_path):
    memory = mem(tmp_path)
    out = memory.remember("s1", SESSION_TURNS[:1])
    assert out["session_id"] == "s1"
    assert out["turn_ids"]
    assert out["memory_ids"] == []
    dump = memory.dump("s1")
    assert dump["stats"]["turns"] == 1
    assert dump["stats"]["cards"] == 0


def test_status_alias_for_lifecycle(tmp_path):
    memory = mem(tmp_path)
    out = memory.remember(
        "s1",
        SESSION_TURNS[:1],
        [
            {
                "subject": "User",
                "fact": "prefers small, quiet hotels",
                "kind": "factual",
                "status": "stable",
                "event_date": "unknown",
                "turn_start": 0,
                "turn_end": 0,
            }
        ],
    )
    card = memory.get(out["memory_ids"][0])
    assert card["kind"] == "fact"
    assert card["lifecycle"] == "stable"


def test_lookup_returns_preference(loaded: Memory):
    result = loaded.recall("Would a quiet Kyoto guesthouse suit me?")
    assert result["mode"] == "lookup"
    assert len(result["cards"]) <= 10
    assert result["memory_ids"] == [c["memory_id"] for c in result["cards"]]
    texts = " ".join(c["text"].lower() for c in result["cards"])
    assert "quiet" in texts or "hotel" in texts
    assert "source_span" not in result["cards"][0]


def test_compose_keeps_budget_and_finds_trips(loaded: Memory):
    result = loaded.recall("Write a one-line summary of my 2025 trips")
    assert result["mode"] == "compose"
    assert len(result["cards"]) <= 10
    texts = " ".join(c["text"].lower() for c in result["cards"])
    assert "kyoto" in texts
    assert "lisbon" in texts


def test_replay_includes_source_span(loaded: Memory):
    result = loaded.recall("Pull up the schedule for my last morning in Kyoto")
    assert result["mode"] == "replay"
    assert result["cards"]
    assert "source_span" in result["cards"][0]
    spans = " ".join(
        turn["content"]
        for card in result["cards"]
        for turn in card["source_span"]
    )
    assert "8:00" in spans or "Kamo" in spans or "itinerary" in spans.lower()


def test_correct_drops_from_recall_not_get(loaded: Memory):
    hit = loaded.recall("What hotel style does the user prefer?")
    target = next(c for c in hit["cards"] if "hotel" in c["text"].lower())
    updated = loaded.correct(target["memory_id"], "wrong", note="stale")
    assert updated["validity"] == "wrong"
    later = loaded.recall("What hotel style does the user prefer?")
    assert target["memory_id"] not in later["memory_ids"]
    fetched = loaded.get(target["memory_id"])
    assert fetched["validity"] == "wrong"
    assert fetched["source_span"]


def test_dump_includes_inactive_and_omits_embeddings(loaded: Memory):
    mid = loaded.dump()["cards"][0]["memory_id"]
    loaded.correct(mid, "dead")
    dump = loaded.dump()
    assert dump["stats"]["cards"] == 4
    assert dump["stats"]["by_validity"]["dead"] == 1
    assert dump["stats"]["turns"] == 5
    for card in dump["cards"]:
        assert "embedding" not in card
        assert "memory_id" in card
        assert "text" in card


def test_top_k_budget(tmp_path):
    memory = mem(tmp_path)
    turns = [
        {"role": "user", "content": f"Note number {i} about travel plans.", "index": i}
        for i in range(15)
    ]
    cards = [
        {
            "subject": "User",
            "fact": f"travel note {i} about itineraries and hotels",
            "kind": "fact",
            "lifecycle": "stable",
            "event_date": "unknown",
            "turn_start": i,
            "turn_end": i,
        }
        for i in range(15)
    ]
    memory.remember("bulk", turns, cards)
    result = memory.recall("What travel notes and hotel facts do I have?")
    assert result["mode"] == "lookup"
    assert len(result["cards"]) == 10
