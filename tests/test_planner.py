from jm.planner import plan, rewrites
from jm.types import Mode


def test_lookup_is_default():
    assert plan("Would a quiet Kyoto guesthouse suit me?") is Mode.lookup
    assert plan("What hotel style does the user prefer?") is Mode.lookup


def test_compose_for_distributed_questions():
    assert plan("Can you write a one-line summary of my 2025 trips?") is Mode.compose
    assert plan("List all of my visits across sessions") is Mode.compose
    assert plan("How many trips did I take?") is Mode.compose
    assert plan("Compare Kyoto and Lisbon") is Mode.compose


def test_replay_precedes_compose():
    assert plan("Pull up the schedule for my last morning in Kyoto") is Mode.replay
    assert plan("What did I say about the quiet hotel?") is Mode.replay
    assert plan("Show me the original itinerary") is Mode.replay
    assert plan("Quote the exact plan") is Mode.replay


def test_compose_rewrites_are_answer_free_and_capped():
    views = rewrites("Can you write a one-line summary of my 2025 trips to Kyoto?")
    assert 1 <= len(views) <= 2
    joined = " ".join(views).lower()
    assert "2025" in joined or "kyoto" in joined
    assert "yes" not in joined
