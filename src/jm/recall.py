"""Plan then fetch. The host never chooses lookup / compose / replay."""

from __future__ import annotations

from jm.embed import Embedder
from jm.planner import plan, rewrites
from jm.search import retrieve
from jm.store import Store
from jm.types import TOP_K, Mode, RecallResult


def run_recall(
    store: Store,
    embedder: Embedder,
    question: str,
    *,
    as_of: str | None = None,
    k: int = TOP_K,
) -> RecallResult:
    mode = plan(question)
    extra = rewrites(question) if mode is Mode.compose else None
    cards = retrieve(
        store,
        embedder,
        question,
        extra_queries=extra,
        as_of=as_of,
        limit=k,
    )
    if mode is Mode.replay:
        for card in cards:
            card.source_span = store.source_span(
                card.session_id, card.turn_start, card.turn_end
            )
    return RecallResult(
        mode=mode,
        cards=cards,
        memory_ids=[card.memory_id for card in cards],
    )
