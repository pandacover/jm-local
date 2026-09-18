"""Facade: remember, recall, correct, get, dump."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jm.embed import Embedder, FakeEmbedder, OllamaEmbedder
from jm.extract import parse_cards, parse_turns
from jm.recall import run_recall
from jm.store import Store
from jm.types import CardIn, Kind, Lifecycle, Validity


class Memory:
    def __init__(self, db_path: str | Path, embedder: Embedder | None = None) -> None:
        self.store = Store(db_path)
        self.embedder = embedder or OllamaEmbedder()

    def remember(
        self,
        session_id: str,
        turns: list[dict[str, Any]],
        cards: list[dict[str, Any]] | None = None,
        session_time: str | None = None,
    ) -> dict[str, Any]:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        parsed_turns = parse_turns(turns)
        parsed_cards = parse_cards(cards)
        self.store.upsert_session(session_id, session_time)
        turn_ids = self.store.insert_turns(session_id, parsed_turns)
        memory_ids: list[str] = []
        if parsed_cards:
            texts = [card.compact() for card in parsed_cards]
            vectors = self.embedder.embed_docs(texts)
            memory_ids = self.store.insert_cards(session_id, parsed_cards, vectors)
        return {
            "session_id": session_id,
            "turn_ids": turn_ids,
            "memory_ids": memory_ids,
        }

    def recall(self, question: str, as_of: str | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        result = run_recall(self.store, self.embedder, question, as_of=as_of)
        return result.to_public()

    def correct(
        self,
        memory_id: str,
        status: str | None = None,
        note: str | None = None,
        superseded_by: str | None = None,
        *,
        subject: str | None = None,
        fact: str | None = None,
        kind: str | None = None,
        lifecycle: str | None = None,
        event_date: str | None = None,
    ) -> dict[str, Any]:
        amending = any(
            value is not None
            for value in (subject, fact, kind, lifecycle, event_date)
        )
        if status is None and not amending:
            raise ValueError("status or an amendment field is required")

        card = None
        if amending:
            current = self.store.get_card(memory_id, with_span=False)
            if current is None:
                raise KeyError(memory_id)
            next_subject = subject if subject is not None else current.subject
            next_fact = fact if fact is not None else current.fact
            next_kind = Kind(kind) if kind is not None else current.kind
            next_lifecycle = (
                Lifecycle(lifecycle) if lifecycle is not None else current.lifecycle
            )
            next_event = event_date if event_date is not None else current.event_date
            parsed = CardIn(
                subject=next_subject,
                fact=next_fact,
                kind=next_kind,
                lifecycle=next_lifecycle,
                event_date=next_event,
                turn_start=current.turn_start,
                turn_end=current.turn_end,
            )
            vector = self.embedder.embed_docs([parsed.compact()])[0]
            card = self.store.amend_card(
                memory_id,
                subject=parsed.subject,
                fact=parsed.fact,
                kind=parsed.kind.value,
                lifecycle=parsed.lifecycle.value,
                event_date=parsed.event_date,
                embedding=vector,
                note=note,
            )

        if status is not None:
            validity = Validity(status)
            card = self.store.set_validity(
                memory_id, validity, note=note, superseded_by=superseded_by
            )

        assert card is not None
        return card.to_public(include_span=False)

    def get(self, memory_id: str) -> dict[str, Any]:
        card = self.store.get_card(memory_id, with_span=True)
        if card is None:
            raise KeyError(memory_id)
        return card.to_public(include_span=True)

    def dump(self, session_id: str | None = None) -> dict[str, Any]:
        return self.store.dump(session_id)

    def close(self) -> None:
        self.store.close()


def default_memory() -> Memory:
    import os

    db = os.environ.get("JM_DB") or str(
        Path.home() / ".local" / "share" / "jm" / "memory.db"
    )
    embedder: Embedder
    if os.environ.get("JM_FAKE_EMBED") == "1":
        embedder = FakeEmbedder()
    else:
        embedder = OllamaEmbedder()
    return Memory(db, embedder=embedder)
