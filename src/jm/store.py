"""SQLite persistence for sessions, turns, and cards (FTS5 + embedding blobs)."""

from __future__ import annotations

import sqlite3
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jm.types import (
    CORRECT_STATUSES,
    CardIn,
    CardOut,
    Kind,
    Lifecycle,
    SourceTurn,
    TurnIn,
    Validity,
    compact_text,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts TEXT,
    UNIQUE(session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    kind TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    event_date TEXT,
    validity TEXT NOT NULL DEFAULT 'active',
    note TEXT,
    superseded_by TEXT,
    turn_start INTEGER NOT NULL,
    turn_end INTEGER NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cards_session ON cards(session_id);
CREATE INDEX IF NOT EXISTS idx_cards_validity ON cards(validity);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_index);

CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    id UNINDEXED,
    text
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pack_embedding(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def new_memory_id() -> str:
    return "m_" + uuid.uuid4().hex[:16]


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_session(self, session_id: str, started_at: str | None) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (id, started_at, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    started_at = COALESCE(excluded.started_at, sessions.started_at)
                """,
                (session_id, started_at, utc_now()),
            )
            self._conn.commit()

    def insert_turns(self, session_id: str, turns: list[TurnIn]) -> list[int]:
        indices = _assign_indices(self._max_turn_index(session_id), turns)
        ids: list[int] = []
        with self._lock:
            for turn, index in zip(turns, indices, strict=True):
                row = self._conn.execute(
                    """
                    INSERT INTO turns (session_id, turn_index, role, content, ts)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, turn_index) DO UPDATE SET
                        role = excluded.role,
                        content = excluded.content,
                        ts = COALESCE(excluded.ts, turns.ts)
                    RETURNING id
                    """,
                    (session_id, index, turn.role, turn.content, turn.ts),
                ).fetchone()
                ids.append(int(row["id"]))
            self._conn.commit()
        return ids

    def insert_cards(
        self,
        session_id: str,
        cards: list[CardIn],
        embeddings: list[list[float]],
    ) -> list[str]:
        if len(cards) != len(embeddings):
            raise ValueError("each card needs one embedding")
        known = set(self.turn_indices(session_id))
        ids: list[str] = []
        with self._lock:
            for card, vector in zip(cards, embeddings, strict=True):
                span = range(card.turn_start, card.turn_end + 1)
                if known and not set(span).issubset(known):
                    raise ValueError(
                        f"card provenance {card.turn_start}-{card.turn_end} "
                        "is not in this session"
                    )
                memory_id = new_memory_id()
                text = card.compact()
                self._conn.execute(
                    """
                    INSERT INTO cards (
                        id, session_id, subject, fact, kind, lifecycle,
                        event_date, validity, turn_start, turn_end,
                        embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        session_id,
                        card.subject,
                        card.fact,
                        card.kind.value,
                        card.lifecycle.value,
                        card.event_date,
                        card.turn_start,
                        card.turn_end,
                        pack_embedding(vector),
                        utc_now(),
                    ),
                )
                self._conn.execute(
                    "INSERT INTO cards_fts (id, text) VALUES (?, ?)",
                    (memory_id, text),
                )
                ids.append(memory_id)
            self._conn.commit()
        return ids

    def turn_indices(self, session_id: str) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_index FROM turns WHERE session_id = ? ORDER BY turn_index",
                (session_id,),
            ).fetchall()
        return [int(row["turn_index"]) for row in rows]

    def get_card(self, memory_id: str, *, with_span: bool = True) -> CardOut | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cards WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            return None
        card = _row_to_card(row)
        if with_span:
            card.source_span = self.source_span(
                card.session_id, card.turn_start, card.turn_end
            )
        return card

    def set_validity(
        self,
        memory_id: str,
        status: Validity,
        note: str | None = None,
        superseded_by: str | None = None,
    ) -> CardOut:
        if status not in CORRECT_STATUSES:
            raise ValueError("status must be wrong, dead, or superseded")
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM cards WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(memory_id)
            self._conn.execute(
                """
                UPDATE cards
                SET validity = ?, note = COALESCE(?, note), superseded_by = ?
                WHERE id = ?
                """,
                (status.value, note, superseded_by, memory_id),
            )
            self._conn.commit()
        card = self.get_card(memory_id, with_span=False)
        assert card is not None
        return card

    def source_span(
        self, session_id: str, turn_start: int, turn_end: int
    ) -> list[SourceTurn]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT turn_index, role, content, ts
                FROM turns
                WHERE session_id = ? AND turn_index BETWEEN ? AND ?
                ORDER BY turn_index
                """,
                (session_id, turn_start, turn_end),
            ).fetchall()
        return [
            SourceTurn(
                index=int(row["turn_index"]),
                role=row["role"],
                content=row["content"],
                ts=row["ts"],
            )
            for row in rows
        ]

    def active_cards(self) -> list[CardOut]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM cards WHERE validity = 'active'"
            ).fetchall()
        return [_row_to_card(row) for row in rows]

    def embeddings_for(self, memory_ids: Iterable[str]) -> dict[str, list[float]]:
        ids = list(memory_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, embedding FROM cards WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        out: dict[str, list[float]] = {}
        for row in rows:
            vector = unpack_embedding(row["embedding"])
            if vector is not None:
                out[row["id"]] = vector
        return out

    def fts_ids(self, match: str, limit: int, *, active_only: bool = True) -> list[str]:
        if not match.strip():
            return []
        where = "AND cards.validity = 'active'" if active_only else ""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"""
                    SELECT cards_fts.id AS id
                    FROM cards_fts
                    JOIN cards ON cards.id = cards_fts.id
                    WHERE cards_fts MATCH ? {where}
                    ORDER BY bm25(cards_fts)
                    LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [row["id"] for row in rows]

    def cards_by_ids(self, memory_ids: list[str]) -> list[CardOut]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM cards WHERE id IN ({placeholders})",
                memory_ids,
            ).fetchall()
        by_id = {row["id"]: _row_to_card(row) for row in rows}
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def dump(self, session_id: str | None = None) -> dict[str, Any]:
        params: tuple[Any, ...] = ()
        session_clause = ""
        if session_id:
            session_clause = "WHERE id = ?"
            params = (session_id,)
        with self._lock:
            sessions = [
                dict(row)
                for row in self._conn.execute(
                    f"SELECT id, started_at, created_at FROM sessions {session_clause} ORDER BY created_at",
                    params,
                ).fetchall()
            ]
            if session_id:
                turns = self._conn.execute(
                    """
                    SELECT id, session_id, turn_index AS "index", role, content, ts
                    FROM turns WHERE session_id = ? ORDER BY turn_index
                    """,
                    (session_id,),
                ).fetchall()
                cards = self._conn.execute(
                    "SELECT * FROM cards WHERE session_id = ? ORDER BY created_at",
                    (session_id,),
                ).fetchall()
            else:
                turns = self._conn.execute(
                    """
                    SELECT id, session_id, turn_index AS "index", role, content, ts
                    FROM turns ORDER BY session_id, turn_index
                    """
                ).fetchall()
                cards = self._conn.execute(
                    "SELECT * FROM cards ORDER BY created_at"
                ).fetchall()
        turn_payload = [dict(row) for row in turns]
        card_payload = []
        validity_counts: dict[str, int] = {v.value: 0 for v in Validity}
        for row in cards:
            item = {k: row[k] for k in row.keys() if k != "embedding"}
            item["memory_id"] = item.pop("id")
            item["text"] = compact_text(item["subject"], item["fact"])
            validity_counts[item["validity"]] = (
                validity_counts.get(item["validity"], 0) + 1
            )
            card_payload.append(item)
        return {
            "stats": {
                "sessions": len(sessions),
                "turns": len(turn_payload),
                "cards": len(card_payload),
                "by_validity": validity_counts,
            },
            "sessions": sessions,
            "turns": turn_payload,
            "cards": card_payload,
        }

    def _max_turn_index(self, session_id: str) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(turn_index) AS m FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None or row["m"] is None:
            return None
        return int(row["m"])


def _assign_indices(existing_max: int | None, turns: list[TurnIn]) -> list[int]:
    specified = [turn.index is not None for turn in turns]
    if any(specified) and not all(specified):
        raise ValueError("either every turn has index, or none do")
    if all(specified):
        return [int(turn.index) for turn in turns]  # type: ignore[arg-type]
    start = 0 if existing_max is None else existing_max + 1
    return list(range(start, start + len(turns)))


def _row_to_card(row: sqlite3.Row) -> CardOut:
    return CardOut(
        memory_id=row["id"],
        session_id=row["session_id"],
        subject=row["subject"],
        fact=row["fact"],
        text=compact_text(row["subject"], row["fact"]),
        kind=Kind(row["kind"]),
        lifecycle=Lifecycle(row["lifecycle"]),
        event_date=row["event_date"],
        validity=Validity(row["validity"]),
        turn_start=int(row["turn_start"]),
        turn_end=int(row["turn_end"]),
        note=row["note"],
        superseded_by=row["superseded_by"],
    )
