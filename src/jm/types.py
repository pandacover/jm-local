"""Shared models for turns, cards, and recall results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Kind(str, Enum):
    fact = "fact"
    event = "event"
    preference = "preference"
    state = "state"
    measurement = "measurement"
    relation = "relation"
    list = "list"
    interaction = "interaction"


class Lifecycle(str, Enum):
    stable = "stable"
    completed = "completed"
    planned = "planned"
    ongoing = "ongoing"


class Validity(str, Enum):
    active = "active"
    wrong = "wrong"
    dead = "dead"
    superseded = "superseded"


class Mode(str, Enum):
    lookup = "lookup"
    compose = "compose"
    replay = "replay"


class TurnIn(BaseModel):
    role: str
    content: str
    ts: str | None = None
    index: int | None = None

    @field_validator("role")
    @classmethod
    def role_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("role must not be empty")
        return value

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        return value


class CardIn(BaseModel):
    subject: str
    fact: str
    kind: Kind
    lifecycle: Lifecycle = Lifecycle.stable
    event_date: str | None = None
    turn_start: int
    turn_end: int

    @field_validator("subject", "fact")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"factual", "facts"}:
            return Kind.fact
        return value

    @field_validator("event_date")
    @classmethod
    def normalize_event_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or value.lower() == "unknown":
            return "unknown"
        return value

    @model_validator(mode="before")
    @classmethod
    def lifecycle_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "lifecycle" not in data and "status" in data:
            data = dict(data)
            data["lifecycle"] = data.pop("status")
        return data

    @model_validator(mode="after")
    def span_order(self) -> CardIn:
        if self.turn_start < 0 or self.turn_end < 0:
            raise ValueError("turn indices must be >= 0")
        if self.turn_end < self.turn_start:
            raise ValueError("turn_end must be >= turn_start")
        return self

    def compact(self) -> str:
        return compact_text(self.subject, self.fact)


def compact_text(subject: str, fact: str) -> str:
    return f"{subject}: {fact}"


class SourceTurn(BaseModel):
    index: int
    role: str
    content: str
    ts: str | None = None


class CardOut(BaseModel):
    memory_id: str
    session_id: str
    subject: str
    fact: str
    text: str
    kind: Kind
    lifecycle: Lifecycle
    event_date: str | None = None
    validity: Validity
    turn_start: int
    turn_end: int
    note: str | None = None
    superseded_by: str | None = None
    source_span: list[SourceTurn] | None = None

    def to_public(self, *, include_span: bool) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_span:
            payload.pop("source_span", None)
        elif payload.get("source_span") is None:
            payload["source_span"] = []
        return payload


class RecallResult(BaseModel):
    mode: Mode
    cards: list[CardOut]
    memory_ids: list[str]

    def to_public(self) -> dict[str, Any]:
        include_span = self.mode is Mode.replay
        return {
            "mode": self.mode.value,
            "cards": [card.to_public(include_span=include_span) for card in self.cards],
            "memory_ids": self.memory_ids,
        }


TOP_K = 10
EMBED_DIM = 1024
QUERY_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)

KIND_VALUES = tuple(k.value for k in Kind)
LIFECYCLE_VALUES = tuple(k.value for k in Lifecycle)
VALIDITY_VALUES = tuple(k.value for k in Validity)
CORRECT_STATUSES = (Validity.wrong, Validity.dead, Validity.superseded)
