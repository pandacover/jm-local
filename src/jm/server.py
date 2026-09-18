"""MCP server: write, read, fix, inspect, dump. Routing stays inside recall."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from jm.extract import EXTRACTION_CONTRACT
from jm.memory import Memory, default_memory
from jm.types import CORRECT_STATUSES, KIND_VALUES, LIFECYCLE_VALUES

TOOL_NAMES = (
    "memory_remember",
    "memory_recall",
    "memory_correct",
    "memory_get",
    "memory_dump",
)

INSTRUCTIONS = """\
Local atomic memory for chat sessions.

Tools:
- memory_remember: write a turn or session slice (raw always; cards when you extracted them)
- memory_recall: read. Pass the question only. Routing is internal.
- memory_correct: fix a card (wrong / dead / superseded)
- memory_get: inspect one card by id, including source turns
- memory_dump: debug snapshot of the store; not for answering questions

Do not try to choose lookup, compose, or replay. memory_recall does that.
"""

REMEMBER_DOC = f"""\
Ingest a turn or a session slice. Always stores the raw turns. Optionally stores pre-extracted atomic cards.

You are the extractor. When the slice contains durable information, extract cards in this same call and pass them as `cards`. Do not call a separate extract tool.

{EXTRACTION_CONTRACT}

Arguments:
- session_id: stable id for this conversation session
- turns: list of {{role, content, optional ts, optional index}}. `index` is the session turn index; if omitted, indices are assigned sequentially.
- session_time: optional timestamp for the session
- cards: optional list of {{subject, fact, kind, lifecycle, event_date, turn_start, turn_end}}
  kind: {' | '.join(KIND_VALUES)}
  lifecycle: {' | '.join(LIFECYCLE_VALUES)}
"""


def build_server(memory: Memory | None = None) -> FastMCP:
    memory = memory or default_memory()
    mcp = FastMCP("jm", instructions=INSTRUCTIONS)

    def memory_remember(
        session_id: str,
        turns: list[dict[str, Any]],
        cards: list[dict[str, Any]] | None = None,
        session_time: str | None = None,
    ) -> dict[str, Any]:
        return memory.remember(
            session_id=session_id,
            turns=turns,
            cards=cards,
            session_time=session_time,
        )

    memory_remember.__doc__ = REMEMBER_DOC
    mcp.tool(name="memory_remember")(memory_remember)

    @mcp.tool(name="memory_recall")
    def memory_recall(question: str, as_of: str | None = None) -> dict[str, Any]:
        """Question in. Plans with heuristics and runs lookup, compose, or replay internally.

        Returns compact cards, their memory_ids, and the mode that was used.
        Do not pick a mode; pass only the question (and optional as_of timestamp).
        Replay responses include source_span on each card.
        """
        return memory.recall(question=question, as_of=as_of)

    @mcp.tool(name="memory_correct")
    def memory_correct(
        memory_id: str,
        status: str,
        note: str | None = None,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        """Mark a card wrong, dead, or superseded. Does not delete source turns.

        Corrected cards drop out of memory_recall. memory_get still returns them.
        status: wrong | dead | superseded
        """
        allowed = {item.value for item in CORRECT_STATUSES}
        if status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        try:
            return memory.correct(
                memory_id=memory_id,
                status=status,
                note=note,
                superseded_by=superseded_by,
            )
        except KeyError as exc:
            raise ValueError(f"unknown memory_id: {memory_id}") from exc

    @mcp.tool(name="memory_get")
    def memory_get(memory_id: str) -> dict[str, Any]:
        """Fetch one card by id, including source_span (original turns)."""
        try:
            return memory.get(memory_id)
        except KeyError as exc:
            raise ValueError(f"unknown memory_id: {memory_id}") from exc

    @mcp.tool(name="memory_dump")
    def memory_dump(session_id: str | None = None) -> dict[str, Any]:
        """Debug snapshot of the store. Not a retrieval path — use memory_recall to answer questions.

        Optional session_id limits the dump. Embeddings are omitted.
        """
        return memory.dump(session_id)

    return mcp


def registered_tool_names(mcp: FastMCP) -> tuple[str, ...]:
    manager = getattr(mcp, "_tool_manager", None)
    if manager is not None and hasattr(manager, "list_tools"):
        tools = manager.list_tools()
        names = []
        for tool in tools:
            names.append(getattr(tool, "name", None) or str(tool))
        return tuple(names)
    return TOOL_NAMES
