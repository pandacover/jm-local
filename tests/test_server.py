import asyncio

from jm.embed import FakeEmbedder
from jm.memory import Memory
from jm.server import TOOL_NAMES, build_server, registered_tool_names


def test_exactly_five_tools(tmp_path):
    memory = Memory(tmp_path / "m.db", embedder=FakeEmbedder())
    server = build_server(memory)
    names = registered_tool_names(server)
    assert names == TOOL_NAMES
    assert names == (
        "memory_remember",
        "memory_recall",
        "memory_correct",
        "memory_get",
        "memory_dump",
    )
    forbidden = {
        "lookup",
        "compose",
        "replay",
        "memory_lookup",
        "memory_compose",
        "memory_replay",
        "embed",
        "rerank",
        "plan",
        "dream",
        "sleep",
    }
    assert forbidden.isdisjoint(names)


def test_mcp_handlers_write_read_fix_inspect_dump(tmp_path):
    memory = Memory(tmp_path / "m.db", embedder=FakeEmbedder())
    server = build_server(memory)
    tools = server._tool_manager

    async def roundtrip():
        remembered = await tools.call_tool(
            "memory_remember",
            {
                "session_id": "s1",
                "turns": [
                    {
                        "role": "user",
                        "content": "I prefer small, quiet hotels.",
                        "index": 0,
                    }
                ],
                "cards": [
                    {
                        "subject": "User",
                        "fact": "prefers small, quiet hotels",
                        "kind": "preference",
                        "lifecycle": "stable",
                        "event_date": "unknown",
                        "turn_start": 0,
                        "turn_end": 0,
                    }
                ],
            },
        )
        memory_id = remembered["memory_ids"][0]

        recalled = await tools.call_tool(
            "memory_recall", {"question": "What hotel style does the user prefer?"}
        )
        assert recalled["mode"] == "lookup"
        assert memory_id in recalled["memory_ids"]
        assert "mode" not in tools.get_tool("memory_recall").parameters["properties"]

        fetched = await tools.call_tool("memory_get", {"memory_id": memory_id})
        assert fetched["source_span"][0]["content"].startswith("I prefer")

        corrected = await tools.call_tool(
            "memory_correct", {"memory_id": memory_id, "status": "superseded"}
        )
        assert corrected["validity"] == "superseded"

        dumped = await tools.call_tool("memory_dump", {})
        assert dumped["stats"]["by_validity"]["superseded"] == 1
        assert "embedding" not in dumped["cards"][0]

    asyncio.run(roundtrip())

