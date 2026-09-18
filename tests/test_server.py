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
