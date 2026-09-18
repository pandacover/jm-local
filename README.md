# jm

Local atomic memory for chat sessions. A session (or a slice of dialogue) is stored as raw turns plus optional compact cards. Each card is one claim, with type metadata, optional time, and a link back to the original turns.

The host extracts cards. This process stores them, embeds them, and retrieves them. At question time a small rule planner chooses how to read the store; the host does not pick a mode.

## Tools

| Tool | Role |
| --- | --- |
| `memory_remember` | Write. Ingest a turn or session slice. Always stores raw turns. Optional pre-extracted cards. |
| `memory_recall` | Read. Question in. Heuristics choose lookup, compose, or replay internally. Returns cards, memory ids, and the mode used. |
| `memory_correct` | Fix. Mark a card `wrong`, `dead`, or `superseded`. Corrected cards leave recall. |
| `memory_get` | Inspect. Fetch one card by id, including `source_span`. |
| `memory_dump` | Debug snapshot of the store. Not a retrieval path. |

Lookup, compose, and replay are not tools. Embed, rerank, and plan are not tools.

## Setup

1. Install [Ollama](https://ollama.com) and pull the embedding model:

```bash
ollama pull qwen3-embedding:0.6b
```

2. Install this package:

```bash
pip install .
```

3. Point your MCP host at the stdio server:

```json
{
  "mcpServers": {
    "jm": {
      "command": "jm",
      "env": {
        "JM_DB": "~/.local/share/jm/memory.db",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

`python -m jm` is equivalent to `jm`.

Set `JM_FAKE_EMBED=1` to skip Ollama (deterministic vectors, useful in tests).

## Cards

The host should extract cards in the same `memory_remember` call when the slice has durable information:

- `subject` and `fact` — compact, subject-explicit note
- `kind` — `fact`, `event`, `preference`, `state`, `measurement`, `relation`, `list`, `interaction`
- `lifecycle` — `stable`, `completed`, `planned`, `ongoing`
- `event_date` — `YYYY-MM-DD` or `unknown`
- `turn_start` / `turn_end` — inclusive session turn indices (provenance)

Recall reads at most 10 cards. Compose only widens the candidate pool. Replay adds `source_span` from the stored turns.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
