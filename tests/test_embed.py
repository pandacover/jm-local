from jm.embed import FakeEmbedder, OllamaEmbedder
from jm.types import EMBED_DIM


def test_default_ollama_model_is_nomic(monkeypatch):
    monkeypatch.delenv("JM_EMBED_MODEL", raising=False)
    embedder = OllamaEmbedder()
    assert embedder.model == "nomic-embed-text:v1.5"
    assert embedder.dim == 768
    assert EMBED_DIM == 768
    assert FakeEmbedder().dim == 768
    assert embedder.query_prefix.startswith("search_query:")
    assert embedder.doc_prefix.startswith("search_document:")
