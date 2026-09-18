"""Local embeddings via Ollama, plus a deterministic fake for tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from jm.types import EMBED_DIM

TOKEN = re.compile(r"[A-Za-z0-9_]+")


class Embedder(Protocol):
    dim: int

    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class FakeEmbedder:
    """Bag-of-tokens hashing so overlapping text is nearby. No network."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return l2_normalize(vector)


class OllamaEmbedder:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        dim: int = EMBED_DIM,
        timeout: float = 60.0,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip(
            "/"
        )
        self.model = (
            model
            or os.environ.get("JM_EMBED_MODEL")
            or "nomic-embed-text:v1.5"
        )
        self.dim = dim
        self.timeout = timeout
        self.doc_prefix = "search_document: "
        self.query_prefix = "search_query: "

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed([f"{self.doc_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self.query_prefix}{text}"])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": inputs}).encode()
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embed failed at {self.host} ({self.model}): {exc}"
            ) from exc
        vectors = body.get("embeddings")
        if vectors is None and "embedding" in body:
            vectors = [body["embedding"]]
        if not vectors:
            raise RuntimeError("Ollama embed returned no vectors")
        return [l2_normalize([float(x) for x in vector]) for vector in vectors]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b, strict=True)))
