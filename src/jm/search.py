"""Hybrid FTS + vector search with RRF and a compact Top-K budget."""

from __future__ import annotations

import re
from datetime import datetime

from jm.embed import Embedder, cosine
from jm.store import Store
from jm.types import TOP_K, CardOut, Kind

CANDIDATE_K = 50
RRF_K = 60
KIND_INTENT_BOOST = 0.25
TOKEN = re.compile(r"[A-Za-z0-9_]+")
YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
PREFERENCE_INTENT = re.compile(
    r"\b(prefer|prefers|preferred|preference|preferences|like|likes|liked|"
    r"dislike|want|wants|wanted|rather|favorite|favourite)\b",
    re.I,
)

QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "it",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "at",
    "as",
    "by",
    "do",
    "did",
    "does",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "can",
    "could",
    "would",
    "should",
    "please",
    "tell",
    "about",
    "user",
    "users",
}


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def query_terms(text: str) -> list[str]:
    return [token for token in tokenize(text) if len(token) > 1 and token not in QUERY_STOPWORDS]


def fts_match(text: str) -> str:
    terms = query_terms(text)
    return " OR ".join(f'"{token}"' for token in terms)


def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, memory_id in enumerate(ranking):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def years_in(text: str) -> set[str]:
    return set(YEAR.findall(text))


def _as_of_ok(card: CardOut, as_of: str | None) -> bool:
    if not as_of or not card.event_date or card.event_date == "unknown":
        return True
    try:
        datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        card_day = card.event_date[:10]
        as_of_day = as_of[:10]
        return card_day <= as_of_day
    except ValueError:
        return True


def _related(query_token: str, doc_token: str) -> bool:
    if query_token == doc_token:
        return True
    if len(query_token) >= 4 and (
        doc_token.startswith(query_token) or query_token.startswith(doc_token)
    ):
        return True
    return False


def _overlap_count(query_tokens: set[str], fact: str) -> int:
    doc_tokens = set(tokenize(fact))
    return sum(
        1
        for query_token in query_tokens
        if any(_related(query_token, doc_token) for doc_token in doc_tokens)
    )


def _overlap_ids(query: str, cards: list[CardOut]) -> list[str]:
    query_tokens = set(query_terms(query))
    scored: list[tuple[int, str]] = []
    for card in cards:
        overlap = _overlap_count(query_tokens, card.fact)
        if overlap:
            scored.append((overlap, card.memory_id))
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:CANDIDATE_K]]


def _year_ids(query: str, cards: list[CardOut]) -> list[str]:
    query_years = years_in(query)
    if not query_years:
        return []
    hits = [
        card.memory_id
        for card in cards
        if card.event_date
        and any(year in card.event_date for year in query_years)
    ]
    return hits[:CANDIDATE_K]


def retrieve(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    extra_queries: list[str] | None = None,
    as_of: str | None = None,
    limit: int = TOP_K,
) -> list[CardOut]:
    views = [query]
    for extra in extra_queries or []:
        extra = extra.strip()
        if extra and extra.lower() not in {v.lower() for v in views}:
            views.append(extra)

    score: dict[str, float] = {}
    query_vec = embedder.embed_query(query)
    active = [card for card in store.active_cards() if _as_of_ok(card, as_of)]
    active_by_id = {card.memory_id: card for card in active}
    embeddings = store.embeddings_for(active_by_id.keys())

    for view in views:
        lexical = [
            mid
            for mid in store.fts_ids(fts_match(view), CANDIDATE_K)
            if mid in active_by_id
        ]
        view_vec = embedder.embed_query(view) if view != query else query_vec
        vector_ranked = sorted(
            (
                (mid, cosine(view_vec, embeddings[mid]))
                for mid in embeddings
                if mid in active_by_id
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        vector_ids = [mid for mid, sim in vector_ranked[:CANDIDATE_K] if sim > 0]
        overlap_ids = _overlap_ids(view, active)
        year_ids = _year_ids(view, active)
        fused = rrf_fuse([lexical, vector_ids, overlap_ids, year_ids])
        for mid, value in fused.items():
            score[mid] = max(score.get(mid, 0.0), value)

    query_years = years_in(query)
    query_tokens = set(query_terms(query))
    prefers = bool(PREFERENCE_INTENT.search(query))
    for mid, value in list(score.items()):
        card = active_by_id[mid]
        if query_years and card.event_date:
            if any(year in card.event_date for year in query_years):
                score[mid] = value + 0.05
        if prefers and card.kind is Kind.preference:
            score[mid] += KIND_INTENT_BOOST
        elif card.kind.value in query_tokens:
            score[mid] += 0.01
        overlap = _overlap_count(query_tokens, card.fact)
        if overlap:
            score[mid] += min(0.03, 0.005 * overlap)

    ranked_ids = sorted(score, key=lambda mid: score[mid], reverse=True)[:limit]
    return store.cards_by_ids(ranked_ids)
