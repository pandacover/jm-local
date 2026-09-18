"""Heuristic access planner. Routing is internal; the host never picks a mode."""

from __future__ import annotations

import re

from jm.search import tokenize
from jm.types import Mode

REPLAY_PATTERNS = [
    r"\bverbatim\b",
    r"\bword for word\b",
    r"\bexact (words|wording|text|quote|quotes|message|code|plan)\b",
    r"\bquot(?:e|ed|ing)\b",
    r"\boriginal (message|chat|conversation|email|turn|plan|code|itinerary|schedule|text)\b",
    r"\bwhat did (?:i|you|we) (?:just )?(?:say|write|tell|reply)\b",
    r"\bpull up\b",
    r"\bshow me the original\b",
    r"\bfull details\b",
    r"\bthe exact\b",
    r"\bgenerated (?:code|plan|itinerary|schedule)\b",
    r"\bschedule for\b",
    r"\bsource (?:turn|message|conversation|span)\b",
    r"\bshow me the (?:code|plan|itinerary|schedule|snippet)\b",
]

COMPOSE_PATTERNS = [
    r"\b(?:all|every|each)\b",
    r"\blist\b",
    r"\bsummar(?:y|ize|ise)\b",
    r"\bacross\b",
    r"\bover time\b",
    r"\bhistor(?:y|ies)\b",
    r"\bcompar(?:e|ison|ed)\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference(?:s)?\b",
    r"\bwhich of\b",
    r"\bhow many\b",
    r"\bcount\b",
    r"\bin (?:19|20)\d{2}\b",
    r"\blast year\b",
    r"\bthis year\b",
    r"\btrips?\b",
    r"\bvisits?\b",
    r"\bbetween\b.+\band\b",
]

STOPWORDS = {
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
}

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
PROPER = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def plan(question: str) -> Mode:
    text = question.strip().lower()
    if any(re.search(pattern, text) for pattern in REPLAY_PATTERNS):
        return Mode.replay
    if any(re.search(pattern, text) for pattern in COMPOSE_PATTERNS):
        return Mode.compose
    return Mode.lookup


def rewrites(question: str) -> list[str]:
    """Up to two answer-free views: keywords, then time/entity focus."""
    out: list[str] = []
    keywords = [
        token
        for token in tokenize(question)
        if token not in STOPWORDS and len(token) > 1
    ]
    keyword_view = " ".join(keywords)
    if keyword_view and keyword_view.lower() != question.strip().lower():
        out.append(keyword_view)

    years = YEAR.findall(question)
    entities = [match.group(0) for match in PROPER.finditer(question)]
    entity_view = " ".join(entities + years)
    if (
        entity_view
        and entity_view.lower() != question.strip().lower()
        and entity_view.lower() not in {item.lower() for item in out}
    ):
        out.append(entity_view)
    return out[:2]
