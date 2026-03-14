import re
from typing import Dict, List, Tuple

from config import config

_UNIT_RE = re.compile(r"\bunit\s*[-\u2013\u2014:]?\s*(\d+)\b", re.IGNORECASE)
_EXPERIMENT_RE = re.compile(
    r"\bexperiment\s*[-\u2013\u2014:*]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE
)


def extract_unit_number(text: str | None) -> str | None:
    if not text:
        return None
    match = _UNIT_RE.search(text)
    return match.group(1) if match else None


def extract_experiment_number(text: str | None) -> str | None:
    if not text:
        return None
    match = _EXPERIMENT_RE.search(text)
    return match.group(1) if match else None


def _matches_filters(metadata: Dict, filters: Dict) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        if key == "unit_number":
            if extract_unit_number(metadata.get("unit_title")) != str(value):
                return False
            continue
        if key == "experiment_number":
            if extract_experiment_number(metadata.get("experiment_name")) != str(value):
                return False
            continue
        if metadata.get(key) != value:
            return False
    return True


def retrieve_with_scores(vector_store, query: str, top_k: int, filters: Dict | None = None):
    filters = filters or {}
    candidate_k = max(top_k * config.SIMILARITY_K_MULTIPLIER, top_k)
    results = vector_store.similarity_search_with_score(query, k=candidate_k)

    filtered: List[Tuple] = []
    for doc, score in results:
        if not _matches_filters(doc.metadata, filters):
            continue
        filtered.append((doc, float(score)))
        if len(filtered) >= top_k:
            break
    return filtered


def retrieve(vector_store, query: str, top_k: int, filters: Dict | None = None):
    results = retrieve_with_scores(vector_store, query, top_k, filters)
    return [doc for doc, _ in results]
