import json
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).resolve().parent / "query_expansion_rules.json"

def _load_rules() -> dict[str, Any]:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

RULES = _load_rules()
CANONICAL_MAP: dict[str, str] = RULES.get("canonical_map", {})
CANONICAL_SYNONYMS: dict[str, list[str]] = RULES.get("canonical_synonyms", {})

def normalize_query(query: str) -> dict[str, Any]:
    found_canonical: list[str] = []
    q = (query or "").strip()
    if not q:
        return {"raw_query": query, "canonical_terms": []}
    q_lower = q.lower()
    for raw_term, canonical_term in CANONICAL_MAP.items():
        if raw_term.lower() in q_lower and canonical_term not in found_canonical:
            found_canonical.append(canonical_term)
    return {"raw_query": query, "canonical_terms": found_canonical}

def build_search_queries(query: str, max_synonyms_per_term: int = 3) -> list[str]:
    normalized = normalize_query(query)
    queries: list[str] = []
    raw_query = (query or "").strip()
    if raw_query:
        queries.append(raw_query)
    for canonical in normalized["canonical_terms"]:
        queries.append(canonical)
        synonyms = CANONICAL_SYNONYMS.get(canonical, [])[:max_synonyms_per_term]
        queries.extend(synonyms)
    deduped: list[str] = []
    seen: set[str] = set()
    for q in queries:
        q_clean = q.strip()
        if q_clean:
            key = q_clean.lower()
            if key not in seen:
                deduped.append(q_clean)
                seen.add(key)
    return deduped
