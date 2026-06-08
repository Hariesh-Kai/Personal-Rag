from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

LATE_INTERACTION_SCHEMA_VERSION = "engineering-late-interaction-v1"
LATE_INTERACTION_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "between",
    "does",
    "each",
    "explain",
    "from",
    "give",
    "have",
    "how",
    "into",
    "need",
    "show",
    "that",
    "the",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}
ENGINEERING_TOKEN_BOOSTS = {
    "asme",
    "api",
    "iso",
    "iec",
    "norsok",
    "p&id",
    "pid",
    "mm",
    "bar",
    "psi",
    "slope",
    "valve",
    "relief",
    "drain",
    "flare",
    "pump",
    "vessel",
    "line",
}


def late_interaction_plan(query: str, route: Any) -> dict[str, Any]:
    active = "late_interaction" in set(getattr(route, "retrievers", ()))
    query_tokens = weighted_query_tokens(query)
    return {
        "schema": LATE_INTERACTION_SCHEMA_VERSION,
        "active": active,
        "query_tokens": [item["token"] for item in query_tokens],
        "weighted_query_tokens": query_tokens,
        "route_primary": getattr(route, "primary", ""),
        "strategy": "token_maxsim_exact_stem_identifier_numeric_char_ngram",
    }


def late_interaction_score(plan: dict[str, Any], text: str, metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matches": [], "coverage": 0.0}
    query_tokens = plan.get("weighted_query_tokens") or []
    if not query_tokens:
        return 0.0, {"active": True, "matches": [], "coverage": 0.0}
    candidate_tokens = candidate_interaction_tokens(text, metadata)
    if not candidate_tokens:
        return 0.0, {"active": True, "matches": [], "coverage": 0.0}

    matches = []
    weighted_total = 0.0
    weight_sum = 0.0
    exact_hits = 0
    numeric_hits = 0
    identifier_hits = 0
    for query_token in query_tokens:
        best = best_token_match(query_token["token"], candidate_tokens)
        weighted_total += best["score"] * float(query_token["weight"])
        weight_sum += float(query_token["weight"])
        exact_hits += 1 if best["match_type"] == "exact" else 0
        numeric_hits += 1 if best["match_type"] == "numeric" else 0
        identifier_hits += 1 if best["match_type"] == "identifier" else 0
        matches.append({**best, "query_token": query_token["token"], "weight": query_token["weight"]})

    coverage = weighted_total / max(0.001, weight_sum)
    phrase_score = phrase_alignment_score(plan, text, metadata)
    proximity_score = proximity_alignment_score([item["token"] for item in query_tokens], candidate_tokens)
    exact_bonus = min(0.08, exact_hits * 0.015)
    numeric_bonus = min(0.05, numeric_hits * 0.02)
    identifier_bonus = min(0.05, identifier_hits * 0.02)
    raw = coverage * 0.58 + phrase_score * 0.17 + proximity_score * 0.15 + exact_bonus + numeric_bonus + identifier_bonus
    score = round(min(0.22, raw * 0.22), 5)
    return score, {
        "schema": LATE_INTERACTION_SCHEMA_VERSION,
        "active": True,
        "coverage": round(coverage, 4),
        "phrase_alignment": round(phrase_score, 4),
        "proximity_alignment": round(proximity_score, 4),
        "exact_hits": exact_hits,
        "numeric_hits": numeric_hits,
        "identifier_hits": identifier_hits,
        "matches": sorted(matches, key=lambda item: (item["score"], item["weight"]), reverse=True)[:16],
    }


def weighted_query_tokens(query: str) -> list[dict[str, Any]]:
    tokens = []
    for token in tokenize_interaction_text(query):
        if token in LATE_INTERACTION_STOPWORDS or len(token) < 2:
            continue
        weight = 1.0
        if is_identifier(token):
            weight += 1.1
        if is_numeric_like(token):
            weight += 0.9
        if token in ENGINEERING_TOKEN_BOOSTS:
            weight += 0.5
        if len(token) >= 8:
            weight += 0.2
        tokens.append({"token": token, "weight": round(weight, 2)})
    deduped: dict[str, dict[str, Any]] = {}
    for token in tokens:
        current = deduped.get(token["token"])
        if not current or token["weight"] > current["weight"]:
            deduped[token["token"]] = token
    return sorted(deduped.values(), key=lambda item: item["weight"], reverse=True)[:32]


def candidate_interaction_tokens(text: str, metadata: dict[str, Any]) -> list[str]:
    surfaces = [
        text,
        metadata.get("section_title") or "",
        metadata.get("current_section_title") or "",
        " ".join(metadata.get("section_path") or []),
        metadata.get("table_title") or "",
        " ".join(metadata.get("table_columns") or []),
        " ".join(metadata.get("table_rows") or []),
        " ".join(metadata.get("keywords") or []),
        " ".join(metadata.get("keyphrases") or []),
        " ".join(metadata.get("exact_terms") or []),
        " ".join(metadata.get("technical_identifiers") or []),
        " ".join(metadata.get("standards") or []),
        " ".join(metadata.get("engineering_entities") or []),
        " ".join(metadata.get("engineering_canonical_entities") or []),
        " ".join(metadata.get("primary_entities") or []),
        " ".join(metadata.get("engineering_entity_aliases") or []),
        " ".join(metadata.get("relationship_types") or []),
        " ".join(metadata.get("semantic_labels") or []),
    ]
    for record in metadata.get("engineering_entity_records") or []:
        if isinstance(record, dict):
            surfaces.extend(
                [
                    str(record.get("text") or ""),
                    str(record.get("canonical") or ""),
                    " ".join(str(alias) for alias in record.get("aliases") or []),
                ]
            )
    tokens = tokenize_interaction_text("\n".join(str(surface) for surface in surfaces if surface))
    return unique_preserve(tokens)[:1200]


def best_token_match(query_token: str, candidate_tokens: list[str]) -> dict[str, Any]:
    best = {"candidate_token": "", "score": 0.0, "match_type": "none"}
    query_stem = stem_token(query_token)
    query_ngrams = char_ngrams(query_token)
    for candidate in candidate_tokens:
        score, match_type = token_similarity(query_token, query_stem, query_ngrams, candidate)
        if score > best["score"]:
            best = {"candidate_token": candidate, "score": round(score, 4), "match_type": match_type}
            if score >= 1.0:
                break
    return best


def token_similarity(query_token: str, query_stem: str, query_ngrams: set[str], candidate: str) -> tuple[float, str]:
    if query_token == candidate:
        return 1.0, "exact"
    if is_numeric_like(query_token) and query_token == normalize_numeric(candidate):
        return 0.98, "numeric"
    if is_identifier(query_token) and identifier_equal(query_token, candidate):
        return 0.96, "identifier"
    candidate_stem = stem_token(candidate)
    if query_stem and query_stem == candidate_stem:
        return 0.88, "stem"
    if len(query_token) >= 4 and (query_token in candidate or candidate in query_token):
        return 0.78, "substring"
    ngram_score = ngram_similarity(query_ngrams, char_ngrams(candidate))
    if ngram_score >= 0.72:
        return 0.70 + min(0.12, (ngram_score - 0.72) * 0.4), "char_ngram"
    fuzzy = SequenceMatcher(None, query_token, candidate).ratio()
    if fuzzy >= 0.82:
        return min(0.74, fuzzy * 0.82), "fuzzy"
    return 0.0, "none"


def phrase_alignment_score(plan: dict[str, Any], text: str, metadata: dict[str, Any]) -> float:
    query_tokens = plan.get("query_tokens") or []
    if len(query_tokens) < 2:
        return 0.0
    haystack = " ".join(candidate_interaction_tokens(text, metadata))
    bigrams = [" ".join(query_tokens[index : index + 2]) for index in range(len(query_tokens) - 1)]
    trigrams = [" ".join(query_tokens[index : index + 3]) for index in range(len(query_tokens) - 2)]
    hits = sum(1 for phrase in bigrams + trigrams if phrase in haystack)
    return min(1.0, hits / max(1, len(bigrams) + len(trigrams)))


def proximity_alignment_score(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    positions: list[int] = []
    candidate_index: dict[str, list[int]] = {}
    for index, token in enumerate(candidate_tokens[:900]):
        candidate_index.setdefault(token, []).append(index)
    for token in query_tokens:
        if token in candidate_index:
            positions.append(candidate_index[token][0])
    if len(positions) < 2:
        return 0.0
    span = max(positions) - min(positions) + 1
    return max(0.0, min(1.0, len(positions) / max(1, span)))


def tokenize_interaction_text(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{1,}", str(text or "")):
        normalized = normalize_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def normalize_token(token: str) -> str:
    token = token.lower().replace("&", "and")
    token = token.strip(" .,:;()[]{}")
    token = re.sub(r"[^a-z0-9_.:/+-]+", "", token)
    return normalize_numeric(token)


def normalize_numeric(token: str) -> str:
    return re.sub(r"\s+", "", str(token or "").lower())


def is_identifier(token: str) -> bool:
    return bool(re.search(r"[a-z]{2,}[-_/][a-z0-9]", token) or re.search(r"\b(?:asme|norsok|api|iso|iec)[-_:/a-z0-9.]*\b", token))


def identifier_equal(left: str, right: str) -> bool:
    return re.sub(r"[-_/:.]+", "", left.lower()) == re.sub(r"[-_/:.]+", "", right.lower())


def is_numeric_like(token: str) -> bool:
    return bool(re.search(r"\d", token) and re.search(r"^(?:\d+(?:\.\d+)?|1:\d+|\d+(?:mm|bar|psi|deg|%))$", token))


def stem_token(token: str) -> str:
    for suffix in ("ing", "tion", "ions", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            if suffix == "ies":
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token


def char_ngrams(token: str, size: int = 3) -> set[str]:
    if len(token) <= size:
        return {token}
    return {token[index : index + size] for index in range(len(token) - size + 1)}


def ngram_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
