from __future__ import annotations

import re
from typing import Any

ONTOLOGY_SCHEMA_VERSION = "engineering-ontology-retriever-v1"

ENGINEERING_ONTOLOGY: dict[str, dict[str, Any]] = {
    "valve": {
        "synonyms": ["valves", "manual valve", "actuated valve", "control valve", "isolation valve", "check valve", "globe valve", "butterfly valve"],
        "broader": ["piping component", "flow control"],
        "related": ["isolation", "throttling", "shut-off", "actuator", "pressure relief"],
    },
    "isolation": {
        "synonyms": ["isolate", "isolating", "isolation philosophy", "system isolation", "shutdown isolation"],
        "broader": ["safety function", "operational control"],
        "related": ["valve", "shutdown", "maintenance", "barrier", "control"],
    },
    "pressure relief": {
        "synonyms": ["relief valve", "pressure relief device", "rupture disc", "psv", "relief devices"],
        "broader": ["safety system", "overpressure protection"],
        "related": ["flare", "vent", "discharge", "safety", "pressure"],
    },
    "piping": {
        "synonyms": ["pipe", "pipeline", "piping system", "line", "pipework"],
        "broader": ["mechanical system"],
        "related": ["routing", "support", "slope", "flexibility", "valve", "drain", "vent"],
    },
    "slope": {
        "synonyms": ["sloping", "gradient", "fall", "inclination", "pipe slope"],
        "broader": ["layout requirement"],
        "related": ["drain", "vent", "direction", "table", "low point", "high point"],
    },
    "drain": {
        "synonyms": ["drains", "draining", "operational drain", "closed drain", "open drain"],
        "broader": ["piping function"],
        "related": ["vent", "slope", "liquid removal", "low point"],
    },
    "vent": {
        "synonyms": ["vents", "venting", "operational vent", "atmospheric vent"],
        "broader": ["piping function"],
        "related": ["drain", "slope", "gas release", "high point"],
    },
    "flare": {
        "synonyms": ["flare header", "flare system", "flare line", "discharge to flare"],
        "broader": ["relief system"],
        "related": ["pressure relief", "vent", "discharge", "safety"],
    },
    "safety": {
        "synonyms": ["safe", "hazard", "risk", "emergency", "shutdown", "fire", "explosion"],
        "broader": ["engineering requirement"],
        "related": ["shall", "must", "relief", "isolation", "fire", "explosion"],
    },
    "layout": {
        "synonyms": ["arrangement", "routing", "location", "placement", "orientation"],
        "broader": ["design requirement"],
        "related": ["piping", "access", "support", "equipment", "maintenance"],
    },
    "support": {
        "synonyms": ["pipe support", "supported", "supporting", "restraint", "guide"],
        "broader": ["mechanical integrity"],
        "related": ["vibration", "load", "flexibility", "piping"],
    },
    "vibration": {
        "synonyms": ["vibrations", "vibrating", "dynamic load", "fatigue"],
        "broader": ["mechanical integrity"],
        "related": ["support", "dead leg", "pump", "compressor", "piping"],
    },
    "equipment": {
        "synonyms": ["pump", "compressor", "vessel", "module", "skid", "instrument"],
        "broader": ["plant asset"],
        "related": ["piping", "layout", "maintenance", "access", "valve"],
    },
    "standard": {
        "synonyms": ["code", "specification", "asme", "norsok", "iso", "api", "iec", "requirement"],
        "broader": ["compliance basis"],
        "related": ["shall", "must", "design code", "engineering requirement"],
    },
}


def build_ontology_plan(query: str, route: Any, profile: Any) -> dict[str, Any]:
    normalized = normalize_text(query)
    query_terms = tokenize(normalized)
    concepts = detect_concepts(normalized, query_terms)
    active = bool(concepts) or "ontology" in set(getattr(route, "retrievers", ()) or ())
    expanded_terms = ontology_terms_for_concepts(concepts)
    return {
        "schema": ONTOLOGY_SCHEMA_VERSION,
        "active": active,
        "route_primary": getattr(route, "primary", ""),
        "profile_type": getattr(profile, "type_id", ""),
        "concepts": concepts,
        "expanded_terms": expanded_terms,
        "query_terms": query_terms[:30],
        "strategy": "engineering_taxonomy_concept_synonym_broader_related_expansion_and_scoring",
    }


def ontology_expanded_queries(seed_queries: list[str], plan: dict[str, Any]) -> list[str]:
    queries = list(seed_queries)
    if not plan.get("active"):
        return unique_preserve(queries)[:14]
    concepts = plan.get("concepts") or []
    terms = plan.get("expanded_terms") or []
    if concepts:
        queries.append(f"ontology concepts {' '.join(concepts)} {seed_queries[0] if seed_queries else ''}".strip())
    if terms:
        queries.append(f"engineering synonyms related terms {' '.join(terms[:24])}")
    for concept in concepts[:6]:
        data = ENGINEERING_ONTOLOGY.get(concept) or {}
        related = " ".join((data.get("related") or [])[:8])
        broader = " ".join((data.get("broader") or [])[:4])
        queries.append(f"{concept} {broader} {related}".strip())
    return unique_preserve(query for query in queries if str(query).strip())[:16]


def ontology_candidate_score(plan: dict[str, Any], metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched_concepts": []}
    haystack = ontology_haystack(metadata, text)
    matched = []
    score = 0.0
    for concept in plan.get("concepts") or []:
        data = ENGINEERING_ONTOLOGY.get(concept) or {}
        concept_terms = [concept] + list(data.get("synonyms") or [])
        broader_terms = list(data.get("broader") or [])
        related_terms = list(data.get("related") or [])
        concept_hits = matching_terms(concept_terms, haystack)
        broader_hits = matching_terms(broader_terms, haystack)
        related_hits = matching_terms(related_terms, haystack)
        if concept_hits or broader_hits or related_hits:
            matched.append(
                {
                    "concept": concept,
                    "concept_hits": concept_hits[:8],
                    "broader_hits": broader_hits[:6],
                    "related_hits": related_hits[:8],
                }
            )
            score += min(0.06, len(concept_hits) * 0.022 + len(broader_hits) * 0.012 + len(related_hits) * 0.008)

    expanded_hits = matching_terms(plan.get("expanded_terms") or [], haystack)
    metadata_bonus = ontology_metadata_bonus(plan, metadata)
    score += min(0.05, len(expanded_hits) * 0.006)
    score += metadata_bonus
    return round(min(0.18, score), 5), {
        "schema": ONTOLOGY_SCHEMA_VERSION,
        "active": True,
        "matched_concepts": matched,
        "expanded_term_hits": expanded_hits[:24],
        "metadata_bonus": round(metadata_bonus, 5),
        "coverage": round(len(matched) / max(1, len(plan.get("concepts") or [])), 4),
    }


def detect_concepts(normalized_query: str, query_terms: list[str]) -> list[str]:
    concepts = []
    term_text = " ".join(query_terms)
    for concept, data in ENGINEERING_ONTOLOGY.items():
        candidates = [concept] + list(data.get("synonyms") or [])
        if any(term_in_text(candidate, normalized_query) or term_in_text(candidate, term_text) for candidate in candidates):
            concepts.append(concept)
            continue
        if any(len(term) > 4 and (term in concept or concept in term) for term in query_terms):
            concepts.append(concept)
    return unique_preserve(concepts)[:10]


def ontology_terms_for_concepts(concepts: list[str]) -> list[str]:
    terms = []
    for concept in concepts:
        data = ENGINEERING_ONTOLOGY.get(concept) or {}
        terms.append(concept)
        terms.extend(data.get("synonyms") or [])
        terms.extend(data.get("broader") or [])
        terms.extend(data.get("related") or [])
    return unique_preserve(normalize_text(term) for term in terms if normalize_text(term))[:80]


def ontology_metadata_bonus(plan: dict[str, Any], metadata: dict[str, Any]) -> float:
    metadata_terms = normalize_text(
        " ".join(
            [
                flatten_values(metadata.get("domain_terms") or []),
                flatten_values(metadata.get("engineering_entities") or []),
                flatten_values(metadata.get("engineering_canonical_entities") or []),
                flatten_values(metadata.get("semantic_labels") or []),
                flatten_values(metadata.get("retrieval_tags") or []),
                flatten_values(metadata.get("keywords") or []),
            ]
        )
    )
    if not metadata_terms:
        return 0.0
    hits = matching_terms((plan.get("concepts") or []) + (plan.get("expanded_terms") or []), metadata_terms)
    return min(0.045, len(hits) * 0.007)


def ontology_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(
        " ".join(
            [
                text,
                metadata.get("section_title") or "",
                metadata.get("table_title") or "",
                flatten_values(metadata.get("section_path") or []),
                flatten_values(metadata.get("domain_terms") or []),
                flatten_values(metadata.get("engineering_entities") or []),
                flatten_values(metadata.get("engineering_canonical_entities") or []),
                flatten_values(metadata.get("primary_entities") or []),
                flatten_values(metadata.get("semantic_labels") or []),
                flatten_values(metadata.get("keywords") or []),
                flatten_values(metadata.get("table_terms") or []),
            ]
        )
    )


def matching_terms(terms: list[str], haystack: str) -> list[str]:
    hits = []
    for term in terms:
        normalized = normalize_text(term)
        if normalized and term_in_text(normalized, haystack):
            hits.append(normalized)
    return unique_preserve(hits)


def term_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    if " " in term or "-" in term or "/" in term:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return bool(re.search(rf"\b{re.escape(term)}s?\b", text))


def tokenize(text: str) -> list[str]:
    stopwords = {"and", "are", "for", "need", "what", "where", "which", "about", "explain", "show", "tell"}
    return unique_preserve(token for token in re.findall(r"[a-z0-9][a-z0-9_.:/&+-]{2,}", text) if token not in stopwords)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.:/&+-]+", " ", str(value or "").lower())).strip()


def flatten_values(values: Any) -> str:
    if isinstance(values, dict):
        return " ".join(flatten_values(value) for value in values.values())
    if isinstance(values, (list, tuple, set)):
        return " ".join(flatten_values(value) for value in values)
    return str(values or "")


def unique_preserve(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
