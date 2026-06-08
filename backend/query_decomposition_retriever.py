from __future__ import annotations

import re
from typing import Any

QUERY_DECOMPOSITION_SCHEMA_VERSION = "engineering-query-decomposition-v1"
DECOMPOSITION_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "does",
    "each",
    "explain",
    "for",
    "from",
    "give",
    "have",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "short",
    "show",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "where",
    "which",
    "with",
}
SUBQUESTION_PATTERNS = {
    "definition": r"\b(what is|define|definition|meaning|what are)\b",
    "enumeration": r"\b(types?|different|list|enumerate|include|mentioned)\b",
    "comparison": r"\b(compare|difference|versus|vs\.?|between|whereas)\b",
    "table_numeric": r"\b(table|row|column|value|numeric|number|dimension|slope|rating|mm|bar|psi|1:\d+)\b",
    "safety_requirement": r"\b(safety|hazard|fire|explosion|shutdown|relief|shall|must|required|mandatory)\b",
    "location_section": r"\b(where|section|page|clause|location|located)\b",
    "procedure": r"\b(procedure|steps?|sequence|workflow|method|how to)\b",
    "relationship": r"\b(related|connected|upstream|downstream|associated|dependency|relationship)\b",
    "exception": r"\b(exception|except|unless|not allowed|shall not|must not|prohibited)\b",
}


def query_decomposition_plan(query: str, route: Any, profile: Any) -> dict[str, Any]:
    active = "query_decomposition" in set(getattr(route, "retrievers", ())) or is_complex_question(query, profile)
    subquestions = build_subquestions(query)
    return {
        "schema": QUERY_DECOMPOSITION_SCHEMA_VERSION,
        "active": bool(active and subquestions),
        "route_primary": getattr(route, "primary", ""),
        "profile_type": getattr(profile, "type_id", ""),
        "subquestion_count": len(subquestions),
        "subquestions": subquestions,
        "subquestion_queries": [item["query"] for item in subquestions],
        "required_terms": unique_preserve(term for item in subquestions for term in item.get("terms", []))[:40],
        "required_types": unique_preserve(item.get("type") for item in subquestions if item.get("type")),
        "strategy": "typed_subquestions_query_expansion_subquestion_coverage_scoring",
    }


def decomposition_expanded_queries(seed_queries: list[str], plan: dict[str, Any]) -> list[str]:
    queries = list(seed_queries)
    if not plan.get("active"):
        return unique_preserve(queries)[:10]
    for item in plan.get("subquestions") or []:
        prefix = subquestion_prefix(item.get("type") or "semantic")
        queries.append(f"{prefix} {item.get('query') or ''}".strip())
    if plan.get("required_terms"):
        queries.append(f"answer all subquestions {' '.join(plan['required_terms'][:18])}")
    if plan.get("required_types"):
        queries.append(f"evidence types {' '.join(plan['required_types'][:10])} {' '.join(plan.get('required_terms') or [])}")
    return unique_preserve(query for query in queries if str(query).strip())[:12]


def decomposition_candidate_score(plan: dict[str, Any], metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "subquestion_matches": []}
    haystack = decomposition_haystack(metadata, text)
    matches = []
    covered = 0
    score = 0.0
    for item in plan.get("subquestions") or []:
        terms = item.get("terms") or []
        term_hits = sum(1 for term in terms if normalize_text(term) in haystack)
        type_hit = subquestion_type_hit(item.get("type") or "", metadata, haystack)
        coverage = term_hits / max(1, len(terms))
        if coverage >= 0.34 or type_hit:
            covered += 1
        item_score = coverage * 0.045 + (0.025 if type_hit else 0)
        score += item_score * float(item.get("weight") or 1.0)
        matches.append(
            {
                "id": item.get("id"),
                "type": item.get("type"),
                "term_hits": term_hits,
                "term_count": len(terms),
                "coverage": round(coverage, 4),
                "type_hit": type_hit,
            }
        )
    coverage_bonus = min(0.06, covered * 0.018)
    score = round(min(0.18, score + coverage_bonus), 5)
    return score, {
        "schema": QUERY_DECOMPOSITION_SCHEMA_VERSION,
        "active": True,
        "covered_subquestions": covered,
        "subquestion_count": int(plan.get("subquestion_count") or len(plan.get("subquestions") or [])),
        "coverage": round(covered / max(1, int(plan.get("subquestion_count") or 1)), 4),
        "subquestion_matches": matches,
    }


def build_subquestions(query: str) -> list[dict[str, Any]]:
    pieces = split_question_parts(query)
    subquestions = []
    for index, piece in enumerate(pieces[:8], start=1):
        terms = important_terms(piece)
        if not terms:
            continue
        question_type = classify_subquestion(piece)
        subquestions.append(
            {
                "id": f"Q{index}",
                "query": piece,
                "type": question_type,
                "terms": terms,
                "weight": subquestion_weight(question_type),
            }
        )
    if len(subquestions) == 1:
        subquestions.extend(infer_missing_subquestions(query, subquestions[0])[:4])
    return subquestions[:8]


def split_question_parts(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(query or "")).strip()
    if not normalized:
        return []
    parts = re.split(
        r"\?(?=\s*\w)|[;\n]+|\b(?:and also|also|then|next|plus|as well as)\b|\b(?:what are|what is|where is|where are|explain|compare|list|do we need)\b",
        normalized,
        flags=re.I,
    )
    rebuilt = []
    for part in parts:
        cleaned = part.strip(" .,:;-?")
        if len(cleaned) > 4:
            rebuilt.append(cleaned)
    if not rebuilt:
        rebuilt = [normalized]
    if len(rebuilt) == 1:
        more = re.split(r"\b(?:and|or|with|between|versus|vs\.?|compared to)\b", normalized, flags=re.I)
        more = [part.strip(" .,:;-?") for part in more if len(part.strip()) > 6]
        if len(more) > 1:
            rebuilt = more
    return unique_preserve(rebuilt)


def infer_missing_subquestions(query: str, anchor: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_text(query)
    terms = anchor.get("terms") or important_terms(query)
    inferred = []
    if re.search(r"\b(types?|different|each)\b", normalized):
        inferred.append(make_inferred("enumeration", f"types or categories for {query}", terms))
    if re.search(r"\b(explain|what is|what are|define)\b", normalized):
        inferred.append(make_inferred("definition", f"definition and meaning for {query}", terms))
    if re.search(r"\b(do we need|necessary|required)\b", normalized):
        inferred.append(make_inferred("safety_requirement", f"document requirement necessity for {query}", terms))
    if re.search(r"\b(short paragraph|each of them|each)\b", normalized):
        inferred.append(make_inferred("enumeration", f"short explanation for each item in {query}", terms))
    return inferred


def make_inferred(question_type: str, query: str, terms: list[str]) -> dict[str, Any]:
    return {
        "id": f"QX-{question_type}",
        "query": query,
        "type": question_type,
        "terms": unique_preserve(terms + important_terms(query))[:18],
        "weight": subquestion_weight(question_type) * 0.9,
        "inferred": True,
    }


def classify_subquestion(text: str) -> str:
    normalized = normalize_text(text)
    for question_type, pattern in SUBQUESTION_PATTERNS.items():
        if re.search(pattern, normalized):
            return question_type
    return "semantic"


def subquestion_prefix(question_type: str) -> str:
    return {
        "definition": "definition meaning document-grounded",
        "enumeration": "list types categories mentioned",
        "comparison": "compare differences side by side",
        "table_numeric": "table numeric exact values",
        "safety_requirement": "shall must safety requirement",
        "location_section": "section page location",
        "procedure": "procedure steps sequence",
        "relationship": "related connected upstream downstream",
        "exception": "exception unless shall not",
    }.get(question_type, "semantic evidence")


def subquestion_type_hit(question_type: str, metadata: dict[str, Any], haystack: str) -> bool:
    if question_type == "table_numeric":
        return bool(metadata.get("contains_table") or metadata.get("has_numeric_constraints"))
    if question_type == "safety_requirement":
        return bool(metadata.get("safety_critical") or metadata.get("has_requirement") or re.search(r"\b(shall|must|required|safety)\b", haystack))
    if question_type == "location_section":
        return bool(metadata.get("section_title") or metadata.get("current_section_id") or metadata.get("page_start"))
    if question_type == "relationship":
        return bool(metadata.get("engineering_entity_relationships") or metadata.get("relationship_count"))
    if question_type == "procedure":
        return bool(re.search(r"\b(step|procedure|sequence|shall|ensure)\b", haystack))
    if question_type == "exception":
        return bool(re.search(r"\b(except|unless|shall not|must not|prohibited|not allowed)\b", haystack))
    if question_type == "enumeration":
        return bool(re.search(r"\b(types?|include|following|listed|categories)\b", haystack))
    if question_type == "definition":
        return bool(re.search(r"\b(is|are|means|defined|refers)\b", haystack))
    return bool(metadata.get("keywords") or metadata.get("engineering_entities"))


def subquestion_weight(question_type: str) -> float:
    return {
        "table_numeric": 1.3,
        "safety_requirement": 1.25,
        "exception": 1.22,
        "comparison": 1.15,
        "relationship": 1.12,
        "location_section": 1.08,
        "procedure": 1.08,
        "enumeration": 1.05,
        "definition": 1.0,
        "semantic": 1.0,
    }.get(question_type, 1.0)


def is_complex_question(query: str, profile: Any) -> bool:
    normalized = normalize_text(query)
    return (
        getattr(profile, "type_id", "") in {"multi_part", "comparison", "multi_constraint", "troubleshooting", "calculation"}
        or len(re.findall(r"\?", str(query or ""))) >= 2
        or bool(re.search(r"\b(and also|each of them|different types|compare|between|step by step|explain each)\b", normalized))
    )


def important_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", str(text or "")):
        value = normalize_text(token)
        if value and value not in DECOMPOSITION_STOPWORDS:
            terms.append(value)
    return unique_preserve(terms)[:18]


def decomposition_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(
        " ".join(
            [
                text,
                metadata.get("filename") or "",
                metadata.get("section_title") or "",
                metadata.get("current_section_title") or "",
                metadata.get("current_section_id") or "",
                metadata.get("table_title") or "",
                " ".join(metadata.get("section_path") or []),
                " ".join(metadata.get("table_rows") or []),
                " ".join(metadata.get("standards") or []),
                " ".join(metadata.get("technical_identifiers") or []),
                " ".join(metadata.get("engineering_entities") or []),
                " ".join(metadata.get("engineering_canonical_entities") or []),
                " ".join(metadata.get("primary_entities") or []),
                " ".join(metadata.get("relationship_types") or []),
                " ".join(metadata.get("keywords") or []),
                " ".join(metadata.get("semantic_labels") or []),
            ]
        )
    )


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.:/&+-]+", " ", str(value or "").lower())).strip()


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
