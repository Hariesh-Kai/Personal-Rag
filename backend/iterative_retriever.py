from __future__ import annotations

import json
import re
from typing import Any

ITERATIVE_RETRIEVER_SCHEMA_VERSION = "engineering-iterative-retriever-v1"
ITERATIVE_STOPWORDS = {
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


def iterative_plan(query: str, route: Any, profile: Any) -> dict[str, Any]:
    active = "iterative" in set(getattr(route, "retrievers", ())) or getattr(profile, "type_id", "") in {
        "multi_part",
        "troubleshooting",
        "conflict_detection",
        "multi_document",
        "cross_section",
        "knowledge_gap",
    }
    terms = important_terms(query)
    return {
        "schema": ITERATIVE_RETRIEVER_SCHEMA_VERSION,
        "active": active,
        "route_primary": getattr(route, "primary", ""),
        "profile_type": getattr(profile, "type_id", ""),
        "query_terms": terms,
        "max_iterations": 2,
        "strategy": "first_pass_analyze_missing_terms_retry_merge_rescore",
    }


def build_iterative_retry_queries(plan: dict[str, Any], rows: list[Any], query: str) -> dict[str, Any]:
    if not plan.get("active"):
        return {**plan, "retry_needed": False, "retry_queries": [], "missing_terms": []}
    row_dicts = [dict(row) for row in rows[:30]]
    coverage = coverage_report(plan.get("query_terms") or [], row_dicts)
    retry_queries = []
    missing_terms = coverage["missing_terms"]
    weak_signals = detect_weak_signals(query, row_dicts)
    if missing_terms:
        retry_queries.append(f"exact missing terms {' '.join(missing_terms[:12])} {query}")
    if weak_signals.get("needs_table"):
        retry_queries.append(f"table rows numeric values {' '.join((plan.get('query_terms') or [])[:12])}")
    if weak_signals.get("needs_safety"):
        retry_queries.append(f"safety shall must exception surrounding context {' '.join((plan.get('query_terms') or [])[:12])}")
    if weak_signals.get("needs_section"):
        retry_queries.append(f"section title page document location {' '.join((plan.get('query_terms') or [])[:12])}")
    if weak_signals.get("needs_relationship"):
        retry_queries.append(f"related connected upstream downstream relationship {' '.join((plan.get('query_terms') or [])[:12])}")
    if coverage["coverage"] < 0.45:
        retry_queries.append(f"broader engineering context {' '.join((plan.get('query_terms') or [])[:16])}")
    retry_queries = unique_preserve(retry_queries)[:5]
    return {
        **plan,
        "retry_needed": bool(retry_queries),
        "retry_queries": retry_queries,
        "missing_terms": missing_terms,
        "coverage": coverage,
        "weak_signals": weak_signals,
    }


def iterative_candidate_score(plan: dict[str, Any], row: dict[str, Any], metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False}
    haystack = iterative_haystack(metadata, text)
    missing_hits = sum(1 for term in plan.get("missing_terms") or [] if normalize_text(term) in haystack)
    query_hits = sum(1 for term in plan.get("query_terms") or [] if normalize_text(term) in haystack)
    recovered_by_retry = bool(row.get("iterative_retrieved"))
    weak_signal_hits = []
    weak = plan.get("weak_signals") or {}
    if weak.get("needs_table") and metadata.get("contains_table"):
        weak_signal_hits.append("table")
    if weak.get("needs_safety") and (metadata.get("safety_critical") or metadata.get("has_requirement")):
        weak_signal_hits.append("safety")
    if weak.get("needs_section") and (metadata.get("section_title") or metadata.get("current_section_id") or metadata.get("page_start")):
        weak_signal_hits.append("section")
    if weak.get("needs_relationship") and (metadata.get("engineering_entity_relationships") or metadata.get("relationship_count")):
        weak_signal_hits.append("relationship")

    score = 0.0
    score += min(0.08, missing_hits * 0.025)
    score += min(0.05, query_hits * 0.006)
    score += min(0.05, len(weak_signal_hits) * 0.018)
    if recovered_by_retry:
        score += 0.045
    if row.get("iterative_retry_query"):
        score += 0.015
    return round(min(0.18, score), 5), {
        "schema": ITERATIVE_RETRIEVER_SCHEMA_VERSION,
        "active": True,
        "missing_hits": missing_hits,
        "query_hits": query_hits,
        "weak_signal_hits": weak_signal_hits,
        "recovered_by_retry": recovered_by_retry,
        "retry_query": row.get("iterative_retry_query") or "",
    }


def merge_iterative_rows(rows: list[Any], retry_rows: list[dict[str, Any]]) -> list[Any]:
    if not retry_rows:
        return rows
    merged = list(rows)
    seen_ids = {str(dict(row).get("id")) for row in rows}
    for row in retry_rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            continue
        row["iterative_retrieved"] = True
        merged.append(row)
        seen_ids.add(row_id)
    return merged


def annotate_retry_rows(rows: list[dict[str, Any]], retry_query: str) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item["iterative_retrieved"] = True
        item["iterative_retry_query"] = retry_query
        annotated.append(item)
    return annotated


def coverage_report(query_terms: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_terms:
        return {"coverage": 1.0, "covered_terms": [], "missing_terms": []}
    haystack = normalize_text(
        " ".join(
            " ".join(
                [
                    str(row.get("text") or ""),
                    metadata_text(row_metadata(row)),
                ]
            )
            for row in rows
        )
    )
    covered = [term for term in query_terms if normalize_text(term) in haystack]
    missing = [term for term in query_terms if normalize_text(term) not in haystack]
    return {
        "coverage": round(len(covered) / max(1, len(query_terms)), 4),
        "covered_terms": covered[:32],
        "missing_terms": missing[:32],
    }


def detect_weak_signals(query: str, rows: list[dict[str, Any]]) -> dict[str, bool]:
    normalized = normalize_text(query)
    metadata_list = [row_metadata(row) for row in rows]
    has_table = any(metadata.get("contains_table") for metadata in metadata_list)
    has_safety = any(metadata.get("safety_critical") or metadata.get("has_requirement") for metadata in metadata_list)
    has_section = any(metadata.get("section_title") or metadata.get("current_section_id") for metadata in metadata_list)
    has_relationship = any(metadata.get("engineering_entity_relationships") or metadata.get("relationship_count") for metadata in metadata_list)
    return {
        "needs_table": bool(re.search(r"\b(table|row|column|numeric|value|slope|dimension|rating)\b", normalized)) and not has_table,
        "needs_safety": bool(re.search(r"\b(safety|hazard|fire|explosion|shutdown|relief|shall|must|exception)\b", normalized)) and not has_safety,
        "needs_section": bool(re.search(r"\b(section|page|where|location|document)\b", normalized)) and not has_section,
        "needs_relationship": bool(re.search(r"\b(related|connected|upstream|downstream|dependency|relationship|conflict)\b", normalized)) and not has_relationship,
    }


def important_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", str(text or "")):
        value = normalize_text(token)
        if value and value not in ITERATIVE_STOPWORDS:
            terms.append(value)
    return unique_preserve(terms)[:32]


def iterative_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(f"{text} {metadata_text(metadata)}")


def metadata_text(metadata: dict[str, Any]) -> str:
    parts = [
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
    return " ".join(str(part) for part in parts if part)


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            return json.loads(metadata or "{}")
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


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
