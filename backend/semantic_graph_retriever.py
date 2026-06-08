from __future__ import annotations

import re
from typing import Any

SEMANTIC_GRAPH_SCHEMA_VERSION = "engineering-semantic-graph-retriever-v1"


def semantic_graph_plan(query: str, route: Any, profile: Any, knowledge_graph_plan: dict[str, Any], ontology_plan: dict[str, Any]) -> dict[str, Any]:
    route_retrievers = set(getattr(route, "retrievers", ()) or ())
    query_terms = important_terms(query)
    relation_intent = detect_relation_intent(query)
    active = (
        "semantic_graph" in route_retrievers
        or bool(knowledge_graph_plan.get("active"))
        or bool(ontology_plan.get("active") and relation_intent)
        or getattr(route, "graph_mode", False)
    )
    graph_nodes = unique_preserve((knowledge_graph_plan.get("seed_nodes") or []) + (knowledge_graph_plan.get("expanded_nodes") or []))[:40]
    ontology_terms = unique_preserve((ontology_plan.get("concepts") or []) + (ontology_plan.get("expanded_terms") or []))[:60]
    return {
        "schema": SEMANTIC_GRAPH_SCHEMA_VERSION,
        "active": active,
        "route_primary": getattr(route, "primary", ""),
        "profile_type": getattr(profile, "type_id", ""),
        "query_terms": query_terms,
        "relation_intent": relation_intent,
        "graph_nodes": graph_nodes,
        "ontology_terms": ontology_terms,
        "relation_types": knowledge_graph_plan.get("relation_types") or [],
        "strategy": "semantic_dense_relevance_plus_entity_relationship_ontology_section_link_graph_scoring",
    }


def semantic_graph_candidate_score(
    plan: dict[str, Any],
    metadata: dict[str, Any],
    text: str,
    component_scores: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched": {}}
    component_scores = component_scores or {}
    haystack = semantic_graph_haystack(metadata, text)
    graph_hits = matching_terms(plan.get("graph_nodes") or [], haystack)
    ontology_hits = matching_terms(plan.get("ontology_terms") or [], haystack)
    relation_hits = matching_terms(plan.get("relation_types") or [], haystack)
    query_hits = matching_terms(plan.get("query_terms") or [], haystack)
    link_bonus = linked_context_bonus(metadata)
    section_bonus = 0.012 if metadata.get("section_title") or metadata.get("current_section_id") else 0.0
    relationship_bonus = 0.025 if metadata.get("engineering_entity_relationships") or metadata.get("relationship_graph_ready") else 0.0
    semantic_component = min(0.045, float(component_scores.get("vector") or 0) * 0.045)
    graph_component = min(0.05, float(component_scores.get("knowledge_graph") or 0) * 0.6 + len(graph_hits) * 0.008)
    ontology_component = min(0.04, float(component_scores.get("ontology") or 0) * 0.5 + len(ontology_hits) * 0.004)
    lexical_component = min(0.035, float(component_scores.get("keyword") or 0) * 0.025 + len(query_hits) * 0.004)
    relation_component = min(0.04, len(relation_hits) * 0.012 + relationship_bonus)
    total = semantic_component + graph_component + ontology_component + lexical_component + relation_component + link_bonus + section_bonus
    if plan.get("relation_intent") and (relation_hits or relationship_bonus):
        total += 0.018
    return round(min(0.18, total), 5), {
        "schema": SEMANTIC_GRAPH_SCHEMA_VERSION,
        "active": True,
        "matched": {
            "graph_nodes": graph_hits[:16],
            "ontology_terms": ontology_hits[:16],
            "relation_types": relation_hits[:12],
            "query_terms": query_hits[:16],
        },
        "components": {
            "semantic": round(semantic_component, 5),
            "graph": round(graph_component, 5),
            "ontology": round(ontology_component, 5),
            "lexical": round(lexical_component, 5),
            "relation": round(relation_component, 5),
            "link": round(link_bonus, 5),
            "section": round(section_bonus, 5),
        },
        "relation_intent": bool(plan.get("relation_intent")),
    }


def detect_relation_intent(query: str) -> bool:
    return bool(re.search(r"\b(relationship|related|connected|depends?|dependency|upstream|downstream|associated|between|link(?:ed)?)\b", str(query or ""), flags=re.I))


def linked_context_bonus(metadata: dict[str, Any]) -> float:
    link_count = 0
    for key in ("outbound_link_chunk_ids", "inbound_link_chunk_ids", "same_section_chunk_ids", "document_reference_chunk_ids"):
        values = metadata.get(key) or []
        if isinstance(values, list):
            link_count += len(values)
    return min(0.025, link_count * 0.004)


def semantic_graph_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(
        " ".join(
            [
                text,
                metadata.get("section_title") or "",
                metadata.get("current_section_id") or "",
                metadata.get("document_identifier") or "",
                flatten_values(metadata.get("section_path") or []),
                flatten_values(metadata.get("engineering_entities") or []),
                flatten_values(metadata.get("engineering_canonical_entities") or []),
                flatten_values(metadata.get("primary_entities") or []),
                flatten_values(metadata.get("engineering_entity_aliases") or []),
                flatten_values(metadata.get("engineering_entity_relationships") or []),
                flatten_values(metadata.get("relationship_types") or []),
                flatten_values(metadata.get("domain_terms") or []),
                flatten_values(metadata.get("semantic_labels") or []),
                flatten_values(metadata.get("keywords") or []),
                flatten_values(metadata.get("standards") or []),
            ]
        )
    )


def matching_terms(terms: list[str], haystack: str) -> list[str]:
    hits = []
    for term in terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        if " " in normalized or "/" in normalized or "-" in normalized:
            matched = normalized in haystack
        else:
            matched = bool(re.search(rf"\b{re.escape(normalized)}s?\b", haystack))
        if matched:
            hits.append(normalized)
    return unique_preserve(hits)


def important_terms(text: str) -> list[str]:
    stopwords = {"about", "and", "are", "does", "explain", "from", "how", "is", "show", "the", "this", "what", "where", "which", "with"}
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", str(text or "")):
        normalized = normalize_text(term)
        if normalized and normalized not in stopwords:
            terms.append(normalized)
    return unique_preserve(terms)[:30]


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
