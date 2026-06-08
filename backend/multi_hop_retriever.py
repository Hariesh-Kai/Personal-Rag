from __future__ import annotations

import re
from typing import Any

MULTI_HOP_SCHEMA_VERSION = "engineering-multi-hop-retriever-v1"
MULTI_HOP_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "between",
    "does",
    "each",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "have",
    "how",
    "related",
    "show",
    "the",
    "then",
    "this",
    "to",
    "what",
    "where",
    "which",
    "with",
}
RELATION_MARKERS = {
    "related": "related_context",
    "connected": "connected_context",
    "upstream": "upstream_context",
    "downstream": "downstream_context",
    "associated": "associated_context",
    "requires": "requirement_context",
    "because": "causal_context",
    "therefore": "causal_context",
    "safety": "safety_context",
    "exception": "exception_context",
    "conflict": "conflict_context",
}


def multi_hop_plan(query: str, route: Any) -> dict[str, Any]:
    active = "multi_hop" in set(getattr(route, "retrievers", ())) or is_multi_hop_question(query)
    hops = build_hops(query)
    return {
        "schema": MULTI_HOP_SCHEMA_VERSION,
        "active": bool(active and hops),
        "route_primary": getattr(route, "primary", ""),
        "hop_count": len(hops),
        "hops": hops,
        "hop_queries": [hop["query"] for hop in hops],
        "required_terms": unique_preserve(term for hop in hops for term in hop.get("terms", []))[:32],
        "relation_types": unique_preserve(hop.get("relation_type") for hop in hops if hop.get("relation_type")),
        "strategy": "decompose_retrieve_link_expand_score_by_hop_coverage",
    }


def multi_hop_expanded_queries(seed_queries: list[str], plan: dict[str, Any]) -> list[str]:
    queries = list(seed_queries)
    if not plan.get("active"):
        return unique_preserve(queries)[:8]
    queries.extend(plan.get("hop_queries") or [])
    relation_terms = " ".join(plan.get("relation_types") or [])
    required_terms = " ".join(plan.get("required_terms") or [])
    if relation_terms or required_terms:
        queries.append(f"multi hop relationship context {relation_terms} {required_terms}")
        queries.append(f"linked section related requirements {required_terms}")
    return unique_preserve(query for query in queries if str(query).strip())[:10]


def multi_hop_candidate_score(plan: dict[str, Any], row: dict[str, Any], metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "hop_matches": []}
    haystack = multi_hop_haystack(metadata, text)
    hop_matches = []
    covered_hops = 0
    weighted_score = 0.0
    for hop in plan.get("hops") or []:
        terms = hop.get("terms") or []
        term_hits = sum(1 for term in terms if normalize_text(term) in haystack)
        relation_hit = bool(hop.get("relation_type") and normalize_text(hop["relation_type"]) in haystack)
        metadata_hit = hop_metadata_hit(hop, metadata)
        coverage = term_hits / max(1, len(terms))
        hop_score = coverage * 0.065 + (0.025 if relation_hit else 0) + (0.025 if metadata_hit else 0)
        if coverage >= 0.34 or relation_hit or metadata_hit:
            covered_hops += 1
        weighted_score += hop_score * float(hop.get("weight") or 1.0)
        hop_matches.append(
            {
                "hop_id": hop.get("id"),
                "intent": hop.get("intent"),
                "relation_type": hop.get("relation_type"),
                "term_hits": term_hits,
                "term_count": len(terms),
                "coverage": round(coverage, 4),
                "relation_hit": relation_hit,
                "metadata_hit": metadata_hit,
            }
        )

    link_bonus = linked_context_bonus(metadata)
    graph_bonus = 0.025 if metadata.get("engineering_entity_relationships") or metadata.get("relationship_graph_ready") else 0.0
    section_bonus = 0.018 if metadata.get("same_section_chunk_ids") or metadata.get("outbound_link_chunk_ids") else 0.0
    coverage_bonus = min(0.06, covered_hops * 0.018)
    score = round(min(0.22, weighted_score + link_bonus + graph_bonus + section_bonus + coverage_bonus), 5)
    return score, {
        "schema": MULTI_HOP_SCHEMA_VERSION,
        "active": True,
        "covered_hops": covered_hops,
        "hop_count": int(plan.get("hop_count") or len(plan.get("hops") or [])),
        "coverage": round(covered_hops / max(1, int(plan.get("hop_count") or 1)), 4),
        "hop_matches": hop_matches,
        "link_bonus": round(link_bonus, 4),
        "graph_bonus": round(graph_bonus, 4),
        "section_bonus": round(section_bonus, 4),
    }


def expand_multi_hop_candidates(ranked: list[dict[str, Any]], all_scored: list[dict[str, Any]], plan: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if not plan.get("active") or not ranked:
        return ranked
    selected = list(ranked)
    seen_ids = {item["id"] for item in selected}
    by_chunk_id = {}
    by_doc_index = {}
    by_section = {}
    for item in all_scored:
        metadata = item.get("metadata") or {}
        chunk_id = str(metadata.get("chunk_id") or "")
        if chunk_id:
            by_chunk_id[chunk_id] = item
        by_doc_index[(int(item.get("document_id") or 0), int(item.get("chunk_index") or 0))] = item
        section = metadata.get("section_title") or metadata.get("current_section_title") or ""
        if section:
            by_section.setdefault(section, []).append(item)

    for anchor in ranked[: min(4, len(ranked))]:
        metadata = anchor.get("metadata") or {}
        linked_ids = unique_preserve(
            (metadata.get("outbound_link_chunk_ids") or [])
            + (metadata.get("inbound_link_chunk_ids") or [])
            + (metadata.get("same_section_chunk_ids") or [])[:6]
            + (metadata.get("resolved_section_chunk_ids") or [])[:6]
        )
        for linked_id in linked_ids[:16]:
            linked = by_chunk_id.get(str(linked_id))
            if linked and linked["id"] not in seen_ids:
                selected.append(promote_multi_hop_candidate(linked, anchor, "linked_chunk"))
                seen_ids.add(linked["id"])
        doc_id = int(anchor.get("document_id") or 0)
        chunk_index = int(anchor.get("chunk_index") or 0)
        for offset in (-2, -1, 1, 2):
            neighbor = by_doc_index.get((doc_id, chunk_index + offset))
            if neighbor and neighbor["id"] not in seen_ids:
                selected.append(promote_multi_hop_candidate(neighbor, anchor, "neighbor_chunk"))
                seen_ids.add(neighbor["id"])
        section = metadata.get("section_title") or metadata.get("current_section_title") or ""
        for sibling in (by_section.get(section) or [])[:8]:
            if sibling["id"] not in seen_ids:
                selected.append(promote_multi_hop_candidate(sibling, anchor, "same_section"))
                seen_ids.add(sibling["id"])
        if len(selected) >= limit * 3:
            break
    return sorted(selected, key=lambda item: float(item.get("score", 0)), reverse=True)[: max(limit, len(ranked))]


def promote_multi_hop_candidate(candidate: dict[str, Any], anchor: dict[str, Any], reason: str) -> dict[str, Any]:
    promoted = dict(candidate)
    promoted["score"] = max(float(promoted.get("score", 0)), float(anchor.get("score", 0)) * 0.76)
    promoted["multi_hop_expanded_from"] = (anchor.get("metadata") or {}).get("chunk_id") or anchor.get("id")
    promoted["multi_hop_expansion_reason"] = reason
    return promoted


def build_hops(query: str) -> list[dict[str, Any]]:
    parts = split_hop_parts(query)
    if len(parts) == 1:
        parts.extend(infer_secondary_hops(query))
    hops = []
    for index, part in enumerate(parts[:6], start=1):
        terms = important_terms(part)
        if not terms:
            continue
        relation_type = detect_relation_type(part)
        intent = detect_hop_intent(part, relation_type)
        hops.append(
            {
                "id": f"H{index}",
                "query": part,
                "terms": terms,
                "relation_type": relation_type,
                "intent": intent,
                "weight": hop_weight(intent, index),
            }
        )
    return hops


def split_hop_parts(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(query or "")).strip()
    parts = re.split(r"\b(?:then|after that|next|and then|before|after|because|therefore|so that|related to|connected to|depends on|requires)\b|[;?]", normalized, flags=re.I)
    parts = [part.strip(" .,:;-") for part in parts if len(part.strip()) > 5]
    if len(parts) <= 1:
        parts = re.split(r"\b(?:and|with|between|versus|vs\.?|compared to)\b", normalized, flags=re.I)
        parts = [part.strip(" .,:;-") for part in parts if len(part.strip()) > 5]
    return unique_preserve(parts)[:6]


def infer_secondary_hops(query: str) -> list[str]:
    lowered = query.lower()
    hops = []
    if re.search(r"\b(safety|hazard|fire|explosion|shutdown|relief)\b", lowered):
        hops.append(f"safety requirements surrounding {query}")
    if re.search(r"\b(table|numeric|slope|value|dimension|rating)\b", lowered):
        hops.append(f"numeric table values related to {query}")
    if re.search(r"\b(upstream|downstream|connected|associated|dependency|relationship)\b", lowered):
        hops.append(f"connected upstream downstream related systems for {query}")
    if re.search(r"\b(conflict|contradict|difference|mismatch)\b", lowered):
        hops.append(f"conflicting related sections and documents for {query}")
    return hops


def detect_relation_type(text: str) -> str:
    normalized = normalize_text(text)
    for marker, relation_type in RELATION_MARKERS.items():
        if marker in normalized:
            return relation_type
    return "semantic_context"


def detect_hop_intent(text: str, relation_type: str) -> str:
    normalized = normalize_text(text)
    if re.search(r"\b(table|row|column|slope|value|dimension|rating|numeric)\b", normalized):
        return "table_numeric_evidence"
    if re.search(r"\b(section|page|document|where|location)\b", normalized):
        return "metadata_location_evidence"
    if re.search(r"\b(safety|hazard|fire|explosion|shutdown|relief|shall|must)\b", normalized):
        return "safety_requirement_evidence"
    if relation_type != "semantic_context":
        return "relationship_evidence"
    return "semantic_evidence"


def hop_metadata_hit(hop: dict[str, Any], metadata: dict[str, Any]) -> bool:
    intent = hop.get("intent")
    if intent == "table_numeric_evidence":
        return bool(metadata.get("contains_table") or metadata.get("has_numeric_constraints"))
    if intent == "metadata_location_evidence":
        return bool(metadata.get("section_title") or metadata.get("current_section_id") or metadata.get("page_start"))
    if intent == "safety_requirement_evidence":
        return bool(metadata.get("safety_critical") or metadata.get("has_requirement") or metadata.get("requirement_modalities"))
    if intent == "relationship_evidence":
        return bool(metadata.get("relationship_count") or metadata.get("engineering_entity_relationships"))
    return bool(metadata.get("keywords") or metadata.get("engineering_entities"))


def hop_weight(intent: str, index: int) -> float:
    base = {
        "table_numeric_evidence": 1.25,
        "safety_requirement_evidence": 1.2,
        "relationship_evidence": 1.15,
        "metadata_location_evidence": 1.05,
        "semantic_evidence": 1.0,
    }.get(intent, 1.0)
    return round(base * (1.0 if index == 1 else 0.92), 2)


def linked_context_bonus(metadata: dict[str, Any]) -> float:
    link_count = len(metadata.get("outbound_link_chunk_ids") or []) + len(metadata.get("inbound_link_chunk_ids") or [])
    same_section = len(metadata.get("same_section_chunk_ids") or [])
    references = len(metadata.get("referenced_section_ids") or []) + len(metadata.get("resolved_section_chunk_ids") or [])
    return min(0.045, link_count * 0.008 + same_section * 0.004 + references * 0.008)


def is_multi_hop_question(query: str) -> bool:
    return bool(
        re.search(
            r"\b(then|after that|multi[- ]?hop|related to|connected to|depends on|requires|upstream|downstream|because|therefore|conflict|across documents)\b",
            str(query or ""),
            flags=re.I,
        )
    )


def multi_hop_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(
        " ".join(
            [
                text,
                metadata.get("filename") or "",
                metadata.get("section_title") or "",
                metadata.get("current_section_title") or "",
                metadata.get("current_section_id") or "",
                " ".join(metadata.get("section_path") or []),
                metadata.get("table_title") or "",
                " ".join(metadata.get("table_rows") or []),
                " ".join(metadata.get("standards") or []),
                " ".join(metadata.get("technical_identifiers") or []),
                " ".join(metadata.get("engineering_entities") or []),
                " ".join(metadata.get("engineering_canonical_entities") or []),
                " ".join(metadata.get("primary_entities") or []),
                " ".join(metadata.get("relationship_types") or []),
                " ".join(metadata.get("semantic_labels") or []),
                " ".join(metadata.get("keywords") or []),
                relationship_text(metadata),
            ]
        )
    )


def relationship_text(metadata: dict[str, Any]) -> str:
    parts = []
    for relation in (metadata.get("engineering_entity_relationships") or []) + (metadata.get("relationship_records") or []):
        if isinstance(relation, dict):
            parts.append(" ".join(str(relation.get(key) or "") for key in ("left", "relation", "right")))
    return " ".join(parts)


def important_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", str(text or "")):
        value = normalize_text(token)
        if value and value not in MULTI_HOP_STOPWORDS:
            terms.append(value)
    return unique_preserve(terms)[:18]


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
