from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

KNOWLEDGE_GRAPH_SCHEMA_VERSION = "engineering-knowledge-graph-v1"
GRAPH_QUERY_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "between",
    "connected",
    "dependency",
    "does",
    "each",
    "explain",
    "from",
    "have",
    "how",
    "is",
    "related",
    "relationship",
    "show",
    "the",
    "this",
    "to",
    "upstream",
    "what",
    "where",
    "which",
    "with",
}
GRAPH_RELATION_TERMS = {
    "associated",
    "connected",
    "downstream",
    "from",
    "located",
    "related",
    "requires",
    "supports",
    "to",
    "upstream",
}


@dataclass
class KnowledgeGraph:
    nodes: dict[str, dict[str, Any]]
    edges: dict[str, list[dict[str, Any]]]
    chunk_nodes: dict[int, set[str]]
    section_nodes: dict[str, set[str]]
    document_nodes: dict[int, set[str]]
    relation_counts: Counter
    query_terms: list[str]


def build_knowledge_graph(rows: list[Any], query: str) -> KnowledgeGraph:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_nodes: dict[int, set[str]] = defaultdict(set)
    section_nodes: dict[str, set[str]] = defaultdict(set)
    document_nodes: dict[int, set[str]] = defaultdict(set)
    relation_counts: Counter = Counter()

    for raw_row in rows:
        row = dict(raw_row)
        metadata = row_metadata(row)
        chunk_id = safe_int(row.get("id"))
        document_id = safe_int(row.get("document_id"))
        section_id = str(metadata.get("current_section_id") or metadata.get("section_title") or "")
        filename = str(row.get("filename") or metadata.get("filename") or "")
        chunk_text = str(row.get("text") or "")

        chunk_entity_keys = set()
        for record in entity_records_from_metadata(metadata):
            key = node_key(record.get("canonical") or record.get("text"))
            if not key:
                continue
            node = nodes.setdefault(
                key,
                {
                    "key": key,
                    "label": record.get("text") or record.get("canonical") or key,
                    "canonical": record.get("canonical") or key,
                    "type": record.get("type") or "entity",
                    "aliases": set(),
                    "chunks": set(),
                    "documents": set(),
                    "sections": set(),
                    "confidence": 0.0,
                    "mention_count": 0,
                },
            )
            node["aliases"].update(str(alias) for alias in record.get("aliases") or [])
            node["chunks"].add(chunk_id)
            node["documents"].add(document_id)
            if section_id:
                node["sections"].add(section_id)
            node["confidence"] = max(float(node.get("confidence") or 0), safe_float(record.get("confidence"), 0.5))
            node["mention_count"] += int(record.get("mention_count") or 1)
            chunk_entity_keys.add(key)

        for entity in metadata.get("engineering_entities") or []:
            key = node_key(entity)
            if key:
                node = nodes.setdefault(
                    key,
                    {
                        "key": key,
                        "label": str(entity),
                        "canonical": key,
                        "type": "entity",
                        "aliases": set(),
                        "chunks": set(),
                        "documents": set(),
                        "sections": set(),
                        "confidence": 0.45,
                        "mention_count": 0,
                    },
                )
                node["chunks"].add(chunk_id)
                node["documents"].add(document_id)
                if section_id:
                    node["sections"].add(section_id)
                node["mention_count"] += 1
                chunk_entity_keys.add(key)

        for standard in metadata.get("standards") or []:
            key = node_key(standard)
            if key:
                chunk_entity_keys.add(key)
                nodes.setdefault(
                    key,
                    {
                        "key": key,
                        "label": str(standard),
                        "canonical": key,
                        "type": "standard",
                        "aliases": {str(standard)},
                        "chunks": {chunk_id},
                        "documents": {document_id},
                        "sections": {section_id} if section_id else set(),
                        "confidence": 0.95,
                        "mention_count": 1,
                    },
                )

        chunk_nodes[chunk_id].update(chunk_entity_keys)
        if section_id:
            section_nodes[section_id].update(chunk_entity_keys)
        document_nodes[document_id].update(chunk_entity_keys)

        for relation in relationships_from_metadata(metadata):
            left = closest_graph_node(relation.get("left"), chunk_entity_keys)
            right = closest_graph_node(relation.get("right"), chunk_entity_keys)
            relation_type = normalize_relation(relation.get("relation"))
            if not left or not right or left == right:
                continue
            edge = {
                "left": left,
                "right": right,
                "relation": relation_type,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "section_id": section_id,
                "filename": filename,
            }
            add_edge(edges, left, edge)
            add_edge(edges, right, {**edge, "left": right, "right": left})
            relation_counts[relation_type] += 1

        inferred_pairs = infer_cooccurrence_edges(chunk_entity_keys, chunk_text, chunk_id, document_id, section_id, filename)
        for edge in inferred_pairs:
            add_edge(edges, edge["left"], edge)
            add_edge(edges, edge["right"], {**edge, "left": edge["right"], "right": edge["left"]})
            relation_counts[edge["relation"]] += 1

    normalize_graph_nodes(nodes)
    return KnowledgeGraph(
        nodes=nodes,
        edges=dict(edges),
        chunk_nodes=dict(chunk_nodes),
        section_nodes=dict(section_nodes),
        document_nodes=dict(document_nodes),
        relation_counts=relation_counts,
        query_terms=important_graph_terms(query),
    )


def knowledge_graph_query_plan(graph: KnowledgeGraph, query: str) -> dict[str, Any]:
    seeds = query_seed_nodes(graph, query)
    expansions = graph_expand(graph, seeds, max_depth=2)
    return {
        "schema": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
        "active": bool(seeds),
        "seed_nodes": seeds[:12],
        "expanded_nodes": expansions[:32],
        "query_terms": graph.query_terms,
        "relation_types": [relation for relation, _ in graph.relation_counts.most_common(12)],
        "node_count": len(graph.nodes),
        "edge_count": sum(len(edges) for edges in graph.edges.values()) // 2,
        "retrieval_strategy": "entity_relationship_section_document_graph",
    }


def knowledge_graph_candidate_score(plan: dict[str, Any], metadata: dict[str, Any], text: str) -> float:
    if not plan.get("active"):
        return 0.0
    candidate_nodes = metadata_node_keys(metadata)
    if not candidate_nodes:
        return 0.0
    seed_nodes = set(plan.get("seed_nodes") or [])
    expanded_nodes = set(plan.get("expanded_nodes") or [])
    exact_hits = len(candidate_nodes & seed_nodes)
    expanded_hits = len(candidate_nodes & expanded_nodes)
    relation_text = " ".join(metadata.get("relationship_types") or []) + " " + entity_relationship_text(metadata)
    relation_hits = sum(1 for term in plan.get("relation_types") or [] if term and term in relation_text.lower())
    term_hits = sum(1 for term in plan.get("query_terms") or [] if term in graph_haystack(metadata, text))
    score = 0.0
    score += min(0.10, exact_hits * 0.04)
    score += min(0.08, expanded_hits * 0.015)
    score += min(0.04, relation_hits * 0.012)
    score += min(0.04, term_hits * 0.01)
    if metadata.get("relationship_graph_ready") or metadata.get("engineering_entity_relationships"):
        score += 0.025
    if metadata.get("current_section_id") or metadata.get("section_title"):
        score += 0.01
    return round(min(0.18, score), 5)


def graph_expand(graph: KnowledgeGraph, seeds: list[str], max_depth: int = 2) -> list[str]:
    visited = set(seeds)
    queue = deque((seed, 0) for seed in seeds)
    expanded = list(seeds)
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in graph.edges.get(node, [])[:24]:
            neighbor = edge.get("right")
            if not neighbor or neighbor in visited:
                continue
            visited.add(neighbor)
            expanded.append(neighbor)
            queue.append((neighbor, depth + 1))
    return expanded


def query_seed_nodes(graph: KnowledgeGraph, query: str) -> list[str]:
    query_text = normalize_graph_text(query)
    scored = []
    for key, node in graph.nodes.items():
        aliases = [key, str(node.get("label") or ""), str(node.get("canonical") or ""), *list(node.get("aliases") or [])]
        alias_hits = sum(1 for alias in aliases if alias and normalize_graph_text(alias) in query_text)
        token_hits = sum(1 for term in graph.query_terms if term in normalize_graph_text(" ".join(aliases)))
        if alias_hits or token_hits:
            scored.append((alias_hits * 2 + token_hits + float(node.get("confidence") or 0), key))
    return [key for _, key in sorted(scored, reverse=True)[:16]]


def entity_records_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = [record for record in metadata.get("engineering_entity_records") or [] if isinstance(record, dict)]
    if records:
        return records
    return [{"text": entity, "canonical": entity, "type": "entity", "aliases": [entity], "confidence": 0.45} for entity in metadata.get("engineering_entities") or []]


def relationships_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    relationships = []
    for source in [metadata.get("engineering_entity_relationships") or [], metadata.get("relationship_records") or []]:
        relationships.extend(item for item in source if isinstance(item, dict))
    return relationships


def infer_cooccurrence_edges(entity_keys: set[str], text: str, chunk_id: int, document_id: int, section_id: str, filename: str) -> list[dict[str, Any]]:
    if len(entity_keys) < 2:
        return []
    relation = "co_occurs"
    lowered = text.lower()
    if "upstream" in lowered:
        relation = "upstream_downstream_context"
    elif "downstream" in lowered:
        relation = "downstream_context"
    elif "associated" in lowered or "connected" in lowered:
        relation = "associated_with"
    entities = sorted(entity_keys)[:10]
    edges = []
    for index, left in enumerate(entities):
        for right in entities[index + 1 : index + 4]:
            edges.append(
                {
                    "left": left,
                    "right": right,
                    "relation": relation,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "section_id": section_id,
                    "filename": filename,
                }
            )
    return edges[:24]


def add_edge(edges: dict[str, list[dict[str, Any]]], node: str, edge: dict[str, Any]) -> None:
    existing = edges.setdefault(node, [])
    signature = (edge.get("left"), edge.get("right"), edge.get("relation"), edge.get("chunk_id"))
    if any((item.get("left"), item.get("right"), item.get("relation"), item.get("chunk_id")) == signature for item in existing):
        return
    existing.append(edge)


def metadata_node_keys(metadata: dict[str, Any]) -> set[str]:
    keys = set()
    for record in entity_records_from_metadata(metadata):
        key = node_key(record.get("canonical") or record.get("text"))
        if key:
            keys.add(key)
    for entity in (metadata.get("engineering_entities") or []) + (metadata.get("engineering_canonical_entities") or []) + (metadata.get("primary_entities") or []):
        key = node_key(entity)
        if key:
            keys.add(key)
    for standard in metadata.get("standards") or []:
        key = node_key(standard)
        if key:
            keys.add(key)
    return keys


def entity_relationship_text(metadata: dict[str, Any]) -> str:
    parts = []
    for relation in relationships_from_metadata(metadata):
        parts.append(" ".join(str(relation.get(key) or "") for key in ("left", "relation", "right")))
    return " ".join(parts)


def graph_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_graph_text(
        " ".join(
            [
                text,
                " ".join(metadata.get("engineering_entities") or []),
                " ".join(metadata.get("engineering_canonical_entities") or []),
                " ".join(metadata.get("primary_entities") or []),
                " ".join(metadata.get("engineering_entity_aliases") or []),
                " ".join(metadata.get("relationship_types") or []),
                entity_relationship_text(metadata),
                metadata.get("section_title") or "",
                metadata.get("document_identifier") or "",
            ]
        )
    )


def closest_graph_node(value: Any, allowed: set[str]) -> str:
    key = node_key(value)
    if key in allowed:
        return key
    for candidate in allowed:
        if key and (key in candidate or candidate in key):
            return candidate
    return key if key else ""


def normalize_graph_nodes(nodes: dict[str, dict[str, Any]]) -> None:
    for node in nodes.values():
        node["aliases"] = sorted(str(alias) for alias in node.get("aliases") or [] if alias)
        node["chunks"] = sorted(int(value) for value in node.get("chunks") or [] if value is not None)
        node["documents"] = sorted(int(value) for value in node.get("documents") or [] if value is not None)
        node["sections"] = sorted(str(value) for value in node.get("sections") or [] if value)
        node["confidence"] = round(float(node.get("confidence") or 0), 3)


def node_key(value: Any) -> str:
    normalized = normalize_graph_text(value)
    if len(normalized) < 2:
        return ""
    return normalized


def normalize_relation(value: Any) -> str:
    relation = normalize_graph_text(value)
    for term in GRAPH_RELATION_TERMS:
        if term in relation:
            return term
    return relation or "related"


def important_graph_terms(text: str) -> list[str]:
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,}", str(text or "")):
        normalized = normalize_graph_text(term)
        if normalized and normalized not in GRAPH_QUERY_STOPWORDS:
            terms.append(normalized)
    return list(dict.fromkeys(terms))[:24]


def normalize_graph_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.:/-]+", " ", str(value or "").lower())).strip()


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            return json.loads(metadata or "{}")
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
