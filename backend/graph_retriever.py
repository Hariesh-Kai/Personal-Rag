from __future__ import annotations

from collections import Counter, deque
from typing import Any

from .knowledge_graph_retriever import KnowledgeGraph, graph_haystack, metadata_node_keys

GRAPH_RETRIEVER_SCHEMA_VERSION = "engineering-graph-retriever-v1"


def graph_retrieval_plan(graph: KnowledgeGraph | None, query: str, route: Any, knowledge_graph_plan: dict[str, Any]) -> dict[str, Any]:
    if graph is None:
        return {"schema": GRAPH_RETRIEVER_SCHEMA_VERSION, "active": False, "reason": "graph_not_built"}
    route_retrievers = set(getattr(route, "retrievers", ()) or ())
    seed_nodes = list(knowledge_graph_plan.get("seed_nodes") or [])
    expanded_nodes = graph_expand_with_edges(graph, seed_nodes, max_depth=3)
    chunk_scores = graph_chunk_scores(graph, seed_nodes, expanded_nodes)
    edge_evidence = graph_edge_evidence(graph, seed_nodes, expanded_nodes)
    active = "graph" in route_retrievers or getattr(route, "graph_mode", False) or bool(seed_nodes)
    return {
        "schema": GRAPH_RETRIEVER_SCHEMA_VERSION,
        "active": active,
        "route_primary": getattr(route, "primary", ""),
        "seed_nodes": seed_nodes[:16],
        "expanded_nodes": expanded_nodes[:80],
        "edge_evidence": edge_evidence[:40],
        "chunk_scores": {str(key): round(value, 5) for key, value in chunk_scores.items()},
        "node_count": len(graph.nodes),
        "edge_count": sum(len(edges) for edges in graph.edges.values()) // 2,
        "query_terms": graph.query_terms,
        "strategy": "entity_edge_graph_seed_expansion_chunk_distance_scoring",
    }


def graph_candidate_score(plan: dict[str, Any], metadata: dict[str, Any], text: str, row_id: Any) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched_nodes": []}
    candidate_nodes = metadata_node_keys(metadata)
    seed_nodes = set(plan.get("seed_nodes") or [])
    expanded_nodes = set(plan.get("expanded_nodes") or [])
    exact_hits = sorted(candidate_nodes & seed_nodes)
    expanded_hits = sorted((candidate_nodes & expanded_nodes) - set(exact_hits))
    chunk_graph_score = float((plan.get("chunk_scores") or {}).get(str(row_id)) or 0.0)
    haystack = graph_haystack(metadata, text)
    query_hits = [term for term in plan.get("query_terms") or [] if term and term in haystack]
    edge_hits = edge_hits_for_candidate(plan, candidate_nodes)
    score = 0.0
    score += min(0.08, len(exact_hits) * 0.035)
    score += min(0.07, len(expanded_hits) * 0.014)
    score += min(0.06, chunk_graph_score)
    score += min(0.035, len(edge_hits) * 0.009)
    score += min(0.03, len(query_hits) * 0.006)
    if metadata.get("relationship_graph_ready") or metadata.get("engineering_entity_relationships"):
        score += 0.02
    return round(min(0.18, score), 5), {
        "schema": GRAPH_RETRIEVER_SCHEMA_VERSION,
        "active": True,
        "matched_seed_nodes": exact_hits[:12],
        "matched_expanded_nodes": expanded_hits[:16],
        "matched_edges": edge_hits[:12],
        "query_hits": query_hits[:12],
        "chunk_graph_score": round(chunk_graph_score, 5),
    }


def expand_graph_candidates(ranked: list[dict], all_scored: list[dict], plan: dict[str, Any], limit: int) -> list[dict]:
    if not ranked or not plan.get("active"):
        return ranked
    graph_scores = plan.get("chunk_scores") or {}
    if not graph_scores:
        return ranked
    by_id = {str(item.get("id")): item for item in all_scored}
    selected = list(ranked)
    seen_ids = {str(item.get("id")) for item in selected}
    candidate_ids = sorted(graph_scores, key=lambda key: float(graph_scores.get(key) or 0), reverse=True)
    for chunk_id in candidate_ids[: max(limit * 3, 12)]:
        if chunk_id in seen_ids:
            continue
        item = by_id.get(chunk_id)
        if not item:
            continue
        item = dict(item)
        item["score"] = max(float(item.get("score") or 0), float(graph_scores.get(chunk_id) or 0) + 0.1)
        item["graph_expanded"] = True
        selected.append(item)
        seen_ids.add(chunk_id)
        if len(selected) >= limit * 2:
            break
    return sorted(selected, key=lambda row: float(row.get("score", 0)), reverse=True)[: max(limit, len(ranked))]


def graph_expand_with_edges(graph: KnowledgeGraph, seeds: list[str], max_depth: int = 3) -> list[str]:
    if not seeds:
        return []
    visited = set(seeds)
    queue = deque((seed, 0) for seed in seeds)
    expanded = list(seeds)
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        edges = sorted(graph.edges.get(node, []), key=lambda edge: relation_weight(edge.get("relation")), reverse=True)
        for edge in edges[:32]:
            neighbor = edge.get("right")
            if not neighbor or neighbor in visited:
                continue
            visited.add(neighbor)
            expanded.append(neighbor)
            queue.append((neighbor, depth + 1))
    return expanded


def graph_chunk_scores(graph: KnowledgeGraph, seeds: list[str], expanded_nodes: list[str]) -> dict[int, float]:
    seed_set = set(seeds)
    expanded_set = set(expanded_nodes)
    scores: Counter[int] = Counter()
    for node in seed_set:
        for chunk_id in graph.nodes.get(node, {}).get("chunks") or []:
            scores[int(chunk_id)] += 0.06
    for node in expanded_set - seed_set:
        for chunk_id in graph.nodes.get(node, {}).get("chunks") or []:
            scores[int(chunk_id)] += 0.018
    for node in expanded_set:
        for edge in graph.edges.get(node, [])[:24]:
            chunk_id = edge.get("chunk_id")
            if chunk_id is not None:
                scores[int(chunk_id)] += 0.012 * relation_weight(edge.get("relation"))
    return {chunk_id: min(0.16, score) for chunk_id, score in scores.items()}


def graph_edge_evidence(graph: KnowledgeGraph, seeds: list[str], expanded_nodes: list[str]) -> list[dict[str, Any]]:
    allowed = set(expanded_nodes) or set(seeds)
    evidence = []
    seen = set()
    for node in list(seeds) + list(expanded_nodes):
        for edge in graph.edges.get(node, [])[:24]:
            if edge.get("right") not in allowed:
                continue
            signature = (edge.get("left"), edge.get("relation"), edge.get("right"), edge.get("chunk_id"))
            if signature in seen:
                continue
            seen.add(signature)
            evidence.append(
                {
                    "left": edge.get("left"),
                    "relation": edge.get("relation"),
                    "right": edge.get("right"),
                    "chunk_id": edge.get("chunk_id"),
                    "section_id": edge.get("section_id"),
                    "filename": edge.get("filename"),
                }
            )
    return evidence


def edge_hits_for_candidate(plan: dict[str, Any], candidate_nodes: set[str]) -> list[dict[str, Any]]:
    hits = []
    for edge in plan.get("edge_evidence") or []:
        if edge.get("left") in candidate_nodes or edge.get("right") in candidate_nodes:
            hits.append(edge)
    return hits


def relation_weight(relation: Any) -> float:
    value = str(relation or "").lower()
    if any(term in value for term in ("requires", "supports", "connected", "associated", "upstream", "downstream")):
        return 1.25
    if "co_occurs" in value:
        return 0.75
    return 1.0
