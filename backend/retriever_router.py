from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .question_types import QuestionProfile


@dataclass(frozen=True)
class RetrieverRoute:
    primary: str
    retrievers: tuple[str, ...]
    semantic_weight: float
    keyword_weight: float
    phrase_weight: float
    rerank_weight: float
    table_weight: float
    metadata_weight: float = 0.0
    exact_boost: float = 0.0
    window_size: int = 0
    diversify_sections: bool = False
    include_siblings: bool = False
    multi_query: bool = False
    parent_child: bool = False
    citation_mode: bool = False
    cache_mode: bool = False
    graph_mode: bool = False
    route_notes: str = ""


@dataclass(frozen=True)
class RetrieverCapability:
    key: str
    label: str
    implemented_as: str
    signal: str


RETRIEVER_CAPABILITIES: dict[str, RetrieverCapability] = {
    "semantic_vector": RetrieverCapability("semantic_vector", "Semantic / vector retriever", "E5 dense embeddings over enriched chunks", "conceptual and paraphrase similarity"),
    "keyword": RetrieverCapability("keyword", "Keyword retriever", "exact lexical scoring", "exact engineering words"),
    "hybrid": RetrieverCapability("hybrid", "Hybrid retriever", "combined dense, keyword, phrase, rerank, table, metadata scoring", "mixed enterprise QA"),
    "table": RetrieverCapability("table", "Table retriever", "table metadata, rows, columns, title, numeric boost", "tables, rows, values, slopes, matrices"),
    "metadata": RetrieverCapability("metadata", "Metadata retriever", "section, filename, revision, document identifier scoring", "document scope and section filters"),
    "reranker": RetrieverCapability("reranker", "Reranker retriever", "local lexical+dense precision rerank", "similar-topic precision"),
    "window": RetrieverCapability("window", "Window retriever", "neighbor chunk expansion", "surrounding context"),
    "parent_child": RetrieverCapability("parent_child", "Parent-child retriever", "same-section sibling/parent context expansion", "hierarchy preservation"),
    "hierarchical": RetrieverCapability("hierarchical", "Hierarchical retriever", "document > section > subsection routing", "section navigation"),
    "multi_query": RetrieverCapability("multi_query", "Multi-query retriever", "rule-based query rewrites", "vague and messy questions"),
    "self_query": RetrieverCapability("self_query", "Self-query retriever", "metadata-intent routing without external LLM filters", "scoped metadata filtering"),
    "contextual_compression": RetrieverCapability("contextual_compression", "Contextual compression retriever", "retrieves broader context then answer prompt compresses evidence", "large sections and procedures"),
    "query_router": RetrieverCapability("query_router", "Query router retriever", "question profile to route selection", "dynamic retriever choice"),
    "multimodal": RetrieverCapability("multimodal", "Multimodal retriever", "OCR/image/layout metadata route", "diagrams, figures, P&ID/layout questions"),
    "agentic": RetrieverCapability("agentic", "Agentic retriever", "bounded iterative expansion, not autonomous web/tool use", "not-enough-context expansion"),
    "graph": RetrieverCapability("graph", "Graph retriever", "entity-relation scoring approximation", "connected engineering systems"),
    "knowledge_graph": RetrieverCapability("knowledge_graph", "Knowledge graph retriever", "structured entity relationship approximation", "enterprise knowledge relationships"),
    "dense_passage": RetrieverCapability("dense_passage", "Dense passage retriever", "E5 query/passage embedding search", "semantic QA"),
    "sparse": RetrieverCapability("sparse", "Sparse retriever", "token/phrase weighting", "interpretable exact scoring"),
    "late_interaction": RetrieverCapability("late_interaction", "Late interaction retriever", "token-level lexical overlap proxy", "precise term alignment"),
    "memory": RetrieverCapability("memory", "Memory retriever", "recent chat follow-up resolution", "conversation continuity"),
    "time_aware": RetrieverCapability("time_aware", "Time-aware retriever", "revision/date/version metadata boost", "revision-aware retrieval"),
    "citation": RetrieverCapability("citation", "Citation retriever", "page/section/source evidence boost", "audit and traceability"),
    "sql_database": RetrieverCapability("sql_database", "SQL/database retriever", "local SQLite metadata/chunk database lookup", "structured local database facts"),
    "api": RetrieverCapability("api", "API retriever", "guarded unavailable external API slot", "live systems only when connected"),
    "multi_hop": RetrieverCapability("multi_hop", "Multi-hop retriever", "query decomposition plus related-section expansion", "staged relationship retrieval"),
    "iterative": RetrieverCapability("iterative", "Iterative retriever", "expanded query pass and context expansion", "retrieve-analyze-expand loop"),
    "query_decomposition": RetrieverCapability("query_decomposition", "Query decomposition retriever", "split complex questions into focused route queries", "multi-part questions"),
    "tool_aware": RetrieverCapability("tool_aware", "Tool-aware retriever", "routes to vector/table/metadata/database-safe behavior", "choosing retrieval tools"),
    "cache": RetrieverCapability("cache", "Cache retriever", "recent interaction awareness and route metadata", "latency and repeated questions"),
    "multi_vector": RetrieverCapability("multi_vector", "Multi-vector retriever", "text + title + section + table metadata searchable text", "multi-view chunk retrieval"),
    "section_aware": RetrieverCapability("section_aware", "Section-aware retriever", "heading hierarchy metadata boost", "section/subsection weighting"),
    "entity": RetrieverCapability("entity", "Entity retriever", "identifier/entity term extraction and exact boosts", "tags, systems, equipment names"),
    "ontology": RetrieverCapability("ontology", "Ontology retriever", "domain term dictionary expansion", "engineering taxonomy approximation"),
    "symbolic": RetrieverCapability("symbolic", "Symbolic retriever", "shall/must/if/unless/exception rule-word boosts", "rule-based retrieval"),
    "semantic_graph": RetrieverCapability("semantic_graph", "Semantic graph retriever", "graph terms plus dense similarity", "hybrid graph and embeddings"),
    "document_map": RetrieverCapability("document_map", "Document map retriever", "document structure and linked-section routing", "document navigation"),
    "layout_aware": RetrieverCapability("layout_aware", "Layout-aware retriever", "table/image/page/layout metadata", "layout-sensitive PDFs"),
    "image_region": RetrieverCapability("image_region", "Image region retriever", "OCR/image block metadata route", "image areas and diagram evidence"),
    "retriever_routing_system": RetrieverCapability("retriever_routing_system", "Retriever routing system", "profile classifier chooses route and active retrievers", "full dynamic routing"),
}


ROUTES: dict[str, RetrieverRoute] = {
    "semantic_vector": RetrieverRoute("semantic_vector", ("retriever_routing_system", "semantic_vector", "dense_passage", "multi_vector"), 0.58, 0.18, 0.06, 0.14, 0.02, 0.02, route_notes="Conceptual, paraphrase, explanation, workflow retrieval."),
    "keyword_sparse": RetrieverRoute("keyword_sparse", ("retriever_routing_system", "keyword", "sparse", "entity", "late_interaction"), 0.18, 0.50, 0.16, 0.10, 0.01, 0.05, 0.08, route_notes="Exact terms, identifiers, section titles, document IDs."),
    "hybrid": RetrieverRoute("hybrid", ("retriever_routing_system", "hybrid", "semantic_vector", "keyword", "sparse", "reranker", "multi_vector"), 0.38, 0.32, 0.10, 0.15, 0.03, 0.02, route_notes="Default enterprise QA retrieval."),
    "table": RetrieverRoute("table", ("retriever_routing_system", "table", "multi_vector", "keyword", "sparse", "reranker", "late_interaction"), 0.14, 0.22, 0.18, 0.12, 0.30, 0.04, 0.10, route_notes="Tables, rows, numeric values, directional specifications."),
    "metadata": RetrieverRoute("metadata", ("retriever_routing_system", "metadata", "section_aware", "document_map", "self_query", "sql_database"), 0.22, 0.34, 0.12, 0.10, 0.02, 0.20, 0.06, diversify_sections=True, route_notes="Sections, revisions, document coverage, metadata navigation."),
    "reranker": RetrieverRoute("reranker", ("retriever_routing_system", "hybrid", "reranker", "late_interaction", "sparse"), 0.30, 0.30, 0.12, 0.24, 0.02, 0.02, 0.04, route_notes="Precision pass for similar-topic engineering retrieval."),
    "window": RetrieverRoute("window", ("retriever_routing_system", "hybrid", "window", "parent_child", "symbolic", "citation"), 0.24, 0.34, 0.16, 0.16, 0.02, 0.08, 0.08, 2, parent_child=True, route_notes="Nearby chunks for safety, conditional, exception, and negative wording."),
    "parent_child": RetrieverRoute("parent_child", ("retriever_routing_system", "semantic_vector", "section_aware", "parent_child", "hierarchical"), 0.32, 0.28, 0.10, 0.16, 0.02, 0.12, 0.04, 2, True, True, route_notes="Small-chunk match with larger section context."),
    "hierarchical": RetrieverRoute("hierarchical", ("retriever_routing_system", "hierarchical", "section_aware", "document_map", "metadata", "parent_child"), 0.28, 0.30, 0.10, 0.14, 0.02, 0.16, 0.04, 1, True, True, route_notes="Document > section > subsection navigation."),
    "multi_query": RetrieverRoute("multi_query", ("retriever_routing_system", "multi_query", "semantic_vector", "keyword", "query_decomposition"), 0.40, 0.28, 0.10, 0.16, 0.02, 0.04, 0.04, 1, multi_query=True, route_notes="Vague or messy query rewriting."),
    "self_query": RetrieverRoute("self_query", ("retriever_routing_system", "self_query", "metadata", "keyword", "section_aware", "sql_database"), 0.24, 0.34, 0.12, 0.12, 0.02, 0.16, 0.08, diversify_sections=True, route_notes="Metadata-filter-like behavior from query intent."),
    "contextual_compression": RetrieverRoute("contextual_compression", ("retriever_routing_system", "hybrid", "contextual_compression", "reranker", "parent_child"), 0.34, 0.30, 0.10, 0.20, 0.02, 0.04, route_notes="Large-section and long-procedure retrieval with context trimming."),
    "multimodal": RetrieverRoute("multimodal", ("retriever_routing_system", "multimodal", "layout_aware", "image_region", "semantic_vector", "document_map"), 0.24, 0.30, 0.10, 0.14, 0.02, 0.20, 0.08, 1, True, route_notes="OCR/image/diagram/layout retrieval."),
    "agentic_iterative": RetrieverRoute("agentic_iterative", ("retriever_routing_system", "agentic", "iterative", "multi_hop", "query_decomposition", "hybrid", "api"), 0.32, 0.30, 0.12, 0.18, 0.02, 0.06, 0.04, 1, True, True, True, route_notes="Bounded retry/expand strategy for complex cross-document or research-style retrieval."),
    "graph": RetrieverRoute("graph", ("retriever_routing_system", "graph", "knowledge_graph", "semantic_graph", "entity", "ontology", "symbolic"), 0.28, 0.36, 0.10, 0.16, 0.02, 0.08, 0.06, 1, True, True, graph_mode=True, route_notes="Entity and relationship style retrieval approximation."),
    "memory": RetrieverRoute("memory", ("retriever_routing_system", "memory", "cache", "hybrid", "multi_query"), 0.30, 0.30, 0.10, 0.15, 0.02, 0.13, 0.04, multi_query=True, cache_mode=True, route_notes="Conversational memory and previous interactions."),
    "time_aware": RetrieverRoute("time_aware", ("retriever_routing_system", "time_aware", "metadata", "citation", "document_map"), 0.20, 0.34, 0.16, 0.12, 0.02, 0.16, 0.08, diversify_sections=True, citation_mode=True, route_notes="Revision/date/version evidence."),
    "citation": RetrieverRoute("citation", ("retriever_routing_system", "citation", "keyword", "metadata", "document_map"), 0.20, 0.40, 0.16, 0.12, 0.02, 0.10, 0.08, citation_mode=True, route_notes="Evidence-first retrieval for audit/trust."),
    "tool_aware": RetrieverRoute("tool_aware", ("retriever_routing_system", "tool_aware", "sql_database", "api", "metadata", "hybrid", "symbolic"), 0.26, 0.34, 0.12, 0.14, 0.04, 0.10, 0.06, diversify_sections=True, route_notes="Chooses structured/meta/vector behavior. SQL/API are guarded unless connected."),
    "cache": RetrieverRoute("cache", ("retriever_routing_system", "cache", "memory", "citation", "metadata"), 0.24, 0.36, 0.12, 0.12, 0.02, 0.14, 0.04, cache_mode=True, citation_mode=True, route_notes="Previously answered/retrieved result preference."),
    "document_map": RetrieverRoute("document_map", ("retriever_routing_system", "document_map", "hierarchical", "layout_aware", "section_aware", "metadata"), 0.24, 0.34, 0.12, 0.12, 0.02, 0.16, 0.06, 1, True, True, route_notes="Document structure maps and linked sections."),
}


PROFILE_ROUTE: dict[str, str] = {
    "direct_fact": "hybrid",
    "location_section": "metadata",
    "multi_part": "agentic_iterative",
    "comparison": "reranker",
    "table_numeric": "table",
    "procedural": "contextual_compression",
    "yes_no": "hybrid",
    "safety_critical": "window",
    "explanation": "semantic_vector",
    "search_discovery": "hierarchical",
    "conditional": "window",
    "exception": "window",
    "cross_section": "hierarchical",
    "document_coverage": "metadata",
    "enumeration": "hierarchical",
    "identifier": "keyword_sparse",
    "negative": "window",
    "ambiguous": "multi_query",
    "follow_up": "memory",
    "multi_constraint": "parent_child",
    "out_of_document": "citation",
    "safety_interpretation": "window",
    "troubleshooting": "agentic_iterative",
    "meta_document": "metadata",
    "engineering_decision": "citation",
    "temporal_revision": "time_aware",
    "conflict_detection": "agentic_iterative",
    "multi_document": "agentic_iterative",
    "calculation": "tool_aware",
    "image_diagram": "multimodal",
    "regulation_compliance": "citation",
    "uncertainty": "citation",
    "workflow": "contextual_compression",
    "messy_language": "multi_query",
    "false_assumption": "citation",
    "audit_traceability": "citation",
    "partial_match": "multi_query",
    "knowledge_gap": "citation",
    "role_based": "hybrid",
    "adversarial_stress": "citation",
}


def route_for_question(profile: QuestionProfile, question: str) -> RetrieverRoute:
    normalized = question.lower()
    if re.search(r"\b(table|row|column|slope|rating|dimension|value|mm|1:\d+|matrix)\b", normalized):
        return ROUTES["table"]
    if re.search(r"\b(image|diagram|figure|drawing|p&id|layout)\b", normalized):
        return ROUTES["multimodal"]
    if re.search(r"\b(conflict|contradict|inconsistent|mismatch|difference between documents)\b", normalized):
        return ROUTES["agentic_iterative"]
    if re.search(r"\b(revision|version|latest|changed|date|rev\.)\b", normalized):
        return ROUTES["time_aware"]
    if profile.type_id == "location_section" and re.search(r"\b(where|which section|section)\b", normalized):
        return ROUTES["metadata"]
    if re.search(r"\b(tag|document id|asme|norsok|iso|p&id|section \d|line no|equipment no)\b", normalized):
        return ROUTES["keyword_sparse"]
    if re.search(r"\b(entity|relationship|dependency|connected|upstream|downstream)\b", normalized):
        return ROUTES["graph"]
    return ROUTES[PROFILE_ROUTE.get(profile.type_id, "hybrid")]


def route_payload(route: RetrieverRoute) -> dict[str, Any]:
    active_capabilities = [
        {
            "key": capability.key,
            "label": capability.label,
            "implemented_as": capability.implemented_as,
            "signal": capability.signal,
        }
        for key in route.retrievers
        if (capability := RETRIEVER_CAPABILITIES.get(key))
    ]
    return {
        "primary": route.primary,
        "retrievers": list(route.retrievers),
        "active_capabilities": active_capabilities,
        "supported_retriever_count": len(RETRIEVER_CAPABILITIES),
        "weights": {
            "semantic": route.semantic_weight,
            "keyword": route.keyword_weight,
            "phrase": route.phrase_weight,
            "rerank": route.rerank_weight,
            "table": route.table_weight,
            "metadata": route.metadata_weight,
        },
        "window_size": route.window_size,
        "diversify_sections": route.diversify_sections,
        "include_siblings": route.include_siblings,
        "multi_query": route.multi_query,
        "parent_child": route.parent_child,
        "citation_mode": route.citation_mode,
        "cache_mode": route.cache_mode,
        "graph_mode": route.graph_mode,
        "route_notes": route.route_notes,
    }


def expanded_queries(question: str, route: RetrieverRoute) -> list[str]:
    queries = [question]
    normalized = question.strip()
    if route.multi_query:
        queries.extend(
            [
                re.sub(r"\b(what|where|which|how|tell me|show me)\b", "requirements", normalized, flags=re.I),
                f"section topic requirements {normalized}",
                f"exact terms identifiers {normalized}",
            ]
        )
    if route.graph_mode:
        queries.append(f"related systems entities dependencies {normalized}")
    if route.parent_child:
        queries.append(f"parent section context surrounding requirements {normalized}")
    if "query_decomposition" in route.retrievers or "multi_hop" in route.retrievers:
        parts = re.split(r"\b(?:and|or|with|between|versus|vs\.?|compared to)\b|[;?]", normalized, flags=re.I)
        queries.extend(part.strip() for part in parts if len(part.strip()) > 8)
    if "self_query" in route.retrievers:
        queries.append(f"metadata section revision document identifier {normalized}")
    if "time_aware" in route.retrievers:
        queries.append(f"revision version date latest changed {normalized}")
    if "citation" in route.retrievers:
        queries.append(f"exact evidence page section source {normalized}")
    return list(dict.fromkeys(query for query in queries if query.strip()))[:5]


def route_capability_score(question: str, text: str, metadata: dict[str, Any], route: RetrieverRoute) -> float:
    normalized_question = question.lower()
    haystack = " ".join(
        [
            text,
            " ".join(metadata.get("section_path") or []),
            metadata.get("section_title") or "",
            metadata.get("parent_section") or "",
            metadata.get("table_title") or "",
            " ".join(metadata.get("table_columns") or []),
            " ".join(metadata.get("table_rows") or []),
            " ".join(metadata.get("keywords") or []),
            metadata.get("revision") or "",
            metadata.get("document_identifier") or "",
            metadata.get("filename") or "",
        ]
    ).lower()
    retrievers = set(route.retrievers)
    score = 0.0

    if retrievers & {"keyword", "sparse", "late_interaction"}:
        score += exact_overlap_score(normalized_question, haystack) * 0.10
    if "entity" in retrievers and re.search(r"\b[A-Z]{2,}[-A-Z0-9_/]{2,}|\b\d+(?:\.\d+){1,}\b", text):
        score += 0.06
    if "table" in retrievers and metadata.get("contains_table"):
        score += 0.12
    if "section_aware" in retrievers and (metadata.get("section_title") or metadata.get("section_path")):
        score += 0.05
    if "document_map" in retrievers and (metadata.get("page_start") or metadata.get("page_label_start")):
        score += 0.04
    if "layout_aware" in retrievers and (metadata.get("contains_table") or metadata.get("images") or metadata.get("layout_blocks")):
        score += 0.05
    if "image_region" in retrievers and re.search(r"\b(image|figure|diagram|drawing|p&id|layout)\b", haystack):
        score += 0.06
    if "time_aware" in retrievers and (metadata.get("revision") or re.search(r"\b(rev(?:ision)?|date|version|latest|changed)\b", haystack)):
        score += 0.06
    if "citation" in retrievers and (metadata.get("page_start") or metadata.get("filename")):
        score += 0.04
    if "symbolic" in retrievers and re.search(r"\b(shall|must|required|if|when|unless|except|shall not|not allowed)\b", haystack):
        score += 0.07
    if retrievers & {"graph", "knowledge_graph", "semantic_graph", "ontology"} and re.search(r"\b(system|equipment|valve|piping|drain|flare|pump|vessel|module|instrument|line)\b", haystack):
        score += 0.06
    if "metadata" in retrievers and any(metadata.get(key) for key in ("filename", "document_identifier", "revision", "section_title")):
        score += 0.05
    if "sql_database" in retrievers and metadata:
        score += 0.03
    return min(score, 0.25)


def exact_overlap_score(question: str, haystack: str) -> float:
    terms = [
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", question)
        if term not in {"what", "where", "which", "when", "does", "need", "show", "tell", "about", "explain"}
    ]
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in haystack) / len(terms)
