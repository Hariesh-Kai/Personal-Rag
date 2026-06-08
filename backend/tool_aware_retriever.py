from __future__ import annotations

import re
from typing import Any

TOOL_AWARE_SCHEMA_VERSION = "engineering-tool-aware-retriever-v1"

TOOL_RULES: dict[str, dict[str, Any]] = {
    "semantic_vector": {
        "patterns": [r"\b(explain|describe|meaning|concept|overview|why|how)\b"],
        "profiles": {"explanation", "workflow", "multi_part", "troubleshooting"},
        "query": "semantic conceptual paraphrase evidence",
    },
    "keyword_sparse": {
        "patterns": [r"\b(tag|identifier|document id|section\s+\d|asme|norsok|iso|api|iec|line no|equipment no)\b"],
        "profiles": {"identifier", "location_section"},
        "query": "exact keyword identifier section title",
    },
    "table_numeric": {
        "patterns": [r"\b(table|row|column|matrix|value|numeric|slope|dimension|rating|mm|bar|psi|1:\d+)\b"],
        "profiles": {"table_numeric", "calculation"},
        "query": "table row column numeric exact value",
    },
    "metadata": {
        "patterns": [r"\b(section|page|revision|version|document|file|where|coverage|latest)\b"],
        "profiles": {"location_section", "document_coverage", "temporal_revision", "meta_document"},
        "query": "metadata section page revision document",
    },
    "sql_database": {
        "patterns": [r"\b(database|sql|sqlite|how many|count|list uploaded|retrieval logs?|chat history)\b"],
        "profiles": {"meta_document", "document_coverage"},
        "query": "database metadata count document chunk log",
    },
    "api": {
        "patterns": [r"\b(api|live|external system|monitoring|current sensor|real[- ]?time|from system)\b"],
        "profiles": {"troubleshooting", "workflow"},
        "query": "api live external system evidence",
    },
    "knowledge_graph": {
        "patterns": [r"\b(relationship|connected|dependency|depends?|associated|upstream|downstream|between systems?)\b"],
        "profiles": {"cross_section", "conflict_detection", "multi_document"},
        "query": "relationship entity graph connected system",
    },
    "memory": {
        "patterns": [r"\b(previous|earlier|that|those|same|above|follow[- ]?up|continue)\b"],
        "profiles": {"follow_up", "messy_language"},
        "query": "conversation memory previous source section",
    },
    "iterative": {
        "patterns": [r"\b(not enough|find again|broader|missing|complete|all related)\b"],
        "profiles": {"ambiguous", "knowledge_gap", "troubleshooting", "conflict_detection"},
        "query": "retrieve analyze expand missing terms",
    },
    "multi_hop": {
        "patterns": [r"\b(related to|linked to|causes?|affects?|from .* to|find .* then)\b"],
        "profiles": {"cross_section", "multi_constraint", "multi_document"},
        "query": "multi hop linked section relationship",
    },
    "query_decomposition": {
        "patterns": [r"\b(and|also|compare|difference|each|types?|steps?|explain each)\b"],
        "profiles": {"multi_part", "comparison", "enumeration", "calculation"},
        "query": "subquestion decomposition all parts",
    },
    "multimodal_layout": {
        "patterns": [r"\b(image|figure|diagram|drawing|p&id|layout|ocr|caption)\b"],
        "profiles": {"image_diagram"},
        "query": "image figure diagram layout ocr caption",
    },
    "citation": {
        "patterns": [r"\b(cite|source|evidence|audit|trace|prove|reference|where exactly)\b"],
        "profiles": {"audit_traceability", "uncertainty", "safety_critical", "regulation_compliance", "false_assumption", "adversarial_stress"},
        "query": "citation evidence page section source",
    },
    "late_interaction": {
        "patterns": [r"\b(exact|precise|shall|must|unless|except|not allowed|prohibited)\b"],
        "profiles": {"exception", "negative", "procedural", "safety_interpretation"},
        "query": "precise token alignment shall must exception",
    },
}


def tool_aware_plan(query: str, route: Any, profile: Any, dependent_plans: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_text(query)
    route_retrievers = set(getattr(route, "retrievers", ()) or ())
    profile_type = getattr(profile, "type_id", "")
    dependent_plans = dependent_plans or {}
    selected: list[str] = []
    reasons: dict[str, list[str]] = {}

    for tool_name, rule in TOOL_RULES.items():
        hits = []
        if profile_type in set(rule.get("profiles") or []):
            hits.append(f"profile:{profile_type}")
        for pattern in rule.get("patterns") or []:
            if re.search(pattern, normalized):
                hits.append(f"pattern:{pattern}")
                break
        if route_matches_tool(tool_name, route_retrievers):
            hits.append(f"route:{getattr(route, 'primary', '')}")
        if plan_depends_on_tool(tool_name, dependent_plans):
            hits.append("dependent-plan")
        if hits:
            selected.append(tool_name)
            reasons[tool_name] = hits

    if not selected:
        selected = ["semantic_vector", "keyword_sparse"]
        reasons = {"semantic_vector": ["fallback"], "keyword_sparse": ["fallback"]}

    selected = unique_preserve(selected)[:10]
    priorities = {tool: round(tool_priority(tool, profile_type, normalized), 4) for tool in selected}
    return {
        "schema": TOOL_AWARE_SCHEMA_VERSION,
        "active": True,
        "route_primary": getattr(route, "primary", ""),
        "profile_type": profile_type,
        "selected_tools": selected,
        "tool_reasons": reasons,
        "tool_priorities": priorities,
        "query_intent": classify_tool_intent(selected),
        "strategy": "dynamic_local_tool_selection_query_expansion_and_candidate_rescoring",
    }


def tool_aware_expanded_queries(seed_queries: list[str], plan: dict[str, Any]) -> list[str]:
    queries = list(seed_queries)
    if not plan.get("active"):
        return unique_preserve(queries)[:12]
    for tool in plan.get("selected_tools") or []:
        rule = TOOL_RULES.get(tool) or {}
        prefix = rule.get("query") or tool.replace("_", " ")
        queries.append(f"{prefix} {seed_queries[0] if seed_queries else ''}".strip())
    if "citation" in plan.get("selected_tools", []):
        queries.append(f"exact evidence page section source {seed_queries[0] if seed_queries else ''}".strip())
    return unique_preserve(query for query in queries if str(query).strip())[:14]


def tool_aware_candidate_score(plan: dict[str, Any], metadata: dict[str, Any], text: str, component_scores: dict[str, float] | None = None) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched_tools": []}
    component_scores = component_scores or {}
    haystack = tool_haystack(metadata, text)
    matched = []
    score = 0.0
    for tool in plan.get("selected_tools") or []:
        tool_score = candidate_tool_score(tool, metadata, haystack, component_scores)
        if tool_score > 0:
            matched.append({"tool": tool, "score": round(tool_score, 5)})
            score += tool_score * float((plan.get("tool_priorities") or {}).get(tool, 1.0))
    score = round(min(0.2, score), 5)
    return score, {
        "schema": TOOL_AWARE_SCHEMA_VERSION,
        "active": True,
        "selected_tools": plan.get("selected_tools") or [],
        "matched_tools": matched,
        "matched_tool_count": len(matched),
        "coverage": round(len(matched) / max(1, len(plan.get("selected_tools") or [])), 4),
    }


def candidate_tool_score(tool: str, metadata: dict[str, Any], haystack: str, component_scores: dict[str, float]) -> float:
    if tool == "semantic_vector":
        return 0.012 if metadata.get("semantic_labels") or metadata.get("keywords") or component_scores.get("vector", 0) > 0.35 else 0.0
    if tool == "keyword_sparse":
        return min(0.04, component_scores.get("keyword", 0) * 0.06 + component_scores.get("phrase", 0) * 0.05)
    if tool == "table_numeric":
        value = 0.0
        if metadata.get("contains_table"):
            value += 0.04
        if metadata.get("has_numeric_constraints") or re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|bar|psi|%)?\b|1:\d+", haystack):
            value += 0.025
        return min(0.07, value + component_scores.get("table", 0) * 0.05)
    if tool == "metadata":
        return 0.035 if any(metadata.get(key) for key in ("filename", "section_title", "current_section_id", "page_start", "revision")) else 0.0
    if tool == "sql_database":
        return min(0.055, 0.025 + component_scores.get("sql_database", 0) * 0.6) if component_scores.get("sql_database", 0) or metadata else 0.0
    if tool == "api":
        return min(0.055, component_scores.get("api", 0) * 0.65 + 0.025) if component_scores.get("api", 0) or metadata.get("source_type") == "api" else 0.0
    if tool == "knowledge_graph":
        return min(0.055, component_scores.get("knowledge_graph", 0) + 0.025) if metadata.get("engineering_entity_relationships") or metadata.get("relationship_count") or re.search(r"\b(upstream|downstream|connected|associated)\b", haystack) else 0.0
    if tool == "memory":
        return min(0.045, component_scores.get("memory", 0) * 0.7 + 0.012) if component_scores.get("memory", 0) else 0.0
    if tool == "iterative":
        return min(0.045, component_scores.get("iterative", 0) * 0.7 + 0.012) if component_scores.get("iterative", 0) else 0.0
    if tool == "multi_hop":
        return min(0.05, component_scores.get("multi_hop", 0) * 0.7 + 0.018) if component_scores.get("multi_hop", 0) or metadata.get("outbound_link_chunk_ids") or metadata.get("same_section_chunk_ids") else 0.0
    if tool == "query_decomposition":
        return min(0.05, component_scores.get("query_decomposition", 0) * 0.7 + 0.014) if component_scores.get("query_decomposition", 0) else 0.0
    if tool == "multimodal_layout":
        return 0.045 if metadata.get("image_block_count") or metadata.get("figure_references") or metadata.get("modalities") or metadata.get("layout_blocks") else 0.0
    if tool == "citation":
        return 0.032 if metadata.get("filename") and (metadata.get("page_start") or metadata.get("section_title") or metadata.get("page_label_start")) else 0.0
    if tool == "late_interaction":
        return min(0.05, component_scores.get("late_interaction", 0) * 0.7 + 0.012) if component_scores.get("late_interaction", 0) or re.search(r"\b(shall|must|unless|except|not allowed|prohibited)\b", haystack) else 0.0
    return 0.0


def route_matches_tool(tool: str, retrievers: set[str]) -> bool:
    mapping = {
        "semantic_vector": {"semantic_vector", "dense_passage", "multi_vector"},
        "keyword_sparse": {"keyword", "sparse", "entity"},
        "table_numeric": {"table"},
        "metadata": {"metadata", "section_aware", "document_map"},
        "sql_database": {"sql_database"},
        "api": {"api"},
        "knowledge_graph": {"graph", "knowledge_graph", "semantic_graph"},
        "memory": {"memory", "cache"},
        "iterative": {"iterative", "agentic"},
        "multi_hop": {"multi_hop"},
        "query_decomposition": {"query_decomposition"},
        "multimodal_layout": {"multimodal", "layout_aware", "image_region"},
        "citation": {"citation"},
        "late_interaction": {"late_interaction", "symbolic"},
    }
    return bool(retrievers & mapping.get(tool, set()))


def plan_depends_on_tool(tool: str, dependent_plans: dict[str, Any]) -> bool:
    key_map = {
        "memory": "memory",
        "iterative": "iterative",
        "multi_hop": "multi_hop",
        "query_decomposition": "decomposition",
        "sql_database": "sql_database",
        "api": "api",
    }
    key = key_map.get(tool)
    if not key:
        return False
    plan = dependent_plans.get(key) or {}
    return bool(plan.get("active") or plan.get("retry_needed"))


def tool_priority(tool: str, profile_type: str, normalized_query: str) -> float:
    priority = 1.0
    if tool == "table_numeric" and profile_type in {"table_numeric", "calculation"}:
        priority += 0.35
    if tool == "citation" and profile_type in {"safety_critical", "audit_traceability", "uncertainty", "regulation_compliance"}:
        priority += 0.25
    if tool in {"keyword_sparse", "late_interaction"} and re.search(r"\b(exact|shall|must|section|tag|id)\b", normalized_query):
        priority += 0.2
    if tool in {"knowledge_graph", "multi_hop"} and re.search(r"\b(relationship|connected|upstream|downstream)\b", normalized_query):
        priority += 0.2
    if tool == "sql_database" and re.search(r"\b(database|how many|count|list)\b", normalized_query):
        priority += 0.2
    if tool == "api" and re.search(r"\b(api|live|real[- ]?time|external system|monitoring)\b", normalized_query):
        priority += 0.25
    return min(priority, 1.45)


def classify_tool_intent(selected_tools: list[str]) -> str:
    selected = set(selected_tools)
    if "table_numeric" in selected:
        return "table_or_numeric_tooling"
    if "sql_database" in selected:
        return "structured_database_tooling"
    if "api" in selected:
        return "configured_api_tooling"
    if selected & {"knowledge_graph", "multi_hop"}:
        return "relationship_tooling"
    if "multimodal_layout" in selected:
        return "layout_or_image_tooling"
    if "citation" in selected:
        return "evidence_tooling"
    return "hybrid_search_tooling"


def tool_haystack(metadata: dict[str, Any], text: str) -> str:
    values = [
        text,
        metadata.get("filename") or "",
        metadata.get("section_title") or "",
        metadata.get("current_section_id") or "",
        metadata.get("table_title") or "",
        flatten_values(metadata.get("section_path") or []),
        flatten_values(metadata.get("table_columns") or []),
        flatten_values(metadata.get("table_rows") or []),
        flatten_values(metadata.get("engineering_entities") or []),
        flatten_values(metadata.get("engineering_entity_relationships") or []),
        flatten_values(metadata.get("semantic_labels") or []),
        flatten_values(metadata.get("keywords") or []),
    ]
    return normalize_text(" ".join(str(value) for value in values if value))


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
