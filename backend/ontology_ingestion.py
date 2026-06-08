from __future__ import annotations

import hashlib
import re
from typing import Any

from .ontology_retriever import ENGINEERING_ONTOLOGY


ONTOLOGY_INGESTION_SCHEMA_VERSION = "engineering-ontology-ingestion-v1"


def ontology_ingestion_metadata(
    text: str,
    engineering_meta: dict[str, Any] | None = None,
    semantic_meta: dict[str, Any] | None = None,
    safety_meta: dict[str, Any] | None = None,
    table_meta: dict[str, Any] | None = None,
    schema_meta: dict[str, Any] | None = None,
    code_meta: dict[str, Any] | None = None,
    unit_meta: dict[str, Any] | None = None,
    knowledge_graph_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engineering_meta = engineering_meta or {}
    semantic_meta = semantic_meta or {}
    safety_meta = safety_meta or {}
    table_meta = table_meta or {}
    schema_meta = schema_meta or {}
    code_meta = code_meta or {}
    unit_meta = unit_meta or {}
    knowledge_graph_meta = knowledge_graph_meta or {}
    haystack = ontology_haystack(
        text,
        engineering_meta,
        semantic_meta,
        safety_meta,
        table_meta,
        schema_meta,
        code_meta,
        unit_meta,
        knowledge_graph_meta,
    )
    matches = []
    for concept, data in ENGINEERING_ONTOLOGY.items():
        match = concept_match(concept, data, haystack)
        if match["score"] > 0:
            matches.append(match)
    matches.sort(key=lambda item: (-float(item["score"]), item["concept"]))
    primary = [item["concept"] for item in matches[:10]]
    broader = unique(term for item in matches for term in item.get("broader", []))
    related = unique(term for item in matches for term in item.get("related", []))
    synonyms = unique(term for item in matches for term in item.get("synonyms", []))
    facets = ontology_facets(primary, engineering_meta, semantic_meta, safety_meta, table_meta, schema_meta, unit_meta)
    retrieval_text = ontology_retrieval_text(primary, broader, related, synonyms, facets, haystack)
    return {
        "ontology_ingestion_schema_version": ONTOLOGY_INGESTION_SCHEMA_VERSION,
        "ontology_ingestion_ready": True,
        "ontology_concepts": primary,
        "ontology_matches": matches[:30],
        "ontology_broader_terms": broader[:60],
        "ontology_related_terms": related[:80],
        "ontology_synonyms": synonyms[:100],
        "ontology_facets": facets,
        "ontology_concept_count": len(primary),
        "ontology_retrieval_text": retrieval_text,
        "ontology_graph_node_keys": ontology_graph_node_keys(primary, knowledge_graph_meta),
        "ontology_hash": hashlib.sha1(retrieval_text.encode("utf-8")).hexdigest()[:16] if retrieval_text else "",
    }


def concept_match(concept: str, data: dict[str, Any], haystack: str) -> dict[str, Any]:
    synonyms = [str(item) for item in data.get("synonyms") or []]
    broader = [str(item) for item in data.get("broader") or []]
    related = [str(item) for item in data.get("related") or []]
    concept_hits = matching_terms([concept] + synonyms, haystack)
    broader_hits = matching_terms(broader, haystack)
    related_hits = matching_terms(related, haystack)
    score = min(1.0, len(concept_hits) * 0.45 + len(broader_hits) * 0.22 + len(related_hits) * 0.12)
    return {
        "concept": concept,
        "score": round(score, 4),
        "concept_hits": concept_hits[:12],
        "broader_hits": broader_hits[:8],
        "related_hits": related_hits[:12],
        "broader": broader,
        "related": related,
        "synonyms": synonyms,
    }


def ontology_haystack(
    text: str,
    engineering_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    safety_meta: dict[str, Any],
    table_meta: dict[str, Any],
    schema_meta: dict[str, Any],
    code_meta: dict[str, Any],
    unit_meta: dict[str, Any],
    knowledge_graph_meta: dict[str, Any],
) -> str:
    parts = [
        text,
        flatten(engineering_meta.get("domain_terms") or []),
        flatten(engineering_meta.get("engineering_entities") or []),
        flatten(engineering_meta.get("engineering_canonical_entities") or []),
        flatten(engineering_meta.get("primary_entities") or []),
        flatten(engineering_meta.get("standards") or []),
        flatten(engineering_meta.get("retrieval_tags") or []),
        flatten(semantic_meta.get("semantic_labels") or []),
        flatten(safety_meta.get("safety_flags") or []),
        flatten(table_meta.get("table_columns") or []),
        flatten(table_meta.get("table_terms") or []),
        flatten(schema_meta.get("schema_names") or []),
        flatten(schema_meta.get("schema_field_names") or []),
        flatten(code_meta.get("code_functions") or []),
        flatten(code_meta.get("code_classes") or []),
        flatten(unit_meta.get("numeric_unit_families") or []),
        flatten(unit_meta.get("numeric_units") or []),
        flatten(knowledge_graph_meta.get("knowledge_graph_node_types") or []),
        flatten(knowledge_graph_meta.get("knowledge_graph_relation_types") or []),
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def ontology_facets(
    concepts: list[str],
    engineering_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    safety_meta: dict[str, Any],
    table_meta: dict[str, Any],
    schema_meta: dict[str, Any],
    unit_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "has_safety_concept": bool({"safety", "pressure relief", "isolation"} & set(concepts)) or bool(safety_meta.get("safety_critical")),
        "has_piping_concept": bool({"piping", "valve", "drain", "vent", "slope", "flare"} & set(concepts)),
        "has_compliance_concept": "standard" in concepts or bool(engineering_meta.get("compliance_related")),
        "has_table_concept": bool(table_meta.get("contains_table") or table_meta.get("table_columns")),
        "has_schema_concept": bool(schema_meta.get("schema_detected")),
        "has_numeric_concept": bool(unit_meta.get("unit_normalization_applied") or engineering_meta.get("has_numeric_constraints")),
        "semantic_labels": semantic_meta.get("semantic_labels") or [],
        "unit_families": unit_meta.get("numeric_unit_families") or [],
    }


def ontology_graph_node_keys(concepts: list[str], knowledge_graph_meta: dict[str, Any]) -> list[str]:
    graph_keys = [str(item) for item in knowledge_graph_meta.get("knowledge_graph_node_keys") or []]
    normalized_concepts = [normalize_key(concept) for concept in concepts]
    return [
        key
        for key in graph_keys
        if any(concept and concept in normalize_key(key) for concept in normalized_concepts)
    ][:40]


def ontology_retrieval_text(concepts: list[str], broader: list[str], related: list[str], synonyms: list[str], facets: dict[str, Any], haystack: str) -> str:
    parts = [
        "ontology concepts " + " ".join(concepts) if concepts else "",
        "ontology broader " + " ".join(broader) if broader else "",
        "ontology related " + " ".join(related) if related else "",
        "ontology synonyms " + " ".join(synonyms) if synonyms else "",
        flatten(facets),
        haystack[:1800],
    ]
    return "\n".join(part for part in parts if part)


def matching_terms(terms: list[str], haystack: str) -> list[str]:
    hits = []
    for term in terms:
        normalized = normalize_text(term)
        if normalized and term_in_text(normalized, haystack):
            hits.append(normalized)
    return unique(hits)


def term_in_text(term: str, text: str) -> bool:
    if " " in term or "-" in term or "/" in term:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return bool(re.search(rf"\b{re.escape(term)}s?\b", text))


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.:/&+-]+", " ", str(value or "").lower())).strip()


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def flatten(values: Any) -> str:
    if isinstance(values, dict):
        return " ".join(f"{key} {flatten(value)}" for key, value in values.items())
    if isinstance(values, (list, tuple, set)):
        return " ".join(flatten(value) for value in values)
    return str(values or "")


def unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = re.sub(r"\s+", " ", str(value or "")).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
