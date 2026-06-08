from __future__ import annotations

import hashlib
import re
from typing import Any


HIERARCHICAL_EMBEDDING_SCHEMA_VERSION = "engineering-hierarchical-embedding-v1"
HIERARCHICAL_TEXT_LIMIT = 14000


def hierarchical_embedding_metadata(
    text: str,
    blocks: list[Any],
    doc_meta: dict[str, Any],
    keyword_set: Any,
    engineering_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    table_meta: dict[str, Any],
) -> dict[str, Any]:
    section_path = first_non_empty(getattr(block, "section_path", []) for block in blocks)
    structure = first_structure_metadata(blocks)
    metadata = {
        "document_title": structure.get("document_title") or doc_meta.get("title") or "",
        "document_identifier": doc_meta.get("document_identifier") or structure.get("document_identifier") or "",
        "section_path": section_path,
        "section_breadcrumb": structure.get("section_breadcrumb") or " > ".join(section_path),
        "section_ids": structure.get("section_ids") or [],
        "section_titles": structure.get("section_titles") or section_path,
        "current_section_id": structure.get("current_section_id") or "",
        "current_section_title": structure.get("current_section_title") or (section_path[-1] if section_path else ""),
        "parent_section_id": structure.get("parent_section_id") or "",
        "parent_section_title": structure.get("parent_section_title") or (section_path[-2] if len(section_path) > 1 else ""),
        "page_start": min((getattr(block, "page", 0) for block in blocks), default=None),
        "page_end": max((getattr(block, "page", 0) for block in blocks), default=None),
        "keywords": getattr(keyword_set, "keywords", []) or [],
        "keyphrases": getattr(keyword_set, "keyphrases", []) or [],
        "exact_terms": getattr(keyword_set, "exact_terms", []) or [],
        "primary_entities": engineering_meta.get("primary_entities") or [],
        "engineering_entities": engineering_meta.get("engineering_entities") or [],
        "technical_identifiers": engineering_meta.get("technical_identifiers") or [],
        "standards": engineering_meta.get("standards") or [],
        "semantic_labels": semantic_meta.get("semantic_labels") or [],
        "contains_table": bool(table_meta),
        "table_title": table_meta.get("table_title") or "",
        "table_columns": table_meta.get("table_columns") or [],
    }
    return prepare_hierarchical_embedding_metadata(text, metadata)


def prepare_hierarchical_embedding_metadata(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    layers = hierarchical_layers(metadata, text)
    surfaces = hierarchical_surfaces(text, metadata, layers)
    active_layers = [name for name, value in layers.items() if value]
    dense_text = weighted_hierarchical_text(surfaces)
    sparse_text = sparse_hierarchical_text(surfaces, metadata)
    return {
        **metadata,
        "hierarchical_embedding_schema_version": HIERARCHICAL_EMBEDDING_SCHEMA_VERSION,
        "hierarchical_embedding_ready": True,
        "hierarchical_embedding_layers": layers,
        "hierarchical_embedding_active_layers": active_layers,
        "hierarchical_embedding_layer_count": len(active_layers),
        "hierarchical_embedding_surfaces": {
            name: surface_payload(name, value)
            for name, value in surfaces.items()
        },
        "hierarchical_embedding_dense_text": limit_text(dense_text),
        "hierarchical_embedding_sparse_text": limit_text(sparse_text),
        "hierarchical_embedding_strategy": "document_section_parent_child_weighted_e5_passage_surface",
        "hierarchical_embedding_hash": hashlib.sha1(dense_text.encode("utf-8")).hexdigest()[:16] if dense_text else "",
    }


def hierarchical_embedding_text(text: str, metadata: dict[str, Any]) -> str:
    if metadata.get("hierarchical_embedding_dense_text"):
        return str(metadata.get("hierarchical_embedding_dense_text") or "")
    prepared = prepare_hierarchical_embedding_metadata(text, metadata)
    return str(prepared.get("hierarchical_embedding_dense_text") or text)


def hierarchical_embedding_candidate_score(query: str, metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    terms = query_terms(query)
    if not terms:
        return 0.0, {"matched_layers": [], "matched_terms": []}
    layers = metadata.get("hierarchical_embedding_layers") or hierarchical_layers(metadata, text)
    matched_layers = []
    matched_terms = []
    score = 0.0
    weights = {
        "document": 0.012,
        "section": 0.028,
        "parent": 0.018,
        "breadcrumb": 0.026,
        "entity": 0.02,
        "identifier": 0.024,
        "table": 0.022,
        "chunk": 0.01,
    }
    for layer_name, layer_value in layers.items():
        haystack = normalize(" ".join(layer_value) if isinstance(layer_value, list) else str(layer_value or "")).lower()
        layer_hits = [term for term in terms if term in haystack]
        if layer_hits:
            matched_layers.append(layer_name)
            matched_terms.extend(layer_hits)
            score += min(weights.get(layer_name, 0.01), len(set(layer_hits)) * weights.get(layer_name, 0.01) * 0.6)
    return round(min(0.09, score), 5), {
        "matched_layers": unique_preserve(matched_layers),
        "matched_terms": unique_preserve(matched_terms)[:20],
        "schema": HIERARCHICAL_EMBEDDING_SCHEMA_VERSION,
    }


def hierarchical_layers(metadata: dict[str, Any], text: str) -> dict[str, Any]:
    section_path = metadata.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [part.strip() for part in re.split(r">|/", section_path) if part.strip()]
    return {
        "document": compact_join([metadata.get("document_title"), metadata.get("document_identifier"), metadata.get("filename")]),
        "section": compact_join([metadata.get("current_section_id"), metadata.get("current_section_title") or metadata.get("section_title")]),
        "parent": compact_join([metadata.get("parent_section_id"), metadata.get("parent_section_title") or metadata.get("parent_section")]),
        "breadcrumb": metadata.get("section_breadcrumb") or " > ".join(section_path),
        "entity": compact_join((metadata.get("primary_entities") or []) + (metadata.get("engineering_entities") or [])),
        "identifier": compact_join((metadata.get("technical_identifiers") or []) + (metadata.get("standards") or [])),
        "table": compact_join([metadata.get("table_title"), " ".join(metadata.get("table_columns") or [])]),
        "chunk": compact_join([metadata.get("chunk_strategy"), metadata.get("chunk_boundary_reason"), text[:900]]),
    }


def hierarchical_surfaces(text: str, metadata: dict[str, Any], layers: dict[str, Any]) -> dict[str, str]:
    keywords = compact_join((metadata.get("keywords") or []) + (metadata.get("keyphrases") or []) + (metadata.get("exact_terms") or []))
    semantic = compact_join(metadata.get("semantic_labels") or [])
    return {
        "document_surface": compact_join([layers.get("document"), layers.get("breadcrumb")]),
        "section_surface": compact_join([layers.get("section"), layers.get("parent"), layers.get("breadcrumb"), keywords]),
        "entity_surface": compact_join([layers.get("entity"), layers.get("identifier"), semantic]),
        "table_surface": compact_join([layers.get("table"), " ".join(metadata.get("table_rows") or [])]),
        "chunk_surface": compact_join([layers.get("chunk"), text]),
    }


def weighted_hierarchical_text(surfaces: dict[str, str]) -> str:
    parts = [
        labeled("document", surfaces.get("document_surface", "")),
        labeled("section", surfaces.get("section_surface", "")),
        labeled("section", surfaces.get("section_surface", "")),
        labeled("entity", surfaces.get("entity_surface", "")),
        labeled("table", surfaces.get("table_surface", "")),
        labeled("chunk", surfaces.get("chunk_surface", "")),
    ]
    return "\n".join(part for part in parts if part.strip())


def sparse_hierarchical_text(surfaces: dict[str, str], metadata: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            surfaces.get("document_surface", ""),
            surfaces.get("section_surface", ""),
            surfaces.get("entity_surface", ""),
            surfaces.get("table_surface", ""),
            compact_join(metadata.get("section_ids") or []),
            compact_join(metadata.get("section_titles") or []),
            compact_join(metadata.get("reference_section_ids") or []),
        ]
        if part
    )


def surface_payload(name: str, text: str) -> dict[str, Any]:
    text = limit_text(text)
    return {
        "name": name,
        "chars": len(text),
        "enabled": bool(text),
        "hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else "",
    }


def first_structure_metadata(blocks: list[Any]) -> dict[str, Any]:
    for block in reversed(blocks):
        metadata = getattr(block, "metadata", {}) or {}
        if metadata.get("section_breadcrumb") or metadata.get("document_title"):
            return metadata
    return {}


def first_non_empty(values: Any) -> list[str]:
    for value in values:
        if value:
            return list(value)
    return []


def compact_join(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return normalize(values)
    return normalize(" ".join(str(value) for value in values if value))


def labeled(label: str, text: str) -> str:
    text = normalize(text)
    return f"{label}: {text}" if text else ""


def limit_text(text: str) -> str:
    text = normalize(text)
    if len(text) <= HIERARCHICAL_TEXT_LIMIT:
        return text
    return text[:HIERARCHICAL_TEXT_LIMIT].rsplit(" ", 1)[0] or text[:HIERARCHICAL_TEXT_LIMIT]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def query_terms(query: str) -> list[str]:
    return unique_preserve(token.lower() for token in re.findall(r"[A-Za-z0-9_.-]{2,}", query or ""))


def unique_preserve(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value).lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
