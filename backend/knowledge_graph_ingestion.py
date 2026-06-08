from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any


KNOWLEDGE_GRAPH_INGESTION_SCHEMA_VERSION = "engineering-knowledge-graph-ingestion-v1"


def knowledge_graph_ingestion_metadata(
    text: str,
    structure_meta: dict[str, Any] | None = None,
    engineering_meta: dict[str, Any] | None = None,
    relationship_meta: dict[str, Any] | None = None,
    reference_meta: dict[str, Any] | None = None,
    table_meta: dict[str, Any] | None = None,
    schema_meta: dict[str, Any] | None = None,
    code_meta: dict[str, Any] | None = None,
    numeric_meta: dict[str, Any] | None = None,
    doc_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structure_meta = structure_meta or {}
    engineering_meta = engineering_meta or {}
    relationship_meta = relationship_meta or {}
    reference_meta = reference_meta or {}
    table_meta = table_meta or {}
    schema_meta = schema_meta or {}
    code_meta = code_meta or {}
    numeric_meta = numeric_meta or {}
    doc_meta = doc_meta or {}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    section_node = add_node(nodes, "section", structure_meta.get("current_section_id") or structure_meta.get("section_title"), structure_meta.get("current_section_title") or structure_meta.get("section_title"))
    document_node = add_node(nodes, "document", doc_meta.get("filename") or structure_meta.get("document_title"), structure_meta.get("document_title") or doc_meta.get("filename"))

    for record in engineering_meta.get("engineering_entity_records") or []:
        if not isinstance(record, dict):
            continue
        node = add_node(nodes, record.get("type") or "entity", record.get("canonical") or record.get("text"), record.get("text") or record.get("canonical"), aliases=record.get("aliases") or [])
        if section_node:
            edges.append(edge(section_node, node, "mentioned_in_section", "structure"))
        if document_node:
            edges.append(edge(document_node, node, "contains_entity", "document"))

    for entity in engineering_meta.get("engineering_entities") or []:
        node = add_node(nodes, "entity", entity, entity)
        if section_node:
            edges.append(edge(section_node, node, "mentioned_in_section", "structure"))

    for standard in engineering_meta.get("standards") or []:
        node = add_node(nodes, "standard", standard, standard)
        if document_node:
            edges.append(edge(document_node, node, "references_standard", "standard"))

    for relation in (engineering_meta.get("engineering_entity_relationships") or []) + (relationship_meta.get("relationship_records") or []):
        if not isinstance(relation, dict):
            continue
        left = add_node(nodes, "entity", relation.get("left"), relation.get("left"))
        right = add_node(nodes, "entity", relation.get("right"), relation.get("right"))
        if left and right and left != right:
            edges.append(edge(left, right, relation.get("relation") or relation.get("type") or "related_to", "explicit_relationship"))

    for ref in reference_meta.get("reference_section_ids") or []:
        ref_node = add_node(nodes, "section_reference", ref, ref)
        if section_node and ref_node:
            edges.append(edge(section_node, ref_node, "references_section", "reference"))
    for ref in reference_meta.get("reference_document_ids") or []:
        ref_node = add_node(nodes, "document_reference", ref, ref)
        if document_node and ref_node:
            edges.append(edge(document_node, ref_node, "references_document", "reference"))

    table_node = add_node(nodes, "table", table_meta.get("table_title"), table_meta.get("table_title")) if table_meta.get("contains_table") or table_meta.get("table_columns") else ""
    if table_node and section_node:
        edges.append(edge(section_node, table_node, "contains_table", "table"))
    for column in table_meta.get("table_columns") or []:
        column_node = add_node(nodes, "table_column", column, column)
        if table_node and column_node:
            edges.append(edge(table_node, column_node, "has_column", "table"))

    for schema_name in schema_meta.get("schema_names") or []:
        schema_node = add_node(nodes, "schema", schema_name, schema_name)
        if document_node and schema_node:
            edges.append(edge(document_node, schema_node, "defines_schema", "schema"))
        for field in schema_meta.get("schema_fields") or []:
            if not isinstance(field, dict):
                continue
            field_node = add_node(nodes, "schema_field", field.get("name"), field.get("name"), properties={"field_type": field.get("type"), "required": field.get("required")})
            if schema_node and field_node:
                edges.append(edge(schema_node, field_node, "has_field", "schema"))

    for function_name in code_meta.get("code_functions") or []:
        function_node = add_node(nodes, "code_function", function_name, function_name)
        if document_node and function_node:
            edges.append(edge(document_node, function_node, "defines_function", "code"))
    for class_name in code_meta.get("code_classes") or []:
        class_node = add_node(nodes, "code_class", class_name, class_name)
        if document_node and class_node:
            edges.append(edge(document_node, class_node, "defines_class", "code"))
    for endpoint in code_meta.get("code_endpoints") or []:
        endpoint_node = add_node(nodes, "api_endpoint", endpoint, endpoint)
        if document_node and endpoint_node:
            edges.append(edge(document_node, endpoint_node, "defines_endpoint", "code"))

    for item in numeric_meta.get("normalized_numeric_constraints") or engineering_meta.get("numeric_constraints") or []:
        if not isinstance(item, dict):
            continue
        label = " ".join(str(item.get(key) or "") for key in ("value", "unit", "context", "normalized_unit")).strip()
        numeric_node = add_node(nodes, "numeric_constraint", label, label, properties=item)
        if section_node and numeric_node:
            edges.append(edge(section_node, numeric_node, "has_numeric_constraint", "numeric"))

    nodes = unique_nodes(nodes)
    edges = unique_edges([item for item in edges if item.get("source") and item.get("target")])
    relation_counts = Counter(item["relation"] for item in edges)
    node_types = sorted({node["type"] for node in nodes})
    graph_text = graph_retrieval_text(nodes, edges, text)
    return {
        "knowledge_graph_ingestion_schema_version": KNOWLEDGE_GRAPH_INGESTION_SCHEMA_VERSION,
        "knowledge_graph_ingestion_ready": True,
        "knowledge_graph_ready": bool(nodes),
        "knowledge_graph_nodes": nodes[:220],
        "knowledge_graph_edges": edges[:360],
        "knowledge_graph_node_keys": [node["key"] for node in nodes[:220]],
        "knowledge_graph_edge_keys": [item["key"] for item in edges[:360]],
        "knowledge_graph_node_types": node_types,
        "knowledge_graph_relation_types": [relation for relation, _ in relation_counts.most_common(40)],
        "knowledge_graph_node_count": len(nodes),
        "knowledge_graph_edge_count": len(edges),
        "knowledge_graph_retrieval_text": graph_text,
        "knowledge_graph_hash": hashlib.sha1(graph_text.encode("utf-8")).hexdigest()[:16] if graph_text else "",
    }


def add_node(
    nodes: list[dict[str, Any]],
    node_type: str,
    key_value: Any,
    label: Any,
    aliases: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> str:
    label_text = re.sub(r"\s+", " ", str(label or key_value or "")).strip()
    key = node_key(node_type, key_value or label_text)
    if not key:
        return ""
    nodes.append(
        {
            "key": key,
            "type": normalize_token(node_type) or "entity",
            "label": label_text or key,
            "aliases": unique(aliases or []),
            "properties": properties or {},
        }
    )
    return key


def edge(source: str, target: str, relation: Any, source_type: str) -> dict[str, Any]:
    relation_name = normalize_relation(relation)
    key = f"{source}|{relation_name}|{target}"
    return {"key": key, "source": source, "target": target, "relation": relation_name, "source_type": source_type}


def node_key(node_type: str, value: Any) -> str:
    normalized = normalize_token(value)
    node_type = normalize_token(node_type)
    return f"{node_type}:{normalized}" if normalized else ""


def normalize_relation(value: Any) -> str:
    relation = normalize_token(value)
    return relation or "related_to"


def normalize_token(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", str(value or "").lower()).strip("_")
    return re.sub(r"_+", "_", text)[:120]


def unique_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for node in nodes:
        key = node.get("key")
        if not key:
            continue
        existing = merged.setdefault(key, {**node, "aliases": [], "properties": {}})
        existing["aliases"] = unique((existing.get("aliases") or []) + (node.get("aliases") or []))
        existing["properties"] = {**(existing.get("properties") or {}), **(node.get("properties") or {})}
        if len(str(node.get("label") or "")) > len(str(existing.get("label") or "")):
            existing["label"] = node.get("label")
    return list(merged.values())


def unique_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for item in edges:
        key = str(item.get("key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def graph_retrieval_text(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], text: str) -> str:
    node_text = " ".join(" ".join(str(node.get(key) or "") for key in ("key", "type", "label")) for node in nodes[:120])
    edge_text = " ".join(" ".join(str(edge_item.get(key) or "") for key in ("source", "relation", "target")) for edge_item in edges[:160])
    return "\n".join(part for part in [node_text, edge_text, text[:1800]] if part)


def unique(values: list[Any]) -> list[str]:
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
