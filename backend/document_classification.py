from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any


DOCUMENT_CLASSIFICATION_SCHEMA_VERSION = "engineering-document-classification-v1"

CLASS_RULES: dict[str, dict[str, Any]] = {
    "piping_design_philosophy": {
        "terms": ["piping design philosophy", "pipe routing", "piping system", "piping layout", "valves", "vents", "drains"],
        "weight": 1.0,
    },
    "engineering_specification": {
        "terms": ["specification", "shall", "requirements", "design basis", "asme", "norsok", "iso", "api"],
        "weight": 0.95,
    },
    "procedure_or_work_instruction": {
        "terms": ["procedure", "step", "workflow", "instruction", "shall be carried out", "method", "sequence"],
        "weight": 0.9,
    },
    "safety_critical_document": {
        "terms": ["safety", "shutdown", "emergency", "fire", "explosion", "hazard", "relief", "flare"],
        "weight": 1.05,
    },
    "table_numeric_reference": {
        "terms": ["table", "slope", "rating", "dimension", "minimum", "maximum", "mm", "bar", "value"],
        "weight": 0.9,
    },
    "inspection_testing_document": {
        "terms": ["inspection", "testing", "test", "acceptance", "verification", "commissioning", "check"],
        "weight": 0.85,
    },
    "drawing_or_layout_document": {
        "terms": ["drawing", "layout", "p&id", "pid", "diagram", "figure", "skid", "equipment layout"],
        "weight": 0.8,
    },
    "revision_control_document": {
        "terms": ["revision", "rev.", "issue", "status", "validity status", "document id"],
        "weight": 0.75,
    },
    "general_engineering_document": {
        "terms": ["engineering", "design", "equipment", "system", "module", "facility"],
        "weight": 0.55,
    },
}


def document_classification_metadata(
    text: str,
    blocks: list[Any],
    doc_meta: dict[str, Any],
    structure_meta: dict[str, Any],
    keyword_set: Any,
    engineering_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    table_meta: dict[str, Any],
    revision_meta: dict[str, Any],
    safety_meta: dict[str, Any],
) -> dict[str, Any]:
    haystack = classification_haystack(text, blocks, doc_meta, structure_meta, keyword_set, engineering_meta, semantic_meta, table_meta, revision_meta)
    scores = class_scores(haystack, engineering_meta, semantic_meta, table_meta, revision_meta, safety_meta)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = ranked[0] if ranked else ("general_engineering_document", 0.0)
    secondary = [{"label": label, "score": score} for label, score in ranked[1:5] if score >= 0.18]
    confidence = classification_confidence(primary_score, ranked[1][1] if len(ranked) > 1 else 0.0)
    signals = classification_signals(haystack, primary, scores, doc_meta, semantic_meta, engineering_meta)
    return {
        "document_classification_schema_version": DOCUMENT_CLASSIFICATION_SCHEMA_VERSION,
        "document_classification_ready": True,
        "document_class": primary,
        "document_class_label": readable_label(primary),
        "document_class_confidence": confidence,
        "document_class_score": round(primary_score, 4),
        "document_class_secondary": secondary,
        "document_class_scores": {label: round(score, 4) for label, score in ranked},
        "document_class_signals": signals,
        "document_class_retrieval_tags": classification_retrieval_tags(primary, secondary, semantic_meta, engineering_meta),
        "document_class_routing_hint": classification_routing_hint(primary),
        "document_class_hash": hashlib.sha1(f"{primary}|{primary_score:.4f}|{haystack[:400]}".encode("utf-8")).hexdigest()[:16],
    }


def document_classification_candidate_score(query: str, metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not metadata.get("document_classification_ready"):
        return 0.0, {"matched_terms": [], "document_class": ""}
    query_text = normalize(query).lower()
    class_text = normalize(
        " ".join(
            [
                metadata.get("document_class") or "",
                metadata.get("document_class_label") or "",
                metadata.get("document_class_routing_hint") or "",
                " ".join(metadata.get("document_class_retrieval_tags") or []),
                " ".join(metadata.get("semantic_labels") or []),
            ]
        )
    ).lower()
    matched = [term for term in query_terms(query_text) if term in class_text]
    class_score = float(metadata.get("document_class_score") or 0.0)
    confidence = float(metadata.get("document_class_confidence") or 0.0)
    base = min(0.04, class_score * confidence * 0.035)
    if matched:
        base += min(0.05, len(set(matched)) * 0.012)
    elif not any(word in query_text for word in ["document", "type", "class", "specification", "procedure", "safety", "table"]):
        base *= 0.45
    return round(min(0.09, base), 5), {
        "matched_terms": sorted(set(matched)),
        "document_class": metadata.get("document_class") or "",
        "document_class_label": metadata.get("document_class_label") or "",
        "schema": DOCUMENT_CLASSIFICATION_SCHEMA_VERSION,
    }


def classification_haystack(
    text: str,
    blocks: list[Any],
    doc_meta: dict[str, Any],
    structure_meta: dict[str, Any],
    keyword_set: Any,
    engineering_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    table_meta: dict[str, Any],
    revision_meta: dict[str, Any],
) -> str:
    block_kinds = " ".join(str(getattr(block, "kind", "")) for block in blocks)
    parts = [
        doc_meta.get("extractor", ""),
        doc_meta.get("ocr", ""),
        structure_meta.get("document_title", ""),
        structure_meta.get("section_breadcrumb", ""),
        structure_meta.get("current_section_title", ""),
        " ".join(getattr(keyword_set, "keywords", []) or []),
        " ".join(getattr(keyword_set, "keyphrases", []) or []),
        " ".join(getattr(keyword_set, "exact_terms", []) or []),
        " ".join(engineering_meta.get("primary_entities") or []),
        " ".join(engineering_meta.get("standards") or []),
        " ".join(engineering_meta.get("requirement_modalities") or []),
        " ".join(semantic_meta.get("semantic_labels") or []),
        table_meta.get("table_title", ""),
        " ".join(table_meta.get("table_columns") or []),
        revision_meta.get("revision", ""),
        revision_meta.get("document_id", ""),
        block_kinds,
        text[:6000],
    ]
    return normalize(" ".join(str(part) for part in parts if part))


def class_scores(
    haystack: str,
    engineering_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    table_meta: dict[str, Any],
    revision_meta: dict[str, Any],
    safety_meta: dict[str, Any],
) -> dict[str, float]:
    normalized = haystack.lower()
    scores: dict[str, float] = {}
    for label, rule in CLASS_RULES.items():
        hits = [term for term in rule["terms"] if term in normalized]
        score = len(hits) * 0.11 * float(rule["weight"])
        if hits:
            score += 0.12
        scores[label] = score
    if engineering_meta.get("has_mandatory_requirement") or engineering_meta.get("requirement_modalities"):
        scores["engineering_specification"] += 0.16
    if safety_meta.get("safety_critical") or engineering_meta.get("safety_critical"):
        scores["safety_critical_document"] += 0.22
    if table_meta:
        scores["table_numeric_reference"] += 0.14
    if revision_meta.get("revision") or revision_meta.get("document_id"):
        scores["revision_control_document"] += 0.12
    if "procedure" in semantic_meta.get("semantic_labels", []):
        scores["procedure_or_work_instruction"] += 0.16
    if "requirement" in semantic_meta.get("semantic_labels", []):
        scores["engineering_specification"] += 0.12
    return {label: round(min(1.0, score), 4) for label, score in scores.items()}


def classification_confidence(primary_score: float, secondary_score: float) -> float:
    margin = max(0.0, primary_score - secondary_score)
    return round(min(1.0, 0.35 + primary_score * 0.45 + margin * 0.35), 3)


def classification_signals(
    haystack: str,
    primary: str,
    scores: dict[str, float],
    doc_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    engineering_meta: dict[str, Any],
) -> dict[str, Any]:
    normalized = haystack.lower()
    terms = [term for term in CLASS_RULES.get(primary, {}).get("terms", []) if term in normalized]
    token_counts = Counter(query_terms(normalized))
    return {
        "matched_primary_terms": terms[:20],
        "top_scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]),
        "extractor": doc_meta.get("extractor", ""),
        "semantic_labels": semantic_meta.get("semantic_labels") or [],
        "requirement_modalities": engineering_meta.get("requirement_modalities") or [],
        "dominant_tokens": [token for token, _count in token_counts.most_common(12)],
    }


def classification_retrieval_tags(primary: str, secondary: list[dict[str, Any]], semantic_meta: dict[str, Any], engineering_meta: dict[str, Any]) -> list[str]:
    tags = [primary, readable_label(primary)]
    tags.extend(item["label"] for item in secondary[:3])
    tags.extend(semantic_meta.get("semantic_labels") or [])
    tags.extend(engineering_meta.get("engineering_entity_types") or [])
    return unique_preserve(tags)


def classification_routing_hint(primary: str) -> str:
    if primary == "table_numeric_reference":
        return "table_numeric_retrieval"
    if primary == "safety_critical_document":
        return "safety_window_retrieval"
    if primary == "procedure_or_work_instruction":
        return "procedural_requirement_retrieval"
    if primary == "revision_control_document":
        return "metadata_revision_retrieval"
    if primary == "drawing_or_layout_document":
        return "layout_multimodal_retrieval"
    return "hybrid_engineering_retrieval"


def readable_label(label: str) -> str:
    return label.replace("_", " ").title()


def query_terms(text: str) -> list[str]:
    return unique_preserve(re.findall(r"[a-z0-9_.&/-]{2,}", str(text or "").lower()))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def unique_preserve(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value).lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
