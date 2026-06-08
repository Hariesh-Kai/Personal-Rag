from __future__ import annotations

import hashlib
import re
from typing import Any


INGESTION_QUALITY_SCHEMA_VERSION = "engineering-ingestion-quality-v1"


def ingestion_quality_metadata(
    text: str,
    blocks: list[Any],
    doc_meta: dict[str, Any],
    contains_table: bool,
    table_meta: dict[str, Any],
    engineering_meta: dict[str, Any],
    image_meta: dict[str, Any],
    validation_meta: dict[str, Any],
    metadata_quality_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    numeric_meta: dict[str, Any],
    section_importance_meta: dict[str, Any],
    document_classification_meta: dict[str, Any],
    hierarchical_embedding_meta: dict[str, Any],
) -> dict[str, Any]:
    components = quality_components(
        text,
        blocks,
        doc_meta,
        contains_table,
        table_meta,
        engineering_meta,
        image_meta,
        validation_meta,
        metadata_quality_meta,
        semantic_meta,
        numeric_meta,
        section_importance_meta,
        document_classification_meta,
        hierarchical_embedding_meta,
    )
    overall = weighted_quality_score(components)
    warnings = quality_warnings(components, validation_meta)
    return {
        "ingestion_quality_schema_version": INGESTION_QUALITY_SCHEMA_VERSION,
        "ingestion_quality_ready": True,
        "ingestion_quality_score": overall,
        "ingestion_quality_label": quality_label(overall),
        "ingestion_quality_components": components,
        "ingestion_quality_warnings": warnings,
        "ingestion_quality_warning_count": len(warnings),
        "ingestion_quality_indexable": overall >= 0.45 and "empty_or_near_empty_text" not in warnings,
        "ingestion_quality_answerable": overall >= 0.55 and not hard_answer_warnings(warnings),
        "ingestion_quality_retrieval_boost": round(max(-0.06, min(0.06, (overall - 0.55) * 0.09)), 5),
        "ingestion_quality_reason": quality_reason(components, warnings),
        "ingestion_quality_hash": hashlib.sha1(f"{overall}|{sorted(warnings)}|{text[:300]}".encode("utf-8")).hexdigest()[:16],
    }


def ingestion_quality_candidate_score(metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not metadata.get("ingestion_quality_ready"):
        return 0.0, {"label": "", "warnings": []}
    boost = float(metadata.get("ingestion_quality_retrieval_boost") or 0.0)
    if not metadata.get("ingestion_quality_answerable") and boost > 0:
        boost *= 0.4
    return round(boost, 5), {
        "label": metadata.get("ingestion_quality_label") or "",
        "score": float(metadata.get("ingestion_quality_score") or 0),
        "warnings": metadata.get("ingestion_quality_warnings") or [],
        "answerable": bool(metadata.get("ingestion_quality_answerable")),
        "schema": INGESTION_QUALITY_SCHEMA_VERSION,
    }


def quality_components(
    text: str,
    blocks: list[Any],
    doc_meta: dict[str, Any],
    contains_table: bool,
    table_meta: dict[str, Any],
    engineering_meta: dict[str, Any],
    image_meta: dict[str, Any],
    validation_meta: dict[str, Any],
    metadata_quality_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    numeric_meta: dict[str, Any],
    section_importance_meta: dict[str, Any],
    document_classification_meta: dict[str, Any],
    hierarchical_embedding_meta: dict[str, Any],
) -> dict[str, float]:
    words = re.findall(r"\S+", text or "")
    has_text = 1.0 if len(words) >= 8 else min(1.0, len(words) / 8)
    block_score = min(1.0, len(blocks) / 4) if blocks else (0.65 if words else 0.0)
    ocr_confidence = float(doc_meta.get("ocr_confidence_avg") or 0)
    has_ocr = bool(doc_meta.get("ocr_pages") or doc_meta.get("ocr_block_count") or any(getattr(block, "metadata", {}).get("ocr") for block in blocks))
    ocr_score = ocr_confidence if has_ocr and ocr_confidence else 1.0
    layout_score = 1.0 if doc_meta.get("layout_aware") or any(getattr(block, "metadata", {}).get("layout_aware") for block in blocks) else 0.62
    table_score = float(table_meta.get("table_quality_score") or 0.0) if contains_table else 1.0
    validation_score = float(validation_meta.get("ingestion_validation_score") or 0.0)
    metadata_score = float(metadata_quality_meta.get("metadata_quality_score") or 0.0)
    semantic_score = min(1.0, 0.35 + len(semantic_meta.get("semantic_labels") or []) * 0.14)
    entity_score = min(1.0, 0.35 + int(engineering_meta.get("entity_count") or len(engineering_meta.get("engineering_entities") or [])) * 0.04)
    numeric_score = 1.0 if not engineering_meta.get("numeric_constraints") else (1.0 if numeric_meta.get("numeric_constraint_ingestion_ready") else 0.45)
    safety_score = 1.0 if not engineering_meta.get("safety_critical") else min(1.0, 0.55 + float(engineering_meta.get("safety_score") or 0) * 0.45)
    classification_score = float(document_classification_meta.get("document_class_confidence") or 0.0)
    hierarchy_score = 1.0 if hierarchical_embedding_meta.get("hierarchical_embedding_ready") else 0.35
    section_score = float(section_importance_meta.get("section_importance_score") or 0.0)
    image_score = 1.0 if not image_meta.get("has_images") else (0.75 if image_meta.get("figure_ingestion_status") else 0.45)
    return {
        "text_presence": round(has_text, 3),
        "block_extraction": round(block_score, 3),
        "ocr": round(max(0.0, min(1.0, ocr_score)), 3),
        "layout": round(layout_score, 3),
        "table": round(table_score, 3),
        "validation": round(validation_score, 3),
        "metadata": round(metadata_score, 3),
        "semantic": round(semantic_score, 3),
        "entity": round(entity_score, 3),
        "numeric": round(numeric_score, 3),
        "safety": round(safety_score, 3),
        "document_classification": round(classification_score, 3),
        "hierarchical_embedding": round(hierarchy_score, 3),
        "section_importance": round(section_score, 3),
        "image_figure": round(image_score, 3),
    }


def weighted_quality_score(components: dict[str, float]) -> float:
    weights = {
        "text_presence": 0.08,
        "block_extraction": 0.06,
        "ocr": 0.07,
        "layout": 0.05,
        "table": 0.09,
        "validation": 0.12,
        "metadata": 0.09,
        "semantic": 0.07,
        "entity": 0.06,
        "numeric": 0.07,
        "safety": 0.08,
        "document_classification": 0.05,
        "hierarchical_embedding": 0.05,
        "section_importance": 0.04,
        "image_figure": 0.02,
    }
    score = sum(float(components.get(name, 0.0)) * weight for name, weight in weights.items())
    return round(max(0.0, min(1.0, score)), 3)


def quality_warnings(components: dict[str, float], validation_meta: dict[str, Any]) -> list[str]:
    warnings = list(validation_meta.get("ingestion_validation_warnings") or [])
    thresholds = {
        "text_presence": (0.25, "empty_or_near_empty_text"),
        "ocr": (0.35, "low_quality_ocr"),
        "table": (0.55, "weak_table_quality"),
        "metadata": (0.55, "weak_metadata_quality"),
        "semantic": (0.45, "weak_semantic_labeling"),
        "numeric": (0.6, "numeric_constraints_not_normalized"),
        "document_classification": (0.45, "weak_document_classification"),
        "hierarchical_embedding": (0.6, "hierarchy_embedding_not_ready"),
    }
    for name, (threshold, warning) in thresholds.items():
        if float(components.get(name, 0.0)) < threshold:
            warnings.append(warning)
    return unique_preserve(warnings)


def hard_answer_warnings(warnings: list[str]) -> bool:
    return any(warning in set(warnings) for warning in {"empty_or_near_empty_text", "weak_table_quality", "low_quality_ocr"})


def quality_label(score: float) -> str:
    if score >= 0.82:
        return "excellent"
    if score >= 0.68:
        return "good"
    if score >= 0.52:
        return "usable-check"
    if score >= 0.38:
        return "weak"
    return "poor"


def quality_reason(components: dict[str, float], warnings: list[str]) -> str:
    weakest = sorted(components.items(), key=lambda item: item[1])[:3]
    strongest = sorted(components.items(), key=lambda item: item[1], reverse=True)[:3]
    return (
        f"strong={','.join(name for name, _ in strongest)}; "
        f"weak={','.join(name for name, _ in weakest)}; "
        f"warnings={','.join(warnings[:5]) if warnings else 'none'}"
    )


def unique_preserve(values: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value).lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
