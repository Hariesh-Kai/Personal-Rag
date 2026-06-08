from __future__ import annotations

import hashlib
import re
from typing import Any


SECTION_IMPORTANCE_SCHEMA_VERSION = "engineering-section-importance-v1"

IMPORTANT_SECTION_TERMS = {
    "basis": 0.05,
    "design": 0.05,
    "requirement": 0.08,
    "requirements": 0.08,
    "safety": 0.1,
    "shutdown": 0.08,
    "isolation": 0.08,
    "valve": 0.06,
    "valves": 0.06,
    "relief": 0.08,
    "vent": 0.06,
    "drain": 0.06,
    "slope": 0.07,
    "pressure": 0.06,
    "fire": 0.08,
    "explosion": 0.08,
    "corrosion": 0.06,
    "vibration": 0.06,
    "inspection": 0.05,
    "testing": 0.05,
    "standard": 0.05,
    "code": 0.05,
}


def section_importance_metadata(
    text: str,
    blocks: list[Any],
    section_path: list[str],
    structure_meta: dict[str, Any],
    contains_table: bool,
    engineering_meta: dict[str, Any],
    table_meta: dict[str, Any],
    reference_meta: dict[str, Any],
    relationship_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    numeric_meta: dict[str, Any],
    safety_meta: dict[str, Any],
) -> dict[str, Any]:
    section_title = (
        structure_meta.get("current_section_title")
        or structure_meta.get("section_title")
        or (section_path[-1] if section_path else "")
    )
    parent_title = structure_meta.get("parent_section_title") or (section_path[-2] if len(section_path) > 1 else "")
    breadcrumb = structure_meta.get("section_breadcrumb") or " > ".join(section_path)
    section_id = structure_meta.get("current_section_id") or ""
    signals = section_importance_signals(
        text,
        blocks,
        section_title,
        parent_title,
        breadcrumb,
        contains_table,
        engineering_meta,
        table_meta,
        reference_meta,
        relationship_meta,
        semantic_meta,
        numeric_meta,
        safety_meta,
    )
    score = section_importance_score(signals)
    return {
        "section_importance_schema_version": SECTION_IMPORTANCE_SCHEMA_VERSION,
        "section_importance_ready": True,
        "section_importance_score": score,
        "section_importance_label": section_importance_label(score),
        "section_importance_signals": signals,
        "section_importance_retrieval_boost": round(min(0.1, score * 0.09), 5),
        "section_importance_reason": section_importance_reason(signals),
        "section_importance_scope": {
            "section_id": section_id,
            "section_title": section_title,
            "parent_section_title": parent_title,
            "breadcrumb": breadcrumb,
            "depth": len(section_path),
        },
        "section_importance_hash": stable_hash([section_id, section_title, parent_title, breadcrumb, str(score)]),
    }


def section_importance_candidate_score(query: str, metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    base = float(metadata.get("section_importance_retrieval_boost") or 0)
    if base <= 0:
        return 0.0, {"matched_terms": [], "label": ""}
    scope = metadata.get("section_importance_scope") or {}
    terms = query_terms(query)
    section_text = normalize(
        " ".join(
            str(item)
            for item in [
                metadata.get("current_section_id"),
                metadata.get("current_section_title"),
                metadata.get("section_title"),
                metadata.get("parent_section_title"),
                metadata.get("section_breadcrumb"),
                scope.get("section_id"),
                scope.get("section_title"),
                scope.get("parent_section_title"),
                scope.get("breadcrumb"),
                " ".join(metadata.get("section_keyword_terms") or []),
                " ".join(metadata.get("semantic_labels") or []),
                " ".join(metadata.get("primary_entities") or []),
            ]
        )
    ).lower()
    matched_terms = [term for term in terms if term in section_text]
    direct_multiplier = 1.0 + min(0.6, len(set(matched_terms)) * 0.15)
    if not matched_terms:
        direct_multiplier = 0.55
    score = min(0.12, base * direct_multiplier)
    return round(score, 5), {
        "matched_terms": sorted(set(matched_terms)),
        "label": metadata.get("section_importance_label") or "",
        "reason": metadata.get("section_importance_reason") or "",
        "schema": SECTION_IMPORTANCE_SCHEMA_VERSION,
    }


def section_importance_signals(
    text: str,
    blocks: list[Any],
    section_title: str,
    parent_title: str,
    breadcrumb: str,
    contains_table: bool,
    engineering_meta: dict[str, Any],
    table_meta: dict[str, Any],
    reference_meta: dict[str, Any],
    relationship_meta: dict[str, Any],
    semantic_meta: dict[str, Any],
    numeric_meta: dict[str, Any],
    safety_meta: dict[str, Any],
) -> dict[str, Any]:
    title_text = normalize(f"{section_title} {parent_title} {breadcrumb}").lower()
    title_hits = {
        term: weight
        for term, weight in IMPORTANT_SECTION_TERMS.items()
        if term in title_text
    }
    requirement_count = len(engineering_meta.get("requirement_modalities") or [])
    numeric_count = int(numeric_meta.get("numeric_constraint_count") or len(engineering_meta.get("numeric_constraints") or []))
    reference_count = int(reference_meta.get("reference_count") or 0)
    relationship_count = int(relationship_meta.get("relationship_count") or 0)
    entity_count = int(engineering_meta.get("entity_count") or len(engineering_meta.get("engineering_entities") or []))
    word_count = len(re.findall(r"\S+", text or ""))
    return {
        "title_hits": title_hits,
        "has_safety": bool(engineering_meta.get("safety_critical") or safety_meta.get("safety_critical")),
        "safety_score": float(safety_meta.get("safety_score") or 0),
        "has_requirement": bool(requirement_count or engineering_meta.get("has_requirement")),
        "requirement_count": requirement_count,
        "contains_table": bool(contains_table),
        "table_quality_score": float(table_meta.get("table_quality_score") or 0),
        "numeric_constraint_count": numeric_count,
        "reference_count": reference_count,
        "relationship_count": relationship_count,
        "entity_count": entity_count,
        "semantic_labels": semantic_meta.get("semantic_labels") or [],
        "section_depth": len([part for part in breadcrumb.split(">") if part.strip()]),
        "word_count": word_count,
        "heading_block_count": sum(1 for block in blocks if getattr(block, "is_heading", False)),
    }


def section_importance_score(signals: dict[str, Any]) -> float:
    score = 0.12
    score += min(0.2, sum(float(value) for value in (signals.get("title_hits") or {}).values()))
    score += 0.16 if signals.get("has_safety") else 0
    score += min(0.12, float(signals.get("safety_score") or 0) * 0.12)
    score += 0.12 if signals.get("has_requirement") else 0
    score += min(0.08, int(signals.get("requirement_count") or 0) * 0.025)
    score += 0.08 if signals.get("contains_table") else 0
    score += min(0.08, float(signals.get("table_quality_score") or 0) * 0.08)
    score += min(0.1, int(signals.get("numeric_constraint_count") or 0) * 0.025)
    score += min(0.06, int(signals.get("reference_count") or 0) * 0.015)
    score += min(0.06, int(signals.get("relationship_count") or 0) * 0.012)
    score += min(0.06, int(signals.get("entity_count") or 0) * 0.008)
    if "procedure" in signals.get("semantic_labels", []) or "requirement" in signals.get("semantic_labels", []):
        score += 0.05
    if int(signals.get("word_count") or 0) < 35:
        score -= 0.06
    return round(max(0.0, min(1.0, score)), 3)


def section_importance_label(score: float) -> str:
    if score >= 0.78:
        return "critical"
    if score >= 0.58:
        return "high"
    if score >= 0.36:
        return "medium"
    return "low"


def section_importance_reason(signals: dict[str, Any]) -> str:
    reasons = []
    if signals.get("title_hits"):
        reasons.append("important_section_title_terms")
    if signals.get("has_safety"):
        reasons.append("safety_critical")
    if signals.get("has_requirement"):
        reasons.append("requirements_present")
    if signals.get("contains_table"):
        reasons.append("table_or_matrix_content")
    if int(signals.get("numeric_constraint_count") or 0):
        reasons.append("numeric_constraints_present")
    if int(signals.get("reference_count") or 0):
        reasons.append("references_present")
    return ", ".join(reasons) or "standard_section"


def query_terms(query: str) -> list[str]:
    seen = set()
    terms = []
    for token in re.findall(r"[A-Za-z0-9_.-]{2,}", query or ""):
        term = token.lower()
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def stable_hash(values: list[str]) -> str:
    return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:16]
