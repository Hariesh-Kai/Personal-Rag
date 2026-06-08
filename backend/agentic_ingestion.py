from __future__ import annotations

import os
import re
from typing import Any


AGENTIC_INGESTION_SCHEMA_VERSION = "engineering-agentic-ingestion-v1"


def agentic_ingestion_metadata(
    text: str,
    blocks: list[Any],
    doc_meta: dict[str, Any],
    contains_table: bool,
    table_meta: dict[str, Any],
    engineering_meta: dict[str, Any],
    validation_meta: dict[str, Any],
    quality_meta: dict[str, Any],
) -> dict[str, Any]:
    """Create deterministic ingestion-agent decisions for each chunk.

    This is intentionally local and bounded: it does not let an LLM invent
    metadata, but it behaves like a small ingestion review crew by inspecting
    extraction, OCR, layout, table, safety, validation, and retrieval-readiness
    signals.
    """

    agents = [
        extraction_auditor(text, blocks, doc_meta),
        structure_auditor(blocks),
        table_auditor(contains_table, table_meta),
        retrieval_preparer(text, engineering_meta, quality_meta),
        safety_guard(engineering_meta, validation_meta),
    ]
    decisions = [decision for agent in agents for decision in agent["decisions"]]
    warnings = [warning for agent in agents for warning in agent["warnings"]]
    retry_steps = unique_preserve(
        step
        for decision in decisions
        for step in decision.get("recommended_steps", [])
    )
    quality_score = agentic_quality_score(agents, validation_meta, quality_meta)
    crew_plan = optional_crewai_ingestion_plan(text, agents)

    return {
        "agentic_ingestion_schema_version": AGENTIC_INGESTION_SCHEMA_VERSION,
        "agentic_ingestion_ready": True,
        "agentic_ingestion_mode": crew_plan.get("mode", "deterministic_local_audit"),
        "agentic_ingestion_agents": agents,
        "agentic_ingestion_decisions": decisions,
        "agentic_ingestion_warnings": warnings,
        "agentic_ingestion_retry_required": bool(retry_steps),
        "agentic_ingestion_retry_steps": retry_steps,
        "agentic_ingestion_quality_score": quality_score,
        "agentic_ingestion_strategy": "extract_audit_structure_audit_table_audit_retrieval_prepare_safety_guard",
        "agentic_ingestion_crewai": crew_plan,
    }


def agentic_pipeline_metadata(stages: list[Any], doc_meta: dict[str, Any], chunk_count: int) -> dict[str, Any]:
    stage_summaries = [
        {
            "name": getattr(stage, "name", ""),
            "status": getattr(stage, "status", ""),
            "input_count": int(getattr(stage, "input_count", 0) or 0),
            "output_count": int(getattr(stage, "output_count", 0) or 0),
            "duration_ms": float(getattr(stage, "duration_ms", 0) or 0),
        }
        for stage in stages
    ]
    failed = [stage["name"] for stage in stage_summaries if stage["status"] != "complete"]
    slow = [stage["name"] for stage in stage_summaries if stage["duration_ms"] > 120000]
    zero_output = [
        stage["name"]
        for stage in stage_summaries
        if stage["status"] == "complete" and stage["output_count"] == 0 and stage["name"] != "chunk_serialization"
    ]
    actions = []
    if failed:
        actions.append({"issue": "failed_stage", "stage_names": failed, "action": "inspect_stage_error_before_indexing"})
    if slow:
        actions.append({"issue": "slow_stage", "stage_names": slow, "action": "review_ocr_or_table_extraction_runtime"})
    if zero_output:
        actions.append({"issue": "zero_output_stage", "stage_names": zero_output, "action": "rerun_with_extraction_debug_metadata"})
    if chunk_count == 0:
        actions.append({"issue": "no_chunks_created", "stage_names": [], "action": "block_indexing_until_chunks_exist"})

    return {
        "agentic_ingestion_schema_version": AGENTIC_INGESTION_SCHEMA_VERSION,
        "agentic_ingestion_pipeline_ready": not failed and chunk_count > 0,
        "agentic_ingestion_pipeline_agents": [
            "pipeline_supervisor",
            "stage_runtime_auditor",
            "indexing_readiness_guard",
        ],
        "agentic_ingestion_pipeline_decisions": actions,
        "agentic_ingestion_pipeline_retry_required": bool(actions),
        "agentic_ingestion_pipeline_stage_count": len(stage_summaries),
        "agentic_ingestion_pipeline_chunk_count": chunk_count,
        "agentic_ingestion_pipeline_source": {
            "extractor": doc_meta.get("extractor", ""),
            "ocr": doc_meta.get("ocr", ""),
            "layout_aware": bool(doc_meta.get("layout_aware")),
        },
    }


def extraction_auditor(text: str, blocks: list[Any], doc_meta: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    ocr_confidence = float(doc_meta.get("ocr_confidence_avg") or 0)
    has_ocr = bool(doc_meta.get("ocr_pages") or doc_meta.get("ocr_block_count"))
    if len(text.strip()) < 40:
        warnings.append("short_extracted_text")
        decisions.append(decision("extraction", "short_text", "review_source_extraction", ["rerun_with_ocr_or_layout_extraction"]))
    if has_ocr and ocr_confidence and ocr_confidence < 0.35:
        warnings.append("low_ocr_confidence")
        decisions.append(decision("extraction", "low_ocr_confidence", "prefer_native_text_or_higher_dpi_ocr", ["rerun_ocr_higher_resolution"]))
    if not blocks:
        warnings.append("no_content_blocks")
        decisions.append(decision("extraction", "no_blocks", "block_indexing", ["inspect_loader_support"]))
    return agent("extraction_auditor", decisions, warnings)


def structure_auditor(blocks: list[Any]) -> dict[str, Any]:
    warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    sectioned = sum(1 for block in blocks if getattr(block, "section_path", None))
    heading_count = sum(1 for block in blocks if getattr(block, "is_heading", False))
    if blocks and not sectioned and not heading_count:
        warnings.append("weak_structure_detection")
        decisions.append(decision("structure", "no_section_context", "keep_page_and_layout_metadata", ["review_heading_detection"]))
    return agent("structure_auditor", decisions, warnings)


def table_auditor(contains_table: bool, table_meta: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    integrity = table_meta.get("table_integrity")
    if contains_table and integrity not in {"complete", "usable"}:
        warnings.append("table_needs_review")
        decisions.append(decision("table", "weak_table_integrity", "preserve_raw_table_text_and_boost_table_retrieval", ["inspect_table_rows"]))
    if contains_table and not table_meta.get("table_rows"):
        warnings.append("table_rows_missing")
        decisions.append(decision("table", "missing_rows", "do_not_trust_numeric_table_answers_without_neighbor_chunks", ["rerun_table_extraction"]))
    return agent("table_auditor", decisions, warnings)


def retrieval_preparer(text: str, engineering_meta: dict[str, Any], quality_meta: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    exact_signals = len(engineering_meta.get("technical_identifiers") or []) + len(engineering_meta.get("standards") or [])
    entity_signals = len(engineering_meta.get("primary_entities") or [])
    if exact_signals:
        decisions.append(decision("retrieval", "identifier_ready", "boost_keyword_sparse_and_identifier_retrieval", []))
    if entity_signals:
        decisions.append(decision("retrieval", "entity_ready", "enable_entity_graph_and_section_retrieval", []))
    if len(re.findall(r"\S+", text)) > 240:
        decisions.append(decision("retrieval", "large_context", "prefer_contextual_compression_and_window_retrieval", []))
    if float(quality_meta.get("metadata_quality_score") or 0) < 0.55:
        warnings.append("weak_metadata_quality")
        decisions.append(decision("retrieval", "weak_metadata", "avoid_metadata_only_filtering", ["review_metadata_enrichment"]))
    return agent("retrieval_preparer", decisions, warnings)


def safety_guard(engineering_meta: dict[str, Any], validation_meta: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    if engineering_meta.get("safety_critical"):
        decisions.append(decision("safety", "safety_critical", "require_citation_window_and_no_speculation", []))
    if validation_meta.get("ingestion_validation_warnings"):
        warnings.extend(validation_meta.get("ingestion_validation_warnings") or [])
        decisions.append(decision("validation", "validation_warning", "surface_ingestion_warning_in_debug_metadata", ["inspect_validation_warnings"]))
    return agent("safety_guard", decisions, warnings)


def optional_crewai_ingestion_plan(text: str, agents: list[dict[str, Any]]) -> dict[str, Any]:
    if os.environ.get("RAG_AGENTIC_INGESTION_CREWAI", "").lower() not in {"1", "true", "yes"}:
        return {"enabled": False, "mode": "deterministic_local_audit", "reason": "crewai_disabled"}
    try:
        import crewai  # type: ignore  # noqa: F401
    except Exception as exc:
        return {"enabled": False, "mode": "deterministic_local_audit", "reason": f"crewai_unavailable:{type(exc).__name__}"}
    return {
        "enabled": True,
        "mode": "crewai_available_deterministic_guarded",
        "reason": "crewai_import_available; local deterministic guards remain authoritative",
        "agent_count": len(agents),
        "text_words": len(re.findall(r"\S+", text)),
    }


def decision(stage: str, issue: str, action: str, recommended_steps: list[str]) -> dict[str, Any]:
    return {
        "stage": stage,
        "issue": issue,
        "action": action,
        "recommended_steps": recommended_steps,
    }


def agent(name: str, decisions: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "reviewed",
        "decision_count": len(decisions),
        "warning_count": len(warnings),
        "decisions": decisions,
        "warnings": unique_preserve(warnings),
    }


def agentic_quality_score(agents: list[dict[str, Any]], validation_meta: dict[str, Any], quality_meta: dict[str, Any]) -> float:
    warning_count = sum(int(agent.get("warning_count") or 0) for agent in agents)
    base = 0.55
    base += 0.25 * float(validation_meta.get("ingestion_validation_score") or 0)
    base += 0.2 * float(quality_meta.get("metadata_quality_score") or 0)
    base -= min(0.35, warning_count * 0.06)
    return round(max(0.0, min(1.0, base)), 3)


def unique_preserve(values: list[Any] | Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value).lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
