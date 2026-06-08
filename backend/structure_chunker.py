from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .abbreviation_detection import ABBREVIATION_SCHEMA_VERSION, abbreviation_metadata
from .access_control import ACCESS_CONTROL_SCHEMA_VERSION, access_control_metadata as policy_access_control_metadata
from .agentic_ingestion import agentic_ingestion_metadata, agentic_pipeline_metadata
from .change_detection import CHANGE_DETECTION_SCHEMA_VERSION, change_detection_metadata
from .code_ingestion import CODE_INGESTION_SCHEMA_VERSION, code_ingestion_metadata
from .document_classification import DOCUMENT_CLASSIFICATION_SCHEMA_VERSION, document_classification_metadata
from .formula_ingestion import FORMULA_SCHEMA_VERSION, formula_ingestion_metadata
from .hierarchical_embedding import HIERARCHICAL_EMBEDDING_SCHEMA_VERSION, hierarchical_embedding_metadata
from .ingestion_quality import INGESTION_QUALITY_SCHEMA_VERSION, ingestion_quality_metadata
from .knowledge_graph_ingestion import KNOWLEDGE_GRAPH_INGESTION_SCHEMA_VERSION, knowledge_graph_ingestion_metadata
from .language_detection import LANGUAGE_DETECTION_SCHEMA_VERSION, detected_scripts as rich_detected_scripts, language_detection_metadata
from .ingestion_events import (
    EVENT_DRIVEN_INGESTION_SCHEMA_VERSION,
    EventDrivenIngestionBus,
    event_driven_ingestion_metadata,
    publish_ingestion_event,
)
from .ocr import easyocr_available, read_image_file, read_pdf_page_image
from .ontology_ingestion import ONTOLOGY_INGESTION_SCHEMA_VERSION, ontology_ingestion_metadata
from .section_importance import SECTION_IMPORTANCE_SCHEMA_VERSION, section_importance_metadata
from .schema_ingestion import SCHEMA_INGESTION_SCHEMA_VERSION, schema_ingestion_metadata
from .streaming_ingestion import STREAMING_INGESTION_SCHEMA_VERSION, stream_ingestion_events, streaming_ingestion_metadata
from .translation_ingestion import TRANSLATION_SCHEMA_VERSION, translation_ingestion_metadata as rich_translation_ingestion_metadata
from .unit_normalization import UNIT_NORMALIZATION_SCHEMA_VERSION, unit_normalization_metadata
from .document_loader import EXCEL_SUFFIXES, HTML_SUFFIXES, _read_docx, _read_excel, _read_html, _read_json, _read_text


HEADING_RE = re.compile(r"^((?:\d+\.)+\d*|[A-Z]\.|\d+|APPENDIX\s+[A-Z0-9]+)\s+[A-Z][A-Za-z0-9 /,&()'_.:-]{3,}$", re.IGNORECASE)
SECTION_NUMBER_RE = re.compile(r"^(?P<num>(?:\d+\.)*\d+)\s+(?P<title>.+)$")
APPENDIX_RE = re.compile(r"^(?P<num>appendix\s+[A-Z0-9]+)\s*[-: ]\s*(?P<title>.+)$", re.IGNORECASE)
LETTER_HEADING_RE = re.compile(r"^(?P<num>[A-Z])\.\s+(?P<title>[A-Z][A-Za-z0-9 /,&()'_.:-]{3,})$")
INLINE_SECTION_RE = re.compile(r"(?=\b(?:\d+\.\d+(?:\.\d+)*|APPENDIX\s+[A-Z0-9]+)\s+[A-Z][A-Za-z0-9 /,&()'_.:-]{3,})", re.IGNORECASE)
MAX_CHUNK_CHARS = 1700
MIN_CHUNK_CHARS = 180
TABLE_CONTEXT_CHARS = 700
OVERLAP_MAX_CHARS = 260
OVERLAP_MIN_CHARS = 45
OVERLAP_TARGET_RATIO = 0.12
OVERLAP_MAX_RATIO = 0.20
OVERLAP_SIMILARITY_SKIP = 0.82
CHUNK_SYSTEM_VERSION = "engineering-semantic-v2"
DEDUPE_SCHEMA_VERSION = "engineering-dedupe-v2"
DEDUPE_JACCARD_THRESHOLD = 0.9
DEDUPE_SHINGLE_THRESHOLD = 0.86
DEDUPE_CONTAINMENT_THRESHOLD = 0.92
DEDUPE_MIN_WORDS = 12
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TABLE_HEADER_RE = re.compile(r"^TABLE:\n(?P<header>.+?)(?:\n|$)", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^[^\n|]+(?:\|[^\n|]+)+$", re.MULTILINE)
CONTAMINATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwhat to tell management\b",
        r"\bimmediate rule for debugging\b",
        r"\bstage-level failure record\b",
        r"\brag frameworks\b",
        r"\bupload lanes to ingest\b",
        r"\bretrieval grounded\b",
        r"\bhallucination control\b",
    ]
]
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "being",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "may",
    "not",
    "shall",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "will",
}

ENGINEERING_DOMAIN_TERMS = {
    "valve",
    "isolation",
    "pressure",
    "relief",
    "drain",
    "vent",
    "slope",
    "pipe",
    "piping",
    "header",
    "flare",
    "support",
    "vibration",
    "corrosion",
    "actuator",
    "equipment",
    "module",
    "layout",
    "fire",
    "explosion",
    "safety",
    "shutdown",
    "control",
    "instrument",
}
ENTITY_TYPE_TERMS: dict[str, set[str]] = {
    "valve": {"valve", "valves", "control valve", "isolation valve", "actuated valve", "manual valve", "choke valve", "throttle valve", "anti-surge valve"},
    "pressure_relief": {"pressure relief", "relief valve", "relief valves", "rupture disc", "rupture discs", "pressure relief device", "pressure relief devices"},
    "drain_vent": {"drain", "drains", "vent", "vents", "closed drain", "open drain", "overboard drain"},
    "piping": {"pipe", "piping", "line", "lines", "header", "headers", "flare header", "flare piping"},
    "equipment": {"equipment", "pump", "pumps", "compressor", "compressors", "vessel", "vessels", "skid", "module"},
    "instrumentation": {"instrument", "instrumentation", "control", "signal", "probe", "probes"},
    "safety_system": {"safety", "shutdown", "emergency shutdown", "fire", "explosion", "flare", "hazard"},
    "corrosion_vibration": {"corrosion", "vibration", "dead leg", "fatigue"},
    "standard": {"asme", "norsok", "iso", "iec", "api", "standard", "code"},
}
ENTITY_ALIAS_MAP = {
    "esd": "emergency shutdown",
    "psv": "pressure safety valve",
    "prv": "pressure relief valve",
    "sdv": "shutdown valve",
    "bdv": "blowdown valve",
    "rov": "remotely operated valve",
    "cv": "control valve",
    "pid": "p&id",
    "pids": "p&id",
}
ENTITY_RELATION_RE = re.compile(
    r"\b(?P<left>[A-Za-z][A-Za-z0-9 /&-]{2,60}?)\s+"
    r"(?P<relation>connected to|located in|located on|located at|installed in|installed on|associated with|upstream of|downstream of|vents? to|drains? to|connected with|provided with)\s+"
    r"(?P<right>[A-Za-z0-9][A-Za-z0-9 /&().-]{2,70})",
    re.IGNORECASE,
)
ENTITY_SCHEMA_VERSION = "engineering-entities-v2"
ENTITY_CONTEXT_WINDOW = 90
ENTITY_MAX_RECORDS = 100
LANGUAGE_SCHEMA_VERSION = LANGUAGE_DETECTION_SCHEMA_VERSION
INGESTION_EVENT_SCHEMA_VERSION = "engineering-ingestion-events-v1"
EVENT_DRIVEN_INGESTION_SCHEMA_VERSION_LOCAL = EVENT_DRIVEN_INGESTION_SCHEMA_VERSION
IMAGE_FIGURE_SCHEMA_VERSION = "engineering-image-figure-v1"
REVISION_MANAGEMENT_SCHEMA_VERSION = "engineering-revision-management-v1"
INGESTION_VALIDATION_SCHEMA_VERSION = "engineering-ingestion-validation-v1"
MULTIMODAL_SCHEMA_VERSION = "engineering-multimodal-v1"
REFERENCE_SCHEMA_VERSION = "engineering-references-v1"
RELATIONSHIP_SCHEMA_VERSION = "engineering-relationships-v1"
QUERY_OPTIMIZATION_SCHEMA_VERSION = "engineering-query-optimization-v1"
SEMANTIC_LABEL_SCHEMA_VERSION = "engineering-semantic-labels-v1"
SAFETY_TAG_SCHEMA_VERSION = "engineering-safety-tags-v1"
NUMERIC_CONSTRAINT_SCHEMA_VERSION = "engineering-numeric-constraints-v1"
CONTEXT_WINDOW_SCHEMA_VERSION = "engineering-context-window-v1"
VECTOR_OPTIMIZATION_SCHEMA_VERSION = "engineering-vector-optimization-v1"
HIERARCHICAL_EMBEDDING_SCHEMA_VERSION_LOCAL = HIERARCHICAL_EMBEDDING_SCHEMA_VERSION
SECTION_IMPORTANCE_SCHEMA_VERSION_LOCAL = SECTION_IMPORTANCE_SCHEMA_VERSION
DOCUMENT_CLASSIFICATION_SCHEMA_VERSION_LOCAL = DOCUMENT_CLASSIFICATION_SCHEMA_VERSION
INGESTION_QUALITY_SCHEMA_VERSION_LOCAL = INGESTION_QUALITY_SCHEMA_VERSION
CHANGE_DETECTION_SCHEMA_VERSION_LOCAL = CHANGE_DETECTION_SCHEMA_VERSION
ABBREVIATION_SCHEMA_VERSION_LOCAL = ABBREVIATION_SCHEMA_VERSION
FORMULA_SCHEMA_VERSION_LOCAL = FORMULA_SCHEMA_VERSION
CODE_INGESTION_SCHEMA_VERSION_LOCAL = CODE_INGESTION_SCHEMA_VERSION
SCHEMA_INGESTION_SCHEMA_VERSION_LOCAL = SCHEMA_INGESTION_SCHEMA_VERSION
KNOWLEDGE_GRAPH_INGESTION_SCHEMA_VERSION_LOCAL = KNOWLEDGE_GRAPH_INGESTION_SCHEMA_VERSION
UNIT_NORMALIZATION_SCHEMA_VERSION_LOCAL = UNIT_NORMALIZATION_SCHEMA_VERSION
ONTOLOGY_INGESTION_SCHEMA_VERSION_LOCAL = ONTOLOGY_INGESTION_SCHEMA_VERSION
REQUIREMENT_TERMS = {"shall", "must", "required", "requirement", "mandatory", "ensure", "to be"}
RECOMMENDATION_TERMS = {"should", "preferably", "recommended", "consider", "may"}
PROHIBITION_TERMS = {"shall not", "must not", "not permitted", "prohibited", "avoid"}
SAFETY_TERMS = {"safety", "fire", "explosion", "emergency", "shutdown", "relief", "hazard", "critical", "flare", "overpressure"}
COMPLIANCE_TERMS = {"asme", "norsok", "iso", "iec", "api", "standard", "code", "regulation", "philosophy", "specification"}
NUMERIC_UNIT_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)?)\s*(?P<unit>mm|cm|m|bar|barg|psi|kpa|mpa|deg|degree|degrees|%|c|hz|kg|ton|inch|in|n/a)?\b",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(r"\b(?:[A-Z]{1,6}-\d{1,6}[A-Z]?|[A-Z]{2,}[A-Z0-9_-]*(?:-[A-Z0-9]+){1,}|[A-Z]{2,}\d{2,}[A-Z0-9_-]*)\b")

LAYOUT_MAX_BLOCK_BBOXES = 20
LAYOUT_COLUMN_GAP_RATIO = 0.18


@dataclass
class ContentBlock:
    text: str
    page: int
    kind: str = "text"
    bbox: tuple[float, float, float, float] | None = None
    font_size: float = 0.0
    is_heading: bool = False
    heading_level: int = 0
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RichChunk:
    text: str
    metadata: dict[str, Any]


@dataclass
class PipelineStage:
    name: str
    status: str
    started_at: float
    ended_at: float = 0.0
    duration_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    detail: str = ""


@dataclass
class IngestionResult:
    chunks: list[dict[str, Any]]
    metadata: dict[str, Any]
    stages: list[dict[str, Any]]


@dataclass
class KeywordSet:
    keywords: list[str]
    keyphrases: list[str]
    exact_terms: list[str]
    acronyms: list[str]
    identifiers: list[str]
    standards: list[str]
    table_terms: list[str]
    section_terms: list[str]
    domain_terms: list[str]
    weighted_terms: list[dict[str, Any]]


def process_document(path: Path) -> list[dict[str, Any]]:
    return process_document_detailed(path).chunks


def process_document_detailed(path: Path, progress_callback: Any | None = None, event_callback: Any | None = None) -> IngestionResult:
    pipeline_started = time.perf_counter()
    stages: list[PipelineStage] = []
    suffix = path.suffix.lower()
    doc_meta: dict[str, Any] = {}
    event_bus = EventDrivenIngestionBus()
    event_records: list[dict[str, Any]] = []
    event_bus.subscribe("pipeline_event_recorder", event_records.append)

    def emit(event_type: str, stage: str, progress: int, detail: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return publish_ingestion_event(
            event_bus,
            event_type,
            stage,
            progress,
            detail,
            payload,
            source=path.name,
            event_callback=event_callback,
        )

    def notify(progress: int, stage: str, detail: str = "") -> None:
        if progress_callback:
            progress_callback(progress, stage, detail)
        emit("progress", stage, progress, detail)

    def run_stage(name: str, func: Any, input_count: int = 0, progress: int = 0, detail: str = "") -> Any:
        emit("stage_started", name, progress, detail, {"input_count": input_count})
        notify(progress, name, detail)
        stage = PipelineStage(name=name, status="running", started_at=time.perf_counter(), input_count=input_count, detail=detail)
        try:
            result = func()
            stage.status = "complete"
            stage.output_count = result_count(result)
            emit(
                "stage_completed",
                name,
                progress,
                stage.detail or f"output={stage.output_count}",
                {"input_count": input_count, "output_count": stage.output_count, "status": stage.status},
            )
            return result
        except Exception as exc:
            stage.status = "error"
            stage.detail = str(exc)
            emit("stage_error", name, progress, str(exc), {"input_count": input_count, "status": "error"})
            raise
        finally:
            stage.ended_at = time.perf_counter()
            stage.duration_ms = round((stage.ended_at - stage.started_at) * 1000, 3)
            stages.append(stage)

    def extract() -> tuple[list[ContentBlock], dict[str, Any]]:
        if suffix == ".pdf":
            return extract_pdf_blocks(path)
        if suffix in IMAGE_SUFFIXES:
            return extract_image_blocks(path)
        return extract_generic_blocks(path)

    emit("source_received", "document_loading_extraction", 0, f"source={path.name}", {"suffix": suffix or "none"})
    blocks, doc_meta = run_stage("document_loading_extraction", extract, progress=18, detail=f"suffix={suffix or 'none'}")
    cleaned = run_stage("cleaning_normalization", lambda: remove_noise(blocks), input_count=len(blocks), progress=35)
    structured = run_stage("structure_detection", lambda: detect_structure(cleaned), input_count=len(cleaned), progress=48)
    raw_chunks = run_stage("semantic_chunking", lambda: semantic_chunk(structured, doc_meta), input_count=len(structured), progress=62)
    deduped_chunks = run_stage("deduplication", lambda: dedupe_chunks(raw_chunks), input_count=len(raw_chunks), progress=70)
    final_chunks = run_stage("chunk_finalization_metadata", lambda: finalize_chunk_system(deduped_chunks), input_count=len(deduped_chunks), progress=76)
    chunk_dicts = run_stage(
        "chunk_serialization",
        lambda: serialize_pipeline_chunks(final_chunks, path, doc_meta, stages, pipeline_started),
        input_count=len(final_chunks),
        progress=82,
    )
    emit("chunks_ready", "ingestion_pipeline_ready", 100, f"chunks={len(chunk_dicts)}", {"chunk_count": len(chunk_dicts)})
    pipeline_meta = ingestion_pipeline_metadata(path, doc_meta, stages, pipeline_started, len(chunk_dicts), event_records, event_bus.dead_letters)
    notify(86, "ingestion_pipeline_ready", f"chunks={len(chunk_dicts)}")
    return IngestionResult(chunks=chunk_dicts, metadata=pipeline_meta, stages=[stage.__dict__ for stage in stages])


def stream_process_document(path: Path) -> Any:
    yield from stream_ingestion_events(path, process_document_detailed)


def result_count(result: Any) -> int:
    if isinstance(result, tuple) and result and isinstance(result[0], list):
        return len(result[0])
    if isinstance(result, list):
        return len(result)
    return 1 if result is not None else 0


def serialize_pipeline_chunks(
    chunks: list[RichChunk],
    path: Path,
    doc_meta: dict[str, Any],
    stages: list[PipelineStage],
    pipeline_started: float,
) -> list[dict[str, Any]]:
    pipeline_meta = ingestion_pipeline_metadata(path, doc_meta, stages, pipeline_started, len(chunks))
    serialized = []
    for chunk in chunks:
        metadata = {
            **chunk.metadata,
            "ingestion_pipeline": pipeline_meta,
            "ingestion_pipeline_version": pipeline_meta["version"],
            "ingestion_stage_count": len(stages),
            "source_suffix": path.suffix.lower(),
        }
        serialized.append({"text": chunk.text, "metadata": metadata})
    return serialized


def ingestion_pipeline_metadata(
    path: Path,
    doc_meta: dict[str, Any],
    stages: list[PipelineStage],
    pipeline_started: float,
    chunk_count: int,
    event_records: list[dict[str, Any]] | None = None,
    dead_letters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_log = event_records or ingestion_event_log(path, stages, chunk_count)
    event_driven_meta = event_driven_ingestion_metadata(event_log, dead_letters)
    streaming_meta = streaming_ingestion_metadata(event_log, path.name)
    return {
        "version": "engineering-ingestion-v2",
        "source_name": path.name,
        "source_suffix": path.suffix.lower(),
        "extractor": doc_meta.get("extractor", ""),
        "pages": doc_meta.get("pages", 1),
        "ocr": doc_meta.get("ocr", ""),
        "layout_aware": bool(doc_meta.get("layout_aware")),
        "chunk_count": chunk_count,
        "duration_ms": round((time.perf_counter() - pipeline_started) * 1000, 3),
        "stages": [stage_summary(stage) for stage in stages],
        "event_schema_version": INGESTION_EVENT_SCHEMA_VERSION,
        "events": event_log,
        "event_count": len(event_log),
        "streaming_schema_version": STREAMING_INGESTION_SCHEMA_VERSION,
        "streaming_supported": True,
        "streaming_mode": "threaded_queue_event_stream",
        **streaming_meta,
        **event_driven_meta,
        **agentic_pipeline_metadata(stages, doc_meta, chunk_count),
    }


def stage_summary(stage: PipelineStage) -> dict[str, Any]:
    return {
        "name": stage.name,
        "status": stage.status,
        "duration_ms": stage.duration_ms,
        "input_count": stage.input_count,
        "output_count": stage.output_count,
        "detail": stage.detail,
    }


def ingestion_event(
    event_type: str,
    stage: str,
    progress: int,
    detail: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": INGESTION_EVENT_SCHEMA_VERSION,
        "type": event_type,
        "stage": stage,
        "progress": max(0, min(100, int(progress))),
        "detail": detail,
        "payload": payload or {},
    }


def ingestion_event_log(path: Path, stages: list[PipelineStage], chunk_count: int) -> list[dict[str, Any]]:
    events = [ingestion_event("source_received", "document_loading_extraction", 0, f"source={path.name}")]
    total = max(1, len(stages))
    for index, stage in enumerate(stages, start=1):
        progress = int((index / total) * 90)
        events.append(ingestion_event(stage.status, stage.name, progress, stage.detail or f"output={stage.output_count}"))
    events.append(ingestion_event("chunks_ready", "ingestion_pipeline_ready", 100, f"chunks={chunk_count}"))
    return events


def extract_pdf_blocks(path: Path) -> tuple[list[ContentBlock], dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Structure-aware PDF extraction needs PyMuPDF.") from exc

    blocks: list[ContentBlock] = []
    ocr_pages = 0
    ocr_attempted = False
    ocr_block_count = 0
    ocr_confidences: list[float] = []
    with fitz.open(str(path)) as document:
        font_sizes: list[float] = []
        page_labels: dict[int, str] = {}
        for page_index, page in enumerate(document, start=1):
            page_blocks: list[ContentBlock] = []
            page_rect = page.rect
            table_blocks = extract_pdf_tables(page, page_index)
            table_bboxes = [block.bbox for block in table_blocks if block.bbox]
            page_text = page.get_text("dict")
            for raw_block in page_text.get("blocks", []):
                raw_bbox = tuple(raw_block.get("bbox", (0, 0, 0, 0)))
                if raw_block.get("type") == 1:
                    page_blocks.append(
                        ContentBlock(
                            text="[IMAGE]",
                            page=page_index,
                            kind="image",
                            bbox=raw_bbox,
                            metadata={"image": True, "page_width": page_rect.width, "page_height": page_rect.height},
                        )
                    )
                    continue
                if any(overlap_ratio(raw_bbox, table_bbox) > 0.45 for table_bbox in table_bboxes):
                    continue
                lines = []
                sizes = []
                for line in raw_block.get("lines", []):
                    spans = line.get("spans", [])
                    text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                    if text:
                        lines.append(text)
                        sizes.extend(float(span.get("size", 0)) for span in spans)
                text = normalize_line(" ".join(lines))
                if text:
                    size = max(sizes) if sizes else 0.0
                    font_sizes.append(size)
                    page_blocks.append(
                        ContentBlock(
                            text=text,
                            page=page_index,
                            kind="text",
                            bbox=raw_bbox,
                            font_size=size,
                            metadata={"page_width": page_rect.width, "page_height": page_rect.height},
                        )
                    )
            page_blocks.extend(table_blocks)
            page_blocks = assign_layout_metadata(page_blocks, page_rect.width, page_rect.height)
            label = detect_printed_page_label(page_blocks)
            if label:
                page_labels[page_index] = label
            if should_ocr_page(page_blocks):
                ocr_attempted = True
                ocr_blocks = extract_ocr_blocks(page, page_index)
                if ocr_blocks:
                    page_blocks.extend(ocr_blocks)
                    page_blocks = assign_layout_metadata(page_blocks, page_rect.width, page_rect.height)
                    ocr_pages += 1
                    ocr_block_count += len(ocr_blocks)
                    ocr_confidences.extend(
                        float(block.metadata.get("ocr_confidence") or 0)
                        for block in ocr_blocks
                        if block.metadata.get("ocr_confidence") is not None
                    )
            blocks.extend(page_blocks)

        doc_meta = {
            "extractor": "PyMuPDF",
            "pages": len(document),
            "page_labels": page_labels,
            "median_font_size": median(font_sizes) if font_sizes else 0,
            "ocr": f"easyocr_pages:{ocr_pages}" if ocr_pages else ("easyocr_unavailable" if ocr_attempted else "not_needed"),
            "ocr_pages": ocr_pages,
            "ocr_block_count": ocr_block_count,
            "ocr_confidence_avg": round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else 0.0,
            "layout_aware": True,
            "note": "PyMuPDF extracts native PDF text, page geometry, images, and detected tables. EasyOCR is used for scanned or image-heavy pages.",
        }
    return blocks, doc_meta


def should_ocr_page(page_blocks: list[ContentBlock]) -> bool:
    if not easyocr_available():
        return False
    text_chars = sum(len(block.text.strip()) for block in page_blocks if block.kind in {"text", "table"})
    image_blocks = sum(1 for block in page_blocks if block.kind == "image")
    if image_blocks <= 0:
        return text_chars < 40
    image_area = 0.0
    page_area = 0.0
    for block in page_blocks:
        page_width = float(block.metadata.get("page_width", 0) or 0)
        page_height = float(block.metadata.get("page_height", 0) or 0)
        if page_width and page_height:
            page_area = max(page_area, page_width * page_height)
        if block.kind == "image" and block.bbox:
            x0, y0, x1, y1 = block.bbox
            image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    image_coverage = image_area / page_area if page_area else 0.0
    return text_chars < 80 or (text_chars < 600 and image_coverage > 0.25)


def extract_ocr_blocks(page: Any, page_number: int) -> list[ContentBlock]:
    try:
        import numpy as np
    except ImportError:
        return []
    try:
        pixmap = page.get_pixmap(matrix=page.parent.Matrix(2, 2), alpha=False)
    except Exception:
        try:
            import fitz

            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        except Exception:
            return []
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    scale_x = page.rect.width / max(1, pixmap.width)
    scale_y = page.rect.height / max(1, pixmap.height)
    blocks = []
    for item in read_pdf_page_image(image):
        x0, y0, x1, y1 = item["bbox"]
        blocks.append(
            ContentBlock(
                text=item["text"],
                page=page_number,
                kind="text",
                bbox=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                metadata={"ocr": "easyocr", "ocr_confidence": item["confidence"], "page_width": page.rect.width, "page_height": page.rect.height},
            )
        )
    return blocks


def extract_image_blocks(path: Path) -> tuple[list[ContentBlock], dict[str, Any]]:
    ocr_blocks, image_meta = read_image_file(path)
    blocks = [
        ContentBlock(
            text=item["text"],
            page=1,
            kind="text",
            bbox=tuple(item["bbox"]),
            metadata={
                "ocr": "easyocr",
                "ocr_confidence": item["confidence"],
                "source_format": "image",
                "page_width": image_meta.get("image_width", 0),
                "page_height": image_meta.get("image_height", 0),
            },
        )
        for item in ocr_blocks
    ]
    doc_meta = {
        "extractor": "EasyOCR image",
        "pages": 1,
        "ocr": image_meta.get("ocr", "easyocr"),
        "ocr_pages": 1 if blocks else 0,
        "ocr_block_count": len(blocks),
        "ocr_confidence_avg": image_meta.get("ocr_confidence_avg", 0.0),
        "image_width": image_meta.get("image_width", 0),
        "image_height": image_meta.get("image_height", 0),
    }
    return blocks, doc_meta


def assign_layout_metadata(blocks: list[ContentBlock], page_width: float, page_height: float) -> list[ContentBlock]:
    if not blocks:
        return blocks
    column_count = detect_layout_column_count(blocks, page_width)
    ordered = sorted(blocks, key=lambda block: layout_sort_key(block, page_width, column_count))
    for order, block in enumerate(ordered, start=1):
        bbox = block.bbox
        column_index = layout_column_index(bbox, page_width, column_count)
        page_region, horizontal_region = layout_regions(bbox, page_width, page_height)
        block.metadata.update(
            {
                "layout_aware": True,
                "page_width": page_width,
                "page_height": page_height,
                "bbox": bbox_to_dict(bbox),
                "column_index": column_index,
                "column_count": column_count,
                "reading_order": order,
                "page_region": page_region,
                "horizontal_region": horizontal_region,
                "layout_region": f"{page_region}:{horizontal_region}",
            }
        )
    return ordered


def detect_layout_column_count(blocks: list[ContentBlock], page_width: float) -> int:
    if page_width <= 0:
        return 1
    centers = sorted(
        bbox_center(block.bbox)[0]
        for block in blocks
        if block.bbox and block.kind in {"text", "table"} and bbox_area(block.bbox) > 0
    )
    if len(centers) < 6:
        return 1
    gaps = [(centers[index + 1] - centers[index], index) for index in range(len(centers) - 1)]
    largest_gap, gap_index = max(gaps, key=lambda item: item[0])
    left_count = gap_index + 1
    right_count = len(centers) - left_count
    if largest_gap >= page_width * LAYOUT_COLUMN_GAP_RATIO and left_count >= 3 and right_count >= 3:
        return 2
    return 1


def layout_sort_key(block: ContentBlock, page_width: float, column_count: int) -> tuple[int, float, float, float]:
    bbox = block.bbox or (0.0, 0.0, 0.0, 0.0)
    x0, y0, _, _ = bbox
    column_index = layout_column_index(block.bbox, page_width, column_count)
    return (column_index, float(y0), float(x0), 0.0 if block.kind == "table" else 1.0)


def layout_column_index(bbox: tuple[float, float, float, float] | None, page_width: float, column_count: int) -> int:
    if not bbox or page_width <= 0 or column_count <= 1:
        return 1
    center_x, _ = bbox_center(bbox)
    column_width = page_width / column_count
    return max(1, min(column_count, int(center_x // max(column_width, 1.0)) + 1))


def layout_regions(
    bbox: tuple[float, float, float, float] | None,
    page_width: float,
    page_height: float,
) -> tuple[str, str]:
    if not bbox or page_width <= 0 or page_height <= 0:
        return "body", "center"
    center_x, center_y = bbox_center(bbox)
    if center_y < page_height * 0.12:
        page_region = "header"
    elif center_y > page_height * 0.88:
        page_region = "footer"
    else:
        page_region = "body"
    if center_x < page_width * 0.34:
        horizontal_region = "left"
    elif center_x > page_width * 0.66:
        horizontal_region = "right"
    else:
        horizontal_region = "center"
    return page_region, horizontal_region


def bbox_to_dict(bbox: tuple[float, float, float, float] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    return {"x0": round(float(x0), 2), "y0": round(float(y0), 2), "x1": round(float(x1), 2), "y1": round(float(y1), 2)}


def bbox_tuple_from_dict(bbox: dict[str, float] | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    return (float(bbox["x0"]), float(bbox["y0"]), float(bbox["x1"]), float(bbox["y1"]))


def bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if not bbox:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)


def union_bbox(blocks: list[ContentBlock]) -> dict[str, float] | None:
    bboxes = [block.bbox for block in blocks if block.bbox]
    if not bboxes:
        return None
    x0 = min(float(bbox[0]) for bbox in bboxes)
    y0 = min(float(bbox[1]) for bbox in bboxes)
    x1 = max(float(bbox[2]) for bbox in bboxes)
    y1 = max(float(bbox[3]) for bbox in bboxes)
    return bbox_to_dict((x0, y0, x1, y1))


def layout_metadata(blocks: list[ContentBlock]) -> dict[str, Any]:
    layout_blocks = [block for block in blocks if block.metadata.get("layout_aware")]
    if not layout_blocks:
        return {
            "layout_aware": False,
            "layout_regions": [],
            "page_regions": [],
            "horizontal_regions": [],
            "columns": [],
            "column_count": 1,
            "bbox": None,
            "block_bboxes": [],
            "reading_order_start": None,
            "reading_order_end": None,
            "layout_block_count": 0,
        }
    reading_orders = [
        int(block.metadata.get("reading_order"))
        for block in layout_blocks
        if block.metadata.get("reading_order") is not None
    ]
    return {
        "layout_aware": True,
        "layout_regions": sorted({str(block.metadata.get("layout_region")) for block in layout_blocks if block.metadata.get("layout_region")}),
        "page_regions": sorted({str(block.metadata.get("page_region")) for block in layout_blocks if block.metadata.get("page_region")}),
        "horizontal_regions": sorted({str(block.metadata.get("horizontal_region")) for block in layout_blocks if block.metadata.get("horizontal_region")}),
        "columns": sorted({int(block.metadata.get("column_index")) for block in layout_blocks if block.metadata.get("column_index") is not None}),
        "column_count": max((int(block.metadata.get("column_count") or 1) for block in layout_blocks), default=1),
        "bbox": union_bbox(layout_blocks),
        "block_bboxes": block_bbox_payload(layout_blocks),
        "reading_order_start": min(reading_orders) if reading_orders else None,
        "reading_order_end": max(reading_orders) if reading_orders else None,
        "layout_block_count": len(layout_blocks),
    }


def block_bbox_payload(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    payload = []
    for block in blocks[:LAYOUT_MAX_BLOCK_BBOXES]:
        payload.append(
            {
                "page": block.page,
                "kind": block.kind,
                "bbox": bbox_to_dict(block.bbox),
                "column_index": block.metadata.get("column_index"),
                "column_count": block.metadata.get("column_count"),
                "reading_order": block.metadata.get("reading_order"),
                "layout_region": block.metadata.get("layout_region"),
            }
        )
    return payload


def extract_pdf_tables(page: Any, page_number: int) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return blocks
    page_rect = getattr(page, "rect", None)
    page_width = float(getattr(page_rect, "width", 0) or 0)
    page_height = float(getattr(page_rect, "height", 0) or 0)
    try:
        tables = finder()
    except Exception:
        return blocks
    for table_index, table in enumerate(getattr(tables, "tables", []), start=1):
        try:
            rows = table.extract()
        except Exception:
            continue
        rendered = render_table(rows, carry_forward=True)
        if rendered and not is_page_furniture_table(rendered):
            row_count = max(0, len(normalize_table_rows(rows)) - 1)
            blocks.append(
                ContentBlock(
                    text=rendered,
                    page=page_number,
                    kind="table",
                    bbox=tuple(getattr(table, "bbox", (0, 0, 0, 0))),
                    metadata={
                        "table_index": table_index,
                        "preserve_together": True,
                        "source_format": "pdf_table",
                        "raw_row_count": row_count,
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                )
            )
    return blocks


def extract_generic_blocks(path: Path) -> tuple[list[ContentBlock], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = _read_docx(path)
        extractor = "python-docx"
    elif suffix == ".json":
        text = _read_json(path)
        extractor = "json"
    elif suffix == ".csv":
        text = render_table(list(csv.reader(io.StringIO(_read_text(path)))))
        extractor = "csv"
        blocks = [ContentBlock(text=text, page=1, kind="table", metadata={"preserve_together": True, "source_format": "csv"})] if text else []
        return blocks, {"extractor": extractor, "pages": 1, "ocr": "not_applicable"}
    elif suffix in EXCEL_SUFFIXES:
        text = excel_text_to_table_chunks(_read_excel(path))
        extractor = "excel"
        blocks = [
            ContentBlock(text=sheet_text, page=index, kind="table", metadata={"preserve_together": True, "source_format": "excel"})
            for index, sheet_text in enumerate(split_excel_sheets(text), start=1)
            if sheet_text.strip()
        ]
        return blocks, {"extractor": extractor, "pages": max(1, len(blocks)), "ocr": "not_applicable"}
    elif suffix in HTML_SUFFIXES:
        text = _read_html(path)
        extractor = "html"
    else:
        text = _read_text(path)
        extractor = "plain-text"
    blocks = [ContentBlock(text=normalize_line(line), page=1) for line in text.splitlines() if normalize_line(line)]
    return blocks, {"extractor": extractor, "pages": 1, "ocr": "not_applicable"}


def remove_noise(blocks: list[ContentBlock]) -> list[ContentBlock]:
    for block in blocks:
        block.text = normalize_table_text(block.text) if block.kind == "table" else normalize_line(block.text)

    line_counts = Counter(line_fingerprint(block.text) for block in blocks if block.kind == "text")
    pages_by_line: dict[str, set[int]] = defaultdict(set)
    page_count = max((block.page for block in blocks), default=1)
    for block in blocks:
        fingerprint = line_fingerprint(block.text)
        if fingerprint:
            pages_by_line[fingerprint].add(block.page)

    cleaned = []
    seen_on_page: set[tuple[int, str]] = set()
    for block in blocks:
        text = normalize_table_text(block.text) if block.kind == "table" else normalize_line(block.text)
        if not text:
            continue
        if is_generated_or_debug_text(text):
            continue
        if block.kind == "image":
            cleaned.append(block)
            continue
        if block.kind == "table" and is_page_furniture_table(text):
            continue
        fingerprint = line_fingerprint(text)
        page_key = (block.page, fingerprint)
        if fingerprint and page_key in seen_on_page and len(text) < 220:
            continue
        seen_on_page.add(page_key)
        repeated_pages = len(pages_by_line[fingerprint]) if fingerprint else 0
        y0 = block.bbox[1] if block.bbox else 0
        page_height = float(block.metadata.get("page_height", 0) or 0)
        near_edge = page_height and (y0 < page_height * 0.09 or y0 > page_height * 0.90)
        if repeated_pages >= max(2, int(page_count * 0.35)) and (near_edge or len(text) < 160):
            continue
        if is_page_number(text) or is_watermark(text) or is_page_furniture(text) or is_boilerplate(text):
            continue
        if line_counts[fingerprint] > max(4, page_count) and len(text) < 220:
            continue
        block.text = text
        cleaned.append(block)
    cleaned = promote_text_tables(cleaned)
    return remove_empty_repetition(remove_front_matter(split_inline_sections(merge_wrapped_lines(cleaned))))


def detect_structure(blocks: list[ContentBlock]) -> list[ContentBlock]:
    body_size = median([block.font_size for block in blocks if block.font_size > 0] or [0])
    document_title = detect_document_title(blocks, body_size)
    stack: list[dict[str, Any]] = []
    for block in blocks:
        if block.kind == "table":
            apply_structure_metadata(block, stack, document_title)
            continue
        block.is_heading = looks_like_heading(block.text, block.font_size, body_size)
        if document_title and normalize_line(block.text).lower() == normalize_line(document_title).lower() and not (SECTION_NUMBER_RE.match(block.text) or APPENDIX_RE.match(block.text)):
            block.is_heading = False
        if block.is_heading:
            level = heading_level(block.text)
            section_id = section_identifier(block.text) or f"heading-{len(stack) + 1}"
            title = clean_heading_title(block.text)
            while stack and int(stack[-1]["level"]) >= level:
                stack.pop()
            node = {
                "level": level,
                "id": section_id,
                "title": title,
                "heading": block.text,
                "page": block.page,
            }
            stack.append(node)
            block.heading_level = level
        apply_structure_metadata(block, stack, document_title)
    return blocks


def semantic_chunk(blocks: list[ContentBlock], doc_meta: dict[str, Any]) -> list[RichChunk]:
    chunks: list[RichChunk] = []
    current: list[ContentBlock] = []
    current_topic = ""
    previous_overlap: dict[str, Any] | None = None

    def flush(boundary_reason: str = "semantic_boundary") -> None:
        nonlocal current, current_topic, previous_overlap
        if not current:
            return
        text = "\n".join(block.text for block in current if block.text != "[IMAGE]").strip()
        if not text:
            current = []
            return
        meta = build_metadata(current, doc_meta)
        meta.update(chunk_strategy_metadata(current, text, "semantic_section", boundary_reason))
        apply_overlap_metadata(meta, previous_overlap, text, "semantic_section", boundary_reason)
        chunks.append(RichChunk(text=text, metadata=meta))
        previous_overlap = build_overlap_context(text, meta)
        current = []
        current_topic = ""

    consumed_table_indexes: set[int] = set()
    for index, block in enumerate(blocks):
        if block.kind == "image":
            continue
        if index in consumed_table_indexes:
            continue
        if block.kind == "table":
            flush("table_boundary_before")
            table_indexes = related_table_indexes(blocks, index)
            consumed_table_indexes.update(table_indexes)
            table_blocks = table_context_blocks(blocks, table_indexes)
            table_text = table_chunk_text(table_blocks)
            table_meta = build_metadata(table_blocks, doc_meta)
            table_meta.update(chunk_strategy_metadata(table_blocks, table_text, "table_preserve_together", "table_preserved"))
            apply_overlap_metadata(table_meta, previous_overlap, table_text, "table_preserve_together", "table_preserved")
            chunks.append(RichChunk(text=table_text, metadata=table_meta))
            previous_overlap = build_overlap_context(table_text, table_meta)
            continue

        topic = " > ".join(block.section_path)
        new_topic = topic and topic != current_topic
        hard_subsection = block.is_heading and bool(SECTION_NUMBER_RE.match(block.text))
        too_large = current_char_count(current) + len(block.text) > MAX_CHUNK_CHARS
        layout_break = current and layout_boundary_changed(current[-1], block)
        concept_break = current and (block.is_heading or new_topic or too_large or layout_break)
        boundary_reason = chunk_boundary_reason(block, new_topic, hard_subsection, too_large, layout_break)
        if current and hard_subsection:
            concept_break = True
        if concept_break:
            flush(boundary_reason)
        if len(block.text) > MAX_CHUNK_CHARS and not block.is_heading:
            flush("oversized_block_before")
            for split_block in split_long_text_block(block):
                meta = build_metadata([split_block], doc_meta)
                meta.update(chunk_strategy_metadata([split_block], split_block.text, "paragraph_sentence_split", "oversized_block_split"))
                apply_overlap_metadata(meta, previous_overlap, split_block.text, "paragraph_sentence_split", "oversized_block_split")
                chunks.append(RichChunk(text=split_block.text, metadata=meta))
                previous_overlap = build_overlap_context(split_block.text, meta)
            continue
        current.append(block)
        current_topic = topic
    flush("document_end")
    return split_oversized_chunks([chunk for chunk in chunks if keep_chunk(chunk)])


def keep_chunk(chunk: RichChunk) -> bool:
    text = chunk.text.strip()
    if len(text) >= 80:
        return True
    metadata = chunk.metadata or {}
    return bool(len(text) >= 4 and (metadata.get("contains_table") or metadata.get("section_title") or metadata.get("content_types")))


def chunk_strategy_metadata(blocks: list[ContentBlock], text: str, strategy: str, boundary_reason: str) -> dict[str, Any]:
    first = blocks[0] if blocks else None
    last = blocks[-1] if blocks else None
    section_ids = first.metadata.get("section_ids", []) if first else []
    current_section_id = first.metadata.get("current_section_id", "") if first else ""
    parent_section_id = first.metadata.get("parent_section_id", "") if first else ""
    tokens_estimate = max(1, len(re.findall(r"\S+", text)))
    diagnostics = chunk_diagnostics(blocks, text, strategy)
    return {
        "chunk_system": CHUNK_SYSTEM_VERSION,
        "chunk_strategy": strategy,
        "chunk_boundary_reason": boundary_reason,
        "chunk_char_count": len(text),
        "chunk_word_count": tokens_estimate,
        "chunk_size_class": chunk_size_class(len(text)),
        "chunk_has_heading_start": bool(first and first.is_heading),
        "chunk_starts_at_section": bool(first and first.is_heading and first.metadata.get("current_section_id")),
        "chunk_ends_at_section": bool(last and last.is_heading),
        "parent_chunk_key": parent_chunk_key(blocks),
        "parent_section_id": parent_section_id,
        "current_section_id": current_section_id,
        "ancestor_section_ids": section_ids,
        "overlap_policy": {
            "enabled": True,
            "target_ratio": OVERLAP_TARGET_RATIO,
            "max_chars": OVERLAP_MAX_CHARS,
            "max_ratio": OVERLAP_MAX_RATIO,
            "boundary": "sentence_or_bullet",
        },
        "overlap_applied": False,
        "overlap_skip_reason": "",
        "preserve_together": any(block.metadata.get("preserve_together") for block in blocks),
        **diagnostics,
    }


def build_overlap_context(text: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    overlap = clean_overlap_text(tail_overlap(text))
    if not overlap:
        return None
    if len(overlap) < OVERLAP_MIN_CHARS:
        return None
    char_ratio = len(overlap) / max(1, len(text))
    if char_ratio > OVERLAP_MAX_RATIO and len(text) > MIN_CHUNK_CHARS:
        overlap = trim_overlap_to_ratio(overlap, text)
    overlap = clean_overlap_text(overlap)
    if not overlap or len(overlap) < OVERLAP_MIN_CHARS:
        return None
    return {
        "text": overlap,
        "source_chunk_id": metadata.get("chunk_id") or "",
        "source_section_id": metadata.get("current_section_id") or "",
        "source_section_title": metadata.get("current_section_title") or metadata.get("section_title") or "",
        "source_strategy": metadata.get("chunk_strategy") or "",
        "source_boundary_reason": metadata.get("chunk_boundary_reason") or "",
        "char_count": len(overlap),
        "word_count": len(overlap.split()),
        "boundary_type": overlap_boundary_type(overlap),
        "quality_score": overlap_quality_score(overlap),
    }


def apply_overlap_metadata(
    metadata: dict[str, Any],
    previous_overlap: dict[str, Any] | None,
    current_text: str,
    strategy: str,
    boundary_reason: str,
) -> None:
    if not previous_overlap:
        metadata.update(overlap_skip_payload("no_previous_overlap"))
        return
    allowed, reason = overlap_allowed(metadata, previous_overlap, current_text, strategy, boundary_reason)
    if not allowed:
        metadata.update(overlap_skip_payload(reason))
        return
    overlap_text = str(previous_overlap.get("text") or "")
    metadata.update(
        {
            "overlap_context": overlap_text,
            "overlap_applied": True,
            "overlap_skip_reason": "",
            "overlap_char_count": len(overlap_text),
            "overlap_word_count": len(overlap_text.split()),
            "overlap_ratio": round(len(overlap_text) / max(1, len(current_text)), 4),
            "overlap_boundary_type": previous_overlap.get("boundary_type") or "unknown",
            "overlap_quality_score": previous_overlap.get("quality_score") or 0.0,
            "overlap_source_section_id": previous_overlap.get("source_section_id") or "",
            "overlap_source_section_title": previous_overlap.get("source_section_title") or "",
            "overlap_source_strategy": previous_overlap.get("source_strategy") or "",
        }
    )


def overlap_allowed(
    metadata: dict[str, Any],
    previous_overlap: dict[str, Any],
    current_text: str,
    strategy: str,
    boundary_reason: str,
) -> tuple[bool, str]:
    overlap_text = str(previous_overlap.get("text") or "")
    if not overlap_text:
        return False, "empty_overlap"
    if boundary_reason in {"hard_subsection", "heading", "section_path_change", "table_boundary_before"}:
        previous_section = str(previous_overlap.get("source_section_id") or "")
        current_section = str(metadata.get("current_section_id") or "")
        if previous_section and current_section and previous_section != current_section:
            return False, "section_boundary"
    if strategy == "table_preserve_together" and not same_section_overlap(metadata, previous_overlap):
        return False, "table_cross_section_overlap"
    if len(overlap_text) / max(1, len(current_text)) > OVERLAP_MAX_RATIO:
        return False, "overlap_too_large"
    if jaccard_words(normalize_for_dedupe(overlap_text), normalize_for_dedupe(current_text)) > OVERLAP_SIMILARITY_SKIP:
        return False, "duplicate_overlap"
    if is_generated_or_debug_text(overlap_text):
        return False, "generated_overlap"
    return True, ""


def same_section_overlap(metadata: dict[str, Any], previous_overlap: dict[str, Any]) -> bool:
    previous_section = str(previous_overlap.get("source_section_id") or "")
    current_section = str(metadata.get("current_section_id") or "")
    return bool(previous_section and current_section and previous_section == current_section)


def overlap_skip_payload(reason: str) -> dict[str, Any]:
    return {
        "overlap_applied": False,
        "overlap_skip_reason": reason,
        "overlap_char_count": 0,
        "overlap_word_count": 0,
        "overlap_ratio": 0.0,
        "overlap_boundary_type": "",
        "overlap_quality_score": 0.0,
        "overlap_source_section_id": "",
        "overlap_source_section_title": "",
        "overlap_source_strategy": "",
    }


def clean_overlap_text(text: str) -> str:
    text = normalize_line(text)
    if is_page_number(text) or is_watermark(text) or is_page_furniture(text) or is_boilerplate(text):
        return ""
    if is_generated_or_debug_text(text):
        return ""
    return text


def trim_overlap_to_ratio(overlap: str, target_text: str) -> str:
    max_chars = min(OVERLAP_MAX_CHARS, max(OVERLAP_MIN_CHARS, int(len(target_text) * OVERLAP_TARGET_RATIO)))
    if len(overlap) <= max_chars:
        return overlap
    units = sentence_units(overlap)
    selected: list[str] = []
    for unit in reversed(units):
        projected = len(" ".join([unit, *selected]).strip())
        if projected > max_chars and selected:
            break
        selected.insert(0, unit)
    return " ".join(selected).strip() if selected else overlap[-max_chars:].strip()


def overlap_boundary_type(text: str) -> str:
    if is_bullet_or_numbered(text):
        return "bullet"
    if re.search(r"[.!?;:]$", text):
        return "sentence"
    if SECTION_NUMBER_RE.match(text) or HEADING_RE.match(text):
        return "heading"
    return "phrase"


def overlap_quality_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.55
    score += 0.2 if overlap_boundary_type(text) in {"sentence", "bullet"} else 0
    score += 0.15 if OVERLAP_MIN_CHARS <= len(text) <= OVERLAP_MAX_CHARS else 0
    score += 0.1 if len(text.split()) >= 8 else 0
    return round(min(1.0, score), 3)


def chunk_size_class(char_count: int) -> str:
    if char_count < MIN_CHUNK_CHARS:
        return "small"
    if char_count <= MAX_CHUNK_CHARS:
        return "healthy"
    return "large"


def chunk_diagnostics(blocks: list[ContentBlock], text: str, strategy: str) -> dict[str, Any]:
    warnings: list[str] = []
    section_ids = {block.metadata.get("current_section_id") for block in blocks if block.metadata.get("current_section_id")}
    starts_clean = starts_at_clean_boundary(text)
    if not starts_clean:
        warnings.append("weak_chunk_start")
    if len(section_ids) > 1 and strategy != "table_preserve_together":
        warnings.append("mixed_sections")
    if len(text) > MAX_CHUNK_CHARS and strategy != "table_preserve_together":
        warnings.append("oversized_chunk")
    if strategy == "table_preserve_together" and "TABLE:" not in text:
        warnings.append("table_without_serialized_header")
    score = 1.0
    score -= 0.2 if "weak_chunk_start" in warnings else 0
    score -= 0.35 if "mixed_sections" in warnings else 0
    score -= 0.25 if "oversized_chunk" in warnings else 0
    score -= 0.25 if "table_without_serialized_header" in warnings else 0
    return {
        "chunk_quality_score": round(max(0.0, score), 3),
        "chunk_warnings": warnings,
        "starts_clean_boundary": starts_clean,
        "semantic_unit_count": len(semantic_split_units(text)),
        "mixed_section_count": len(section_ids),
    }


def starts_at_clean_boundary(text: str) -> bool:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first:
        return False
    return bool(
        SECTION_NUMBER_RE.match(first)
        or APPENDIX_RE.match(first)
        or LETTER_HEADING_RE.match(first)
        or HEADING_RE.match(first)
        or is_bullet_or_numbered(first)
        or re.match(r"^(table|note|remark|purpose|scope|general|requirement|shall|the|all|for|if|where|when)\b", first, flags=re.I)
    )


def parent_chunk_key(blocks: list[ContentBlock]) -> str:
    section = next((block.metadata.get("section_breadcrumb") for block in blocks if block.metadata.get("section_breadcrumb")), "")
    pages = sorted({block.page for block in blocks})
    page_part = f"p{pages[0]}-{pages[-1]}" if pages else "p0"
    seed = f"{section}|{page_part}".strip("|")
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12] if seed else ""


def chunk_boundary_reason(
    block: ContentBlock,
    new_topic: bool,
    hard_subsection: bool,
    too_large: bool,
    layout_break: bool,
) -> str:
    if hard_subsection:
        return "hard_subsection"
    if block.is_heading:
        return "heading"
    if new_topic:
        return "section_path_change"
    if layout_break:
        return "layout_column_change"
    if too_large:
        return "size_limit"
    return "semantic_boundary"


def layout_boundary_changed(previous: ContentBlock, current: ContentBlock) -> bool:
    if previous.page != current.page:
        return False
    previous_column = previous.metadata.get("column_index")
    current_column = current.metadata.get("column_index")
    if not previous_column or not current_column:
        return False
    if previous_column == current_column:
        return False
    previous_count = int(previous.metadata.get("column_count") or 1)
    current_count = int(current.metadata.get("column_count") or 1)
    return previous_count > 1 and current_count > 1


def split_long_text_block(block: ContentBlock) -> list[ContentBlock]:
    parts = semantic_text_splits(block.text, MAX_CHUNK_CHARS)
    split_blocks = []
    for index, part in enumerate(parts, start=1):
        metadata = dict(block.metadata)
        metadata.update({"split_part": index, "split_total": len(parts), "split_from_long_block": True})
        split_blocks.append(
            ContentBlock(
                text=part,
                page=block.page,
                kind=block.kind,
                bbox=block.bbox,
                font_size=block.font_size,
                is_heading=block.is_heading and index == 1,
                heading_level=block.heading_level,
                section_path=list(block.section_path),
                metadata=metadata,
            )
        )
    return split_blocks


def split_oversized_chunks(chunks: list[RichChunk]) -> list[RichChunk]:
    split_chunks: list[RichChunk] = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if len(chunk.text) <= MAX_CHUNK_CHARS or metadata.get("contains_table"):
            split_chunks.append(chunk)
            continue
        parts = semantic_text_splits(chunk.text, MAX_CHUNK_CHARS)
        previous_overlap: dict[str, Any] | None = None
        for index, part in enumerate(parts, start=1):
            part_meta = dict(metadata)
            part_meta.update(
                {
                    "chunk_strategy": "paragraph_sentence_split",
                    "chunk_boundary_reason": "oversized_chunk_split",
                    "chunk_char_count": len(part),
                    "chunk_word_count": len(re.findall(r"\S+", part)),
                    "chunk_size_class": chunk_size_class(len(part)),
                    "split_part": index,
                    "split_total": len(parts),
                }
            )
            apply_overlap_metadata(part_meta, previous_overlap, part, "paragraph_sentence_split", "oversized_chunk_split")
            split_chunks.append(RichChunk(text=part, metadata=part_meta))
            previous_overlap = build_overlap_context(part, part_meta)
    return split_chunks


def semantic_text_splits(text: str, max_chars: int) -> list[str]:
    units = semantic_split_units(text)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_len = len(unit) + 1
        if current and current_len + unit_len > max_chars:
            parts.append("\n".join(current).strip())
            overlap = overlap_units(current)
            current = overlap.copy()
            current_len = sum(len(item) + 1 for item in current)
        if unit_len > max_chars:
            if current:
                parts.append("\n".join(current).strip())
                current = []
                current_len = 0
            parts.extend(split_long_sentence(unit, max_chars))
            continue
        current.append(unit)
        current_len += unit_len
    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part.strip()]


def semantic_split_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            units.append(paragraph)
            continue
        bullet_lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(bullet_lines) > 1 and all(is_bullet_or_numbered(line) for line in bullet_lines[: min(3, len(bullet_lines))]):
            units.extend(bullet_lines)
            continue
        units.extend(sentence_units(paragraph))
    return units


def sentence_units(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?;:])\s+", text) if sentence.strip()]
    return sentences or [text.strip()]


def split_long_sentence(text: str, max_chars: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        projected = current_len + len(word) + 1
        if current and projected > max_chars:
            parts.append(" ".join(current).strip())
            current = []
            current_len = 0
        current.append(word)
        current_len += len(word) + 1
    if current:
        parts.append(" ".join(current).strip())
    return parts


def overlap_units(units: list[str]) -> list[str]:
    if not units:
        return []
    selected: list[str] = []
    char_limit = min(OVERLAP_MAX_CHARS, max(80, int(MAX_CHUNK_CHARS * OVERLAP_TARGET_RATIO)))
    for unit in reversed(units):
        if not selected and len(unit) > char_limit:
            break
        projected = sum(len(item) + 1 for item in selected) + len(unit)
        if projected > char_limit:
            break
        selected.insert(0, unit)
    return selected


def is_bullet_or_numbered(text: str) -> bool:
    return bool(re.match(r"^(\*|-|•|\d+[.)]|[a-zA-Z][.)])\s+", text))


def finalize_chunk_system(chunks: list[RichChunk]) -> list[RichChunk]:
    finalized: list[RichChunk] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        metadata["chunk_sequence"] = index
        metadata["chunk_id"] = stable_chunk_id(chunk.text, metadata, index)
        metadata["chunk_system"] = metadata.get("chunk_system") or CHUNK_SYSTEM_VERSION
        metadata["chunk_char_count"] = len(chunk.text)
        metadata["chunk_word_count"] = len(re.findall(r"\S+", chunk.text))
        metadata["chunk_size_class"] = chunk_size_class(len(chunk.text))
        finalized.append(RichChunk(text=chunk.text, metadata=metadata))
    return finalized


def stable_chunk_id(text: str, metadata: dict[str, Any], sequence: int) -> str:
    seed = "|".join(
        [
            str(sequence),
            str(metadata.get("page_start") or ""),
            str(metadata.get("page_end") or ""),
            str(metadata.get("current_section_id") or ""),
            normalize_for_dedupe(text)[:240],
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def build_metadata(blocks: list[ContentBlock], doc_meta: dict[str, Any]) -> dict[str, Any]:
    pages = sorted({block.page for block in blocks})
    page_labels = doc_meta.get("page_labels") or {}
    section_path = next((block.section_path for block in reversed(blocks) if block.section_path), [])
    structure_meta = next((block.metadata for block in reversed(blocks) if block.metadata.get("section_breadcrumb") or block.metadata.get("document_title")), {})
    content_types = sorted({block.kind for block in blocks})
    text = "\n".join(block.text for block in blocks)
    contains_table = "table" in content_types or looks_like_table_text(text)
    keyword_set = extract_keywords(text, blocks, contains_table)
    table_meta = table_metadata(text) if contains_table else {}
    revision_meta = revision_metadata(text)
    layout_meta = layout_metadata(blocks)
    engineering_meta = engineering_metadata(text, blocks, contains_table)
    abbreviation_meta = abbreviation_metadata(text, keyword_set, engineering_meta)
    image_figure_meta = image_figure_metadata(text, blocks, doc_meta)
    multimodal_meta = multimodal_ingestion_metadata(blocks, contains_table, image_figure_meta)
    reference_meta = reference_ingestion_metadata(text)
    relationship_meta = relationship_ingestion_metadata(text, engineering_meta.get("engineering_entity_relationships") or [])
    query_meta = query_optimization_ingestion_metadata(text, keyword_set, engineering_meta, table_meta)
    semantic_meta = semantic_label_metadata(text, contains_table, engineering_meta, section_path)
    safety_meta = safety_critical_metadata(text, engineering_meta, semantic_meta)
    numeric_meta = numeric_constraint_ingestion_metadata(engineering_meta.get("numeric_constraints") or [], text)
    formula_meta = formula_ingestion_metadata(text, numeric_meta, table_meta)
    code_meta = code_ingestion_metadata(text, doc_meta)
    schema_meta = schema_ingestion_metadata(text, table_meta, code_meta, doc_meta)
    knowledge_graph_meta = knowledge_graph_ingestion_metadata(
        text,
        structure_meta,
        engineering_meta,
        relationship_meta,
        reference_meta,
        table_meta,
        schema_meta,
        code_meta,
        numeric_meta,
        doc_meta,
    )
    ontology_meta = ontology_ingestion_metadata(
        text,
        engineering_meta,
        semantic_meta,
        safety_meta,
        table_meta,
        schema_meta,
        code_meta,
        numeric_meta,
        knowledge_graph_meta,
    )
    context_meta = context_window_optimization_metadata(text, blocks, engineering_meta, contains_table)
    vector_meta = vector_optimization_metadata(text, keyword_set, engineering_meta, semantic_meta)
    hierarchical_embedding_meta = hierarchical_embedding_metadata(text, blocks, doc_meta, keyword_set, engineering_meta, semantic_meta, table_meta)
    section_importance_meta = section_importance_metadata(
        text,
        blocks,
        section_path,
        structure_meta,
        contains_table,
        engineering_meta,
        table_meta,
        reference_meta,
        relationship_meta,
        semantic_meta,
        numeric_meta,
        safety_meta,
    )
    document_classification_meta = document_classification_metadata(
        text,
        blocks,
        doc_meta,
        structure_meta,
        keyword_set,
        engineering_meta,
        semantic_meta,
        table_meta,
        revision_meta,
        safety_meta,
    )
    language_meta = language_metadata(text)
    translation_meta = translation_ingestion_metadata(text, language_meta)
    access_meta = access_control_metadata(text, doc_meta)
    validation_meta = ingestion_validation_metadata(text, blocks, doc_meta, contains_table, table_meta, engineering_meta, image_figure_meta)
    quality_meta = metadata_quality_metadata(text, blocks, doc_meta, contains_table, validation_meta)
    ingestion_quality_meta = ingestion_quality_metadata(
        text,
        blocks,
        doc_meta,
        contains_table,
        table_meta,
        engineering_meta,
        image_figure_meta,
        validation_meta,
        quality_meta,
        semantic_meta,
        numeric_meta,
        section_importance_meta,
        document_classification_meta,
        hierarchical_embedding_meta,
    )
    base_change_metadata = {
        **structure_meta,
        **revision_meta,
        **engineering_meta,
        "document_title": structure_meta.get("document_title", ""),
        "section_title": section_path[-1] if section_path else "",
        "section_path": section_path,
        "section_breadcrumb": structure_meta.get("section_breadcrumb", " > ".join(section_path)),
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
        "technical_identifiers": engineering_meta.get("technical_identifiers") or [],
        "keyword_identifiers": keyword_set.identifiers,
    }
    change_meta = change_detection_metadata(text, base_change_metadata)
    agentic_meta = agentic_ingestion_metadata(
        text,
        blocks,
        doc_meta,
        contains_table,
        table_meta,
        engineering_meta,
        validation_meta,
        quality_meta,
    )
    ocr_confidences = [
        float(block.metadata.get("ocr_confidence") or 0)
        for block in blocks
        if block.metadata.get("ocr") == "easyocr" and block.metadata.get("ocr_confidence") is not None
    ]
    return {
        "content_types": content_types,
        "document_title": structure_meta.get("document_title", ""),
        "section_path": section_path,
        "section_title": section_path[-1] if section_path else "",
        "parent_section": section_path[-2] if len(section_path) > 1 else "",
        "section_ids": structure_meta.get("section_ids", []),
        "section_titles": structure_meta.get("section_titles", []),
        "section_levels": structure_meta.get("section_levels", []),
        "section_breadcrumb": structure_meta.get("section_breadcrumb", " > ".join(section_path)),
        "section_depth": structure_meta.get("section_depth", len(section_path)),
        "current_section_id": structure_meta.get("current_section_id", ""),
        "current_section_title": structure_meta.get("current_section_title", ""),
        "parent_section_id": structure_meta.get("parent_section_id", ""),
        "parent_section_title": structure_meta.get("parent_section_title", ""),
        "document_map": structure_meta.get("document_map", []),
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
        "page_label_start": page_labels.get(pages[0]) if pages else None,
        "page_label_end": page_labels.get(pages[-1]) if pages else None,
        "contains_table": contains_table,
        "table_index": next((block.metadata.get("table_index") for block in blocks if block.metadata.get("table_index")), None),
        **table_meta,
        **revision_meta,
        **layout_meta,
        **engineering_meta,
        **abbreviation_meta,
        **image_figure_meta,
        **multimodal_meta,
        **reference_meta,
        **relationship_meta,
        **query_meta,
        **semantic_meta,
        **safety_meta,
        **numeric_meta,
        **formula_meta,
        **code_meta,
        **schema_meta,
        **knowledge_graph_meta,
        **ontology_meta,
        **context_meta,
        **vector_meta,
        **hierarchical_embedding_meta,
        **section_importance_meta,
        **document_classification_meta,
        **language_meta,
        **translation_meta,
        **access_meta,
        **validation_meta,
        **quality_meta,
        **ingestion_quality_meta,
        **change_meta,
        **agentic_meta,
        "keywords": keyword_set.keywords,
        "keyphrases": keyword_set.keyphrases,
        "exact_terms": keyword_set.exact_terms,
        "acronyms": keyword_set.acronyms,
        "keyword_identifiers": keyword_set.identifiers,
        "keyword_standards": keyword_set.standards,
        "table_keyword_terms": keyword_set.table_terms,
        "section_keyword_terms": keyword_set.section_terms,
        "keyword_domain_terms": keyword_set.domain_terms,
        "weighted_keywords": keyword_set.weighted_terms,
        "keyword_schema_version": "engineering-keywords-v2",
        "extractor": doc_meta.get("extractor"),
        "ocr": doc_meta.get("ocr"),
        "ocr_block_count": len(ocr_confidences) or doc_meta.get("ocr_block_count", 0),
        "ocr_confidence_avg": round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else doc_meta.get("ocr_confidence_avg", 0.0),
        "ocr_derived": bool(ocr_confidences),
    }


def engineering_metadata(text: str, blocks: list[ContentBlock], contains_table: bool) -> dict[str, Any]:
    normalized = normalize_line(text).lower()
    terms = keywords(text)
    domain_terms = sorted({term for term in terms if term in ENGINEERING_DOMAIN_TERMS})
    requirement_modalities = requirement_modality(normalized)
    numeric_constraints = extract_numeric_constraints(text)
    identifiers = extract_identifiers(text)
    standards = extract_standards(text)
    entity_records = enrich_entity_records(extract_engineering_entity_records(text, terms, identifiers, standards), text, blocks)
    entities = unique_preserve([record["text"] for record in entity_records])
    canonical_entities = unique_preserve([record["canonical"] for record in entity_records if record.get("canonical")])
    entity_aliases = unique_preserve(alias for record in entity_records for alias in record.get("aliases", []))
    entity_types = sorted({record["type"] for record in entity_records if record.get("type")})
    entity_relationships = extract_entity_relationships(text, entity_records)
    primary_entities = rank_primary_entities(entity_records)[:12]
    entity_surface = build_entity_surface(entity_records, entity_relationships)
    safety_flags = sorted(term for term in SAFETY_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized))
    compliance_flags = sorted(set(standards + [term.upper() for term in COMPLIANCE_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized)]))
    section_titles = [block.metadata.get("current_section_title") for block in blocks if block.metadata.get("current_section_title")]
    retrieval_tags = retrieval_metadata_tags(
        domain_terms=domain_terms,
        requirement_modalities=requirement_modalities,
        contains_table=contains_table,
        numeric_constraints=numeric_constraints,
        safety_flags=safety_flags,
        compliance_flags=compliance_flags,
        entities=entities,
        section_titles=section_titles,
    )
    return {
        "entity_schema_version": ENTITY_SCHEMA_VERSION,
        "domain_terms": domain_terms,
        "engineering_entities": entities,
        "engineering_canonical_entities": canonical_entities,
        "engineering_entity_records": entity_records,
        "engineering_entity_aliases": entity_aliases,
        "engineering_entity_types": entity_types,
        "engineering_entity_relationships": entity_relationships,
        "primary_entities": primary_entities,
        "entity_surface_text": entity_surface,
        "entity_facets": entity_facets(entity_records),
        "entity_count": len(entity_records),
        "has_entities": bool(entity_records),
        "entity_extraction_quality": entity_extraction_quality(entity_records, identifiers, standards, numeric_constraints),
        "technical_identifiers": identifiers,
        "standards": standards,
        "requirement_modalities": requirement_modalities,
        "has_requirement": bool(requirement_modalities),
        "has_mandatory_requirement": any(item in requirement_modalities for item in ["shall", "must", "required", "prohibition"]),
        "has_recommendation": any(item in requirement_modalities for item in ["should", "preferably", "may", "consider"]),
        "has_numeric_constraints": bool(numeric_constraints),
        "numeric_constraints": numeric_constraints,
        "numeric_constraint_count": len(numeric_constraints),
        "safety_flags": safety_flags,
        "safety_critical": bool(safety_flags),
        "compliance_flags": compliance_flags,
        "compliance_related": bool(compliance_flags),
        "retrieval_tags": retrieval_tags,
        "answer_scope_hint": answer_scope_hint(requirement_modalities, contains_table, safety_flags, numeric_constraints),
    }


def entity_extraction_quality(
    entity_records: list[dict[str, Any]],
    identifiers: list[str],
    standards: list[str],
    numeric_constraints: list[dict[str, str]],
) -> dict[str, Any]:
    typed = sum(1 for record in entity_records if record.get("type") and record.get("type") != "engineering_term")
    alias_covered = sum(1 for record in entity_records if record.get("aliases"))
    score = 0.25
    score += 0.25 if entity_records else 0
    score += 0.15 if typed else 0
    score += 0.10 if alias_covered else 0
    score += 0.10 if identifiers else 0
    score += 0.10 if standards else 0
    score += 0.05 if numeric_constraints else 0
    return {
        "score": round(min(1.0, score), 3),
        "typed_entity_count": typed,
        "alias_covered_count": alias_covered,
        "identifier_count": len(identifiers),
        "standard_count": len(standards),
    }


def requirement_modality(normalized_text: str) -> list[str]:
    modalities: list[str] = []
    for phrase in PROHIBITION_TERMS:
        if phrase in normalized_text:
            modalities.append("prohibition")
    for phrase in REQUIREMENT_TERMS:
        if re.search(rf"\b{re.escape(phrase)}\b", normalized_text):
            modalities.append(phrase)
    for phrase in RECOMMENDATION_TERMS:
        if re.search(rf"\b{re.escape(phrase)}\b", normalized_text):
            modalities.append(phrase)
    return list(dict.fromkeys(modalities))


def extract_numeric_constraints(text: str) -> list[dict[str, str]]:
    constraints: list[dict[str, str]] = []
    for match in NUMERIC_UNIT_RE.finditer(text):
        value = normalize_line(match.group("value"))
        unit = normalize_line(match.group("unit") or "")
        before = text[max(0, match.start() - 12) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 12)]
        if is_inside_standard_reference(text, match.start()):
            continue
        if value.isdigit() and not unit and len(value) > 4:
            continue
        if not unit and re.match(r"^\d+(?:\.\d+)+$", value):
            continue
        if not unit and re.search(r"(ASME|NORSOK|ISO|IEC|API)\s*$", before, flags=re.I):
            continue
        if not unit and re.match(r"^\s*[-A-Z]", after):
            continue
        start = max(0, match.start() - 45)
        end = min(len(text), match.end() + 55)
        context = normalize_line(text[start:end])
        constraints.append({"value": value, "unit": unit, "context": context})
        if len(constraints) >= 30:
            break
    return constraints


def is_inside_standard_reference(text: str, position: int) -> bool:
    patterns = [
        r"\bASME\s+[A-Z]\d+(?:\.\d+)*[A-Z0-9.-]*",
        r"\bNORSOK\s+[A-Z]-\d+[A-Z0-9.-]*",
        r"\bAPI\s+\d+[A-Z0-9.-]*",
        r"\bISO\s+\d+[A-Z0-9:.-]*",
        r"\bIEC\s+\d+[A-Z0-9:.-]*",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            if match.start() <= position < match.end():
                return True
    return False


def extract_identifiers(text: str) -> list[str]:
    identifiers = []
    for match in IDENTIFIER_RE.finditer(text):
        if is_inside_standard_reference(text, match.start()):
            continue
        identifier = match.group(0).strip("-_")
        if len(identifier) >= 5 and identifier not in identifiers:
            identifiers.append(identifier)
        if len(identifiers) >= 30:
            break
    return identifiers


def extract_standards(text: str) -> list[str]:
    standards: list[str] = []
    patterns = [
        r"\bASME\s+[A-Z]\d+(?:\.\d+)*[A-Z0-9.-]*",
        r"\bNORSOK\s+[A-Z]-\d+[A-Z0-9.-]*",
        r"\bAPI\s+\d+[A-Z0-9.-]*",
        r"\bISO\s+\d+[A-Z0-9:.-]*",
        r"\bIEC\s+\d+[A-Z0-9:.-]*",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            standard = normalize_line(match.group(0)).rstrip(".,;:")
            if standard and standard.upper() not in [item.upper() for item in standards]:
                standards.append(standard.upper())
            if len(standards) >= 20:
                return standards
    return standards


def extract_engineering_entities(text: str, terms: list[str]) -> list[str]:
    return unique_preserve([record["text"] for record in extract_engineering_entity_records(text, terms, extract_identifiers(text), extract_standards(text))])[:25]


def extract_engineering_entity_records(
    text: str,
    terms: list[str],
    identifiers: list[str],
    standards: list[str],
) -> list[dict[str, Any]]:
    normalized = normalize_line(text)
    records: list[dict[str, Any]] = []

    phrase_patterns = [
        r"\b(?!(?:of|and|or|to|for|with|from|in|on|at)\s)[A-Za-z0-9/&().-]+\s+(?:valves?|drains?|vents?|headers?|devices?|systems?|lines?|piping|equipment|actuators?|probes?|pumps?|compressors?|vessels?|skids?|modules?)\b",
        r"\bpressure relief devices?\b",
        r"\bpressure safety valves?\b",
        r"\bcontrol valves?\b",
        r"\bactuated valves?\b",
        r"\bmanual valves?\b",
        r"\bisolation valves?\b",
        r"\brupture discs?\b",
        r"\bflare headers?\b",
        r"\bclosed drains?\b",
        r"\boverboard drains?\b",
        r"\bemergency shutdown\b",
        r"\bp&id(?:s)?\b",
    ]
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, normalized, flags=re.I):
            add_entity_record(records, match.group(0), "phrase", "pattern")
            if len(records) >= ENTITY_MAX_RECORDS:
                return records

    for identifier in identifiers:
        add_entity_record(records, identifier, "identifier", "identifier_regex")

    for standard in standards:
        add_entity_record(records, standard, "standard", "standard_regex")

    for acronym in extract_acronyms(text):
        if acronym.lower() in ENTITY_ALIAS_MAP:
            add_entity_record(records, acronym, "alias", "acronym_alias")

    for term in terms:
        if term in ENGINEERING_DOMAIN_TERMS:
            add_entity_record(records, term, classify_entity_type(term), "domain_term")

    return records[:ENTITY_MAX_RECORDS]


def add_entity_record(records: list[dict[str, Any]], value: str, entity_type: str, source: str) -> None:
    entity = normalize_entity_text(value)
    if not valid_entity_text(entity):
        return
    canonical = canonical_entity(entity)
    existing = next((record for record in records if record.get("canonical") == canonical), None)
    aliases = entity_aliases(entity, canonical)
    if existing:
        existing_aliases = unique_preserve([*existing.get("aliases", []), *aliases, entity])
        existing["aliases"] = existing_aliases[:12]
        existing["sources"] = unique_preserve([*existing.get("sources", []), source])[:6]
        if existing.get("type") in {"phrase", "alias"} and entity_type not in {"phrase", "alias"}:
            existing["type"] = entity_type
        existing["confidence"] = max(float(existing.get("confidence") or 0.5), entity_source_confidence(existing.get("type") or entity_type, source))
        return
    records.append(
        {
            "text": entity,
            "canonical": canonical,
            "type": classify_entity_type(entity) if entity_type in {"phrase", "alias"} else entity_type,
            "aliases": aliases,
            "sources": [source],
            "confidence": entity_source_confidence(entity_type, source),
        }
    )


def enrich_entity_records(records: list[dict[str, Any]], text: str, blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    section = next((block.metadata.get("section_breadcrumb") for block in blocks if block.metadata.get("section_breadcrumb")), "")
    enriched: list[dict[str, Any]] = []
    for record in records:
        aliases = unique_preserve([record.get("text", ""), record.get("canonical", ""), *(record.get("aliases") or [])])
        mentions = entity_mentions(text, aliases)
        contexts = entity_contexts(text, aliases)
        enriched.append(
            {
                **record,
                "aliases": aliases[:12],
                "mention_count": len(mentions),
                "first_mention_char": mentions[0] if mentions else -1,
                "contexts": contexts[:4],
                "section_scope": section,
                "confidence": round(min(1.0, float(record.get("confidence") or 0.5) + min(0.2, len(mentions) * 0.03)), 3),
            }
        )
    return sorted(enriched, key=lambda item: entity_rank_key(item), reverse=True)[:ENTITY_MAX_RECORDS]


def entity_mentions(text: str, aliases: list[str]) -> list[int]:
    mentions: list[int] = []
    lower_text = text.lower()
    for alias in aliases:
        alias_text = str(alias or "").lower().strip()
        if len(alias_text) < 2:
            continue
        for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(alias_text)}(?![A-Za-z0-9])", lower_text):
            mentions.append(match.start())
            if len(mentions) >= 20:
                break
    return sorted(set(mentions))[:20]


def entity_contexts(text: str, aliases: list[str]) -> list[str]:
    contexts: list[str] = []
    for position in entity_mentions(text, aliases)[:4]:
        start = max(0, position - ENTITY_CONTEXT_WINDOW)
        end = min(len(text), position + ENTITY_CONTEXT_WINDOW)
        context = normalize_line(text[start:end])
        if context and context not in contexts:
            contexts.append(context)
    return contexts


def entity_rank_key(record: dict[str, Any]) -> tuple[float, int, int]:
    type_weight = 1.0 if record.get("type") in {"identifier", "standard"} else 0.75
    type_weight += 0.2 if record.get("type") not in {"engineering_term", "phrase", "alias"} else 0
    confidence = float(record.get("confidence") or 0)
    mention_count = int(record.get("mention_count") or 0)
    return (type_weight + confidence, mention_count, len(str(record.get("text") or "")))


def entity_source_confidence(entity_type: str, source: str) -> float:
    if entity_type in {"identifier", "standard"}:
        return 0.95
    if source == "pattern":
        return 0.78
    if source == "acronym_alias":
        return 0.72
    if source == "domain_term":
        return 0.58
    return 0.5


def rank_primary_entities(records: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(records, key=lambda item: entity_rank_key(item), reverse=True)
    return unique_preserve([str(record.get("text") or "") for record in ranked if record.get("text")])


def build_entity_surface(records: list[dict[str, Any]], relationships: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for record in records:
        parts.extend(
            [
                str(record.get("text") or ""),
                str(record.get("canonical") or ""),
                str(record.get("type") or ""),
                " ".join(str(alias) for alias in record.get("aliases") or []),
                " ".join(str(source) for source in record.get("sources") or []),
                " ".join(str(context) for context in record.get("contexts") or []),
            ]
        )
    for relationship in relationships:
        parts.append(" ".join(str(relationship.get(key) or "") for key in ("left", "relation", "right")))
    return normalize_line(" ".join(part for part in parts if part))


def entity_facets(records: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(str(record.get("type") or "unknown") for record in records)
    source_counts = Counter(source for record in records for source in (record.get("sources") or []))
    return {
        "types": dict(type_counts),
        "sources": dict(source_counts),
        "high_confidence_count": sum(1 for record in records if float(record.get("confidence") or 0) >= 0.75),
        "identifier_count": type_counts.get("identifier", 0),
        "standard_count": type_counts.get("standard", 0),
    }


def normalize_entity_text(value: str) -> str:
    entity = normalize_line(str(value or "")).strip(" .,:;()[]{}")
    entity = re.sub(r"\s+", " ", entity)
    return entity.lower() if not re.search(r"[A-Z]{2,}[-A-Z0-9_/]{2,}", entity) else entity.upper()


def valid_entity_text(entity: str) -> bool:
    if not entity or len(entity) < 2:
        return False
    normalized = entity.lower()
    if normalized.split()[0] in {"of", "and", "or", "to", "for", "with", "from", "in", "on", "at"}:
        return False
    if normalized in STOPWORDS:
        return False
    return True


def canonical_entity(entity: str) -> str:
    key = normalize_line(entity).lower().replace("&", "and")
    key = re.sub(r"\bp\s*&\s*id\b", "p&id", key)
    key = ENTITY_ALIAS_MAP.get(key, key)
    words = []
    for word in key.split():
        if word.endswith("s") and len(word) > 4 and not word.endswith(("ss", "us")):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def entity_aliases(entity: str, canonical: str) -> list[str]:
    aliases = [entity]
    for alias, target in ENTITY_ALIAS_MAP.items():
        if canonical == canonical_entity(target):
            aliases.append(alias.upper() if alias.isupper() else alias)
    if canonical != entity.lower():
        aliases.append(canonical)
    return unique_preserve(aliases)[:12]


def unique_preserve(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values:
        key = normalize_line(str(value or "")).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def classify_entity_type(entity: str) -> str:
    normalized = canonical_entity(entity)
    for entity_type, terms in ENTITY_TYPE_TERMS.items():
        if any(canonical_entity(term) in normalized or normalized in canonical_entity(term) for term in terms):
            return entity_type
    if re.search(r"\b[A-Z]{2,}[-A-Z0-9_/]{2,}\b", entity):
        return "identifier"
    return "engineering_term"


def extract_entity_relationships(text: str, entity_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    relationships: list[dict[str, str]] = []
    known_records = [
        record
        for record in entity_records
        if record.get("canonical") and record.get("type") not in {"standard"}
    ]
    for match in ENTITY_RELATION_RE.finditer(text):
        relation = normalize_line(match.group("relation")).lower()
        left_text = closest_entity_before(text, match.start("relation"), known_records)
        right_text = closest_entity_after(text, match.end("relation"), known_records)
        if not left_text or not right_text or left_text == right_text:
            continue
        item = {"left": left_text, "relation": relation, "right": right_text}
        if item not in relationships:
            relationships.append(item)
        if len(relationships) >= 20:
            break
    return relationships


def closest_entity_before(text: str, position: int, records: list[dict[str, Any]]) -> str:
    window = text[max(0, position - 140) : position].lower()
    candidates: list[tuple[int, str]] = []
    for record in records:
        for alias in [record.get("text", ""), record.get("canonical", ""), *(record.get("aliases") or [])]:
            alias_text = str(alias or "").lower()
            if not alias_text:
                continue
            idx = window.rfind(alias_text)
            if idx >= 0:
                candidates.append((idx, str(record.get("text") or alias)))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1] if candidates else ""


def closest_entity_after(text: str, position: int, records: list[dict[str, Any]]) -> str:
    window = text[position : min(len(text), position + 160)].lower()
    candidates: list[tuple[int, str]] = []
    for record in records:
        for alias in [record.get("text", ""), record.get("canonical", ""), *(record.get("aliases") or [])]:
            alias_text = str(alias or "").lower()
            if not alias_text:
                continue
            idx = window.find(alias_text)
            if idx >= 0:
                candidates.append((idx, str(record.get("text") or alias)))
    return sorted(candidates, key=lambda item: item[0])[0][1] if candidates else ""


def retrieval_metadata_tags(
    *,
    domain_terms: list[str],
    requirement_modalities: list[str],
    contains_table: bool,
    numeric_constraints: list[dict[str, str]],
    safety_flags: list[str],
    compliance_flags: list[str],
    entities: list[str],
    section_titles: list[str],
) -> list[str]:
    tags = set(domain_terms)
    tags.update(requirement_modalities)
    tags.update(entity.replace(" ", "_") for entity in entities)
    tags.update(normalize_line(title).lower().replace(" ", "_") for title in section_titles[:5] if title)
    if contains_table:
        tags.add("table")
    if numeric_constraints:
        tags.add("numeric_constraint")
    if safety_flags:
        tags.add("safety_critical")
    if compliance_flags:
        tags.add("compliance")
    return sorted(tag for tag in tags if tag)


def answer_scope_hint(
    modalities: list[str],
    contains_table: bool,
    safety_flags: list[str],
    numeric_constraints: list[dict[str, str]],
) -> str:
    if safety_flags:
        return "safety-critical: answer only from cited document context"
    if contains_table or numeric_constraints:
        return "numeric/table: preserve exact values and units"
    if "prohibition" in modalities:
        return "requirement: preserve prohibition wording"
    if any(item in modalities for item in ["shall", "must", "required"]):
        return "requirement: preserve shall/must wording"
    return "grounded-summary"


def language_metadata(text: str) -> dict[str, Any]:
    return language_detection_metadata(text)


def detected_scripts(text: str) -> list[str]:
    return rich_detected_scripts(text)


def translation_ingestion_metadata(text: str, language_meta: dict[str, Any]) -> dict[str, Any]:
    return rich_translation_ingestion_metadata(text, language_meta)


def access_control_metadata(text: str, doc_meta: dict[str, Any]) -> dict[str, Any]:
    return policy_access_control_metadata(text, doc_meta)


def image_figure_metadata(text: str, blocks: list[ContentBlock], doc_meta: dict[str, Any]) -> dict[str, Any]:
    image_blocks = [block for block in blocks if block.kind == "image" or block.metadata.get("image")]
    figure_refs = unique_preserve(re.findall(r"\b(?:figure|fig\.?|image|diagram|drawing)\s*[#:.-]?\s*([A-Z]?\d+(?:\.\d+)*)\b", text, flags=re.I))
    captions = extract_figure_captions(text)
    regions = unique_preserve([block.metadata.get("layout_region", "") for block in image_blocks if block.metadata.get("layout_region")])
    return {
        "image_figure_schema_version": IMAGE_FIGURE_SCHEMA_VERSION,
        "image_block_count": len(image_blocks),
        "has_images": bool(image_blocks),
        "figure_references": figure_refs[:20],
        "figure_captions": captions[:12],
        "figure_region_hints": regions[:12],
        "image_ocr_available": bool(doc_meta.get("ocr_pages") or any(block.metadata.get("ocr") == "easyocr" for block in blocks)),
        "figure_ingestion_status": "image_blocks_detected" if image_blocks else ("figure_text_references_detected" if figure_refs or captions else "not_detected"),
    }


def extract_figure_captions(text: str) -> list[str]:
    captions: list[str] = []
    for line in text.splitlines():
        normalized = normalize_line(line)
        if re.match(r"^(figure|fig\.?|image|diagram|drawing)\s*[#:.-]?\s*", normalized, flags=re.I):
            captions.append(normalized[:240])
    return unique_preserve(captions)


def multimodal_ingestion_metadata(blocks: list[ContentBlock], contains_table: bool, image_meta: dict[str, Any]) -> dict[str, Any]:
    modalities = ["text"]
    if contains_table:
        modalities.append("table")
    if image_meta.get("has_images") or image_meta.get("figure_references"):
        modalities.append("image")
    if any(block.metadata.get("ocr") == "easyocr" for block in blocks):
        modalities.append("ocr")
    if any(block.metadata.get("layout_aware") for block in blocks):
        modalities.append("layout")
    return {
        "multimodal_schema_version": MULTIMODAL_SCHEMA_VERSION,
        "modalities": unique_preserve(modalities),
        "multimodal": len(set(modalities)) > 1,
        "multimodal_signal_count": len(set(modalities)),
        "multimodal_status": "ready" if len(set(modalities)) > 1 else "text_only",
    }


def reference_ingestion_metadata(text: str) -> dict[str, Any]:
    section_refs = unique_preserve(re.findall(r"\b(?:section|sec\.?|clause|paragraph|para\.?)\s+((?:\d+\.)*\d+[A-Z]?)\b", text, flags=re.I))
    standard_refs = extract_standards(text)
    document_refs = extract_identifiers(text)
    figure_refs = unique_preserve(re.findall(r"\b(?:figure|fig\.?|table|drawing)\s+[A-Z]?\d+(?:\.\d+)*\b", text, flags=re.I))
    return {
        "reference_schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_section_ids": section_refs[:30],
        "reference_standards": standard_refs[:30],
        "reference_document_ids": document_refs[:30],
        "reference_figure_ids": figure_refs[:30],
        "reference_count": min(120, len(section_refs) + len(standard_refs) + len(document_refs) + len(figure_refs)),
        "has_references": bool(section_refs or standard_refs or document_refs or figure_refs),
    }


def relationship_ingestion_metadata(text: str, relationships: list[dict[str, str]]) -> dict[str, Any]:
    relation_types = sorted({str(item.get("relation") or "") for item in relationships if item.get("relation")})
    directional = [item for item in relationships if re.search(r"\b(upstream|downstream|to|from|connected|associated)\b", item.get("relation", ""), flags=re.I)]
    dependency_terms = unique_preserve(re.findall(r"\b(?:upstream|downstream|connected to|associated with|drains? to|vents? to|located (?:in|on|at))\b", text, flags=re.I))
    return {
        "relationship_schema_version": RELATIONSHIP_SCHEMA_VERSION,
        "relationship_records": relationships[:30],
        "relationship_types": relation_types[:20],
        "relationship_count": len(relationships),
        "directional_relationship_count": len(directional),
        "dependency_terms": dependency_terms[:20],
        "relationship_graph_ready": bool(relationships),
    }


def query_optimization_ingestion_metadata(text: str, keyword_set: KeywordSet, engineering_meta: dict[str, Any], table_meta: dict[str, Any]) -> dict[str, Any]:
    expansion_terms = unique_preserve(
        keyword_set.keywords[:18]
        + keyword_set.keyphrases[:12]
        + keyword_set.exact_terms[:12]
        + engineering_meta.get("primary_entities", [])[:12]
        + engineering_meta.get("technical_identifiers", [])[:12]
    )
    intent_hints = []
    if table_meta:
        intent_hints.append("table_numeric")
    if engineering_meta.get("safety_critical"):
        intent_hints.append("safety_critical")
    if engineering_meta.get("has_requirement"):
        intent_hints.append("requirement")
    if engineering_meta.get("standards"):
        intent_hints.append("compliance")
    return {
        "query_optimization_schema_version": QUERY_OPTIMIZATION_SCHEMA_VERSION,
        "query_expansion_terms": expansion_terms[:40],
        "query_exact_match_terms": unique_preserve(keyword_set.exact_terms + engineering_meta.get("standards", []))[:40],
        "query_intent_hints": intent_hints,
        "query_optimization_ready": bool(expansion_terms or intent_hints),
    }


def semantic_label_metadata(text: str, contains_table: bool, engineering_meta: dict[str, Any], section_path: list[str]) -> dict[str, Any]:
    labels: set[str] = set()
    normalized = normalize_line(text).lower()
    if contains_table:
        labels.add("table")
    if engineering_meta.get("has_requirement"):
        labels.add("requirement")
    if engineering_meta.get("safety_critical"):
        labels.add("safety")
    if engineering_meta.get("has_numeric_constraints"):
        labels.add("numeric_constraint")
    if engineering_meta.get("compliance_related"):
        labels.add("compliance")
    if re.search(r"\b(definition|means|refers to|is defined as)\b", normalized):
        labels.add("definition")
    if re.search(r"\b(procedure|step|sequence|method|workflow)\b", normalized):
        labels.add("procedure")
    if section_path:
        labels.add(f"section:{normalize_line(section_path[-1]).lower().replace(' ', '_')}")
    return {
        "semantic_label_schema_version": SEMANTIC_LABEL_SCHEMA_VERSION,
        "semantic_labels": sorted(labels),
        "primary_semantic_label": sorted(labels)[0] if labels else "general_engineering",
        "semantic_label_confidence": round(min(1.0, 0.35 + len(labels) * 0.12), 3),
    }


def safety_critical_metadata(text: str, engineering_meta: dict[str, Any], semantic_meta: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_line(text).lower()
    safety_terms = sorted({term for term in SAFETY_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized)})
    hazard_terms = sorted(set(re.findall(r"\b(?:shall not|must not|hazard|critical|emergency|explosion|fire|overpressure|shutdown|relief|flare)\b", normalized)))
    requirement = bool(engineering_meta.get("has_mandatory_requirement") or "requirement" in semantic_meta.get("semantic_labels", []))
    score = min(1.0, len(safety_terms) * 0.12 + len(hazard_terms) * 0.1 + (0.25 if requirement and safety_terms else 0))
    return {
        "safety_tag_schema_version": SAFETY_TAG_SCHEMA_VERSION,
        "safety_terms": safety_terms,
        "hazard_terms": hazard_terms,
        "safety_score": round(score, 3),
        "safety_critical": bool(engineering_meta.get("safety_critical") or score >= 0.25),
        "safety_answer_policy": "strict_citation_required" if score >= 0.25 else "standard_grounded_answer",
    }


def numeric_constraint_ingestion_metadata(constraints: list[dict[str, str]], text: str = "") -> dict[str, Any]:
    return unit_normalization_metadata(constraints, text)


def normalize_unit(unit: str) -> str:
    unit = unit.lower().strip()
    aliases = {"degree": "deg", "degrees": "deg", "in": "inch", "barg": "bar"}
    return aliases.get(unit, unit)


def unit_family(unit: str, value: str) -> str:
    if ":" in value or "/" in value:
        return "ratio"
    if unit in {"mm", "cm", "m", "inch"}:
        return "length"
    if unit in {"bar", "psi", "kpa", "mpa"}:
        return "pressure"
    if unit in {"deg", "c"}:
        return "temperature_or_angle"
    if unit in {"%", "hz"}:
        return "rate_or_frequency"
    return "number"


def context_window_optimization_metadata(text: str, blocks: list[ContentBlock], engineering_meta: dict[str, Any], contains_table: bool) -> dict[str, Any]:
    safety = bool(engineering_meta.get("safety_critical"))
    requirement = bool(engineering_meta.get("has_requirement"))
    section_ids = unique_preserve([str(block.metadata.get("current_section_id") or "") for block in blocks if block.metadata.get("current_section_id")])
    recommended = 2 if safety or requirement else 1
    if contains_table:
        recommended = max(recommended, 1)
    if len(text) > MAX_CHUNK_CHARS * 0.85:
        recommended = max(recommended, 2)
    return {
        "context_window_schema_version": CONTEXT_WINDOW_SCHEMA_VERSION,
        "recommended_window_before": recommended,
        "recommended_window_after": recommended,
        "context_window_reason": "safety_or_requirement" if safety or requirement else ("large_chunk" if len(text) > MAX_CHUNK_CHARS * 0.85 else "standard"),
        "context_section_scope_ids": section_ids[:8],
        "context_window_optimized": True,
    }


def vector_optimization_metadata(text: str, keyword_set: KeywordSet, engineering_meta: dict[str, Any], semantic_meta: dict[str, Any]) -> dict[str, Any]:
    dense_terms = unique_preserve(
        keyword_set.keyphrases[:12]
        + keyword_set.keywords[:18]
        + engineering_meta.get("primary_entities", [])[:12]
        + semantic_meta.get("semantic_labels", [])[:12]
    )
    sparse_terms = unique_preserve(keyword_set.exact_terms[:20] + engineering_meta.get("technical_identifiers", [])[:20] + engineering_meta.get("standards", [])[:20])
    density = len(set(keyword_tokens(text))) / max(1, len(keyword_tokens(text)))
    return {
        "vector_optimization_schema_version": VECTOR_OPTIMIZATION_SCHEMA_VERSION,
        "vector_dense_terms": dense_terms[:40],
        "vector_sparse_terms": sparse_terms[:40],
        "vector_text_density": round(density, 3),
        "vector_chunk_ready": bool(dense_terms or sparse_terms),
        "vector_optimization_strategy": "e5_passage_plus_hybrid_surfaces",
    }


def ingestion_validation_metadata(
    text: str,
    blocks: list[ContentBlock],
    doc_meta: dict[str, Any],
    contains_table: bool,
    table_meta: dict[str, Any],
    engineering_meta: dict[str, Any],
    image_meta: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    if len(text.strip()) < 40 and not contains_table and not image_meta.get("has_images"):
        warnings.append("short_chunk")
    if contains_table and table_meta.get("table_integrity") not in {"complete", "usable"}:
        warnings.append("weak_table_integrity")
    if doc_meta.get("ocr_pages") and float(doc_meta.get("ocr_confidence_avg") or 0) < 0.35:
        warnings.append("low_ocr_confidence")
    if engineering_meta.get("safety_critical") and not engineering_meta.get("requirement_modalities"):
        warnings.append("safety_context_without_requirement_modal")
    if any(is_generated_or_debug_text(block.text) for block in blocks):
        warnings.append("generated_or_debug_text_detected")
    score = max(0.0, 1.0 - len(warnings) * 0.18)
    return {
        "ingestion_validation_schema_version": INGESTION_VALIDATION_SCHEMA_VERSION,
        "ingestion_validation_passed": not warnings,
        "ingestion_validation_warnings": warnings,
        "ingestion_validation_score": round(score, 3),
        "ingestion_validation_checks": [
            "minimum_text_or_structured_content",
            "table_integrity",
            "ocr_confidence",
            "safety_requirement_signal",
            "contamination_guard",
        ],
    }


def metadata_quality_metadata(
    text: str,
    blocks: list[ContentBlock],
    doc_meta: dict[str, Any],
    contains_table: bool,
    validation_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_section = any(block.metadata.get("current_section_id") or block.metadata.get("section_breadcrumb") for block in blocks)
    has_page = any(block.page for block in blocks)
    has_layout = any(block.metadata.get("layout_aware") for block in blocks)
    has_ocr = any(block.metadata.get("ocr") == "easyocr" for block in blocks)
    quality = 0.35
    quality += 0.15 if has_section else 0
    quality += 0.1 if has_page else 0
    quality += 0.1 if has_layout else 0
    quality += 0.1 if keywords(text) else 0
    quality += 0.1 if contains_table and table_metadata(text).get("table_integrity") == "complete" else 0
    quality += 0.1 if not has_ocr or doc_meta.get("ocr_confidence_avg", 0) >= 0.35 else 0
    quality += 0.1 if (validation_meta or {}).get("ingestion_validation_passed") else 0
    return {
        "metadata_schema_version": "engineering-metadata-v2",
        "metadata_quality_score": round(min(1.0, quality), 3),
        "metadata_has_section": has_section,
        "metadata_has_page": has_page,
        "metadata_has_layout": has_layout,
        "metadata_has_ocr": has_ocr,
    }


def revision_metadata(text: str) -> dict[str, Any]:
    compact = normalize_line(text)
    revision = ""
    document_id = ""
    status = ""
    revision_candidates = unique_preserve(
        match.group(0)
        for match in re.finditer(r"\b(?:rev(?:ision)?\.?|version|issue)\s*[:#]?\s*[A-Z]?\d+[A-Z]?\b", compact, flags=re.I)
    )
    revision_dates = unique_preserve(
        re.findall(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", compact)
    )
    revision_match = re.search(r"\b(?:revision|rev\.?|revision number)\s*[:#]?\s*([A-Z]?\d+[A-Z]?|[A-Z]\d?)\b", compact, flags=re.I)
    if revision_match:
        revision = revision_match.group(1)
    document_match = re.search(r"\b(?:document id|document n\.?|contractor document id|company document id)\s*[:#]?\s*([A-Z0-9_-]{6,})\b", compact, flags=re.I)
    if document_match:
        document_id = document_match.group(1)
    status_match = re.search(r"\b(?:validity status|status)\s*[:#]?\s*([A-Z][A-Z0-9 -]{1,40})\b", compact, flags=re.I)
    if status_match:
        status = status_match.group(1).strip()
    return {
        "revision_management_schema_version": REVISION_MANAGEMENT_SCHEMA_VERSION,
        "revision": revision,
        "revision_candidates": revision_candidates[:20],
        "revision_dates": revision_dates[:20],
        "document_identifier": document_id,
        "validity_status": status,
        "revision_managed": bool(revision or revision_candidates or revision_dates or status),
        "revision_sort_key": revision_sort_key(revision),
    }


def revision_sort_key(revision: str) -> str:
    normalized = normalize_line(revision).upper()
    if not normalized:
        return ""
    match = re.match(r"([A-Z]?)(\d+)([A-Z]?)", normalized)
    if not match:
        return normalized
    prefix, number, suffix = match.groups()
    return f"{prefix}{int(number):04d}{suffix}"


def detect_document_title(blocks: list[ContentBlock], body_size: float) -> str:
    candidates: list[tuple[float, int, str]] = []
    for block in blocks[:80]:
        if block.kind != "text" or not block.text:
            continue
        if is_page_number(block.text) or is_page_furniture(block.text) or is_watermark(block.text):
            continue
        if SECTION_NUMBER_RE.match(block.text) or APPENDIX_RE.match(block.text):
            continue
        words = block.text.split()
        if not 2 <= len(words) <= 18:
            continue
        alpha = [char for char in block.text if char.isalpha()]
        if not alpha:
            continue
        uppercase_ratio = sum(1 for char in alpha if char.isupper()) / len(alpha)
        score = float(block.font_size or 0) + (6 if uppercase_ratio > 0.55 else 0) - (block.page * 0.05)
        if body_size and block.font_size >= body_size:
            score += 3
        candidates.append((score, -block.page, block.text))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[2]


def apply_structure_metadata(block: ContentBlock, stack: list[dict[str, Any]], document_title: str) -> None:
    block.section_path = [node["heading"] for node in stack]
    block.metadata.update(
        {
            "document_title": document_title,
            "section_ids": [node["id"] for node in stack],
            "section_titles": [node["title"] for node in stack],
            "section_levels": [node["level"] for node in stack],
            "section_breadcrumb": " > ".join(node["heading"] for node in stack),
            "section_depth": len(stack),
            "current_section_id": stack[-1]["id"] if stack else "",
            "current_section_title": stack[-1]["title"] if stack else "",
            "parent_section_id": stack[-2]["id"] if len(stack) > 1 else "",
            "parent_section_title": stack[-2]["title"] if len(stack) > 1 else "",
            "document_map": [
                {
                    "id": node["id"],
                    "title": node["title"],
                    "heading": node["heading"],
                    "level": node["level"],
                    "page": node["page"],
                }
                for node in stack
            ],
        }
    )


def looks_like_heading(text: str, font_size: float, body_size: float) -> bool:
    if len(text) > 140 or text.endswith("."):
        return False
    if SECTION_NUMBER_RE.match(text) or APPENDIX_RE.match(text) or LETTER_HEADING_RE.match(text) or HEADING_RE.match(text):
        return True
    if re.match(r"^(purpose|scope|definitions?|references?|abbreviations?|introduction|general|requirements?|notes?)$", text, flags=re.I):
        return True
    alpha = [char for char in text if char.isalpha()]
    uppercase_ratio = sum(1 for char in alpha if char.isupper()) / max(1, len(alpha))
    title_case = sum(1 for word in text.split() if word[:1].isupper()) / max(1, len(text.split()))
    return bool(alpha) and len(text.split()) <= 14 and (
        uppercase_ratio > 0.72
        or (title_case > 0.72 and (not body_size or font_size >= body_size))
        or (body_size and font_size >= body_size * 1.18)
    )


def heading_level(text: str) -> int:
    match = SECTION_NUMBER_RE.match(text)
    if match:
        return min(6, match.group("num").count(".") + 1)
    if APPENDIX_RE.match(text):
        return 1
    letter = LETTER_HEADING_RE.match(text)
    if letter:
        return 2
    return 1


def section_identifier(text: str) -> str:
    match = SECTION_NUMBER_RE.match(text)
    if match:
        return match.group("num").rstrip(".")
    match = APPENDIX_RE.match(text)
    if match:
        return normalize_line(match.group("num")).upper()
    match = LETTER_HEADING_RE.match(text)
    if match:
        return match.group("num")
    return ""


def clean_heading_title(text: str) -> str:
    match = SECTION_NUMBER_RE.match(text)
    if match:
        return normalize_line(match.group("title"))
    match = APPENDIX_RE.match(text)
    if match:
        return normalize_line(match.group("title"))
    match = LETTER_HEADING_RE.match(text)
    if match:
        return normalize_line(match.group("title"))
    return normalize_line(text)


def merge_wrapped_lines(blocks: list[ContentBlock]) -> list[ContentBlock]:
    merged: list[ContentBlock] = []
    for block in blocks:
        if block.kind != "text" or not merged or merged[-1].kind != "text" or merged[-1].page != block.page:
            merged.append(block)
            continue
        previous = merged[-1]
        starts_new_subsection = bool(SECTION_NUMBER_RE.match(block.text))
        same_sectionish = (
            not starts_new_subsection
            and not previous.text.endswith((".", ":", ";", "?", "!"))
            and not looks_like_heading(block.text, block.font_size, 0)
        )
        if same_sectionish and len(previous.text) < 180 and len(block.text) < 180:
            if previous.bbox and block.bbox:
                previous.bbox = bbox_tuple_from_dict(union_bbox([previous, block]))
            previous.text = f"{previous.text} {block.text}"
        else:
            merged.append(block)
    return merged


def split_inline_sections(blocks: list[ContentBlock]) -> list[ContentBlock]:
    split_blocks: list[ContentBlock] = []
    for block in blocks:
        if block.kind != "text":
            split_blocks.append(block)
            continue
        if len(block.text) < 80:
            split_blocks.extend(split_numbered_heading_body(block))
            continue
        parts = [part.strip() for part in INLINE_SECTION_RE.split(block.text) if part.strip()]
        if len(parts) <= 1:
            split_blocks.extend(split_numbered_heading_body(block))
            continue
        for part in parts:
            split_blocks.extend(
                split_numbered_heading_body(
                    ContentBlock(
                        text=part,
                        page=block.page,
                        kind=block.kind,
                        bbox=block.bbox,
                        font_size=block.font_size,
                        metadata=dict(block.metadata),
                    )
                )
            )
    return split_blocks


def split_numbered_heading_body(block: ContentBlock) -> list[ContentBlock]:
    match = SECTION_NUMBER_RE.match(block.text)
    if not match:
        return [block]

    number = match.group("num")
    remainder = match.group("title").strip()
    tokens = remainder.split()
    if len(tokens) < 4:
        return [block]

    title_tokens: list[str] = []
    for token in tokens:
        clean = token.strip(",;:()[]")
        connector = clean.lower() in {"and", "or", "for", "of", "to", "with", "&"}
        title_word = bool(clean) and (clean[0].isupper() or clean.isupper() or connector)
        if not title_word:
            break
        title_tokens.append(token)
        if len(title_tokens) >= 8:
            break

    while title_tokens and title_tokens[-1].strip(",;:()[]").lower() in {"all", "the", "a", "an"}:
        title_tokens.pop()

    if len(title_tokens) < 2 or len(title_tokens) >= len(tokens):
        return [block]

    heading = f"{number} {' '.join(title_tokens)}"
    body = " ".join(tokens[len(title_tokens) :]).strip()
    if len(body) < 20:
        return [block]

    heading_block = ContentBlock(
        text=heading,
        page=block.page,
        kind=block.kind,
        bbox=block.bbox,
        font_size=block.font_size,
        metadata=dict(block.metadata),
    )
    body_block = ContentBlock(
        text=body,
        page=block.page,
        kind=block.kind,
        bbox=block.bbox,
        font_size=block.font_size,
        metadata=dict(block.metadata),
    )
    return [heading_block, body_block]


def render_table(rows: list[list[Any]], carry_forward: bool = False) -> str:
    normalized_rows = normalize_table_rows(rows, carry_forward=carry_forward)
    if not normalized_rows:
        return ""
    column_count = max(len(row) for row in normalized_rows)
    padded = [row + [""] * (column_count - len(row)) for row in normalized_rows]
    widths = [max(len(row[index]) for row in padded) for index in range(column_count)]
    rendered = ["TABLE:"]
    for row_index, row in enumerate(padded):
        rendered.append(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())
        if row_index == 0 and len(padded) > 1:
            rendered.append(" | ".join("-" * min(widths[index], 32) for index in range(column_count)))
    return "\n".join(rendered)


def normalize_table_rows(rows: list[list[Any]], carry_forward: bool = False) -> list[list[str]]:
    normalized_rows = [[normalize_line(str(cell or "")) for cell in row] for row in rows if any(str(cell or "").strip() for cell in row)]
    if not normalized_rows:
        return []
    column_count = max(len(row) for row in normalized_rows)
    padded = [row + [""] * (column_count - len(row)) for row in normalized_rows]
    if not carry_forward:
        return padded
    previous: list[str] = [""] * column_count
    carried: list[list[str]] = []
    for row_index, row in enumerate(padded):
        if row_index == 0:
            carried.append(row)
            previous = row[:]
            continue
        fixed = []
        for cell_index, cell in enumerate(row):
            if cell:
                fixed.append(cell)
                previous[cell_index] = cell
            else:
                fixed.append(previous[cell_index] if should_carry_table_cell(cell_index, row, previous) else "")
        carried.append(fixed)
    return carried


def should_carry_table_cell(cell_index: int, row: list[str], previous: list[str]) -> bool:
    if cell_index == 0 and any(row[1:]) and previous[cell_index]:
        return True
    return False


def promote_text_tables(blocks: list[ContentBlock]) -> list[ContentBlock]:
    promoted: list[ContentBlock] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.kind != "text" or not is_table_like_line(block.text):
            promoted.append(block)
            index += 1
            continue
        group = [block]
        cursor = index + 1
        while cursor < len(blocks):
            candidate = blocks[cursor]
            if candidate.kind != "text" or candidate.page != block.page:
                break
            if candidate.is_heading:
                break
            if is_table_like_line(candidate.text) or is_engineering_table_row(candidate.text) or looks_related_to_table_context(candidate.text, block.text):
                group.append(candidate)
                cursor += 1
                continue
            break
        if len([item for item in group if is_table_like_line(item.text)]) >= 2:
            rows = table_like_rows(group)
            rendered = render_table(rows, carry_forward=True)
            promoted.append(
                ContentBlock(
                    text=rendered,
                    page=block.page,
                    kind="table",
                    bbox=block.bbox,
                    font_size=block.font_size,
                    metadata={**block.metadata, "preserve_together": True, "source_format": "text_table"},
                )
            )
            index = cursor
        else:
            promoted.append(block)
            index += 1
    return promoted


def is_table_like_line(text: str) -> bool:
    normalized = normalize_line(text)
    if "|" in normalized and len([cell for cell in normalized.split("|") if cell.strip()]) >= 2:
        return True
    if looks_like_table_header(normalized):
        return True
    if is_engineering_table_row(normalized):
        return True
    if re.search(r"\s{2,}", normalized) and len(re.split(r"\s{2,}", normalized)) >= 3:
        return True
    return False


def table_like_rows(blocks: list[ContentBlock]) -> list[list[str]]:
    rows: list[list[str]] = []
    for block in blocks:
        text = normalize_line(block.text)
        if "|" in text:
            cells = [cell.strip() for cell in text.split("|")]
        elif looks_like_table_header(text):
            cells = known_table_header_cells(text)
        elif slope_row := parse_slope_table_row(text):
            cells = slope_row
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", text)]
        if any(cells):
            rows.append(cells)
    return rows


def looks_like_table_header(text: str) -> bool:
    normalized = normalize_line(text).lower()
    header_sets = [
        ("pipe direction", "slope", "remarks"),
        ("direction", "slope", "remarks"),
        ("size", "rating", "class"),
        ("tag", "description", "remarks"),
    ]
    return any(all(header in normalized for header in headers) for headers in header_sets)


def known_table_header_cells(text: str) -> list[str]:
    normalized = normalize_line(text).lower()
    if all(term in normalized for term in ("pipe direction", "slope", "remarks")):
        return ["Pipe Direction", "Slope", "Remarks"]
    if all(term in normalized for term in ("direction", "slope", "remarks")):
        return ["Direction", "Slope", "Remarks"]
    if all(term in normalized for term in ("size", "rating", "class")):
        return ["Size", "Rating", "Class"]
    if all(term in normalized for term in ("tag", "description", "remarks")):
        return ["Tag", "Description", "Remarks"]
    return [cell.strip() for cell in re.split(r"\s{2,}", text) if cell.strip()]


def is_engineering_table_row(text: str) -> bool:
    normalized = normalize_line(text)
    if parse_slope_table_row(normalized):
        return True
    return bool(re.search(r"\b\d+\s*[:/]\s*\d+\b|\b\d+(?:\.\d+)?\s*(?:mm|bar|psi|deg|°c|%)\b", normalized, flags=re.I))


def parse_slope_table_row(text: str) -> list[str] | None:
    normalized = normalize_line(text)
    match = re.match(r"^(?P<direction>[A-Za-z ]{3,60}?)\s+(?P<slope>\d+\s*[:/]\s*\d+|N/?A|Flat)\s+(?P<remarks>.+)$", normalized, flags=re.I)
    if not match:
        return None
    return [
        normalize_line(match.group("direction")),
        normalize_line(match.group("slope")),
        normalize_line(match.group("remarks")),
    ]


def excel_text_to_table_chunks(text: str) -> str:
    sheets = []
    for sheet in split_excel_sheets(text):
        lines = [line for line in sheet.splitlines() if normalize_line(line)]
        if not lines:
            continue
        title = lines[0] if lines[0].startswith("SHEET:") else ""
        data_lines = lines[1:] if title else lines
        rows = [[cell.strip() for cell in line.split("|")] for line in data_lines if "|" in line]
        rendered = render_table(rows) if rows else "\n".join(data_lines)
        sheets.append("\n".join(part for part in [title, rendered] if part).strip())
    return "\n\n".join(sheets)


def split_excel_sheets(text: str) -> list[str]:
    parts = re.split(r"(?=^SHEET:\s+.+$)", text, flags=re.MULTILINE)
    sheets = [part.strip() for part in parts if part.strip()]
    return sheets or [text.strip()]


def overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    overlap_x = max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0))
    overlap_y = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    overlap_area = overlap_x * overlap_y
    left_area = max(1.0, (left_x1 - left_x0) * (left_y1 - left_y0))
    return overlap_area / left_area


def tail_overlap(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n+", text) if paragraph.strip()]
    if not paragraphs:
        return ""
    candidates = []
    for paragraph in reversed(paragraphs):
        if is_bullet_or_numbered(paragraph) and 45 <= len(paragraph) <= OVERLAP_MAX_CHARS:
            candidates.append(paragraph)
            break
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if 45 <= len(sentence) <= OVERLAP_MAX_CHARS:
                candidates.append(sentence)
                break
        if candidates:
            break
    overlap = candidates[0] if candidates else paragraphs[-1]
    words = overlap.split()
    if len(words) < 8:
        return ""
    if len(overlap) <= OVERLAP_MAX_CHARS:
        return overlap
    return " ".join(words[-40:])


def current_char_count(blocks: list[ContentBlock]) -> int:
    return sum(len(block.text) for block in blocks)


def related_table_indexes(blocks: list[ContentBlock], table_index: int) -> list[int]:
    table = blocks[table_index]
    indexes = [table_index]
    scan_start = table_index + 1
    for index in range(scan_start, min(len(blocks), table_index + 8)):
        block = blocks[index]
        if block.kind == "text" and block.is_heading:
            break
        if block.page != table.page or block.section_path != table.section_path:
            break
        if block.kind == "table":
            between = blocks[indexes[-1] + 1 : index]
            if any(is_table_title_text(item.text) for item in between if item.kind == "text"):
                break
            if any(item.kind == "text" and looks_related_to_table(item.text, table.text) for item in between) or not between:
                indexes.append(index)
                continue
        if block.kind == "text" and not looks_related_to_table(block.text, table.text):
            break
    return indexes


def table_context_blocks(blocks: list[ContentBlock], table_indexes: list[int]) -> list[ContentBlock]:
    first_index = table_indexes[0]
    last_index = table_indexes[-1]
    table = blocks[first_index]
    grouped: list[ContentBlock] = []
    before_chars = 0
    for block in reversed(blocks[max(0, first_index - 5) : first_index]):
        if block.kind == "table":
            break
        if block.kind != "text" or block.page != table.page or block.section_path != table.section_path:
            continue
        if block.is_heading or looks_related_to_table(block.text, table.text):
            grouped.insert(0, block)
            before_chars += len(block.text)
        if is_table_title_text(block.text):
            break
        if before_chars >= TABLE_CONTEXT_CHARS:
            break
    grouped.extend(
        block
        for block in blocks[first_index : last_index + 1]
        if block.kind == "table" or (block.kind == "text" and looks_related_to_table(block.text, table.text))
    )
    after_chars = 0
    for block in blocks[last_index + 1 : last_index + 5]:
        if block.kind == "table" or is_table_title_text(block.text):
            break
        if block.kind != "text" or block.page != table.page or block.section_path != table.section_path:
            continue
        if looks_related_to_table(block.text, table.text):
            grouped.append(block)
            after_chars += len(block.text)
        if after_chars >= 360:
            break
    return grouped


def table_chunk_text(blocks: list[ContentBlock]) -> str:
    lines: list[str] = []
    previous = ""
    for block in blocks:
        for line in block.text.splitlines():
            normalized = normalize_line(line)
            if not normalized or normalized == previous:
                continue
            previous = normalized
            lines.append(line.rstrip())
    return "\n".join(lines).strip()


def looks_related_to_table(text: str, table_text: str) -> bool:
    terms = set(keywords(table_text)[:10])
    text_terms = set(keywords(text)[:12])
    return bool(terms & text_terms) or len(text) < 180 or text.endswith(":")


def looks_related_to_table_context(text: str, table_text: str) -> bool:
    normalized = normalize_line(text)
    if is_table_title_text(normalized):
        return True
    if re.match(r"^(note|remark|basis|legend|where|unless)\b", normalized, flags=re.I):
        return True
    terms = set(keywords(table_text)[:12])
    text_terms = set(keywords(normalized)[:12])
    return bool(terms & text_terms) and len(normalized) < 260


def is_table_title_text(text: str) -> bool:
    normalized = normalize_line(text).lower()
    return bool(
        re.match(r"^(table\s*)?[\w .-]*requirement\s*:", normalized)
        or re.match(r"^[\w .-]*\btable\s*:", normalized)
        or re.match(r"^table\s+\d+[a-z]?\s*[-:.]\s+.+", normalized)
        or re.search(r"\b(schedule|matrix|register|list|summary|requirements?)\b\s*[:.-]?$", normalized)
    )


def detect_printed_page_label(blocks: list[ContentBlock]) -> str | None:
    page_text = "\n".join(block.text for block in blocks)
    match = re.search(r"sheet\s+of\s+sheets\s+(\d+\s*/\s*\d+)", page_text, flags=re.IGNORECASE)
    if match:
        return normalize_line(match.group(1))
    match = re.search(r"\bpage\s+(\d+\s*(?:of|/)\s*\d+)\b", page_text, flags=re.IGNORECASE)
    if match:
        return normalize_line(match.group(1))
    return None


def table_metadata(text: str) -> dict[str, Any]:
    title = table_title(text)
    columns = table_columns(text)
    rows = table_rows(text)
    records = table_records(columns, rows)
    notes = table_notes(text)
    table_terms = sorted(set(keywords(" ".join([title, " ".join(columns), " ".join(rows)]))))
    quality = table_quality_score(columns, rows, title)
    return {
        "table_title": title,
        "table_columns": columns,
        "table_rows": rows[:200],
        "table_records": records[:200],
        "table_notes": notes[:20],
        "table_row_count": len(rows),
        "table_column_count": len(columns),
        "table_terms": table_terms[:30],
        "table_quality_score": quality,
        "table_integrity": table_integrity_label(quality),
    }


def table_title(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    for index, line in enumerate(lines):
        if re.match(r"^table\s+\d+[a-z]?\s*[-:.]\s+.+", line, flags=re.I):
            return line
        if line == "TABLE:" and index > 0:
            title_lines = []
            for candidate in reversed(lines[:index]):
                if candidate.startswith(("Note:", "TABLE:")):
                    continue
                if len(candidate) <= 180:
                    title_lines.insert(0, candidate)
                if len(title_lines) >= 2:
                    break
            return " ".join(title_lines).strip()
    return ""


def table_columns(text: str) -> list[str]:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    for index, line in enumerate(lines):
        if line == "TABLE:" and index + 1 < len(lines):
            return [cell.strip() for cell in lines[index + 1].split("|") if cell.strip()]
    compact = normalize_line(text)
    known_headers = [
        ["Pipe Direction", "Slope", "Remarks"],
        ["Direction", "Slope", "Remarks"],
        ["Size", "Rating", "Class"],
    ]
    for headers in known_headers:
        if all(header.lower() in compact.lower() for header in headers):
            return headers
    return []


def table_rows(text: str) -> list[str]:
    rows = []
    columns = [column.lower() for column in table_columns(text)]
    for line in text.splitlines():
        row = normalize_line(line)
        if "|" not in row:
            continue
        row_cells = [cell.strip().lower() for cell in row.split("|") if cell.strip()]
        is_separator = bool(re.fullmatch(r"[-|\s]+", row))
        is_header = bool(columns) and row_cells == columns
        if row and not is_separator and not is_header:
            rows.append(row)
    if not rows and looks_like_table_text(text):
        compact = normalize_line(text)
        for marker in ["Aft to Forward", "Forward to Aft", "Transverse"]:
            if marker.lower() in compact.lower():
                rows.append(marker)
    return rows


def table_records(columns: list[str], rows: list[str]) -> list[dict[str, str]]:
    if not columns:
        return []
    records: list[dict[str, str]] = []
    for row in rows:
        cells = [normalize_line(cell) for cell in row.split("|")]
        if not any(cells):
            continue
        padded = cells + [""] * (len(columns) - len(cells))
        records.append({columns[index]: padded[index] for index in range(min(len(columns), len(padded)))})
    return records


def table_notes(text: str) -> list[str]:
    notes = []
    for line in text.splitlines():
        normalized = normalize_line(line)
        if re.match(r"^(note|remark|basis|legend|where|unless)\b", normalized, flags=re.I):
            notes.append(normalized)
    return notes


def table_quality_score(columns: list[str], rows: list[str], title: str) -> float:
    score = 0.0
    if title:
        score += 0.2
    if len(columns) >= 2:
        score += 0.3
    if rows:
        score += 0.3
    if len(rows) >= 2:
        score += 0.1
    if columns and rows:
        width = len(columns)
        aligned = sum(1 for row in rows if abs(len([cell for cell in row.split("|")]) - width) <= 1)
        score += 0.1 * (aligned / max(1, len(rows)))
    return round(min(1.0, score), 4)


def table_integrity_label(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.6:
        return "usable"
    if score >= 0.35:
        return "weak"
    return "poor"


def looks_like_table_text(text: str) -> bool:
    normalized = normalize_line(text).lower()
    if "table:" in normalized:
        return True
    header_sets = [
        {"pipe direction", "slope", "remarks"},
        {"direction", "slope", "remarks"},
        {"size", "rating", "class"},
    ]
    return any(all(header in normalized for header in headers) for headers in header_sets)


def dedupe_chunks(chunks: list[RichChunk]) -> list[RichChunk]:
    kept: list[RichChunk] = []
    exact_signatures: set[str] = set()
    prefix_signatures: set[str] = set()
    fingerprints: list[dict[str, Any]] = []
    duplicate_count = 0
    for source_index, chunk in enumerate(chunks, start=1):
        normalized = normalize_for_dedupe(chunk.text)
        metadata = chunk.metadata or {}
        if len(normalized) < 40 and not (metadata.get("contains_table") or metadata.get("content_types")):
            duplicate_count += 1
            continue
        fingerprint = dedupe_fingerprint(chunk.text, metadata)
        exact_signature = fingerprint["exact_signature"]
        prefix_signature = fingerprint["prefix_signature"]
        duplicate_reason = ""
        duplicate_of = ""
        if exact_signature in exact_signatures:
            duplicate_reason = "exact_signature"
        elif prefix_signature in prefix_signatures and not dedupe_protected(metadata):
            duplicate_reason = "prefix_signature"
        else:
            near_duplicate = find_near_duplicate(fingerprint, fingerprints)
            if near_duplicate:
                duplicate_reason = str(near_duplicate["reason"])
                duplicate_of = str(near_duplicate["chunk_id"])
        if duplicate_reason:
            duplicate_count += 1
            continue
        exact_signatures.add(exact_signature)
        prefix_signatures.add(prefix_signature)
        chunk.metadata = {
            **metadata,
            "dedupe_schema_version": DEDUPE_SCHEMA_VERSION,
            "dedupe_status": "kept",
            "dedupe_source_index": source_index,
            "dedupe_exact_signature": exact_signature,
            "dedupe_prefix_signature": prefix_signature,
            "dedupe_word_count": fingerprint["word_count"],
            "dedupe_unique_word_count": fingerprint["unique_word_count"],
            "dedupe_protected": dedupe_protected(metadata),
            "dedupe_removed_before": duplicate_count,
        }
        fingerprints.append({**fingerprint, "chunk_id": chunk.metadata.get("chunk_id") or f"source-{source_index}"})
        kept.append(chunk)
    return kept


def dedupe_fingerprint(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    normalized = normalize_for_dedupe(text)
    words = normalized.split()
    unique_words = sorted(set(words))
    exact_seed = normalized
    prefix_seed = " ".join(words[:100])
    shingles = word_shingles(words)
    numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)?\b", text.lower())))
    table_key = " ".join((metadata.get("table_columns") or [])[:12]) if metadata.get("contains_table") else ""
    section_key = metadata.get("current_section_id") or metadata.get("section_title") or ""
    return {
        "normalized": normalized,
        "words": words,
        "word_set": set(words),
        "shingles": shingles,
        "numbers": numbers,
        "table_key": normalize_for_dedupe(table_key),
        "section_key": normalize_for_dedupe(str(section_key)),
        "exact_signature": hashlib.sha1(exact_seed.encode("utf-8")).hexdigest()[:20],
        "prefix_signature": hashlib.sha1(prefix_seed.encode("utf-8")).hexdigest()[:20],
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "contains_table": bool(metadata.get("contains_table")),
        "has_numeric_constraints": bool(metadata.get("has_numeric_constraints") or numbers),
    }


def find_near_duplicate(candidate: dict[str, Any], existing: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in existing:
        if not enough_dedupe_words(candidate, item):
            continue
        if dedupe_pair_protected(candidate, item):
            continue
        jaccard = set_similarity(candidate["word_set"], item["word_set"])
        containment = containment_similarity(candidate["word_set"], item["word_set"])
        shingle = set_similarity(candidate["shingles"], item["shingles"])
        if jaccard >= DEDUPE_JACCARD_THRESHOLD:
            return {"reason": "near_duplicate_jaccard", "chunk_id": item.get("chunk_id", ""), "score": jaccard}
        if containment >= DEDUPE_CONTAINMENT_THRESHOLD and shingle >= 0.72:
            return {"reason": "near_duplicate_containment", "chunk_id": item.get("chunk_id", ""), "score": containment}
        if shingle >= DEDUPE_SHINGLE_THRESHOLD:
            return {"reason": "near_duplicate_shingle", "chunk_id": item.get("chunk_id", ""), "score": shingle}
    return None


def enough_dedupe_words(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left.get("word_count") or 0) >= DEDUPE_MIN_WORDS and int(right.get("word_count") or 0) >= DEDUPE_MIN_WORDS


def dedupe_protected(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("contains_table") or metadata.get("has_numeric_constraints") or metadata.get("preserve_together"))


def dedupe_pair_protected(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("contains_table") or right.get("contains_table"):
        return left.get("table_key") != right.get("table_key") or left.get("numbers") != right.get("numbers")
    if left.get("has_numeric_constraints") or right.get("has_numeric_constraints"):
        return left.get("numbers") != right.get("numbers")
    left_section = left.get("section_key") or ""
    right_section = right.get("section_key") or ""
    if left_section and right_section and left_section != right_section:
        return True
    return False


def word_shingles(words: list[str], size: int = 5) -> set[str]:
    if len(words) < size:
        return set(words)
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def normalize_for_dedupe(text: str) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text.lower())
    text = re.sub(r"\W+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jaccard_words(left: str, right: str) -> float:
    left_words = set(left.split())
    right_words = set(right.split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def keywords(text: str) -> list[str]:
    return extract_keywords(text).keywords


def extract_keywords(text: str, blocks: list[ContentBlock] | None = None, contains_table: bool = False) -> KeywordSet:
    normalized = normalize_line(text)
    tokens = keyword_tokens(normalized)
    counts = Counter(token for token in tokens if keyword_keep(token))
    keyphrases = extract_keyphrases(normalized)
    acronyms = extract_acronyms(text)
    identifiers = extract_identifiers(text)
    standards = extract_standards(text)
    table_terms = extract_table_keyword_terms(text) if contains_table or looks_like_table_text(text) else []
    section_terms = extract_section_keyword_terms(blocks or [])
    exact_terms = exact_keyword_terms(text, identifiers, standards, acronyms)
    domain_terms = sorted({term for term in counts if term in ENGINEERING_DOMAIN_TERMS} | {term for term in table_terms if term in ENGINEERING_DOMAIN_TERMS})
    weighted_terms = weighted_keyword_terms(
        counts=counts,
        keyphrases=keyphrases,
        exact_terms=exact_terms,
        table_terms=table_terms,
        section_terms=section_terms,
        domain_terms=domain_terms,
    )
    keyword_list = rank_keyword_list(weighted_terms)
    return KeywordSet(
        keywords=keyword_list,
        keyphrases=keyphrases[:24],
        exact_terms=exact_terms[:30],
        acronyms=acronyms[:20],
        identifiers=identifiers[:30],
        standards=standards[:20],
        table_terms=table_terms[:30],
        section_terms=section_terms[:30],
        domain_terms=domain_terms[:30],
        weighted_terms=weighted_terms[:60],
    )


def keyword_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}", text):
        normalized = normalize_keyword_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def normalize_keyword_token(token: str) -> str:
    token = normalize_line(token).lower().strip("._-")
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us")):
        token = token[:-1]
    return token


def keyword_keep(token: str) -> bool:
    singular = token[:-1] if token.endswith("s") and len(token) > 4 else token
    return bool(token and len(token) >= 2 and token not in STOPWORDS and singular not in STOPWORDS and not token.isdigit())


def extract_keyphrases(text: str) -> list[str]:
    phrases: Counter[str] = Counter()
    for sentence in sentence_units(text):
        words = [normalize_keyword_token(word) for word in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}", sentence)]
        words = [word for word in words if keyword_keep(word)]
        for size in (4, 3, 2):
            for index in range(0, max(0, len(words) - size + 1)):
                phrase_words = words[index : index + size]
                if not phrase_words:
                    continue
                if not any(word in ENGINEERING_DOMAIN_TERMS or word in REQUIREMENT_TERMS or word in SAFETY_TERMS for word in phrase_words):
                    continue
                phrase = " ".join(phrase_words)
                phrases[phrase] += 1 + size * 0.1
    return [phrase for phrase, _ in phrases.most_common(30)]


def extract_acronyms(text: str) -> list[str]:
    acronyms: list[str] = []
    for match in re.finditer(r"\b[A-Z]{2,8}\b", text):
        acronym = match.group(0)
        if acronym not in acronyms and acronym.lower() not in STOPWORDS:
            acronyms.append(acronym)
        if len(acronyms) >= 30:
            break
    return acronyms


def extract_table_keyword_terms(text: str) -> list[str]:
    terms: list[str] = []
    rows = [line for line in text.splitlines() if line.strip()]
    for row in rows[:80]:
        if "|" not in row and not row.lower().startswith("table"):
            continue
        for cell in re.split(r"\s*\|\s*", row):
            for term in keyword_tokens(cell):
                if keyword_keep(term) and term not in terms:
                    terms.append(term)
    return terms


def extract_section_keyword_terms(blocks: list[ContentBlock]) -> list[str]:
    terms: list[str] = []
    for block in blocks:
        for value in block.section_path + [block.metadata.get("current_section_title", ""), block.metadata.get("parent_section_title", "")]:
            for term in keyword_tokens(str(value)):
                if keyword_keep(term) and term not in terms:
                    terms.append(term)
    return terms


def exact_keyword_terms(text: str, identifiers: list[str], standards: list[str], acronyms: list[str]) -> list[str]:
    exact_terms: list[str] = []
    for item in identifiers + standards + acronyms:
        if item and item not in exact_terms:
            exact_terms.append(item)
    quoted = re.findall(r"['\"]([^'\"]{3,80})['\"]", text)
    for item in quoted:
        normalized = normalize_line(item)
        if normalized and normalized not in exact_terms:
            exact_terms.append(normalized)
    return exact_terms


def weighted_keyword_terms(
    *,
    counts: Counter[str],
    keyphrases: list[str],
    exact_terms: list[str],
    table_terms: list[str],
    section_terms: list[str],
    domain_terms: list[str],
) -> list[dict[str, Any]]:
    weights: dict[str, float] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for term, count in counts.items():
        weights[term] = weights.get(term, 0.0) + min(3.0, 0.8 + count * 0.35)
        reasons[term].add("frequency")
    for phrase in keyphrases:
        weights[phrase] = weights.get(phrase, 0.0) + 2.4
        reasons[phrase].add("phrase")
    for term in exact_terms:
        key = term.lower()
        weights[key] = weights.get(key, 0.0) + 3.0
        reasons[key].add("exact")
    for term in table_terms:
        weights[term] = weights.get(term, 0.0) + 1.8
        reasons[term].add("table")
    for term in section_terms:
        weights[term] = weights.get(term, 0.0) + 1.6
        reasons[term].add("section")
    for term in domain_terms:
        weights[term] = weights.get(term, 0.0) + 2.0
        reasons[term].add("domain")
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"term": term, "weight": round(weight, 3), "reasons": sorted(reasons.get(term, []))}
        for term, weight in ranked
        if keyword_keep(term.replace(" ", "a"))
    ]


def rank_keyword_list(weighted_terms: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for item in weighted_terms:
        term = str(item.get("term") or "")
        if term and term not in terms:
            terms.append(term)
        if len(terms) >= 30:
            break
    return terms


def normalize_line(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_soft_hyphenation(text)
    text = normalize_ocr_punctuation(text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n|")


def normalize_table_text(text: str) -> str:
    lines = [normalize_table_line(line) for line in normalize_unicode(text).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def normalize_unicode(text: str) -> str:
    replacements = {
        "\x00": " ",
        "\ufeff": " ",
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u00a0": " ",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return unicodedata.normalize("NFKC", text)


def remove_soft_hyphenation(text: str) -> str:
    return re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)


def normalize_ocr_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\b([A-Za-z])\s*/\s*([A-Za-z])\b", r"\1/\2", text)
    text = re.sub(r"\b(\d+)\s*([°%])", r"\1\2", text)
    return text


def normalize_table_line(line: str) -> str:
    line = normalize_line(line)
    if "|" not in line:
        return line
    cells = [normalize_line(cell) for cell in line.split("|")]
    return " | ".join(cell for cell in cells)


def is_page_number(text: str) -> bool:
    normalized = normalize_line(text).lower()
    return bool(
        re.match(r"^(page|pg\.?|sheet)?\s*\d+\s*(of|/)\s*\d+$", normalized)
        or re.match(r"^(page|pg\.?|sheet)\s+\d+$", normalized)
        or re.match(r"^\d+$", normalized)
        or re.match(r"^-+\s*\d+\s*-+$", normalized)
    )


def is_watermark(text: str) -> bool:
    normalized = normalize_line(text).lower()
    return (
        normalized in {"confidential", "draft", "controlled copy", "uncontrolled copy", "preliminary", "for review"}
        or "this document is property" in normalized
        or "this document is the property" in normalized
        or "shall neither be shown to third parties" in normalized
        or "not to be reproduced" in normalized
        or "all rights reserved" in normalized
        or "commercial in confidence" in normalized
    )


def is_page_furniture(text: str) -> bool:
    normalized = normalize_line(text).lower()
    patterns = [
        "company logo",
        "contractor logo",
        "business name",
        "revision index",
        "issued for tender",
        "contractor prepared",
        "contractor verified",
        "company approved",
        "document n.",
        "document no.",
        "doc. no.",
        "project no.",
        "job n.",
        "sheet of sheets",
        "page of pages",
        "prepared by",
        "checked by",
        "approved by",
        "issued by",
        "signature",
    ]
    return len(text) < 320 and any(pattern in normalized for pattern in patterns)


def is_boilerplate(text: str) -> bool:
    normalized = normalize_line(text).lower()
    if len(normalized) > 420:
        return False
    patterns = [
        r"^rev(?:ision)?\s+[a-z0-9]\b",
        r"^date\s+description\s+prepared",
        r"^prepared\s+checked\s+approved",
        r"^contract(?:or)?\s+document",
        r"^company\s+document",
        r"^printed\s+on\b",
        r"^file\s*name\b",
        r"^electronic\s+file\b",
        r"^copyright\b",
        r"^disclaimer\b",
        r"^table\s+of\s+contents$",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def line_fingerprint(text: str) -> str:
    normalized = normalize_line(text).lower()
    normalized = re.sub(r"\b\d{1,4}([/-]\d{1,2}){1,2}\b", "<date>", normalized)
    normalized = re.sub(r"\b\d+\s*(?:of|/)\s*\d+\b", "<page>", normalized)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    normalized = re.sub(r"[^a-z0-9<>]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def remove_empty_repetition(blocks: list[ContentBlock]) -> list[ContentBlock]:
    cleaned: list[ContentBlock] = []
    previous_fingerprint = ""
    for block in blocks:
        fingerprint = line_fingerprint(block.text)
        if fingerprint and fingerprint == previous_fingerprint and len(block.text) < 160:
            continue
        cleaned.append(block)
        previous_fingerprint = fingerprint
    return cleaned


def is_generated_or_debug_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in CONTAMINATION_PATTERNS)


def is_page_furniture_table(text: str) -> bool:
    normalized = text.lower()
    markers = [
        "company document id",
        "contractor document id",
        "sheet of sheets",
        "validity status",
        "revision number",
    ]
    return sum(1 for marker in markers if marker in normalized) >= 2


def remove_front_matter(blocks: list[ContentBlock]) -> list[ContentBlock]:
    first_real_index = 0
    for index, block in enumerate(blocks):
        if block.kind == "table" and block.page <= 2:
            continue
        if block.is_heading or SECTION_NUMBER_RE.match(block.text) or re.match(r"^(purpose|scope|introduction)\b", block.text, re.I):
            first_real_index = index
            break
    if first_real_index > 0:
        return blocks[first_real_index:]
    return blocks
