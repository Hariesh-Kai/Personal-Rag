from __future__ import annotations

import hashlib
from pathlib import Path

from django.conf import settings

from backend.rag_store import RagStore
from backend.structure_chunker import process_document_detailed


SKIP_NAME_MARKERS = {
    "debug",
    "evaluation",
    "framework_plan",
    "plan_for_hod",
    "rag_ingestion_framework",
}


def rebuild_from_uploads() -> dict:
    store = RagStore(settings.DATA_DIR / "rag.sqlite3", settings.DATA_DIR / "chunks.jsonl")
    uploads = sorted(settings.UPLOAD_DIR.glob("*"))
    selected = select_source_uploads(uploads)

    store.reset()
    results = []
    for upload in selected:
        filename = original_filename(upload)
        result = process_document_detailed(upload)
        chunks = result.chunks
        document_id = store.add_document(filename, content_type_for(upload), chunks)
        results.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunks": len(chunks),
                "storage": store.last_add_document_stats,
                "pipeline": result.metadata,
                "stages": result.stages,
            }
        )

    return {"documents": results, "document_count": len(results), "pipeline_version": "engineering-ingestion-v2"}


def select_source_uploads(paths: list[Path]) -> list[Path]:
    by_fingerprint: dict[tuple[str, str], Path] = {}
    for path in paths:
        if not path.is_file() or should_skip(path.name):
            continue
        key = (original_filename(path).lower(), file_hash(path))
        by_fingerprint[key] = path
    return list(by_fingerprint.values())


def should_skip(name: str) -> bool:
    normalized = name.lower()
    return any(marker in normalized for marker in SKIP_NAME_MARKERS)


def original_filename(path: Path) -> str:
    parts = path.name.split("-", 5)
    return parts[-1] if len(parts) == 6 else path.name


def file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"
