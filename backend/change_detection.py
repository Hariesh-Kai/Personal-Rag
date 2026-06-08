from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Any


CHANGE_DETECTION_SCHEMA_VERSION = "engineering-change-detection-v1"


def change_detection_metadata(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_change_text(text)
    section_key = section_change_key(metadata)
    return {
        "change_detection_schema_version": CHANGE_DETECTION_SCHEMA_VERSION,
        "change_detection_ready": True,
        "change_detection_mode": "fingerprint_revision_section_compare",
        "change_text_hash": stable_hash(normalized),
        "change_structure_hash": stable_hash(
            "|".join(
                [
                    str(metadata.get("document_title") or ""),
                    str(metadata.get("document_identifier") or metadata.get("document_id") or ""),
                    str(metadata.get("revision") or ""),
                    str(metadata.get("current_section_id") or ""),
                    str(metadata.get("current_section_title") or metadata.get("section_title") or ""),
                    str(metadata.get("page_start") or ""),
                    str(metadata.get("page_end") or ""),
                ]
            )
        ),
        "change_section_key": section_key,
        "change_revision": metadata.get("revision") or "",
        "change_revision_sort_key": metadata.get("revision_sort_key") or "",
        "change_document_identifier": metadata.get("document_identifier") or metadata.get("document_id") or "",
        "change_word_count": len(re.findall(r"\S+", normalized)),
        "change_numeric_signature": stable_hash(" ".join(numeric_tokens(text))),
        "change_identifier_signature": stable_hash(" ".join(identifier_tokens(text, metadata))),
        "change_compare_status": "not_compared",
        "change_compare_scope": "pending_database_history",
        "change_detected": False,
        "change_similarity": 0.0,
        "change_summary": "No previous chunk comparison has been run yet.",
    }


def enrich_change_detection_from_history(
    conn: sqlite3.Connection,
    *,
    filename: str,
    index_session: str,
    document_id: int,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    prepared = change_detection_metadata(text, metadata)
    candidates = previous_chunk_candidates(conn, filename, index_session, document_id, prepared, limit=40)
    if not candidates:
        return {
            **metadata,
            **prepared,
            "change_compare_status": "new_or_no_history",
            "change_compare_scope": "same_filename_previous_documents",
            "change_summary": "No previous chunks with the same filename/section were found.",
        }
    current_tokens = token_set(text)
    best = None
    for candidate in candidates:
        candidate_metadata = safe_metadata(candidate["metadata"])
        candidate_text = candidate["text"] or ""
        similarity = jaccard(current_tokens, token_set(candidate_text))
        exact = prepared["change_text_hash"] == (candidate_metadata.get("change_text_hash") or stable_hash(normalize_change_text(candidate_text)))
        section_match = prepared["change_section_key"] and prepared["change_section_key"] == section_change_key(candidate_metadata)
        revision_changed = bool(prepared.get("change_revision") and prepared.get("change_revision") != candidate_metadata.get("revision"))
        candidate_score = similarity + (0.2 if exact else 0) + (0.08 if section_match else 0) + (0.04 if revision_changed else 0)
        item = {
            "candidate": candidate,
            "metadata": candidate_metadata,
            "similarity": round(similarity, 4),
            "exact": exact,
            "section_match": section_match,
            "revision_changed": revision_changed,
            "candidate_score": candidate_score,
        }
        if best is None or item["candidate_score"] > best["candidate_score"]:
            best = item
    assert best is not None
    status = compare_status(best)
    changed = status in {"changed", "revision_changed", "possibly_changed"}
    return {
        **metadata,
        **prepared,
        "change_compare_status": status,
        "change_compare_scope": "same_filename_previous_documents",
        "change_detected": changed,
        "change_similarity": best["similarity"],
        "change_previous_document_id": best["candidate"]["document_id"],
        "change_previous_chunk_id": best["candidate"]["id"],
        "change_previous_chunk_index": best["candidate"]["chunk_index"],
        "change_previous_index_session": best["candidate"]["index_session"],
        "change_previous_revision": best["metadata"].get("revision") or "",
        "change_previous_text_hash": best["metadata"].get("change_text_hash") or stable_hash(normalize_change_text(best["candidate"]["text"] or "")),
        "change_exact_match": bool(best["exact"]),
        "change_section_match": bool(best["section_match"]),
        "change_revision_changed": bool(best["revision_changed"]),
        "change_summary": change_summary(status, best),
    }


def previous_chunk_candidates(
    conn: sqlite3.Connection,
    filename: str,
    index_session: str,
    document_id: int,
    prepared: dict[str, Any],
    limit: int,
) -> list[sqlite3.Row]:
    section_key = prepared.get("change_section_key") or ""
    rows = conn.execute(
        """
        SELECT chunks.id, chunks.document_id, chunks.index_session, chunks.chunk_index, chunks.text, chunks.metadata, documents.filename
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE documents.filename = ?
          AND chunks.document_id != ?
        ORDER BY chunks.created_at DESC, chunks.id DESC
        LIMIT ?
        """,
        (filename, document_id, limit * 3),
    ).fetchall()
    if not section_key:
        return rows[:limit]
    section_rows = [row for row in rows if section_change_key(safe_metadata(row["metadata"])) == section_key]
    return (section_rows or rows)[:limit]


def compare_status(best: dict[str, Any]) -> str:
    if best["exact"]:
        return "unchanged"
    if best["revision_changed"] and best["similarity"] >= 0.82:
        return "revision_changed"
    if best["similarity"] >= 0.82:
        return "minor_or_format_change"
    if best["similarity"] >= 0.45:
        return "possibly_changed"
    return "changed"


def change_summary(status: str, best: dict[str, Any]) -> str:
    previous = best["candidate"]
    return (
        f"{status}; previous chunk {previous['id']} in session {previous['index_session']}; "
        f"similarity={best['similarity']}; exact={best['exact']}; revision_changed={best['revision_changed']}"
    )


def section_change_key(metadata: dict[str, Any]) -> str:
    return normalize_change_text(
        " ".join(
            str(value)
            for value in [
                metadata.get("current_section_id"),
                metadata.get("current_section_title"),
                metadata.get("section_title"),
                metadata.get("section_breadcrumb"),
            ]
            if value
        )
    )


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.:/-]{2,}", normalize_change_text(text)))


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)?\s*(?:mm|cm|m|bar|barg|psi|kpa|mpa|deg|%|c|hz|inch|in)?\b", text or "", flags=re.I)


def identifier_tokens(text: str, metadata: dict[str, Any]) -> list[str]:
    identifiers = list(metadata.get("technical_identifiers") or []) + list(metadata.get("keyword_identifiers") or [])
    identifiers.extend(re.findall(r"\b[A-Z]{2,}[A-Z0-9_-]*(?:-[A-Z0-9]+)+\b", text or ""))
    return sorted(set(str(item).upper() for item in identifiers if item))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def safe_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        import json

        return json.loads(raw or "{}")
    except Exception:
        return {}


def normalize_change_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\x00", " ")).strip().lower()


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:20]
