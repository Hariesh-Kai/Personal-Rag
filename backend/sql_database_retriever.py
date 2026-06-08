from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

SQL_DATABASE_SCHEMA_VERSION = "engineering-sql-database-retriever-v1"
SQL_STOPWORDS = {
    "about",
    "all",
    "and",
    "are",
    "chunks",
    "documents",
    "does",
    "files",
    "from",
    "give",
    "have",
    "how",
    "list",
    "many",
    "me",
    "show",
    "the",
    "this",
    "what",
    "where",
    "which",
    "with",
}


def sql_database_plan(query: str, route: Any) -> dict[str, Any]:
    normalized = normalize_sql_text(query)
    active = "sql_database" in set(getattr(route, "retrievers", ())) or is_database_question(normalized)
    intents = detect_intents(normalized)
    filters = parse_filters(query)
    terms = sql_terms(query)
    return {
        "schema": SQL_DATABASE_SCHEMA_VERSION,
        "active": active,
        "intents": intents,
        "filters": filters,
        "terms": terms,
        "route_primary": getattr(route, "primary", ""),
        "strategy": "safe_parameterized_sql_over_documents_chunks_logs_chat_history",
    }


def run_sql_database_retrieval(conn: sqlite3.Connection, index_session: str, plan: dict[str, Any], limit: int = 80) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    if not plan.get("active"):
        return [], {}, {**plan, "matched_chunk_ids": [], "database_facts": {}}

    facts = database_facts(conn, index_session, plan)
    rows = chunk_lookup_rows(conn, index_session, plan, limit=limit)
    chunk_scores = score_sql_rows(rows, plan)
    details = {
        **plan,
        "matched_chunk_ids": [row["id"] for row in rows[:limit]],
        "matched_chunk_count": len(rows),
        "database_facts": facts,
    }
    return rows, chunk_scores, details


def sql_database_candidate_score(plan: dict[str, Any], detail: dict[str, Any], row_id: Any, metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched": []}
    matched = []
    chunk_scores = detail.get("chunk_scores") or {}
    score = float(chunk_scores.get(str(row_id)) or 0.0)
    haystack = sql_haystack(metadata, text)
    filters = plan.get("filters") or {}

    for label, values in filters.items():
        if not values:
            continue
        if any(normalize_sql_text(value) in haystack for value in values):
            score += 0.025
            matched.append(label)

    term_hits = sum(1 for term in plan.get("terms") or [] if normalize_sql_text(term) in haystack)
    if term_hits:
        score += min(0.06, term_hits * 0.008)
        matched.append("terms")

    if metadata.get("contains_table") and "table_lookup" in plan.get("intents", []):
        score += 0.05
        matched.append("table")
    if (metadata.get("section_title") or metadata.get("current_section_id")) and "section_lookup" in plan.get("intents", []):
        score += 0.035
        matched.append("section")
    if (metadata.get("page_start") or metadata.get("page_label_start")) and "page_lookup" in plan.get("intents", []):
        score += 0.025
        matched.append("page")

    return round(min(0.18, score), 5), {
        "schema": SQL_DATABASE_SCHEMA_VERSION,
        "active": True,
        "matched": unique_preserve(matched),
        "term_hits": term_hits,
        "database_facts": detail.get("database_facts") or {},
        "matched_by_sql_row": str(row_id) in (detail.get("chunk_scores") or {}),
    }


def database_facts(conn: sqlite3.Connection, index_session: str, plan: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    intents = set(plan.get("intents") or [])
    if "document_count" in intents or "document_listing" in intents:
        facts["document_count"] = int(
            conn.execute("SELECT COUNT(*) AS count FROM documents WHERE index_session = ?", (index_session,)).fetchone()["count"]
        )
    if "chunk_count" in intents:
        facts["chunk_count"] = int(
            conn.execute("SELECT COUNT(*) AS count FROM chunks WHERE index_session = ?", (index_session,)).fetchone()["count"]
        )
    if "document_listing" in intents:
        rows = conn.execute(
            """
            SELECT id, filename, content_type, created_at
            FROM documents
            WHERE index_session = ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (index_session,),
        ).fetchall()
        facts["documents"] = [dict(row) for row in rows]
    if "recent_retrievals" in intents:
        rows = conn.execute(
            """
            SELECT question, mode, overall_score, grade, source_count, top_source, created_at
            FROM retrieval_logs
            WHERE index_session = ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (index_session,),
        ).fetchall()
        facts["recent_retrievals"] = [dict(row) for row in rows]
    if "recent_chat" in intents:
        rows = conn.execute(
            """
            SELECT question, resolved_question, created_at
            FROM chat_history
            WHERE index_session = ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (index_session,),
        ).fetchall()
        facts["recent_chat"] = [dict(row) for row in rows]
    return facts


def chunk_lookup_rows(conn: sqlite3.Connection, index_session: str, plan: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    filters = plan.get("filters") or {}
    terms = plan.get("terms") or []
    intents = set(plan.get("intents") or [])
    where = ["chunks.index_session = ?", "documents.index_session = ?"]
    params: list[Any] = [index_session, index_session]

    if filters.get("filenames"):
        filename_clauses = []
        for filename in filters["filenames"][:8]:
            filename_clauses.append("LOWER(documents.filename) LIKE ?")
            params.append(f"%{normalize_sql_text(filename)}%")
        where.append("(" + " OR ".join(filename_clauses) + ")")
    if filters.get("sections"):
        section_clauses = []
        for section in filters["sections"][:12]:
            section_clauses.extend(["LOWER(chunks.metadata) LIKE ?", "LOWER(chunks.text) LIKE ?"])
            value = f"%{normalize_sql_text(section)}%"
            params.extend([value, value])
        where.append("(" + " OR ".join(section_clauses) + ")")
    if filters.get("standards"):
        standard_clauses = []
        for standard in filters["standards"][:12]:
            standard_clauses.extend(["LOWER(chunks.metadata) LIKE ?", "LOWER(chunks.text) LIKE ?"])
            value = f"%{normalize_sql_text(standard)}%"
            params.extend([value, value])
        where.append("(" + " OR ".join(standard_clauses) + ")")
    if filters.get("pages"):
        page_clauses = []
        for page in filters["pages"][:12]:
            page_clauses.append("chunks.metadata LIKE ?")
            params.append(f"%\"page_start\": {page}%")
        where.append("(" + " OR ".join(page_clauses) + ")")
    if "table_lookup" in intents:
        where.append("(chunks.metadata LIKE '%\"contains_table\": true%' OR LOWER(chunks.text) LIKE '%table%')")
    if "safety_lookup" in intents:
        where.append("(chunks.metadata LIKE '%\"safety_critical\": true%' OR LOWER(chunks.text) LIKE '%safety%' OR LOWER(chunks.text) LIKE '%shall%')")

    if terms:
        term_clauses = []
        for term in terms[:10]:
            term_clauses.extend(["LOWER(chunks.text) LIKE ?", "LOWER(chunks.metadata) LIKE ?"])
            value = f"%{normalize_sql_text(term)}%"
            params.extend([value, value])
        where.append("(" + " OR ".join(term_clauses) + ")")

    sql = f"""
        SELECT chunks.id, chunks.document_id, documents.filename, chunks.chunk_index,
               chunks.text, chunks.embedding, chunks.metadata
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE {" AND ".join(where)}
        ORDER BY chunks.document_id DESC, chunks.chunk_index ASC
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def score_sql_rows(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, float]:
    scores = {}
    terms = plan.get("terms") or []
    filters = plan.get("filters") or {}
    intents = set(plan.get("intents") or [])
    for row in rows:
        metadata = row_metadata(row)
        haystack = sql_haystack(metadata, row.get("text") or "")
        score = 0.035
        term_hits = sum(1 for term in terms if normalize_sql_text(term) in haystack)
        score += min(0.07, term_hits * 0.012)
        if any(normalize_sql_text(filename) in normalize_sql_text(row.get("filename")) for filename in filters.get("filenames") or []):
            score += 0.04
        if "table_lookup" in intents and metadata.get("contains_table"):
            score += 0.05
        if "section_lookup" in intents and (metadata.get("section_title") or metadata.get("current_section_id")):
            score += 0.03
        if "document_listing" in intents:
            score += 0.015
        scores[str(row["id"])] = round(min(0.16, score), 5)
    return scores


def detect_intents(normalized: str) -> list[str]:
    intents = []
    if re.search(r"\b(how many|count|number of)\b.*\b(documents?|files?)\b", normalized):
        intents.append("document_count")
    if re.search(r"\b(how many|count|number of)\b.*\b(chunks?|sections?)\b", normalized):
        intents.append("chunk_count")
    if re.search(r"\b(list|show|which)\b.*\b(documents?|files?)\b", normalized):
        intents.append("document_listing")
    if re.search(r"\b(section|clause|paragraph)\b", normalized):
        intents.append("section_lookup")
    if re.search(r"\b(page|pg)\b", normalized):
        intents.append("page_lookup")
    if re.search(r"\b(table|row|column|matrix)\b", normalized):
        intents.append("table_lookup")
    if re.search(r"\b(safety|hazard|fire|explosion|shutdown|relief|shall|must)\b", normalized):
        intents.append("safety_lookup")
    if re.search(r"\b(recent retrieval|retrieval logs?|logs?|previous answers?)\b", normalized):
        intents.append("recent_retrievals")
    if re.search(r"\b(chat history|previous questions?|conversation)\b", normalized):
        intents.append("recent_chat")
    return unique_preserve(intents) or ["chunk_lookup"]


def parse_filters(query: str) -> dict[str, list[str]]:
    quoted = [item.strip() for item in re.findall(r"['\"]([^'\"]{2,160})['\"]", query)]
    filenames = re.findall(r"\b(?:file|filename|document|doc)\s*(?:named|called|id|:)?\s*([A-Za-z0-9_. -]{4,140}\.(?:pdf|docx|xlsx|txt|md|csv|html?))", query, flags=re.I)
    sections = re.findall(r"\b(?:section|sec\.?|clause|paragraph|para\.?)\s*((?:\d+\.)*\d+[A-Z]?|[A-Za-z][A-Za-z0-9 /&-]{2,80})", query, flags=re.I)
    pages = re.findall(r"\b(?:page|pg\.?)\s*(\d+)\b", query, flags=re.I)
    standards = re.findall(r"\b(?:ASME\s+[A-Z]\d+(?:\.\d+)*[A-Z0-9.-]*|NORSOK\s+[A-Z]-\d+[A-Z0-9.-]*|API\s+\d+[A-Z0-9.-]*|ISO\s+\d+[A-Z0-9:.-]*|IEC\s+\d+[A-Z0-9:.-]*)\b", query, flags=re.I)
    return {
        "filenames": unique_preserve(quoted + filenames),
        "sections": unique_preserve(sections),
        "pages": unique_preserve(pages),
        "standards": unique_preserve(standards),
    }


def is_database_question(normalized: str) -> bool:
    return bool(re.search(r"\b(database|sqlite|sql|documents?|files?|chunks?|retrieval logs?|chat history|how many|count|list uploaded)\b", normalized))


def sql_terms(query: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", query):
        normalized = normalize_sql_text(token)
        if normalized and normalized not in SQL_STOPWORDS:
            terms.append(normalized)
    return unique_preserve(terms)[:24]


def sql_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_sql_text(
        " ".join(
            [
                text,
                metadata.get("filename") or "",
                metadata.get("document_title") or "",
                metadata.get("document_identifier") or "",
                metadata.get("section_title") or "",
                metadata.get("current_section_title") or "",
                metadata.get("current_section_id") or "",
                metadata.get("table_title") or "",
                " ".join(metadata.get("section_path") or []),
                " ".join(metadata.get("standards") or []),
                " ".join(metadata.get("technical_identifiers") or []),
                " ".join(metadata.get("engineering_entities") or []),
                " ".join(metadata.get("keywords") or []),
            ]
        )
    )


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            return json.loads(metadata or "{}")
        except json.JSONDecodeError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def normalize_sql_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.:/&+-]+", " ", str(value or "").lower())).strip()


def unique_preserve(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
