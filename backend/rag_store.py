from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .embeddings import embed_passage, embed_query, embedding_status
from .question_types import QuestionProfile, classify_question, profile_payload
from .retriever_router import RetrieverRoute, expanded_queries, route_capability_score, route_for_question, route_payload

VECTOR_SIZE = 768
TABLE_QUERY_TERMS = {
    "aft",
    "class",
    "closed",
    "column",
    "direction",
    "drain",
    "forward",
    "line",
    "list",
    "note",
    "open",
    "pipe",
    "rating",
    "remarks",
    "row",
    "slope",
    "table",
    "transverse",
    "value",
}
QUERY_STOPWORDS = {
    "about",
    "answer",
    "are",
    "do",
    "does",
    "different",
    "each",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "it",
    "is",
    "me",
    "need",
    "of",
    "paragraph",
    "short",
    "show",
    "tell",
    "the",
    "this",
    "to",
    "types",
    "we",
    "what",
    "which",
}
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


class RagStore:
    def __init__(self, db_path: Path, chunks_log_path: Path):
        self.db_path = db_path
        self.chunks_log_path = chunks_log_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_document(self, filename: str, content_type: str | None, chunks: list[str | dict]) -> int:
        index_session = self.current_index_session()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO documents(filename, content_type, index_session, created_at) VALUES (?, ?, ?, ?)",
                (filename, content_type, index_session, _now()),
            )
            document_id = int(cursor.lastrowid)
            rows = []
            seen_signatures: set[str] = set()
            seen_normalized: list[str] = []
            for index, chunk in enumerate(chunks):
                text, metadata = normalize_chunk(chunk)
                if is_generated_or_debug_text(text):
                    continue
                signature = dedupe_signature(text)
                if signature in seen_signatures:
                    continue
                normalized = normalize_for_dedupe(text)
                if any(jaccard_words(normalized, existing) > 0.9 for existing in seen_normalized):
                    continue
                seen_signatures.add(signature)
                seen_normalized.append(normalized)
                metadata = {**metadata, "filename": filename}
                embedding_text = searchable_text(text, metadata)
                embedding = embed_passage(embedding_text)
                rows.append((document_id, index_session, index, text, json.dumps(embedding), json.dumps(metadata), _now()))
            conn.executemany(
                """
                INSERT INTO chunks(document_id, index_session, chunk_index, text, embedding, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        self.export_chunks_log()
        return document_id

    def search(self, query: str, limit: int = 5, profile: QuestionProfile | None = None) -> list[dict]:
        profile = profile or classify_question(query)
        route = route_for_question(profile, query)
        limit = max(limit, profile.context_limit)
        index_session = self.current_index_session()
        query_terms = query_keywords(query)
        route_queries = expanded_queries(query, route)
        expanded_query = "\n".join(self.expand_query(route_query, query_keywords(route_query)) for route_query in route_queries)
        query_vector = embed_query(expanded_query)
        important_terms = important_query_terms(query_terms)
        query_phrases = query_exact_phrases(query)
        table_intent = is_table_query(query_terms)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunks.id, chunks.document_id, documents.filename, chunks.chunk_index,
                       chunks.text, chunks.embedding, chunks.metadata
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.index_session = ? AND documents.index_session = ?
                """
                ,
                (index_session, index_session),
            ).fetchall()
        scored = []
        for row in rows:
            if is_generated_or_debug_text(row["text"]):
                continue
            vector = json.loads(row["embedding"])
            metadata = json.loads(row["metadata"] or "{}")
            vector_score = cosine_similarity(query_vector, vector)
            keyword_score = keyword_match_score(important_terms, row["text"], metadata)
            phrase_score = phrase_match_score(query_phrases, row["text"], metadata)
            rerank_score = rerank(important_terms, row["text"], metadata, vector_score)
            table_score = table_match_score(important_terms, query_phrases, row["text"], metadata)
            metadata_score = metadata_match_score(important_terms, query_phrases, metadata)
            capability_score = route_capability_score(route, query, row["text"], metadata)
            score = route_score(route, vector_score, keyword_score, phrase_score, rerank_score, table_score, metadata_score)
            score += capability_score
            score += profile_score_boost(profile, route, important_terms, query_phrases, row["text"], metadata)
            scored.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "metadata": metadata,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "phrase_score": phrase_score,
                    "table_score": table_score,
                    "metadata_score": metadata_score,
                    "capability_score": capability_score,
                    "retriever_route": route_payload(route),
                    "score": score,
                }
            )
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
        if table_intent or profile.type_id == "table_numeric" or route.primary == "table":
            ranked = merge_table_candidates(ranked, limit)
        if route.include_siblings or is_type_query(query) or profile.type_id in {"enumeration", "multi_part"}:
            ranked = merge_section_siblings(ranked, limit)
        if route.window_size:
            ranked = expand_window_candidates(ranked, scored, route.window_size, limit)
        if route.diversify_sections or profile.type_id in {"comparison", "cross_section", "multi_document", "conflict_detection", "document_coverage"}:
            ranked = diversify_sections(ranked, limit)
        return ranked[:limit]

    def log_retrieval(self, question: str, answer_result: dict, contexts: list[dict]) -> None:
        index_session = self.current_index_session()
        quality = answer_result.get("quality") or {}
        payload = {
            "mode": answer_result.get("mode"),
            "sources": summarize_contexts(contexts),
            "quality": quality,
            "question_profile": answer_result.get("question_profile") or profile_payload(classify_question(question)),
            "retriever_route": contexts[0].get("retriever_route") if contexts else None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_logs(
                    index_session, question, answer, mode, overall_score, grade,
                    source_count, top_source, payload, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index_session,
                    question,
                    answer_result.get("answer", ""),
                    answer_result.get("mode", ""),
                    float(quality.get("overall_score") or 0),
                    str(quality.get("grade") or ""),
                    len(contexts),
                    source_name(contexts[0]) if contexts else "",
                    json.dumps(payload, ensure_ascii=False),
                    _now(),
                ),
            )
            conn.commit()

    def retrieval_logs(self, limit: int = 25) -> list[dict]:
        index_session = self.current_index_session()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, index_session, question, answer, mode, overall_score, grade,
                       source_count, top_source, payload, created_at
                FROM retrieval_logs
                WHERE index_session = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (index_session, limit),
            ).fetchall()
        logs = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.get("payload") or "{}")
            logs.append(item)
        return logs

    def list_documents(self) -> list[dict]:
        index_session = self.current_index_session()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT documents.id, documents.filename, documents.content_type, documents.index_session, documents.created_at,
                       COUNT(chunks.id) AS chunk_count
                FROM documents
                LEFT JOIN chunks ON chunks.document_id = documents.id AND chunks.index_session = documents.index_session
                WHERE documents.index_session = ?
                GROUP BY documents.id
                ORDER BY documents.created_at DESC
                """,
                (index_session,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reset(self) -> str:
        index_session = self.create_index_session()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('chunks', 'documents')")
            conn.commit()
        self.export_chunks_log()
        return index_session

    def list_chunks(self, document_id: int | None = None) -> list[dict]:
        index_session = self.current_index_session()
        sql = """
            SELECT chunks.id, chunks.document_id, documents.filename, chunks.chunk_index,
                   chunks.text, chunks.metadata, chunks.created_at
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.index_session = ? AND documents.index_session = ?
        """
        params: tuple[str, str] | tuple[str, str, int] = (index_session, index_session)
        if document_id is not None:
            sql += " AND chunks.document_id = ?"
            params = (index_session, index_session, document_id)
        sql += " ORDER BY chunks.document_id DESC, chunks.chunk_index ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.get("metadata") or "{}")
            result.append(item)
        return result

    def export_chunks_log(self) -> None:
        chunks = self.list_chunks()
        with self.chunks_log_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def current_index_session(self) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM index_sessions WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row:
                return str(row["id"])
        return self.create_index_session()

    def create_index_session(self) -> str:
        session_id = _session_id()
        with self._connect() as conn:
            conn.execute("UPDATE index_sessions SET active = 0")
            conn.execute(
                "INSERT INTO index_sessions(id, active, created_at) VALUES (?, 1, ?)",
                (session_id, _now()),
            )
            conn.commit()
        return session_id

    def index_status(self) -> dict:
        index_session = self.current_index_session()
        with self._connect() as conn:
            doc_count = conn.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE index_session = ?",
                (index_session,),
            ).fetchone()["count"]
            chunk_count = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE index_session = ?",
                (index_session,),
            ).fetchone()["count"]
            session = conn.execute(
                "SELECT created_at FROM index_sessions WHERE id = ?",
                (index_session,),
            ).fetchone()
        return {
            "index_session": index_session,
            "index_created_at": session["created_at"] if session else None,
            "documents": int(doc_count),
            "chunks": int(chunk_count),
            **embedding_status(),
        }

    def expand_query(self, query: str, query_terms: list[str]) -> str:
        dictionary = self.auto_dictionary(query_terms)
        expansions: list[str] = []
        for term in query_terms:
            expansions.extend(dictionary.get(term, []))
        if not expansions:
            return query
        unique_expansions = list(dict.fromkeys(expansions))[:12]
        return f"{query}\nRelated engineering terms: {' '.join(unique_expansions)}"

    def auto_dictionary(self, query_terms: list[str]) -> dict[str, list[str]]:
        index_session = self.current_index_session()
        normalized_query_terms = {normalize_term(term) for term in query_terms if normalize_term(term)}
        if not normalized_query_terms:
            return {}
        dictionary: dict[str, Counter[str]] = {term: Counter() for term in normalized_query_terms}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT text, metadata
                FROM chunks
                WHERE index_session = ?
                """,
                (index_session,),
            ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            terms = chunk_dictionary_terms(row["text"], metadata)
            normalized_terms = {normalize_term(term) for term in terms if normalize_term(term)}
            matched = normalized_query_terms & normalized_terms
            if not matched:
                continue
            related = [term for term in normalized_terms if term not in normalized_query_terms]
            for query_term in matched:
                dictionary[query_term].update(related)
        return {
            term: [candidate for candidate, _ in counts.most_common(8)]
            for term, counts in dictionary.items()
            if counts
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_sessions (
                    id TEXT PRIMARY KEY,
                    active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    index_session TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    index_session TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    index_session TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    mode TEXT,
                    overall_score REAL NOT NULL DEFAULT 0,
                    grade TEXT,
                    source_count INTEGER NOT NULL DEFAULT 0,
                    top_source TEXT,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    index_session TEXT NOT NULL,
                    question TEXT NOT NULL,
                    resolved_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    profile TEXT NOT NULL DEFAULT '{}',
                    source_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = [row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]
            if "metadata" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            document_columns = [row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()]
            if "index_session" not in document_columns:
                conn.execute("ALTER TABLE documents ADD COLUMN index_session TEXT NOT NULL DEFAULT ''")
            chunk_columns = [row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]
            if "index_session" not in chunk_columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN index_session TEXT NOT NULL DEFAULT ''")
            active = conn.execute("SELECT id FROM index_sessions WHERE active = 1 LIMIT 1").fetchone()
            if active is None:
                session_id = _session_id()
                conn.execute(
                    "INSERT OR IGNORE INTO index_sessions(id, active, created_at) VALUES (?, 1, ?)",
                    (session_id, _now()),
                )
                conn.execute("UPDATE documents SET index_session = ? WHERE index_session = ''", (session_id,))
                conn.execute("UPDATE chunks SET index_session = ? WHERE index_session = ''", (session_id,))
            conn.commit()

    def add_chat_turn(self, question: str, resolved_question: str, answer_result: dict, contexts: list[dict]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_history(index_session, question, resolved_question, answer, profile, source_ids, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.current_index_session(),
                    question,
                    resolved_question,
                    answer_result.get("answer", ""),
                    json.dumps(answer_result.get("question_profile") or {}, ensure_ascii=False),
                    json.dumps([item.get("id") for item in contexts if item.get("id")], ensure_ascii=False),
                    _now(),
                ),
            )
            conn.commit()

    def recent_chat_turns(self, limit: int = 4) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, resolved_question, answer, profile, source_ids, created_at
                FROM chat_history
                WHERE index_session = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self.current_index_session(), limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["profile"] = json.loads(item.get("profile") or "{}")
            item["source_ids"] = json.loads(item.get("source_ids") or "[]")
            result.append(item)
        return result


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def summarize_contexts(contexts: list[dict]) -> list[dict]:
    summaries = []
    for item in contexts:
        metadata = item.get("metadata") or {}
        summaries.append(
            {
                "filename": item.get("filename"),
                "chunk_index": item.get("chunk_index"),
                "score": round(float(item.get("score", 0)), 4),
                "vector_score": round(float(item.get("vector_score", 0)), 4),
                "keyword_score": round(float(item.get("keyword_score", 0)), 4),
                "phrase_score": round(float(item.get("phrase_score", 0)), 4),
                "table_score": round(float(item.get("table_score", 0)), 4),
                "capability_score": round(float(item.get("capability_score", 0)), 4),
                "section": metadata.get("section_title") or "",
                "table_title": metadata.get("table_title") or "",
                "contains_table": bool(metadata.get("contains_table")),
            }
        )
    return summaries


def source_name(item: dict) -> str:
    metadata = item.get("metadata") or {}
    label = metadata.get("table_title") or metadata.get("section_title") or "No section"
    return f"{item.get('filename')} #{item.get('chunk_index')} {label}"


def normalize_chunk(chunk: str | dict) -> tuple[str, dict]:
    if isinstance(chunk, dict):
        return str(chunk.get("text", "")).strip(), dict(chunk.get("metadata") or {})
    return str(chunk).strip(), {}


def searchable_text(text: str, metadata: dict) -> str:
    section = " > ".join(metadata.get("section_path") or [])
    keywords = " ".join(metadata.get("keywords") or [])
    overlap_context = metadata.get("overlap_context") or ""
    table_title = metadata.get("table_title") or ""
    table_columns = " ".join(metadata.get("table_columns") or [])
    table_rows = " ".join(metadata.get("table_rows") or [])
    table_terms = " ".join(metadata.get("table_terms") or [])
    return f"{section}\n{keywords}\n{table_title}\n{table_columns}\n{table_rows}\n{table_terms}\n{overlap_context}\n{text}"


def query_keywords(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_.-]{2,}", query)]


def important_query_terms(query_terms: list[str]) -> list[str]:
    terms = []
    for term in query_terms:
        if term in QUERY_STOPWORDS:
            continue
        if term.endswith("s") and len(term) > 4:
            term = term[:-1]
        terms.append(term)
    return terms or query_terms


def query_exact_phrases(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    phrases: list[str] = []
    known = [
        "open and overboard drain",
        "closed drain",
        "flare piping headers",
        "flare header",
        "aft to forward",
        "forward to aft",
        "pipe direction",
        "slope requirement",
    ]
    for phrase in known:
        if phrase in normalized:
            phrases.append(phrase)
    quoted = re.findall(r"['\"]([^'\"]{3,80})['\"]", query)
    phrases.extend(re.sub(r"\s+", " ", item.lower()).strip() for item in quoted)
    return list(dict.fromkeys(phrases))


def keyword_match_score(query_terms: list[str], text: str, metadata: dict) -> float:
    if not query_terms:
        return 0.0
    haystack = searchable_text(text, metadata).lower()
    exact = sum(1 for term in query_terms if term in haystack)
    phrase_bonus = 1 if " ".join(query_terms) in haystack else 0
    return min(1.0, (exact + phrase_bonus) / max(1, len(query_terms)))


def phrase_match_score(query_phrases: list[str], text: str, metadata: dict) -> float:
    if not query_phrases:
        return 0.0
    haystack = searchable_text(text, metadata).lower()
    hits = sum(1 for phrase in query_phrases if phrase in haystack)
    return hits / len(query_phrases)


def rerank(query_terms: list[str], text: str, metadata: dict, dense_score: float = 0.0) -> float:
    section = " ".join(metadata.get("section_path") or []).lower()
    content_type_bonus = 0.16 if metadata.get("contains_table") and is_table_query(query_terms) else 0
    section_hits = sum(1 for term in query_terms if term in section)
    early_text = text[:500].lower()
    early_hits = sum(1 for term in query_terms if term in early_text)
    lexical = ((section_hits * 1.4) + early_hits) / max(1, len(query_terms))
    dense = (dense_score + 1.0) / 2.0
    return min(1.0, (lexical * 0.55) + (dense * 0.45) + content_type_bonus)


def metadata_match_score(query_terms: list[str], query_phrases: list[str], metadata: dict) -> float:
    metadata_text = " ".join(
        [
            " ".join(metadata.get("section_path") or []),
            metadata.get("section_title") or "",
            metadata.get("parent_section") or "",
            metadata.get("table_title") or "",
            " ".join(metadata.get("keywords") or []),
            metadata.get("revision") or "",
            metadata.get("document_identifier") or "",
            metadata.get("validity_status") or "",
            metadata.get("filename") or "",
        ]
    ).lower()
    if not metadata_text:
        return 0.0
    term_hits = sum(1 for term in query_terms if term in metadata_text)
    phrase_hits = sum(1 for phrase in query_phrases if phrase in metadata_text)
    return min(1.0, (term_hits / max(1, len(query_terms))) * 0.75 + (phrase_hits / max(1, len(query_phrases) or 1)) * 0.25)


def route_score(
    route: RetrieverRoute,
    vector_score: float,
    keyword_score: float,
    phrase_score: float,
    rerank_score: float,
    table_score: float,
    metadata_score: float,
) -> float:
    return (
        route.semantic_weight * vector_score
        + route.keyword_weight * keyword_score
        + route.phrase_weight * phrase_score
        + route.rerank_weight * rerank_score
        + route.table_weight * table_score
        + route.metadata_weight * metadata_score
    )


def is_table_query(query_terms: list[str]) -> bool:
    if not query_terms:
        return False
    term_set = set(query_terms)
    if term_set & TABLE_QUERY_TERMS:
        return True
    return any(term.endswith("s") and term[:-1] in TABLE_QUERY_TERMS for term in query_terms)


def table_match_score(query_terms: list[str], query_phrases: list[str], text: str, metadata: dict) -> float:
    if not query_terms or not metadata.get("contains_table"):
        return 0.0
    title = (metadata.get("table_title") or "").lower()
    rows = " ".join(metadata.get("table_rows") or []).lower()
    table_haystack = " ".join(
        [
            metadata.get("table_title") or "",
            " ".join(metadata.get("table_columns") or []),
            " ".join(metadata.get("table_rows") or []),
            " ".join(metadata.get("table_terms") or []),
            text,
        ]
    ).lower()
    exact_hits = sum(1 for term in query_terms if term in table_haystack)
    column_hits = sum(1 for column in metadata.get("table_columns") or [] if any(term in column.lower() for term in query_terms))
    title_hits = sum(1 for term in query_terms if term in title)
    row_hits = sum(1 for term in query_terms if term in rows)
    phrase_hits = sum(1 for phrase in query_phrases if phrase in title or phrase in rows)
    row_density = min(0.18, float(metadata.get("table_row_count") or 0) * 0.03)
    return min(
        1.0,
        (exact_hits / max(1, len(query_terms))) * 0.35
        + (row_hits / max(1, len(query_terms))) * 0.28
        + (title_hits / max(1, len(query_terms))) * 0.18
        + (phrase_hits / max(1, len(query_phrases) or 1)) * 0.18
        + (column_hits * 0.08)
        + row_density,
    )


def merge_table_candidates(ranked: list[dict], limit: int) -> list[dict]:
    table_candidates = [
        item
        for item in ranked
        if (item.get("metadata") or {}).get("contains_table") and item.get("table_score", 0) >= 0.2
    ]
    if not table_candidates:
        return ranked
    selected: list[dict] = []
    seen_ids: set[int] = set()
    table_candidates = sorted(table_candidates, key=lambda item: (item.get("table_score", 0), item.get("phrase_score", 0), item.get("keyword_score", 0)), reverse=True)
    for item in table_candidates[: max(1, min(2, limit))]:
        selected.append(item)
        seen_ids.add(item["id"])
    for item in ranked:
        if item["id"] not in seen_ids:
            selected.append(item)
            seen_ids.add(item["id"])
        if len(selected) >= limit:
            break
    return selected


def merge_section_siblings(ranked: list[dict], limit: int) -> list[dict]:
    if not ranked:
        return ranked
    anchor = next((item for item in ranked if section_number((item.get("metadata") or {}).get("section_title") or "")), ranked[0])
    anchor_section = (anchor.get("metadata") or {}).get("section_title") or ""
    anchor_major = major_section(anchor_section)
    if not anchor_major:
        return ranked
    siblings = [
        item
        for item in ranked
        if major_section((item.get("metadata") or {}).get("section_title") or "") == anchor_major
    ]
    siblings = sorted(
        siblings,
        key=lambda item: (
            section_number((item.get("metadata") or {}).get("section_title") or "") or (999,),
            -float(item.get("score", 0)),
        ),
    )
    selected: list[dict] = []
    seen_ids: set[int] = set()
    for item in siblings[: max(1, min(5, limit))]:
        selected.append(item)
        seen_ids.add(item["id"])
    for item in ranked:
        if item["id"] in seen_ids:
            continue
        selected.append(item)
        seen_ids.add(item["id"])
        if len(selected) >= limit:
            break
    return selected


def diversify_sections(ranked: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[int] = set()
    section_buckets: dict[str, list[dict]] = {}
    for item in ranked:
        section = (item.get("metadata") or {}).get("section_title") or "No section"
        section_buckets.setdefault(section, []).append(item)
    for section in sorted(section_buckets, key=lambda key: -float(section_buckets[key][0].get("score", 0))):
        item = section_buckets[section][0]
        selected.append(item)
        seen_ids.add(item["id"])
        if len(selected) >= max(2, min(limit, 5)):
            break
    for item in ranked:
        if item["id"] in seen_ids:
            continue
        selected.append(item)
        seen_ids.add(item["id"])
        if len(selected) >= limit:
            break
    return selected


def expand_window_candidates(ranked: list[dict], all_scored: list[dict], window_size: int, limit: int) -> list[dict]:
    if not ranked or window_size <= 0:
        return ranked
    by_doc: dict[int, list[dict]] = {}
    for item in all_scored:
        by_doc.setdefault(int(item.get("document_id") or 0), []).append(item)
    for items in by_doc.values():
        items.sort(key=lambda item: int(item.get("chunk_index") or 0))
    selected = list(ranked)
    seen_ids = {item["id"] for item in selected}
    for item in ranked[: max(1, min(3, len(ranked)))]:
        doc_items = by_doc.get(int(item.get("document_id") or 0), [])
        chunk_index = int(item.get("chunk_index") or 0)
        for neighbor in doc_items:
            if abs(int(neighbor.get("chunk_index") or 0) - chunk_index) > window_size:
                continue
            if neighbor["id"] in seen_ids:
                continue
            neighbor = dict(neighbor)
            neighbor["score"] = max(float(neighbor.get("score", 0)), float(item.get("score", 0)) * 0.82)
            selected.append(neighbor)
            seen_ids.add(neighbor["id"])
            if len(selected) >= limit * 2:
                break
    return sorted(selected, key=lambda row: float(row.get("score", 0)), reverse=True)[: max(limit, len(ranked))]


def profile_score_boost(profile: QuestionProfile, route: RetrieverRoute, query_terms: list[str], query_phrases: list[str], text: str, metadata: dict) -> float:
    haystack = searchable_text(text, metadata).lower()
    section = " ".join(metadata.get("section_path") or []).lower()
    boost = 0.0
    if profile.require_exact and any(term in haystack for term in query_terms):
        boost += route.exact_boost or 0.05
    if profile.type_id in {"exception", "negative"} and re.search(r"\b(except|unless|not|only|prohibited|forbidden|shall not|not allowed)\b", haystack):
        boost += 0.09
    if profile.type_id in {"conditional", "multi_constraint"} and re.search(r"\b(if|when|where|provided that|in case)\b", haystack):
        boost += 0.07
    if profile.type_id in {"procedural", "workflow"} and re.search(r"\b(shall|must|required|step|procedure|sequence|ensure)\b", haystack):
        boost += 0.06
    if profile.type_id in {"safety_critical", "safety_interpretation", "regulation_compliance"} and re.search(r"\b(safety|emergency|shutdown|fire|explosion|relief|hazard|shall|must)\b", haystack):
        boost += 0.08
    if profile.type_id in {"identifier", "temporal_revision"} and re.search(r"\b[A-Z]{2,}[-A-Z0-9_/]{2,}|\b\d+(?:\.\d+){1,}\b", text):
        boost += 0.08
    if profile.type_id in {"location_section", "audit_traceability", "meta_document"} and section:
        boost += 0.04
    if profile.type_id == "table_numeric" and metadata.get("contains_table"):
        boost += 0.12
    if route.citation_mode and (metadata.get("page_start") or metadata.get("page_label_start")):
        boost += 0.04
    if route.parent_child and section:
        boost += 0.04
    if route.graph_mode and re.search(r"\b(system|equipment|valve|piping|drain|flare|pump|vessel|module)\b", haystack):
        boost += 0.05
    if route.cache_mode and metadata.get("section_title"):
        boost += 0.03
    if query_phrases and any(phrase in haystack for phrase in query_phrases):
        boost += 0.05
    return boost


def is_type_query(query: str) -> bool:
    return bool(re.search(r"\b(types?|different|each|list|explain)\b", query, flags=re.IGNORECASE))


def section_number(section: str) -> tuple[int, ...] | None:
    match = re.match(r"^((?:\d+\.)*\d+)", section or "")
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split(".") if part.isdigit())


def major_section(section: str) -> int | None:
    number = section_number(section)
    return number[0] if number else None


def is_generated_or_debug_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in CONTAMINATION_PATTERNS)


def dedupe_signature(text: str) -> str:
    normalized = normalize_for_dedupe(text)
    words = normalized.split()
    return " ".join(words[:100])


def normalize_for_dedupe(text: str) -> str:
    normalized = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text.lower())
    normalized = re.sub(r"\W+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def chunk_dictionary_terms(text: str, metadata: dict) -> list[str]:
    terms: list[str] = []
    terms.extend(metadata.get("keywords") or [])
    terms.extend(metadata.get("table_terms") or [])
    terms.extend(metadata.get("table_columns") or [])
    terms.extend(metadata.get("table_rows") or [])
    terms.extend(metadata.get("section_path") or [])
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", text[:1200]))
    return terms


def normalize_term(term: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", " ", str(term).lower()).strip()
    words = [word for word in normalized.split() if word not in QUERY_STOPWORDS and len(word) > 2]
    if len(words) > 4:
        words = words[:4]
    return " ".join(words)


def jaccard_words(left: str, right: str) -> float:
    left_words = set(left.split())
    right_words = set(right.split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_id() -> str:
    timestamp = _now()
    import hashlib

    digest = hashlib.blake2b(timestamp.encode("utf-8"), digest_size=6).hexdigest()
    return f"idx-{timestamp.replace(':', '').replace('.', '')}-{digest}"
