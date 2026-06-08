from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

PGVECTOR_SCHEMA_VERSION = "engineering-pgvector-index-v2"
DEFAULT_PGVECTOR_HOST = os.environ.get("PGVECTOR_HOST", "127.0.0.1")
DEFAULT_PGVECTOR_PORT = int(os.environ.get("PGVECTOR_PORT", "5432"))
DEFAULT_PGVECTOR_DB = os.environ.get("PGVECTOR_DB", "postgres")
DEFAULT_PGVECTOR_USER = os.environ.get("PGVECTOR_USER", "postgres")
DEFAULT_PGVECTOR_PASSWORD = os.environ.get("PGVECTOR_PASSWORD", "1")
DEFAULT_PGVECTOR_TABLE = os.environ.get("PGVECTOR_TABLE", "rag_chunks")
DEFAULT_PGVECTOR_DIMENSIONS = int(os.environ.get("PGVECTOR_DIMENSIONS", "768"))


@dataclass
class PgVectorConfig:
    host: str = DEFAULT_PGVECTOR_HOST
    port: int = DEFAULT_PGVECTOR_PORT
    dbname: str = DEFAULT_PGVECTOR_DB
    user: str = DEFAULT_PGVECTOR_USER
    password: str = DEFAULT_PGVECTOR_PASSWORD
    table: str = DEFAULT_PGVECTOR_TABLE
    dimensions: int = DEFAULT_PGVECTOR_DIMENSIONS


class PgVectorIndex:
    def __init__(self, config: PgVectorConfig | None = None):
        self.config = config or PgVectorConfig()

    @classmethod
    def from_env(cls) -> "PgVectorIndex":
        return cls(PgVectorConfig())

    def enabled(self) -> bool:
        return os.environ.get("RAG_INDEX_BACKEND", "auto").lower() in {"auto", "pgvector"}

    def available(self) -> tuple[bool, str]:
        if not self.enabled():
            return False, "pgvector backend disabled"
        try:
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True, "available"
        except Exception as exc:
            return False, str(exc)

    def status(self) -> dict[str, Any]:
        available, detail = self.available()
        return {
            "index_schema_version": PGVECTOR_SCHEMA_VERSION,
            "index_backend": "pgvector" if available else "sqlite-fallback",
            "pgvector_available": available,
            "pgvector_detail": detail,
            "pgvector_host": self.config.host,
            "pgvector_port": self.config.port,
            "pgvector_db": self.config.dbname,
            "pgvector_user": self.config.user,
            "pgvector_table": self.config.table,
            "pgvector_dimensions": self.config.dimensions,
        }

    def connect(self):
        import psycopg

        return psycopg.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.config.password,
            autocommit=True,
        )

    def ensure_schema(self) -> None:
        table = quote_identifier(self.config.table)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id BIGSERIAL PRIMARY KEY,
                        sqlite_chunk_id BIGINT,
                        document_id BIGINT NOT NULL,
                        index_session TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        filename TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self.config.dimensions}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        dense_text TEXT NOT NULL DEFAULT '',
                        sparse_text TEXT NOT NULL DEFAULT '',
                        exact_phrases TEXT[] NOT NULL DEFAULT '{{}}',
                        table_text TEXT NOT NULL DEFAULT '',
                        metadata_text TEXT NOT NULL DEFAULT '',
                        numeric_text TEXT NOT NULL DEFAULT '',
                        entity_text TEXT NOT NULL DEFAULT '',
                        citation_text TEXT NOT NULL DEFAULT '',
                        index_payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE(index_session, document_id, chunk_index)
                    )
                    """
                )
                self._ensure_multi_index_columns(cur, table)
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_session_idx ON {table}(index_session)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_document_idx ON {table}(document_id)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_metadata_gin_idx ON {table} USING GIN(metadata)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_index_payload_gin_idx ON {table} USING GIN(index_payload)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_sparse_gin_idx ON {table} USING GIN(to_tsvector('english', sparse_text))")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_table_gin_idx ON {table} USING GIN(to_tsvector('english', table_text))")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_numeric_gin_idx ON {table} USING GIN(to_tsvector('english', numeric_text))")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_entity_gin_idx ON {table} USING GIN(to_tsvector('english', entity_text))")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_citation_gin_idx ON {table} USING GIN(to_tsvector('english', citation_text))")
                self._ensure_vector_dimension(cur, table)
                self._ensure_vector_index(cur, table)

    def _ensure_multi_index_columns(self, cur: Any, table: str) -> None:
        columns = {
            "dense_text": "TEXT NOT NULL DEFAULT ''",
            "numeric_text": "TEXT NOT NULL DEFAULT ''",
            "entity_text": "TEXT NOT NULL DEFAULT ''",
            "citation_text": "TEXT NOT NULL DEFAULT ''",
            "index_payload": "JSONB NOT NULL DEFAULT '{}'::jsonb",
        }
        for column, definition in columns.items():
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {quote_identifier(column)} {definition}")
            except Exception:
                pass

    def _ensure_vector_dimension(self, cur: Any, table: str) -> None:
        try:
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({self.config.dimensions}) USING embedding::vector({self.config.dimensions})")
        except Exception:
            pass

    def _ensure_vector_index(self, cur: Any, table: str) -> None:
        try:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_embedding_hnsw_idx ON {table} USING hnsw (embedding vector_cosine_ops)")
        except Exception:
            try:
                cur.execute(f"CREATE INDEX IF NOT EXISTS {self.config.table}_embedding_ivfflat_idx ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
            except Exception:
                pass

    def reset_session(self, index_session: str) -> None:
        if not self.enabled():
            return
        table = quote_identifier(self.config.table)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table} WHERE index_session = %s", (index_session,))

    def clear_all(self) -> None:
        if not self.enabled():
            return
        self.ensure_schema()
        table = quote_identifier(self.config.table)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table}")

    def upsert_chunks(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows or not self.enabled():
            return {"pgvector_enabled": self.enabled(), "pgvector_upserted": 0, "pgvector_skipped": len(rows), "pgvector_error": ""}
        self.ensure_schema()
        table = quote_identifier(self.config.table)
        upserted = 0
        skipped = 0
        error = ""
        with self.connect() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    vector = row.get("embedding") or []
                    if not vector or len(vector) != self.config.dimensions:
                        skipped += 1
                        continue
                    try:
                        cur.execute(
                            f"""
                            INSERT INTO {table} (
                                sqlite_chunk_id, document_id, index_session, chunk_index, filename,
                                text, embedding, metadata, dense_text, sparse_text, exact_phrases,
                                table_text, metadata_text, numeric_text, entity_text, citation_text,
                                index_payload, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                            ON CONFLICT(index_session, document_id, chunk_index)
                            DO UPDATE SET
                                sqlite_chunk_id = EXCLUDED.sqlite_chunk_id,
                                filename = EXCLUDED.filename,
                                text = EXCLUDED.text,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                dense_text = EXCLUDED.dense_text,
                                sparse_text = EXCLUDED.sparse_text,
                                exact_phrases = EXCLUDED.exact_phrases,
                                table_text = EXCLUDED.table_text,
                                metadata_text = EXCLUDED.metadata_text,
                                numeric_text = EXCLUDED.numeric_text,
                                entity_text = EXCLUDED.entity_text,
                                citation_text = EXCLUDED.citation_text,
                                index_payload = EXCLUDED.index_payload,
                                created_at = EXCLUDED.created_at
                            """,
                            (
                                row.get("sqlite_chunk_id"),
                                row["document_id"],
                                row["index_session"],
                                row["chunk_index"],
                                row["filename"],
                                row["text"],
                                vector_literal(vector),
                                json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                                row.get("dense_text") or "",
                                row.get("sparse_text") or "",
                                row.get("exact_phrases") or [],
                                row.get("table_text") or "",
                                row.get("metadata_text") or "",
                                row.get("numeric_text") or "",
                                row.get("entity_text") or "",
                                row.get("citation_text") or "",
                                json.dumps(row.get("index_payload") or {}, ensure_ascii=False),
                                row["created_at"],
                            ),
                        )
                        upserted += 1
                    except Exception as exc:
                        skipped += 1
                        error = str(exc)
        return {"pgvector_enabled": True, "pgvector_upserted": upserted, "pgvector_skipped": skipped, "pgvector_error": error}

    def search(self, index_session: str, query_vector: list[float], query_text: str, limit: int = 80) -> list[dict[str, Any]]:
        if not self.enabled() or not query_vector:
            return []
        self.ensure_schema()
        table = quote_identifier(self.config.table)
        candidate_limit = max(limit, 50)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id, sqlite_chunk_id, document_id, filename, chunk_index, text, metadata,
                        1 - (embedding <=> %s::vector) AS vector_score,
                        ts_rank_cd(to_tsvector('english', sparse_text), plainto_tsquery('english', %s)) AS sparse_score,
                        ts_rank_cd(to_tsvector('english', table_text), plainto_tsquery('english', %s)) AS table_index_score,
                        ts_rank_cd(to_tsvector('english', numeric_text), plainto_tsquery('english', %s)) AS numeric_index_score,
                        ts_rank_cd(to_tsvector('english', entity_text), plainto_tsquery('english', %s)) AS entity_index_score,
                        ts_rank_cd(to_tsvector('english', citation_text), plainto_tsquery('english', %s)) AS citation_index_score
                    FROM {table}
                    WHERE index_session = %s
                    ORDER BY
                        (embedding <=> %s::vector) ASC,
                        (
                            ts_rank_cd(to_tsvector('english', sparse_text), plainto_tsquery('english', %s))
                            + ts_rank_cd(to_tsvector('english', table_text), plainto_tsquery('english', %s)) * 1.2
                            + ts_rank_cd(to_tsvector('english', numeric_text), plainto_tsquery('english', %s)) * 1.1
                            + ts_rank_cd(to_tsvector('english', entity_text), plainto_tsquery('english', %s)) * 0.9
                            + ts_rank_cd(to_tsvector('english', citation_text), plainto_tsquery('english', %s)) * 0.7
                        ) DESC
                    LIMIT %s
                    """,
                    (
                        vector_literal(query_vector),
                        query_text,
                        query_text,
                        query_text,
                        query_text,
                        query_text,
                        index_session,
                        vector_literal(query_vector),
                        query_text,
                        query_text,
                        query_text,
                        query_text,
                        query_text,
                        candidate_limit,
                    ),
                )
                rows = cur.fetchall()
                columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def quote_identifier(value: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    return f'"{safe}"'
