from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .agentic_retriever import agentic_candidate_score, agentic_retrieval_plan
from .api_retriever import api_candidate_score, api_retrieval_plan, run_api_retrieval
from .change_detection import enrich_change_detection_from_history
from .document_classification import document_classification_candidate_score
from .embeddings import embed_texts, embed_query, embedding_status
from .graph_retriever import expand_graph_candidates, graph_candidate_score, graph_retrieval_plan
from .hierarchical_embedding import hierarchical_embedding_candidate_score, hierarchical_embedding_text, prepare_hierarchical_embedding_metadata
from .ingestion_quality import ingestion_quality_candidate_score
from .iterative_retriever import annotate_retry_rows, build_iterative_retry_queries, iterative_candidate_score, iterative_plan, merge_iterative_rows
from .knowledge_graph_retriever import build_knowledge_graph, knowledge_graph_candidate_score, knowledge_graph_query_plan
from .late_interaction_retriever import late_interaction_plan, late_interaction_score
from .memory_retriever import build_chat_memory_payload, memory_candidate_score, memory_expanded_queries, memory_retrieval_plan
from .multi_hop_retriever import expand_multi_hop_candidates, multi_hop_candidate_score, multi_hop_expanded_queries, multi_hop_plan
from .ontology_retriever import build_ontology_plan, ontology_candidate_score, ontology_expanded_queries
from .pgvector_index import PgVectorIndex
from .question_types import QuestionProfile, classify_question, profile_payload
from .query_decomposition_retriever import decomposition_candidate_score, decomposition_expanded_queries, query_decomposition_plan
from .retriever_router import RetrieverRoute, expanded_queries, route_capability_score, route_for_question, route_payload
from .semantic_graph_retriever import semantic_graph_candidate_score, semantic_graph_plan
from .section_importance import section_importance_candidate_score
from .sql_database_retriever import run_sql_database_retrieval, sql_database_candidate_score, sql_database_plan
from .symbolic_retriever import symbolic_candidate_score, symbolic_expanded_queries, symbolic_plan
from .tool_aware_retriever import tool_aware_candidate_score, tool_aware_expanded_queries, tool_aware_plan

VECTOR_SIZE = 768
DEDUPE_SCHEMA_VERSION = "engineering-dedupe-v2"
DEDUPE_JACCARD_THRESHOLD = 0.9
DEDUPE_SHINGLE_THRESHOLD = 0.86
DEDUPE_CONTAINMENT_THRESHOLD = 0.92
DEDUPE_MIN_WORDS = 12
HYBRID_SCHEMA_VERSION = "engineering-hybrid-v2"
DOCUMENT_LINK_SCHEMA_VERSION = "engineering-document-links-v2"
MULTI_INDEX_SCHEMA_VERSION = "engineering-multi-index-v1"
SELF_QUERY_SCHEMA_VERSION = "engineering-self-query-v1"
HYBRID_TEXT_LIMIT = 16000
SPARSE_TERM_LIMIT = 120
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
        self.last_add_document_stats: dict = {}
        self.pgvector = PgVectorIndex.from_env()
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
            seen_exact_signatures: set[str] = set()
            seen_prefix_signatures: set[str] = set()
            seen_fingerprints: list[dict] = []
            skipped_duplicates = 0
            accepted_chunks = []
            for index, chunk in enumerate(chunks):
                text, metadata = normalize_chunk(chunk)
                if is_generated_or_debug_text(text):
                    continue
                fingerprint = dedupe_fingerprint(text, metadata)
                exact_signature = fingerprint["exact_signature"]
                prefix_signature = fingerprint["prefix_signature"]
                duplicate_reason = ""
                if exact_signature in seen_exact_signatures:
                    duplicate_reason = "exact_signature"
                elif prefix_signature in seen_prefix_signatures and not dedupe_protected(metadata):
                    duplicate_reason = "prefix_signature"
                else:
                    near_duplicate = find_near_duplicate(fingerprint, seen_fingerprints)
                    if near_duplicate:
                        duplicate_reason = str(near_duplicate["reason"])
                if duplicate_reason:
                    skipped_duplicates += 1
                    continue
                seen_exact_signatures.add(exact_signature)
                seen_prefix_signatures.add(prefix_signature)
                seen_fingerprints.append({**fingerprint, "chunk_index": index})
                metadata = {
                    **metadata,
                    "filename": filename,
                    "db_dedupe_schema_version": DEDUPE_SCHEMA_VERSION,
                    "db_dedupe_status": "kept",
                    "db_dedupe_exact_signature": exact_signature,
                    "db_dedupe_prefix_signature": prefix_signature,
                    "db_dedupe_word_count": fingerprint["word_count"],
                    "db_dedupe_unique_word_count": fingerprint["unique_word_count"],
                    "db_dedupe_protected": dedupe_protected(metadata),
                    "db_dedupe_skipped_before": skipped_duplicates,
                }
                accepted_chunks.append((index, text, metadata))
            linked_chunks = enrich_document_links(accepted_chunks, filename)
            pending_chunks = []
            for index, text, metadata in linked_chunks:
                metadata = enrich_change_detection_from_history(
                    conn,
                    filename=filename,
                    index_session=index_session,
                    document_id=document_id,
                    text=text,
                    metadata=metadata,
                )
                metadata = prepare_hybrid_metadata(text, metadata)
                metadata = prepare_multi_index_metadata(text, metadata)
                metadata = prepare_hierarchical_embedding_metadata(text, metadata)
                embedding_text = hierarchical_embedding_text(text, metadata)
                pending_chunks.append((index, text, metadata, embedding_text))
            embedding_results = embed_texts([item[3] for item in pending_chunks], kind="passage") if pending_chunks else []
            for (index, text, metadata, _embedding_text), embedding_result in zip(pending_chunks, embedding_results):
                embedding = embedding_result["vector"]
                metadata = {**metadata, **embedding_result["metadata"]}
                rows.append((document_id, index_session, index, text, json.dumps(embedding), json.dumps(metadata), _now()))
            self.last_add_document_stats = {
                "pipeline_stage": "database_storage_embedding",
                "input_chunks": len(chunks),
                "deduped_pending_chunks": len(pending_chunks),
                "stored_chunks": len(rows),
                "skipped_duplicates": skipped_duplicates,
                "embedding_backend": embedding_results[0]["metadata"].get("embedding_backend") if embedding_results else "",
                "embedding_dimensions": embedding_results[0]["metadata"].get("embedding_dimensions") if embedding_results else 0,
                "hybrid_prepared": all((item[2].get("hybrid_prepared") for item in pending_chunks)) if pending_chunks else False,
            }
            pgvector_rows = []
            for row in rows:
                cursor = conn.execute(
                    """
                    INSERT INTO chunks(document_id, index_session, chunk_index, text, embedding, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                metadata = json.loads(row[5] or "{}")
                pgvector_rows.append(
                    {
                        "sqlite_chunk_id": int(cursor.lastrowid),
                        "document_id": row[0],
                        "index_session": row[1],
                        "chunk_index": row[2],
                        "text": row[3],
                        "embedding": json.loads(row[4]),
                        "metadata": metadata,
                        "filename": filename,
                        "sparse_text": metadata.get("hybrid_sparse_text") or searchable_text(row[3], metadata),
                        "dense_text": metadata.get("multi_index_dense_text") or metadata.get("hybrid_dense_text") or "",
                        "exact_phrases": metadata.get("hybrid_exact_phrases") or [],
                        "table_text": metadata.get("hybrid_table_text") or "",
                        "metadata_text": metadata.get("hybrid_metadata_text") or "",
                        "numeric_text": metadata.get("hybrid_numeric_text") or "",
                        "entity_text": metadata.get("multi_index_entity_text") or entity_metadata_text(metadata),
                        "citation_text": metadata.get("multi_index_citation_text") or citation_index_text(metadata),
                        "index_payload": metadata.get("multi_index_payload") or {},
                        "created_at": row[6],
                    }
                )
            pgvector_stats = self._pgvector_upsert(pgvector_rows)
            self.last_add_document_stats.update(pgvector_stats)
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
        memory_plan = memory_retrieval_plan(query, route, self.recent_chat_turns(limit=6))
        route_queries = memory_expanded_queries(query, route_queries, memory_plan)
        decomposition_plan = query_decomposition_plan(query, route, profile)
        route_queries = decomposition_expanded_queries(route_queries, decomposition_plan)
        multi_hop = multi_hop_plan(query, route)
        route_queries = multi_hop_expanded_queries(route_queries, multi_hop)
        iterative = iterative_plan(query, route, profile)
        ontology_plan = build_ontology_plan(query, route, profile)
        route_queries = ontology_expanded_queries(route_queries, ontology_plan)
        symbolic = symbolic_plan(query, route, profile)
        route_queries = symbolic_expanded_queries(route_queries, symbolic)
        sql_plan = sql_database_plan(query, route)
        api_plan = api_retrieval_plan(query, route, profile)
        tool_plan = tool_aware_plan(
            query,
            route,
            profile,
            {
                "memory": memory_plan,
                "decomposition": decomposition_plan,
                "multi_hop": multi_hop,
                "iterative": iterative,
                "sql_database": sql_plan,
                "api": api_plan,
            },
        )
        route_queries = tool_aware_expanded_queries(route_queries, tool_plan)
        agentic_plan = agentic_retrieval_plan(query, profile, route, route_queries)
        planned_queries = agentic_plan.get("queries") or route_queries
        expanded_query = "\n".join(self.expand_query(route_query, query_keywords(route_query)) for route_query in planned_queries)
        query_vector = embed_query(expanded_query)
        important_terms = important_query_terms(query_terms)
        query_phrases = query_exact_phrases(query)
        self_query = parse_self_query_filters(query)
        late_plan = late_interaction_plan(query, route)
        table_intent = is_table_query(query_terms)
        rows = self._pgvector_search_rows(index_session, query_vector, expanded_query, limit=max(limit * 16, 100))
        pgvector_used = bool(rows)
        if not rows:
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
        sql_rows = []
        sql_detail = {**sql_plan, "chunk_scores": {}}
        if sql_plan.get("active"):
            with self._connect() as conn:
                sql_rows, sql_chunk_scores, sql_detail = run_sql_database_retrieval(conn, index_session, sql_plan, limit=max(limit * 12, 80))
                sql_detail["chunk_scores"] = {str(key): value for key, value in sql_chunk_scores.items()}
            rows = merge_sql_rows(rows, sql_rows)
        api_detail = {**api_plan, "chunk_scores": {}}
        if api_plan.get("active"):
            api_rows, api_chunk_scores, api_detail = run_api_retrieval(api_plan, query, query_vector, limit=max(limit * 4, 12))
            api_detail["chunk_scores"] = {str(key): value for key, value in api_chunk_scores.items()}
            rows = merge_api_rows(rows, api_rows)
        iterative = build_iterative_retry_queries(iterative, rows, query)
        if iterative.get("retry_needed"):
            iterative_rows = []
            for retry_query in iterative.get("retry_queries") or []:
                retry_expanded_query = self.expand_query(retry_query, query_keywords(retry_query))
                retry_vector = embed_query(retry_expanded_query)
                retry_rows = self._pgvector_search_rows(index_session, retry_vector, retry_expanded_query, limit=max(limit * 6, 40))
                iterative_rows.extend(annotate_retry_rows(retry_rows, retry_query))
            rows = merge_iterative_rows(rows, iterative_rows)
        knowledge_graph = None
        knowledge_graph_plan = {}
        if "knowledge_graph" in route.retrievers or route.graph_mode:
            knowledge_graph = build_knowledge_graph(rows, query)
            knowledge_graph_plan = knowledge_graph_query_plan(knowledge_graph, query)
        graph_plan = graph_retrieval_plan(knowledge_graph, query, route, knowledge_graph_plan)
        semantic_graph = semantic_graph_plan(query, route, profile, knowledge_graph_plan, ontology_plan)
        scored = []
        for row in rows:
            row_data = dict(row)
            if is_generated_or_debug_text(row_data["text"]):
                continue
            metadata = row_metadata(row_data)
            vector_score = float(row_data.get("vector_score")) if row_data.get("vector_score") is not None else cosine_similarity(query_vector, json.loads(row_data["embedding"]))
            keyword_score = keyword_match_score(important_terms, row_data["text"], metadata)
            phrase_score = phrase_match_score(query_phrases, row_data["text"], metadata)
            rerank_score = rerank(important_terms, row_data["text"], metadata, vector_score)
            table_score = table_match_score(important_terms, query_phrases, row_data["text"], metadata)
            metadata_score = metadata_match_score(important_terms, query_phrases, metadata)
            entity_score = entity_match_score(important_terms, query_phrases, metadata)
            self_query_score, self_query_match = self_query_filter_score(self_query, metadata, row_data["text"])
            multi_index_score = multi_index_score_from_row(row_data)
            capability_score = route_capability_score(query, row_data["text"], metadata, route)
            agentic_score = agentic_candidate_score(agentic_plan, row_data["text"], metadata)
            knowledge_graph_score = knowledge_graph_candidate_score(knowledge_graph_plan, metadata, row_data["text"])
            graph_score, graph_detail = graph_candidate_score(graph_plan, metadata, row_data["text"], row_data.get("id"))
            late_interaction_candidate_score, late_interaction_detail = late_interaction_score(late_plan, row_data["text"], metadata)
            memory_score, memory_detail = memory_candidate_score(memory_plan, row_data.get("id"), metadata, row_data["text"])
            sql_database_score, sql_database_detail = sql_database_candidate_score(sql_plan, sql_detail, row_data.get("id"), metadata, row_data["text"])
            api_score, api_detail_item = api_candidate_score(api_plan, api_detail, row_data.get("id"), metadata, row_data["text"])
            multi_hop_score, multi_hop_detail = multi_hop_candidate_score(multi_hop, row_data, metadata, row_data["text"])
            iterative_score, iterative_detail = iterative_candidate_score(iterative, row_data, metadata, row_data["text"])
            decomposition_score, decomposition_detail = decomposition_candidate_score(decomposition_plan, metadata, row_data["text"])
            ontology_score, ontology_detail = ontology_candidate_score(ontology_plan, metadata, row_data["text"])
            symbolic_score, symbolic_detail = symbolic_candidate_score(symbolic, metadata, row_data["text"])
            hierarchical_embedding_score, hierarchical_embedding_detail = hierarchical_embedding_candidate_score(query, metadata, row_data["text"])
            section_importance_score, section_importance_detail = section_importance_candidate_score(query, metadata, row_data["text"])
            document_classification_score, document_classification_detail = document_classification_candidate_score(query, metadata, row_data["text"])
            ingestion_quality_score, ingestion_quality_detail = ingestion_quality_candidate_score(metadata)
            semantic_graph_score, semantic_graph_detail = semantic_graph_candidate_score(
                semantic_graph,
                metadata,
                row_data["text"],
                {
                    "vector": vector_score,
                    "keyword": keyword_score,
                    "knowledge_graph": knowledge_graph_score,
                    "ontology": ontology_score,
                },
            )
            tool_aware_score, tool_aware_detail = tool_aware_candidate_score(
                tool_plan,
                metadata,
                row_data["text"],
                {
                    "vector": vector_score,
                    "keyword": keyword_score,
                    "phrase": phrase_score,
                    "table": table_score,
                    "metadata": metadata_score,
                    "entity": entity_score,
                    "self_query": self_query_score,
                    "multi_index": multi_index_score,
                    "capability": capability_score,
                    "agentic": agentic_score,
                    "graph": graph_score,
                    "knowledge_graph": knowledge_graph_score,
                    "late_interaction": late_interaction_candidate_score,
                    "memory": memory_score,
                    "sql_database": sql_database_score,
                    "api": api_score,
                    "multi_hop": multi_hop_score,
                    "iterative": iterative_score,
                    "query_decomposition": decomposition_score,
                    "ontology": ontology_score,
                    "symbolic": symbolic_score,
                    "hierarchical_embedding": hierarchical_embedding_score,
                    "section_importance": section_importance_score,
                    "document_classification": document_classification_score,
                    "ingestion_quality": ingestion_quality_score,
                    "semantic_graph": semantic_graph_score,
                },
            )
            score = route_score(route, vector_score, keyword_score, phrase_score, rerank_score, table_score, metadata_score, entity_score)
            score += self_query_score
            if self_query_match.get("hard_filter_missed"):
                score -= 0.18
            score += multi_index_score
            score += capability_score
            score += agentic_score
            score += graph_score
            score += knowledge_graph_score
            score += late_interaction_candidate_score
            score += memory_score
            score += sql_database_score
            score += api_score
            score += multi_hop_score
            score += iterative_score
            score += decomposition_score
            score += ontology_score
            score += symbolic_score
            score += hierarchical_embedding_score
            score += section_importance_score
            score += document_classification_score
            score += ingestion_quality_score
            score += semantic_graph_score
            score += tool_aware_score
            score += profile_score_boost(profile, route, important_terms, query_phrases, row_data["text"], metadata)
            hybrid_breakdown = hybrid_score_breakdown(
                route,
                vector_score,
                keyword_score,
                phrase_score,
                rerank_score,
                table_score,
                metadata_score,
                entity_score,
                capability_score,
                metadata,
            )
            hybrid_breakdown["multi_index"] = round(multi_index_score, 5)
            hybrid_breakdown["self_query"] = round(self_query_score, 5)
            hybrid_breakdown["agentic"] = round(agentic_score, 5)
            hybrid_breakdown["graph"] = round(graph_score, 5)
            hybrid_breakdown["knowledge_graph"] = round(knowledge_graph_score, 5)
            hybrid_breakdown["late_interaction"] = round(late_interaction_candidate_score, 5)
            hybrid_breakdown["memory"] = round(memory_score, 5)
            hybrid_breakdown["sql_database"] = round(sql_database_score, 5)
            hybrid_breakdown["api"] = round(api_score, 5)
            hybrid_breakdown["multi_hop"] = round(multi_hop_score, 5)
            hybrid_breakdown["iterative"] = round(iterative_score, 5)
            hybrid_breakdown["query_decomposition"] = round(decomposition_score, 5)
            hybrid_breakdown["ontology"] = round(ontology_score, 5)
            hybrid_breakdown["symbolic"] = round(symbolic_score, 5)
            hybrid_breakdown["hierarchical_embedding"] = round(hierarchical_embedding_score, 5)
            hybrid_breakdown["section_importance"] = round(section_importance_score, 5)
            hybrid_breakdown["document_classification"] = round(document_classification_score, 5)
            hybrid_breakdown["ingestion_quality"] = round(ingestion_quality_score, 5)
            hybrid_breakdown["semantic_graph"] = round(semantic_graph_score, 5)
            hybrid_breakdown["tool_aware"] = round(tool_aware_score, 5)
            scored.append(
                {
                    "id": row["id"],
                    "document_id": row_data["document_id"],
                    "filename": row_data["filename"],
                    "chunk_index": row_data["chunk_index"],
                    "text": row_data["text"],
                    "metadata": metadata,
                    "index_backend": "pgvector" if pgvector_used else "sqlite",
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "phrase_score": phrase_score,
                    "table_score": table_score,
                    "metadata_score": metadata_score,
                    "entity_score": entity_score,
                    "self_query_score": self_query_score,
                    "self_query": self_query,
                    "self_query_match": self_query_match,
                    "multi_index_score": multi_index_score,
                    "capability_score": capability_score,
                    "agentic_score": agentic_score,
                    "agentic_plan": agentic_plan,
                    "graph_score": graph_score,
                    "graph_detail": graph_detail,
                    "graph_plan": graph_plan,
                    "knowledge_graph_score": knowledge_graph_score,
                    "knowledge_graph_plan": knowledge_graph_plan,
                    "late_interaction_score": late_interaction_candidate_score,
                    "late_interaction_detail": late_interaction_detail,
                    "memory_score": memory_score,
                    "memory_detail": memory_detail,
                    "memory_plan": memory_plan,
                    "sql_database_score": sql_database_score,
                    "sql_database_detail": sql_database_detail,
                    "sql_database_plan": sql_plan,
                    "api_score": api_score,
                    "api_detail": api_detail_item,
                    "api_plan": api_plan,
                    "multi_hop_score": multi_hop_score,
                    "multi_hop_detail": multi_hop_detail,
                    "multi_hop_plan": multi_hop,
                    "iterative_score": iterative_score,
                    "iterative_detail": iterative_detail,
                    "iterative_plan": iterative,
                    "query_decomposition_score": decomposition_score,
                    "query_decomposition_detail": decomposition_detail,
                    "query_decomposition_plan": decomposition_plan,
                    "ontology_score": ontology_score,
                    "ontology_detail": ontology_detail,
                    "ontology_plan": ontology_plan,
                    "symbolic_score": symbolic_score,
                    "symbolic_detail": symbolic_detail,
                    "symbolic_plan": symbolic,
                    "hierarchical_embedding_score": hierarchical_embedding_score,
                    "hierarchical_embedding_detail": hierarchical_embedding_detail,
                    "section_importance_score": section_importance_score,
                    "section_importance_detail": section_importance_detail,
                    "document_classification_score": document_classification_score,
                    "document_classification_detail": document_classification_detail,
                    "ingestion_quality_score": ingestion_quality_score,
                    "ingestion_quality_detail": ingestion_quality_detail,
                    "semantic_graph_score": semantic_graph_score,
                    "semantic_graph_detail": semantic_graph_detail,
                    "semantic_graph_plan": semantic_graph,
                    "tool_aware_score": tool_aware_score,
                    "tool_aware_detail": tool_aware_detail,
                    "tool_aware_plan": tool_plan,
                    "pgvector_sparse_score": row_data.get("pgvector_sparse_score"),
                    "table_index_score": row_data.get("table_index_score"),
                    "numeric_index_score": row_data.get("numeric_index_score"),
                    "entity_index_score": row_data.get("entity_index_score"),
                    "citation_index_score": row_data.get("citation_index_score"),
                    "hybrid_breakdown": hybrid_breakdown,
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
        if route.parent_child or route.graph_mode or route.primary in {"document_map", "hierarchical"}:
            ranked = expand_linked_candidates(ranked, scored, limit)
        if route.graph_mode or "graph" in route.retrievers:
            ranked = expand_graph_candidates(ranked, scored, graph_plan, limit)
        if multi_hop.get("active"):
            ranked = expand_multi_hop_candidates(ranked, scored, multi_hop, limit)
        if route.diversify_sections or profile.type_id in {"comparison", "cross_section", "multi_document", "conflict_detection", "document_coverage"}:
            ranked = diversify_sections(ranked, limit)
        return ranked[:limit]

    def _pgvector_upsert(self, rows: list[dict]) -> dict:
        try:
            return self.pgvector.upsert_chunks(rows)
        except Exception as exc:
            return {
                "pgvector_enabled": self.pgvector.enabled(),
                "pgvector_upserted": 0,
                "pgvector_skipped": len(rows),
                "pgvector_error": str(exc),
            }

    def _pgvector_search_rows(self, index_session: str, query_vector: list[float], query_text: str, limit: int) -> list[dict]:
        try:
            pg_rows = self.pgvector.search(index_session, query_vector, query_text, limit=limit)
        except Exception:
            return []
        rows = []
        for row in pg_rows:
            metadata = row.get("metadata") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            rows.append(
                {
                    "id": row.get("sqlite_chunk_id") or row.get("id"),
                    "document_id": row.get("document_id"),
                    "filename": row.get("filename"),
                    "chunk_index": row.get("chunk_index"),
                    "text": row.get("text") or "",
                    "metadata": metadata,
                    "vector_score": row.get("vector_score"),
                    "pgvector_sparse_score": row.get("sparse_score"),
                    "table_index_score": row.get("table_index_score"),
                    "numeric_index_score": row.get("numeric_index_score"),
                    "entity_index_score": row.get("entity_index_score"),
                    "citation_index_score": row.get("citation_index_score"),
                }
            )
        return rows

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
        try:
            self.pgvector.clear_all()
        except Exception:
            pass
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
            **self.pgvector.status(),
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
                    memory_payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    index_session TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    index_session TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
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
            chat_columns = [row["name"] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()]
            if "memory_payload" not in chat_columns:
                conn.execute("ALTER TABLE chat_history ADD COLUMN memory_payload TEXT NOT NULL DEFAULT '{}'")
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

    def create_chat_session(self, title: str = "") -> dict:
        session_id = str(_session_id())
        now = _now()
        clean_title = chat_session_title(title)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions(id, index_session, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, self.current_index_session(), clean_title, now, now),
            )
            conn.commit()
        return {
            "id": session_id,
            "index_session": self.current_index_session(),
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "latest_preview": "",
        }

    def ensure_chat_session(self, session_id: str | None, title: str = "") -> dict:
        index_session = self.current_index_session()
        if session_id:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, index_session, title, created_at, updated_at
                    FROM chat_sessions
                    WHERE id = ? AND index_session = ?
                    """,
                    (session_id, index_session),
                ).fetchone()
            if row:
                return {**dict(row), "message_count": 0, "latest_preview": ""}
        return self.create_chat_session(title)

    def list_chat_sessions(self, limit: int = 40) -> list[dict]:
        current_index_session = self.current_index_session()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    chat_sessions.id,
                    chat_sessions.index_session,
                    chat_sessions.title,
                    chat_sessions.created_at,
                    chat_sessions.updated_at,
                    COUNT(chat_messages.id) AS message_count,
                    (
                        SELECT text
                        FROM chat_messages AS latest
                        WHERE latest.session_id = chat_sessions.id
                        ORDER BY latest.created_at DESC, latest.id DESC
                        LIMIT 1
                    ) AS latest_preview
                FROM chat_sessions
                LEFT JOIN chat_messages ON chat_messages.session_id = chat_sessions.id
                GROUP BY chat_sessions.id
                ORDER BY chat_sessions.updated_at DESC
                LIMIT ?
                """,
                (max(1, min(100, limit)),),
            ).fetchall()
        return [
            {
                **dict(row),
                "message_count": int(row["message_count"] or 0),
                "latest_preview": compact_preview(row["latest_preview"] or ""),
                "active_index": row["index_session"] == current_index_session,
            }
            for row in rows
        ]

    def get_chat_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            session = conn.execute(
                """
                SELECT id, index_session, title, created_at, updated_at
                FROM chat_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            rows = conn.execute(
                """
                SELECT id, role, text, payload, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.get("payload") or "{}")
            messages.append(item)
        return {**dict(session), "messages": messages}

    def add_chat_message(self, session_id: str, role: str, text: str, payload: dict | None = None) -> dict:
        index_session = self.current_index_session()
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_messages(session_id, index_session, role, text, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, index_session, role, text, json.dumps(payload or {}, ensure_ascii=False), now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ? AND index_session = ?",
                (now, session_id, index_session),
            )
            conn.commit()
            message_id = int(cursor.lastrowid)
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "text": text,
            "payload": payload or {},
            "created_at": now,
        }

    def add_chat_turn(self, question: str, resolved_question: str, answer_result: dict, contexts: list[dict]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_history(index_session, question, resolved_question, answer, profile, source_ids, memory_payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.current_index_session(),
                    question,
                    resolved_question,
                    answer_result.get("answer", ""),
                    json.dumps(answer_result.get("question_profile") or {}, ensure_ascii=False),
                    json.dumps([item.get("id") for item in contexts if item.get("id")], ensure_ascii=False),
                    json.dumps(build_chat_memory_payload(contexts), ensure_ascii=False),
                    _now(),
                ),
            )
            conn.commit()

    def recent_chat_turns(self, limit: int = 4) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, resolved_question, answer, profile, source_ids, memory_payload, created_at
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
            item["memory_payload"] = json.loads(item.get("memory_payload") or "{}")
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
                "index_backend": item.get("index_backend") or "",
                "score": round(float(item.get("score", 0)), 4),
                "vector_score": round(float(item.get("vector_score", 0)), 4),
                "keyword_score": round(float(item.get("keyword_score", 0)), 4),
                "phrase_score": round(float(item.get("phrase_score", 0)), 4),
                "table_score": round(float(item.get("table_score", 0)), 4),
                "entity_score": round(float(item.get("entity_score", 0)), 4),
                "self_query_score": round(float(item.get("self_query_score", 0)), 4),
                "self_query": item.get("self_query") or {},
                "self_query_match": item.get("self_query_match") or {},
                "multi_index_score": round(float(item.get("multi_index_score", 0)), 4),
                "agentic_score": round(float(item.get("agentic_score", 0)), 4),
                "agentic_plan": item.get("agentic_plan") or {},
                "graph_score": round(float(item.get("graph_score", 0)), 4),
                "graph_detail": item.get("graph_detail") or {},
                "graph_plan": item.get("graph_plan") or {},
                "knowledge_graph_score": round(float(item.get("knowledge_graph_score", 0)), 4),
                "knowledge_graph_plan": item.get("knowledge_graph_plan") or {},
                "late_interaction_score": round(float(item.get("late_interaction_score", 0)), 4),
                "late_interaction_detail": item.get("late_interaction_detail") or {},
                "memory_score": round(float(item.get("memory_score", 0)), 4),
                "memory_detail": item.get("memory_detail") or {},
                "memory_plan": item.get("memory_plan") or {},
                "sql_database_score": round(float(item.get("sql_database_score", 0)), 4),
                "sql_database_detail": item.get("sql_database_detail") or {},
                "sql_database_plan": item.get("sql_database_plan") or {},
                "api_score": round(float(item.get("api_score", 0)), 4),
                "api_detail": item.get("api_detail") or {},
                "api_plan": item.get("api_plan") or {},
                "multi_hop_score": round(float(item.get("multi_hop_score", 0)), 4),
                "multi_hop_detail": item.get("multi_hop_detail") or {},
                "multi_hop_plan": item.get("multi_hop_plan") or {},
                "iterative_score": round(float(item.get("iterative_score", 0)), 4),
                "iterative_detail": item.get("iterative_detail") or {},
                "iterative_plan": item.get("iterative_plan") or {},
                "query_decomposition_score": round(float(item.get("query_decomposition_score", 0)), 4),
                "query_decomposition_detail": item.get("query_decomposition_detail") or {},
                "query_decomposition_plan": item.get("query_decomposition_plan") or {},
                "ontology_score": round(float(item.get("ontology_score", 0)), 4),
                "ontology_detail": item.get("ontology_detail") or {},
                "ontology_plan": item.get("ontology_plan") or {},
                "symbolic_score": round(float(item.get("symbolic_score", 0)), 4),
                "symbolic_detail": item.get("symbolic_detail") or {},
                "symbolic_plan": item.get("symbolic_plan") or {},
                "hierarchical_embedding_score": round(float(item.get("hierarchical_embedding_score", 0)), 4),
                "hierarchical_embedding_detail": item.get("hierarchical_embedding_detail") or {},
                "section_importance_score": round(float(item.get("section_importance_score", 0)), 4),
                "section_importance_detail": item.get("section_importance_detail") or {},
                "document_classification_score": round(float(item.get("document_classification_score", 0)), 4),
                "document_classification_detail": item.get("document_classification_detail") or {},
                "ingestion_quality_rank_score": round(float(item.get("ingestion_quality_score", 0)), 4),
                "ingestion_quality_detail": item.get("ingestion_quality_detail") or {},
                "semantic_graph_score": round(float(item.get("semantic_graph_score", 0)), 4),
                "semantic_graph_detail": item.get("semantic_graph_detail") or {},
                "semantic_graph_plan": item.get("semantic_graph_plan") or {},
                "tool_aware_score": round(float(item.get("tool_aware_score", 0)), 4),
                "tool_aware_detail": item.get("tool_aware_detail") or {},
                "tool_aware_plan": item.get("tool_aware_plan") or {},
                "pgvector_sparse_score": round(float(item.get("pgvector_sparse_score") or 0), 4),
                "table_index_score": round(float(item.get("table_index_score") or 0), 4),
                "numeric_index_score": round(float(item.get("numeric_index_score") or 0), 4),
                "entity_index_score": round(float(item.get("entity_index_score") or 0), 4),
                "citation_index_score": round(float(item.get("citation_index_score") or 0), 4),
                "capability_score": round(float(item.get("capability_score", 0)), 4),
                "hybrid_breakdown": item.get("hybrid_breakdown") or {},
                "section": metadata.get("section_title") or "",
                "section_importance_schema_version": metadata.get("section_importance_schema_version") or "",
                "section_importance_ready": bool(metadata.get("section_importance_ready")),
                "section_importance_label": metadata.get("section_importance_label") or "",
                "section_importance_metadata_score": float(metadata.get("section_importance_score") or 0),
                "section_importance_retrieval_boost": float(metadata.get("section_importance_retrieval_boost") or 0),
                "section_importance_reason": metadata.get("section_importance_reason") or "",
                "document_classification_schema_version": metadata.get("document_classification_schema_version") or "",
                "document_classification_ready": bool(metadata.get("document_classification_ready")),
                "document_class": metadata.get("document_class") or "",
                "document_class_label": metadata.get("document_class_label") or "",
                "document_class_confidence": float(metadata.get("document_class_confidence") or 0),
                "document_class_score": float(metadata.get("document_class_score") or 0),
                "document_class_routing_hint": metadata.get("document_class_routing_hint") or "",
                "document_class_retrieval_tags": metadata.get("document_class_retrieval_tags") or [],
                "change_detection_schema_version": metadata.get("change_detection_schema_version") or "",
                "change_detection_ready": bool(metadata.get("change_detection_ready")),
                "change_compare_status": metadata.get("change_compare_status") or "",
                "change_detected": bool(metadata.get("change_detected")),
                "change_similarity": float(metadata.get("change_similarity") or 0),
                "change_revision": metadata.get("change_revision") or metadata.get("revision") or "",
                "change_previous_revision": metadata.get("change_previous_revision") or "",
                "change_previous_chunk_id": metadata.get("change_previous_chunk_id"),
                "change_summary": metadata.get("change_summary") or "",
                "table_title": metadata.get("table_title") or "",
                "contains_table": bool(metadata.get("contains_table")),
                "table_row_count": int(metadata.get("table_row_count") or 0),
                "table_quality_score": float(metadata.get("table_quality_score") or 0),
                "table_integrity": metadata.get("table_integrity") or "",
                "image_block_count": int(metadata.get("image_block_count") or 0),
                "figure_references": metadata.get("figure_references") or [],
                "figure_captions": metadata.get("figure_captions") or [],
                "modalities": metadata.get("modalities") or [],
                "multimodal": bool(metadata.get("multimodal")),
                "layout_aware": bool(metadata.get("layout_aware")),
                "layout_regions": metadata.get("layout_regions") or [],
                "page_regions": metadata.get("page_regions") or [],
                "columns": metadata.get("columns") or [],
                "column_count": int(metadata.get("column_count") or 1),
                "reading_order_start": metadata.get("reading_order_start"),
                "reading_order_end": metadata.get("reading_order_end"),
                "chunk_id": metadata.get("chunk_id") or "",
                "chunk_sequence": metadata.get("chunk_sequence"),
                "chunk_system": metadata.get("chunk_system") or "",
                "chunk_strategy": metadata.get("chunk_strategy") or "",
                "chunk_boundary_reason": metadata.get("chunk_boundary_reason") or "",
                "chunk_size_class": metadata.get("chunk_size_class") or "",
                "chunk_char_count": int(metadata.get("chunk_char_count") or 0),
                "chunk_quality_score": float(metadata.get("chunk_quality_score") or 0),
                "chunk_warnings": metadata.get("chunk_warnings") or [],
                "starts_clean_boundary": bool(metadata.get("starts_clean_boundary")),
                "overlap_applied": bool(metadata.get("overlap_applied")),
                "overlap_skip_reason": metadata.get("overlap_skip_reason") or "",
                "overlap_char_count": int(metadata.get("overlap_char_count") or 0),
                "overlap_ratio": float(metadata.get("overlap_ratio") or 0),
                "overlap_boundary_type": metadata.get("overlap_boundary_type") or "",
                "overlap_quality_score": float(metadata.get("overlap_quality_score") or 0),
                "overlap_source_section": metadata.get("overlap_source_section_title") or "",
                "metadata_schema_version": metadata.get("metadata_schema_version") or "",
                "metadata_quality_score": float(metadata.get("metadata_quality_score") or 0),
                "ingestion_quality_schema_version": metadata.get("ingestion_quality_schema_version") or "",
                "ingestion_quality_ready": bool(metadata.get("ingestion_quality_ready")),
                "ingestion_quality_score": float(metadata.get("ingestion_quality_score") or 0),
                "ingestion_quality_label": metadata.get("ingestion_quality_label") or "",
                "ingestion_quality_indexable": bool(metadata.get("ingestion_quality_indexable")),
                "ingestion_quality_answerable": bool(metadata.get("ingestion_quality_answerable")),
                "ingestion_quality_warnings": metadata.get("ingestion_quality_warnings") or [],
                "ingestion_quality_reason": metadata.get("ingestion_quality_reason") or "",
                "ingestion_validation_passed": bool(metadata.get("ingestion_validation_passed")),
                "ingestion_validation_score": float(metadata.get("ingestion_validation_score") or 0),
                "ingestion_validation_warnings": metadata.get("ingestion_validation_warnings") or [],
                "semantic_labels": metadata.get("semantic_labels") or [],
                "primary_semantic_label": metadata.get("primary_semantic_label") or "",
                "reference_count": int(metadata.get("reference_count") or 0),
                "reference_section_ids": metadata.get("reference_section_ids") or [],
                "reference_standards": metadata.get("reference_standards") or [],
                "relationship_count": int(metadata.get("relationship_count") or 0),
                "relationship_types": metadata.get("relationship_types") or [],
                "domain_terms": metadata.get("domain_terms") or [],
                "engineering_entities": metadata.get("engineering_entities") or [],
                "engineering_canonical_entities": metadata.get("engineering_canonical_entities") or [],
                "primary_entities": metadata.get("primary_entities") or [],
                "entity_facets": metadata.get("entity_facets") or {},
                "entity_extraction_quality": metadata.get("entity_extraction_quality") or {},
                "engineering_entity_types": metadata.get("engineering_entity_types") or [],
                "engineering_entity_aliases": metadata.get("engineering_entity_aliases") or [],
                "engineering_entity_relationships": metadata.get("engineering_entity_relationships") or [],
                "entity_count": int(metadata.get("entity_count") or 0),
                "language_schema_version": metadata.get("language_schema_version") or "",
                "language_detection_ready": bool(metadata.get("language_detection_ready")),
                "language_code": metadata.get("language_code") or "",
                "language_name": metadata.get("language_name") or "",
                "language_confidence": float(metadata.get("language_confidence") or 0),
                "language_detection_method": metadata.get("language_detection_method") or "",
                "detected_scripts": metadata.get("detected_scripts") or [],
                "primary_script": metadata.get("primary_script") or "",
                "multilingual": bool(metadata.get("multilingual")),
                "language_script_ratios": metadata.get("language_script_ratios") or {},
                "english_signal_score": float(metadata.get("english_signal_score") or 0),
                "translation_required": bool(metadata.get("translation_required")),
                "translation_ingestion_ready": bool(metadata.get("translation_ingestion_ready")),
                "translation_status": metadata.get("translation_status") or "",
                "translation_method": metadata.get("translation_method") or "",
                "translation_confidence": float(metadata.get("translation_confidence") or 0),
                "translation_text": metadata.get("translation_text") or "",
                "translation_retrieval_text": metadata.get("translation_retrieval_text") or "",
                "translation_glossary_terms": metadata.get("translation_glossary_terms") or [],
                "access_classification": metadata.get("access_classification") or "",
                "access_control_ready": bool(metadata.get("access_control_ready")),
                "access_sensitivity_level": int(metadata.get("access_sensitivity_level") or 0),
                "access_control_tags": metadata.get("access_control_tags") or [],
                "access_policy_required": bool(metadata.get("access_policy_required")),
                "access_policy_action": metadata.get("access_policy_action") or "",
                "access_allowed_roles": metadata.get("access_allowed_roles") or [],
                "access_redaction_required": bool(metadata.get("access_redaction_required")),
                "access_redaction_fields": metadata.get("access_redaction_fields") or [],
                "access_control_decision": metadata.get("access_control_decision") or {},
                "technical_identifiers": metadata.get("technical_identifiers") or [],
                "standards": metadata.get("standards") or [],
                "requirement_modalities": metadata.get("requirement_modalities") or [],
                "has_numeric_constraints": bool(metadata.get("has_numeric_constraints")),
                "numeric_constraint_count": int(metadata.get("numeric_constraint_count") or 0),
                "normalized_numeric_constraints": metadata.get("normalized_numeric_constraints") or [],
                "numeric_units": metadata.get("numeric_units") or [],
                "unit_normalization_schema_version": metadata.get("unit_normalization_schema_version") or "",
                "unit_normalization_ready": bool(metadata.get("unit_normalization_ready")),
                "unit_normalization_applied": bool(metadata.get("unit_normalization_applied")),
                "numeric_unit_families": metadata.get("numeric_unit_families") or [],
                "canonical_numeric_units": metadata.get("canonical_numeric_units") or [],
                "unit_ratio_constraints": metadata.get("unit_ratio_constraints") or [],
                "formula_schema_version": metadata.get("formula_schema_version") or "",
                "formula_ingestion_ready": bool(metadata.get("formula_ingestion_ready")),
                "formula_count": int(metadata.get("formula_count") or 0),
                "formula_equations": metadata.get("formula_equations") or [],
                "formula_ratios": metadata.get("formula_ratios") or [],
                "formula_variables": metadata.get("formula_variables") or [],
                "formula_units": metadata.get("formula_units") or [],
                "formula_calculation_ready": bool(metadata.get("formula_calculation_ready")),
                "code_ingestion_schema_version": metadata.get("code_ingestion_schema_version") or "",
                "code_ingestion_ready": bool(metadata.get("code_ingestion_ready")),
                "code_detected": bool(metadata.get("code_detected")),
                "code_language": metadata.get("code_language") or "",
                "code_parse_status": metadata.get("code_parse_status") or "",
                "code_ast_available": bool(metadata.get("code_ast_available")),
                "code_snippet_count": int(metadata.get("code_snippet_count") or 0),
                "code_line_count": int(metadata.get("code_line_count") or 0),
                "code_functions": metadata.get("code_functions") or [],
                "code_classes": metadata.get("code_classes") or [],
                "code_imports": metadata.get("code_imports") or [],
                "code_endpoints": metadata.get("code_endpoints") or [],
                "code_symbol_count": int(metadata.get("code_symbol_count") or 0),
                "schema_ingestion_schema_version": metadata.get("schema_ingestion_schema_version") or "",
                "schema_ingestion_ready": bool(metadata.get("schema_ingestion_ready")),
                "schema_detected": bool(metadata.get("schema_detected")),
                "schema_type": metadata.get("schema_type") or "",
                "schema_parse_status": metadata.get("schema_parse_status") or "",
                "schema_names": metadata.get("schema_names") or [],
                "schema_field_names": metadata.get("schema_field_names") or [],
                "schema_required_fields": metadata.get("schema_required_fields") or [],
                "schema_types": metadata.get("schema_types") or [],
                "schema_constraints": metadata.get("schema_constraints") or [],
                "schema_endpoint_paths": metadata.get("schema_endpoint_paths") or [],
                "schema_field_count": int(metadata.get("schema_field_count") or 0),
                "knowledge_graph_ingestion_schema_version": metadata.get("knowledge_graph_ingestion_schema_version") or "",
                "knowledge_graph_ingestion_ready": bool(metadata.get("knowledge_graph_ingestion_ready")),
                "knowledge_graph_ready": bool(metadata.get("knowledge_graph_ready")),
                "knowledge_graph_node_count": int(metadata.get("knowledge_graph_node_count") or 0),
                "knowledge_graph_edge_count": int(metadata.get("knowledge_graph_edge_count") or 0),
                "knowledge_graph_node_types": metadata.get("knowledge_graph_node_types") or [],
                "knowledge_graph_relation_types": metadata.get("knowledge_graph_relation_types") or [],
                "knowledge_graph_node_keys": metadata.get("knowledge_graph_node_keys") or [],
                "ontology_ingestion_schema_version": metadata.get("ontology_ingestion_schema_version") or "",
                "ontology_ingestion_ready": bool(metadata.get("ontology_ingestion_ready")),
                "ontology_concepts": metadata.get("ontology_concepts") or [],
                "ontology_broader_terms": metadata.get("ontology_broader_terms") or [],
                "ontology_related_terms": metadata.get("ontology_related_terms") or [],
                "ontology_concept_count": int(metadata.get("ontology_concept_count") or 0),
                "safety_critical": bool(metadata.get("safety_critical")),
                "safety_flags": metadata.get("safety_flags") or [],
                "safety_score": float(metadata.get("safety_score") or 0),
                "safety_answer_policy": metadata.get("safety_answer_policy") or "",
                "compliance_related": bool(metadata.get("compliance_related")),
                "compliance_flags": metadata.get("compliance_flags") or [],
                "retrieval_tags": metadata.get("retrieval_tags") or [],
                "answer_scope_hint": metadata.get("answer_scope_hint") or "",
                "keyword_schema_version": metadata.get("keyword_schema_version") or "",
                "keywords": metadata.get("keywords") or [],
                "keyphrases": metadata.get("keyphrases") or [],
                "exact_terms": metadata.get("exact_terms") or [],
                "acronyms": metadata.get("acronyms") or [],
                "abbreviation_schema_version": metadata.get("abbreviation_schema_version") or "",
                "abbreviation_detection_ready": bool(metadata.get("abbreviation_detection_ready")),
                "abbreviation_count": int(metadata.get("abbreviation_count") or 0),
                "abbreviation_terms": metadata.get("abbreviation_terms") or [],
                "abbreviation_expansions": metadata.get("abbreviation_expansions") or [],
                "abbreviation_records": metadata.get("abbreviation_records") or [],
                "keyword_identifiers": metadata.get("keyword_identifiers") or [],
                "keyword_standards": metadata.get("keyword_standards") or [],
                "table_keyword_terms": metadata.get("table_keyword_terms") or [],
                "section_keyword_terms": metadata.get("section_keyword_terms") or [],
                "dedupe_schema_version": metadata.get("dedupe_schema_version") or metadata.get("db_dedupe_schema_version") or "",
                "dedupe_status": metadata.get("dedupe_status") or metadata.get("db_dedupe_status") or "",
                "dedupe_protected": bool(metadata.get("dedupe_protected") or metadata.get("db_dedupe_protected")),
                "dedupe_word_count": int(metadata.get("dedupe_word_count") or metadata.get("db_dedupe_word_count") or 0),
                "dedupe_unique_word_count": int(metadata.get("dedupe_unique_word_count") or metadata.get("db_dedupe_unique_word_count") or 0),
                "embedding_schema_version": metadata.get("embedding_schema_version") or "",
                "embedding_model": metadata.get("embedding_model") or "",
                "embedding_backend": metadata.get("embedding_backend") or "",
                "embedding_dimensions": int(metadata.get("embedding_dimensions") or 0),
                "embedding_kind": metadata.get("embedding_kind") or "",
                "embedding_truncated": bool(metadata.get("embedding_truncated")),
                "embedding_fallback": bool(metadata.get("embedding_fallback")),
                "embedding_vector_norm": float(metadata.get("embedding_vector_norm") or 0),
                "embedding_vector_valid": bool(metadata.get("embedding_vector_valid")),
                "embedding_input_chars": int(metadata.get("embedding_input_chars") or 0),
                "embedding_prepared_chars": int(metadata.get("embedding_prepared_chars") or 0),
                "embedding_text_hash": metadata.get("embedding_text_hash") or "",
                "hierarchical_embedding_schema_version": metadata.get("hierarchical_embedding_schema_version") or "",
                "hierarchical_embedding_ready": bool(metadata.get("hierarchical_embedding_ready")),
                "hierarchical_embedding_active_layers": metadata.get("hierarchical_embedding_active_layers") or [],
                "hierarchical_embedding_layer_count": int(metadata.get("hierarchical_embedding_layer_count") or 0),
                "hierarchical_embedding_hash": metadata.get("hierarchical_embedding_hash") or "",
                "hierarchical_embedding_strategy": metadata.get("hierarchical_embedding_strategy") or "",
                "context_window_schema_version": metadata.get("context_window_schema_version") or "",
                "recommended_window_before": int(metadata.get("recommended_window_before") or 0),
                "recommended_window_after": int(metadata.get("recommended_window_after") or 0),
                "vector_optimization_schema_version": metadata.get("vector_optimization_schema_version") or "",
                "vector_dense_terms": metadata.get("vector_dense_terms") or [],
                "vector_sparse_terms": metadata.get("vector_sparse_terms") or [],
                "hybrid_schema_version": metadata.get("hybrid_schema_version") or "",
                "hybrid_prepared": bool(metadata.get("hybrid_prepared")),
                "hybrid_dense_hash": metadata.get("hybrid_dense_hash") or "",
                "hybrid_sparse_hash": metadata.get("hybrid_sparse_hash") or "",
                "hybrid_dense_chars": int(metadata.get("hybrid_dense_chars") or 0),
                "hybrid_sparse_chars": int(metadata.get("hybrid_sparse_chars") or 0),
                "hybrid_sparse_terms": metadata.get("hybrid_sparse_terms") or [],
                "hybrid_exact_phrases": metadata.get("hybrid_exact_phrases") or [],
                "hybrid_facets": metadata.get("hybrid_facets") or {},
                "multi_index_schema_version": metadata.get("multi_index_schema_version") or "",
                "multi_index_prepared": bool(metadata.get("multi_index_prepared")),
                "multi_index_active_surfaces": metadata.get("multi_index_active_surfaces") or [],
                "multi_index_surface_count": int(metadata.get("multi_index_surface_count") or 0),
                "multi_index_payload": metadata.get("multi_index_payload") or {},
                "ingestion_pipeline_version": metadata.get("ingestion_pipeline_version") or "",
                "ingestion_stage_count": int(metadata.get("ingestion_stage_count") or 0),
                "ingestion_pipeline": metadata.get("ingestion_pipeline") or {},
                "document_link_schema_version": metadata.get("document_link_schema_version") or "",
                "document_link_prepared": bool(metadata.get("document_link_prepared")),
                "previous_chunk_id": metadata.get("previous_chunk_id") or "",
                "next_chunk_id": metadata.get("next_chunk_id") or "",
                "same_section_chunk_ids": metadata.get("same_section_chunk_ids") or [],
                "section_chunk_position": int(metadata.get("section_chunk_position") or 0),
                "section_chunk_count": int(metadata.get("section_chunk_count") or 0),
                "referenced_section_ids": metadata.get("referenced_section_ids") or [],
                "referenced_document_ids": metadata.get("referenced_document_ids") or [],
                "referenced_standards": metadata.get("referenced_standards") or [],
                "outbound_link_chunk_ids": metadata.get("outbound_link_chunk_ids") or [],
                "inbound_link_chunk_ids": metadata.get("inbound_link_chunk_ids") or [],
                "link_count": int(metadata.get("link_count") or 0),
                "inbound_link_count": int(metadata.get("inbound_link_count") or 0),
                "link_expanded_from": item.get("link_expanded_from") or "",
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


def row_metadata(row: dict) -> dict:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        return metadata
    return json.loads(metadata or "{}")


def enrich_document_links(chunks: list[tuple[int, str, dict]], filename: str) -> list[tuple[int, str, dict]]:
    if not chunks:
        return chunks
    enriched: list[tuple[int, str, dict]] = []
    chunk_ids = [str(metadata.get("chunk_id") or f"chunk-{index}") for index, _, metadata in chunks]
    section_to_chunks: dict[str, list[str]] = {}
    section_title_to_id: dict[str, str] = {}
    standard_to_chunks: dict[str, list[str]] = {}
    identifier_to_chunks: dict[str, list[str]] = {}
    inbound: dict[str, list[dict]] = {chunk_id: [] for chunk_id in chunk_ids}

    for chunk_id, (_, _, metadata) in zip(chunk_ids, chunks):
        section_id = str(metadata.get("current_section_id") or "")
        section_title = str(metadata.get("current_section_title") or metadata.get("section_title") or "")
        if section_id:
            section_to_chunks.setdefault(section_id.lower(), []).append(chunk_id)
        if section_title and section_id:
            section_title_to_id[normalize_link_key(section_title)] = section_id.lower()
        for standard in metadata.get("standards") or []:
            standard_to_chunks.setdefault(normalize_link_key(standard), []).append(chunk_id)
        for identifier in (metadata.get("technical_identifiers") or []) + ([metadata.get("document_identifier")] if metadata.get("document_identifier") else []):
            identifier_to_chunks.setdefault(normalize_link_key(identifier), []).append(chunk_id)

    link_payloads: list[dict] = []
    for position, (index, text, metadata) in enumerate(chunks):
        chunk_id = chunk_ids[position]
        previous_id = chunk_ids[position - 1] if position > 0 else ""
        next_id = chunk_ids[position + 1] if position + 1 < len(chunk_ids) else ""
        section_id = str(metadata.get("current_section_id") or "")
        section_key = section_id.lower()
        section_members = section_to_chunks.get(section_key, [])
        section_position = section_members.index(chunk_id) + 1 if chunk_id in section_members else 0
        explicit_refs = extract_document_references(text, metadata)
        resolved_section_links = resolve_section_links(explicit_refs["section_refs"], section_to_chunks, section_title_to_id)
        standard_links = resolve_reference_links(explicit_refs["standard_refs"], standard_to_chunks)
        identifier_links = resolve_reference_links(explicit_refs["document_refs"], identifier_to_chunks)
        outbound = linked_targets(resolved_section_links, standard_links, identifier_links, source_chunk_id=chunk_id)
        link_payload = {
            "document_link_schema_version": DOCUMENT_LINK_SCHEMA_VERSION,
            "document_link_prepared": True,
            "document_link_source": filename,
            "previous_chunk_id": previous_id,
            "next_chunk_id": next_id,
            "same_section_chunk_ids": [item for item in section_members if item != chunk_id][:12],
            "section_chunk_position": section_position,
            "section_chunk_count": len(section_members),
            "referenced_section_ids": explicit_refs["section_refs"],
            "referenced_document_ids": explicit_refs["document_refs"],
            "referenced_standards": explicit_refs["standard_refs"],
            "resolved_section_chunk_ids": resolved_section_links,
            "resolved_standard_chunk_ids": standard_links,
            "resolved_document_chunk_ids": identifier_links,
            "outbound_link_chunk_ids": outbound,
            "link_count": len(outbound),
        }
        for target_id in outbound:
            inbound.setdefault(target_id, []).append({"from_chunk_id": chunk_id, "type": "reference"})
        link_payloads.append(link_payload)

    for position, (index, text, metadata) in enumerate(chunks):
        chunk_id = chunk_ids[position]
        inbound_links = inbound.get(chunk_id, [])
        link_payload = {
            **link_payloads[position],
            "inbound_link_chunk_ids": [item["from_chunk_id"] for item in inbound_links][:20],
            "inbound_link_count": len(inbound_links),
        }
        enriched.append((index, text, {**metadata, **link_payload}))
    return enriched


def extract_document_references(text: str, metadata: dict) -> dict[str, list[str]]:
    combined = " ".join(
        [
            text,
            " ".join(metadata.get("exact_terms") or []),
            " ".join(metadata.get("technical_identifiers") or []),
            " ".join(metadata.get("standards") or []),
        ]
    )
    section_refs = []
    for match in re.finditer(r"\b(?:section|sec\.?|clause|para(?:graph)?)\s+((?:\d+\.)*\d+[A-Z]?)\b", combined, flags=re.I):
        section_refs.append(match.group(1).rstrip("."))
    for match in re.finditer(r"\b(?:see|refer(?:\s+to)?|as per|in accordance with)\s+((?:\d+\.)*\d+[A-Z]?)\b", combined, flags=re.I):
        section_refs.append(match.group(1).rstrip("."))
    document_refs = []
    for match in re.finditer(r"\b[A-Z]{2,}[A-Z0-9_-]*(?:-[A-Z0-9]+){2,}\b", combined):
        document_refs.append(match.group(0).strip("-_"))
    standard_refs = []
    for pattern in [
        r"\bASME\s+[A-Z]\d+(?:\.\d+)*[A-Z0-9.-]*",
        r"\bNORSOK\s+[A-Z]-\d+[A-Z0-9.-]*",
        r"\bAPI\s+\d+[A-Z0-9.-]*",
        r"\bISO\s+\d+[A-Z0-9:.-]*",
        r"\bIEC\s+\d+[A-Z0-9:.-]*",
    ]:
        standard_refs.extend(match.group(0).upper().rstrip(".,;:") for match in re.finditer(pattern, combined, flags=re.I))
    return {
        "section_refs": unique_preserve(section_refs)[:20],
        "document_refs": unique_preserve(document_refs)[:20],
        "standard_refs": unique_preserve(standard_refs)[:20],
    }


def resolve_section_links(section_refs: list[str], section_to_chunks: dict[str, list[str]], section_title_to_id: dict[str, str]) -> list[str]:
    targets: list[str] = []
    for ref in section_refs:
        key = normalize_link_key(ref)
        section_key = key if key in section_to_chunks else section_title_to_id.get(key, "")
        targets.extend(section_to_chunks.get(section_key, [])[:6])
    return unique_preserve(targets)[:20]


def resolve_reference_links(refs: list[str], index: dict[str, list[str]]) -> list[str]:
    targets: list[str] = []
    for ref in refs:
        targets.extend(index.get(normalize_link_key(ref), [])[:6])
    return unique_preserve(targets)[:20]


def linked_targets(*groups: list[str], source_chunk_id: str) -> list[str]:
    targets: list[str] = []
    for group in groups:
        for item in group:
            if item != source_chunk_id and item not in targets:
                targets.append(item)
    return targets[:30]


def normalize_link_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip(" .,:;")


def unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        key = normalize_link_key(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def prepare_hybrid_metadata(text: str, metadata: dict) -> dict:
    dense_text = limit_hybrid_text(searchable_text(text, metadata))
    sparse_text = limit_hybrid_text(hybrid_sparse_text(text, metadata))
    exact_phrases = hybrid_exact_phrases(text, metadata)
    sparse_terms = hybrid_sparse_terms(sparse_text, metadata)
    table_surface = limit_hybrid_text(hybrid_table_text(text, metadata))
    metadata_surface = limit_hybrid_text(hybrid_metadata_text(metadata))
    numeric_surface = hybrid_numeric_text(metadata)
    facets = hybrid_facets(metadata)
    prepared = {
        **metadata,
        "hybrid_schema_version": HYBRID_SCHEMA_VERSION,
        "hybrid_dense_text": dense_text,
        "hybrid_sparse_text": sparse_text,
        "hybrid_exact_phrases": exact_phrases,
        "hybrid_sparse_terms": sparse_terms,
        "hybrid_table_text": table_surface,
        "hybrid_metadata_text": metadata_surface,
        "hybrid_numeric_text": numeric_surface,
        "hybrid_facets": facets,
        "hybrid_dense_hash": hashlib.sha1(dense_text.encode("utf-8")).hexdigest()[:16],
        "hybrid_sparse_hash": hashlib.sha1(sparse_text.encode("utf-8")).hexdigest()[:16],
        "hybrid_dense_chars": len(dense_text),
        "hybrid_sparse_chars": len(sparse_text),
        "hybrid_prepared": True,
    }
    return prepared


def prepare_multi_index_metadata(text: str, metadata: dict) -> dict:
    metadata = prepare_hierarchical_embedding_metadata(text, metadata)
    dense_text = metadata.get("hybrid_dense_text") or searchable_text(text, metadata)
    sparse_text = metadata.get("hybrid_sparse_text") or hybrid_sparse_text(text, metadata)
    hierarchical_text = metadata.get("hierarchical_embedding_dense_text") or hierarchical_embedding_text(text, metadata)
    table_text = metadata.get("hybrid_table_text") or hybrid_table_text(text, metadata)
    metadata_text = metadata.get("hybrid_metadata_text") or hybrid_metadata_text(metadata)
    numeric_text = metadata.get("hybrid_numeric_text") or hybrid_numeric_text(metadata)
    entity_text = entity_metadata_text(metadata)
    citation_text = citation_index_text(metadata)
    code_text = code_index_text(metadata)
    schema_text = schema_index_text(metadata)
    knowledge_graph_text = knowledge_graph_index_text(metadata)
    ontology_text = ontology_index_text(metadata)
    payload = {
        "dense": multi_index_surface_payload("dense", dense_text, "pgvector_hnsw"),
        "hierarchical": multi_index_surface_payload("hierarchical", hierarchical_text, "pgvector_hnsw_weighted_hierarchy"),
        "sparse": multi_index_surface_payload("sparse", sparse_text, "postgres_tsvector_gin"),
        "table": multi_index_surface_payload("table", table_text, "postgres_tsvector_gin"),
        "metadata": multi_index_surface_payload("metadata", metadata_text, "postgres_tsvector_gin_jsonb"),
        "numeric": multi_index_surface_payload("numeric", numeric_text, "postgres_tsvector_gin"),
        "entity": multi_index_surface_payload("entity", entity_text, "postgres_tsvector_gin"),
        "citation": multi_index_surface_payload("citation", citation_text, "postgres_tsvector_gin"),
        "code": multi_index_surface_payload("code", code_text, "jsonb_payload_sparse_surface"),
        "schema": multi_index_surface_payload("schema", schema_text, "jsonb_payload_sparse_surface"),
        "knowledge_graph": multi_index_surface_payload("knowledge_graph", knowledge_graph_text, "jsonb_payload_graph_surface"),
        "ontology": multi_index_surface_payload("ontology", ontology_text, "jsonb_payload_taxonomy_surface"),
    }
    active = [name for name, surface in payload.items() if surface["chars"] > 0]
    return {
        **metadata,
        "multi_index_schema_version": MULTI_INDEX_SCHEMA_VERSION,
        "multi_index_prepared": True,
        "multi_index_active_surfaces": active,
        "multi_index_surface_count": len(active),
        "multi_index_payload": payload,
        "multi_index_dense_text": limit_hybrid_text(dense_text),
        "multi_index_hierarchical_text": limit_hybrid_text(hierarchical_text),
        "multi_index_sparse_text": limit_hybrid_text(sparse_text),
        "multi_index_table_text": limit_hybrid_text(table_text),
        "multi_index_metadata_text": limit_hybrid_text(metadata_text),
        "multi_index_numeric_text": limit_hybrid_text(numeric_text),
        "multi_index_entity_text": limit_hybrid_text(entity_text),
        "multi_index_citation_text": limit_hybrid_text(citation_text),
        "multi_index_code_text": limit_hybrid_text(code_text),
        "multi_index_schema_text": limit_hybrid_text(schema_text),
        "multi_index_knowledge_graph_text": limit_hybrid_text(knowledge_graph_text),
        "multi_index_ontology_text": limit_hybrid_text(ontology_text),
    }


def multi_index_surface_payload(name: str, text: str, backend: str) -> dict:
    text = limit_hybrid_text(text)
    return {
        "name": name,
        "backend": backend,
        "chars": len(text),
        "hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else "",
        "enabled": bool(text),
    }


def citation_index_text(metadata: dict) -> str:
    parts = [
        metadata.get("filename") or "",
        metadata.get("document_title") or "",
        metadata.get("document_identifier") or "",
        metadata.get("revision") or "",
        metadata.get("section_breadcrumb") or "",
        metadata.get("current_section_id") or "",
        metadata.get("current_section_title") or "",
        metadata.get("parent_section_id") or "",
        metadata.get("parent_section_title") or "",
        " ".join(metadata.get("section_ids") or []),
        " ".join(metadata.get("referenced_section_ids") or []),
        " ".join(metadata.get("referenced_document_ids") or []),
        " ".join(metadata.get("referenced_standards") or []),
        " ".join(metadata.get("resolved_section_chunk_ids") or []),
        " ".join(metadata.get("resolved_document_chunk_ids") or []),
        " ".join(metadata.get("resolved_standard_chunk_ids") or []),
        " ".join(metadata.get("outbound_link_chunk_ids") or []),
        " ".join(metadata.get("inbound_link_chunk_ids") or []),
    ]
    return normalize_link_key(" ".join(str(part) for part in parts if part))


def hybrid_sparse_text(text: str, metadata: dict) -> str:
    weighted = " ".join(str(item.get("term", "")) for item in (metadata.get("weighted_keywords") or []) if isinstance(item, dict))
    records = " ".join(" ".join(str(value) for value in record.values()) for record in (metadata.get("table_records") or []) if isinstance(record, dict))
    numeric = hybrid_numeric_text(metadata)
    parts = [
        metadata.get("document_title") or "",
        metadata.get("section_breadcrumb") or "",
        metadata.get("section_title") or "",
        metadata.get("parent_section") or "",
        " ".join(metadata.get("section_ids") or []),
        " ".join(metadata.get("keywords") or []),
        " ".join(metadata.get("keyphrases") or []),
        " ".join(metadata.get("exact_terms") or []),
        " ".join(metadata.get("acronyms") or []),
        " ".join(metadata.get("abbreviation_terms") or []),
        " ".join(metadata.get("abbreviation_expansions") or []),
        metadata.get("abbreviation_retrieval_text") or "",
        " ".join(metadata.get("keyword_identifiers") or []),
        " ".join(metadata.get("keyword_standards") or []),
        " ".join(metadata.get("table_keyword_terms") or []),
        " ".join(metadata.get("section_keyword_terms") or []),
        " ".join(metadata.get("domain_terms") or []),
        " ".join(metadata.get("engineering_entities") or []),
        " ".join(metadata.get("engineering_canonical_entities") or []),
        " ".join(metadata.get("primary_entities") or []),
        entity_metadata_text(metadata),
        metadata.get("language_retrieval_text") or "",
        " ".join(metadata.get("detected_scripts") or []),
        metadata.get("primary_script") or "",
        " ".join(metadata.get("technical_identifiers") or []),
        " ".join(metadata.get("standards") or []),
        " ".join(metadata.get("requirement_modalities") or []),
        " ".join(metadata.get("retrieval_tags") or []),
        " ".join(metadata.get("semantic_labels") or []),
        metadata.get("document_class") or "",
        metadata.get("document_class_label") or "",
        metadata.get("document_class_routing_hint") or "",
        " ".join(metadata.get("document_class_retrieval_tags") or []),
        " ".join(metadata.get("query_expansion_terms") or []),
        " ".join(metadata.get("query_exact_match_terms") or []),
        " ".join(metadata.get("figure_references") or []),
        " ".join(metadata.get("figure_captions") or []),
        " ".join(metadata.get("modalities") or []),
        " ".join(metadata.get("reference_section_ids") or []),
        " ".join(metadata.get("reference_standards") or []),
        " ".join(metadata.get("reference_document_ids") or []),
        " ".join(metadata.get("relationship_types") or []),
        " ".join(metadata.get("dependency_terms") or []),
        " ".join(metadata.get("safety_terms") or []),
        " ".join(metadata.get("hazard_terms") or []),
        " ".join(metadata.get("numeric_units") or []),
        " ".join(metadata.get("numeric_unit_families") or []),
        " ".join(metadata.get("canonical_numeric_units") or []),
        metadata.get("unit_normalization_retrieval_text") or "",
        metadata.get("formula_retrieval_text") or "",
        " ".join(metadata.get("formula_equations") or []),
        " ".join(metadata.get("formula_ratios") or []),
        " ".join(metadata.get("formula_variables") or []),
        metadata.get("code_retrieval_text") or "",
        " ".join(metadata.get("code_functions") or []),
        " ".join(metadata.get("code_classes") or []),
        " ".join(metadata.get("code_imports") or []),
        " ".join(metadata.get("code_calls") or []),
        " ".join(metadata.get("code_assignments") or []),
        " ".join(metadata.get("code_endpoints") or []),
        " ".join(metadata.get("code_identifiers") or []),
        metadata.get("schema_retrieval_text") or "",
        " ".join(metadata.get("schema_names") or []),
        " ".join(metadata.get("schema_field_names") or []),
        " ".join(metadata.get("schema_required_fields") or []),
        " ".join(metadata.get("schema_types") or []),
        " ".join(metadata.get("schema_constraints") or []),
        " ".join(metadata.get("schema_endpoint_paths") or []),
        " ".join(metadata.get("schema_component_names") or []),
        " ".join(metadata.get("schema_table_names") or []),
        metadata.get("knowledge_graph_retrieval_text") or "",
        " ".join(metadata.get("knowledge_graph_node_keys") or []),
        " ".join(metadata.get("knowledge_graph_edge_keys") or []),
        " ".join(metadata.get("knowledge_graph_node_types") or []),
        " ".join(metadata.get("knowledge_graph_relation_types") or []),
        metadata.get("ontology_retrieval_text") or "",
        " ".join(metadata.get("ontology_concepts") or []),
        " ".join(metadata.get("ontology_broader_terms") or []),
        " ".join(metadata.get("ontology_related_terms") or []),
        " ".join(metadata.get("ontology_synonyms") or []),
        " ".join(metadata.get("vector_dense_terms") or []),
        " ".join(metadata.get("vector_sparse_terms") or []),
        " ".join(metadata.get("referenced_section_ids") or []),
        " ".join(metadata.get("referenced_document_ids") or []),
        " ".join(metadata.get("referenced_standards") or []),
        " ".join(metadata.get("resolved_section_chunk_ids") or []),
        " ".join(metadata.get("outbound_link_chunk_ids") or []),
        " ".join(metadata.get("inbound_link_chunk_ids") or []),
        metadata.get("table_title") or "",
        " ".join(metadata.get("table_columns") or []),
        " ".join(metadata.get("table_rows") or []),
        records,
        numeric,
        weighted,
        text,
    ]
    return "\n".join(part for part in parts if part)


def hybrid_table_text(text: str, metadata: dict) -> str:
    if not metadata.get("contains_table"):
        return ""
    records = " ".join(" ".join(str(value) for value in record.values()) for record in (metadata.get("table_records") or []) if isinstance(record, dict))
    return "\n".join(
        part
        for part in [
            metadata.get("table_title") or "",
            " ".join(metadata.get("table_columns") or []),
            " ".join(metadata.get("table_rows") or []),
            records,
            " ".join(metadata.get("table_notes") or []),
            " ".join(metadata.get("table_terms") or []),
            " ".join(metadata.get("table_keyword_terms") or []),
            text,
        ]
        if part
    )


def hybrid_metadata_text(metadata: dict) -> str:
    parts = [
        metadata.get("document_title") or "",
        metadata.get("filename") or "",
        metadata.get("section_breadcrumb") or "",
        metadata.get("section_title") or "",
        metadata.get("parent_section") or "",
        metadata.get("current_section_id") or "",
        metadata.get("current_section_title") or "",
        metadata.get("parent_section_id") or "",
        metadata.get("parent_section_title") or "",
        metadata.get("revision") or "",
        metadata.get("document_identifier") or "",
        metadata.get("validity_status") or "",
        " ".join(metadata.get("revision_candidates") or []),
        " ".join(metadata.get("revision_dates") or []),
        entity_metadata_text(metadata),
        metadata.get("language_code") or "",
        metadata.get("language_name") or "",
        metadata.get("language_retrieval_text") or "",
        " ".join(metadata.get("detected_scripts") or []),
        metadata.get("primary_script") or "",
        metadata.get("translation_status") or "",
        metadata.get("translation_method") or "",
        metadata.get("translation_retrieval_text") or "",
        metadata.get("translation_text") or "",
        metadata.get("code_language") or "",
        metadata.get("code_parse_status") or "",
        metadata.get("code_retrieval_text") or "",
        " ".join(metadata.get("code_functions") or []),
        " ".join(metadata.get("code_classes") or []),
        " ".join(metadata.get("code_imports") or []),
        metadata.get("schema_type") or "",
        metadata.get("schema_parse_status") or "",
        metadata.get("schema_retrieval_text") or "",
        " ".join(metadata.get("schema_names") or []),
        " ".join(metadata.get("schema_field_names") or []),
        " ".join(metadata.get("schema_required_fields") or []),
        metadata.get("unit_normalization_retrieval_text") or "",
        " ".join(metadata.get("numeric_units") or []),
        " ".join(metadata.get("numeric_unit_families") or []),
        " ".join(metadata.get("canonical_numeric_units") or []),
        metadata.get("knowledge_graph_retrieval_text") or "",
        " ".join(metadata.get("knowledge_graph_node_keys") or []),
        " ".join(metadata.get("knowledge_graph_relation_types") or []),
        metadata.get("ontology_retrieval_text") or "",
        " ".join(metadata.get("ontology_concepts") or []),
        " ".join(metadata.get("ontology_broader_terms") or []),
        " ".join(metadata.get("ontology_related_terms") or []),
        metadata.get("access_classification") or "",
        " ".join(metadata.get("access_control_tags") or []),
        metadata.get("access_policy_action") or "",
        " ".join(metadata.get("access_allowed_roles") or []),
        " ".join(metadata.get("semantic_labels") or []),
        " ".join(metadata.get("figure_references") or []),
        " ".join(metadata.get("modalities") or []),
        " ".join(metadata.get("reference_section_ids") or []),
        " ".join(metadata.get("relationship_types") or []),
        " ".join(metadata.get("standards") or []),
        " ".join(metadata.get("safety_flags") or []),
        " ".join(metadata.get("compliance_flags") or []),
        " ".join(metadata.get("layout_regions") or []),
        " ".join(metadata.get("page_regions") or []),
        metadata.get("answer_scope_hint") or "",
        " ".join(metadata.get("referenced_section_ids") or []),
        " ".join(metadata.get("referenced_document_ids") or []),
        " ".join(metadata.get("referenced_standards") or []),
    ]
    return "\n".join(part for part in parts if part)


def hybrid_numeric_text(metadata: dict) -> str:
    values = []
    for item in metadata.get("numeric_constraints") or []:
        if isinstance(item, dict):
            values.append(" ".join(str(item.get(key, "")) for key in ["value", "unit", "context"]))
    for item in metadata.get("normalized_numeric_constraints") or []:
        if isinstance(item, dict):
            values.append(
                " ".join(
                    str(item.get(key, ""))
                    for key in [
                        "value",
                        "unit",
                        "normalized_unit",
                        "unit_family",
                        "canonical_value_text",
                        "canonical_unit",
                        "search_token",
                    ]
                )
            )
    values.extend(metadata.get("numeric_units") or [])
    values.extend(metadata.get("numeric_unit_families") or [])
    values.extend(metadata.get("canonical_numeric_units") or [])
    if metadata.get("unit_normalization_retrieval_text"):
        values.append(str(metadata.get("unit_normalization_retrieval_text") or ""))
    values.extend(metadata.get("table_rows") or [])
    return " ".join(value for value in values if value)


def entity_metadata_text(metadata: dict) -> str:
    parts: list[str] = []
    parts.extend(metadata.get("engineering_entities") or [])
    parts.extend(metadata.get("engineering_canonical_entities") or [])
    parts.extend(metadata.get("primary_entities") or [])
    parts.extend(metadata.get("engineering_entity_aliases") or [])
    parts.extend(metadata.get("engineering_entity_types") or [])
    parts.extend(metadata.get("technical_identifiers") or [])
    parts.extend(metadata.get("standards") or [])
    if metadata.get("entity_surface_text"):
        parts.append(str(metadata.get("entity_surface_text") or ""))
    for record in metadata.get("engineering_entity_records") or []:
        if not isinstance(record, dict):
            continue
        parts.extend(
            [
                str(record.get("text") or ""),
                str(record.get("canonical") or ""),
                str(record.get("type") or ""),
                " ".join(str(alias) for alias in record.get("aliases") or []),
                " ".join(str(context) for context in record.get("contexts") or []),
            ]
        )
    for relation in metadata.get("engineering_entity_relationships") or []:
        if isinstance(relation, dict):
            parts.append(" ".join(str(relation.get(key) or "") for key in ("left", "relation", "right")))
    return " ".join(part for part in parts if part)


def code_index_text(metadata: dict) -> str:
    parts = [
        metadata.get("code_language") or "",
        metadata.get("code_parse_status") or "",
        metadata.get("code_retrieval_text") or "",
        " ".join(metadata.get("code_functions") or []),
        " ".join(metadata.get("code_classes") or []),
        " ".join(metadata.get("code_imports") or []),
        " ".join(metadata.get("code_decorators") or []),
        " ".join(metadata.get("code_calls") or []),
        " ".join(metadata.get("code_assignments") or []),
        " ".join(metadata.get("code_endpoints") or []),
        " ".join(metadata.get("code_sql_terms") or []),
        " ".join(metadata.get("code_schema_terms") or []),
        " ".join(metadata.get("code_identifiers") or []),
    ]
    return " ".join(part for part in parts if part)


def schema_index_text(metadata: dict) -> str:
    field_records = []
    for record in metadata.get("schema_fields") or []:
        if isinstance(record, dict):
            field_records.append(" ".join(str(record.get(key) or "") for key in ("name", "type", "source", "required")))
    parts = [
        metadata.get("schema_type") or "",
        metadata.get("schema_parse_status") or "",
        metadata.get("schema_retrieval_text") or "",
        " ".join(metadata.get("schema_names") or []),
        " ".join(metadata.get("schema_field_names") or []),
        " ".join(metadata.get("schema_required_fields") or []),
        " ".join(metadata.get("schema_optional_fields") or []),
        " ".join(metadata.get("schema_types") or []),
        " ".join(metadata.get("schema_constraints") or []),
        " ".join(metadata.get("schema_relations") or []),
        " ".join(metadata.get("schema_endpoint_paths") or []),
        " ".join(metadata.get("schema_component_names") or []),
        " ".join(metadata.get("schema_table_names") or []),
        " ".join(field_records),
    ]
    return " ".join(part for part in parts if part)


def knowledge_graph_index_text(metadata: dict) -> str:
    node_records = []
    for record in metadata.get("knowledge_graph_nodes") or []:
        if isinstance(record, dict):
            node_records.append(" ".join(str(record.get(key) or "") for key in ("key", "type", "label")))
    edge_records = []
    for record in metadata.get("knowledge_graph_edges") or []:
        if isinstance(record, dict):
            edge_records.append(" ".join(str(record.get(key) or "") for key in ("source", "relation", "target", "source_type")))
    parts = [
        metadata.get("knowledge_graph_retrieval_text") or "",
        " ".join(metadata.get("knowledge_graph_node_keys") or []),
        " ".join(metadata.get("knowledge_graph_edge_keys") or []),
        " ".join(metadata.get("knowledge_graph_node_types") or []),
        " ".join(metadata.get("knowledge_graph_relation_types") or []),
        " ".join(node_records),
        " ".join(edge_records),
    ]
    return " ".join(part for part in parts if part)


def ontology_index_text(metadata: dict) -> str:
    match_records = []
    for record in metadata.get("ontology_matches") or []:
        if isinstance(record, dict):
            match_records.append(
                " ".join(
                    [
                        str(record.get("concept") or ""),
                        " ".join(str(item) for item in record.get("concept_hits") or []),
                        " ".join(str(item) for item in record.get("broader_hits") or []),
                        " ".join(str(item) for item in record.get("related_hits") or []),
                    ]
                )
            )
    parts = [
        metadata.get("ontology_retrieval_text") or "",
        " ".join(metadata.get("ontology_concepts") or []),
        " ".join(metadata.get("ontology_broader_terms") or []),
        " ".join(metadata.get("ontology_related_terms") or []),
        " ".join(metadata.get("ontology_synonyms") or []),
        " ".join(metadata.get("ontology_graph_node_keys") or []),
        " ".join(match_records),
    ]
    return " ".join(part for part in parts if part)


def hybrid_exact_phrases(text: str, metadata: dict) -> list[str]:
    phrases: list[str] = []
    for source in [
        metadata.get("keyphrases") or [],
        metadata.get("exact_terms") or [],
        metadata.get("technical_identifiers") or [],
        metadata.get("standards") or [],
        metadata.get("engineering_entities") or [],
        metadata.get("engineering_canonical_entities") or [],
        metadata.get("primary_entities") or [],
        metadata.get("code_functions") or [],
        metadata.get("code_classes") or [],
        metadata.get("code_imports") or [],
        metadata.get("code_endpoints") or [],
        metadata.get("schema_names") or [],
        metadata.get("schema_field_names") or [],
        metadata.get("schema_required_fields") or [],
        metadata.get("schema_component_names") or [],
        metadata.get("schema_endpoint_paths") or [],
        metadata.get("knowledge_graph_node_keys") or [],
        metadata.get("knowledge_graph_relation_types") or [],
        metadata.get("ontology_concepts") or [],
        metadata.get("ontology_broader_terms") or [],
        metadata.get("ontology_related_terms") or [],
        metadata.get("numeric_units") or [],
        metadata.get("numeric_unit_families") or [],
        metadata.get("canonical_numeric_units") or [],
        metadata.get("table_columns") or [],
    ]:
        for phrase in source:
            normalized = re.sub(r"\s+", " ", str(phrase).lower()).strip()
            if len(normalized) >= 3 and normalized not in phrases:
                phrases.append(normalized)
    phrases.extend(query_exact_phrases(text))
    return list(dict.fromkeys(phrases))[:80]


def hybrid_sparse_terms(sparse_text: str, metadata: dict) -> list[str]:
    terms = []
    terms.extend(metadata.get("keywords") or [])
    terms.extend(metadata.get("retrieval_tags") or [])
    terms.extend(metadata.get("domain_terms") or [])
    terms.extend(metadata.get("engineering_entities") or [])
    terms.extend(metadata.get("engineering_canonical_entities") or [])
    terms.extend(metadata.get("primary_entities") or [])
    terms.extend(metadata.get("table_keyword_terms") or [])
    terms.extend(re.findall(r"[A-Za-z0-9_.-]{2,}", sparse_text[:6000]))
    counts = Counter(normalize_sparse_term(term) for term in terms)
    ranked = [term for term, _ in counts.most_common(SPARSE_TERM_LIMIT) if term]
    return ranked


def normalize_sparse_term(term: str) -> str:
    term = re.sub(r"[^A-Za-z0-9_.-]+", " ", str(term).lower()).strip()
    if term.endswith("s") and len(term) > 4:
        term = term[:-1]
    return term


def hybrid_facets(metadata: dict) -> dict:
    return {
        "content_types": metadata.get("content_types") or [],
        "contains_table": bool(metadata.get("contains_table")),
        "safety_critical": bool(metadata.get("safety_critical")),
        "compliance_related": bool(metadata.get("compliance_related")),
        "has_numeric_constraints": bool(metadata.get("has_numeric_constraints")),
        "section_id": metadata.get("current_section_id") or "",
        "section_title": metadata.get("current_section_title") or metadata.get("section_title") or "",
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "column_count": metadata.get("column_count") or 1,
        "chunk_strategy": metadata.get("chunk_strategy") or "",
        "link_count": metadata.get("link_count") or 0,
        "entity_types": metadata.get("engineering_entity_types") or [],
        "entity_count": metadata.get("entity_count") or 0,
        "primary_entities": metadata.get("primary_entities") or [],
        "language_code": metadata.get("language_code") or "",
        "code_detected": bool(metadata.get("code_detected")),
        "code_language": metadata.get("code_language") or "",
        "code_ast_available": bool(metadata.get("code_ast_available")),
        "schema_detected": bool(metadata.get("schema_detected")),
        "schema_type": metadata.get("schema_type") or "",
        "knowledge_graph_ready": bool(metadata.get("knowledge_graph_ready")),
        "knowledge_graph_node_count": int(metadata.get("knowledge_graph_node_count") or 0),
        "knowledge_graph_edge_count": int(metadata.get("knowledge_graph_edge_count") or 0),
        "access_classification": metadata.get("access_classification") or "",
        "access_policy_required": bool(metadata.get("access_policy_required")),
    }


def limit_hybrid_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= HYBRID_TEXT_LIMIT:
        return text
    return text[:HYBRID_TEXT_LIMIT].rsplit(" ", 1)[0] or text[:HYBRID_TEXT_LIMIT]


def searchable_text(text: str, metadata: dict) -> str:
    section = " > ".join(metadata.get("section_path") or [])
    section_ids = " ".join(metadata.get("section_ids") or [])
    section_titles = " ".join(metadata.get("section_titles") or [])
    document_title = metadata.get("document_title") or ""
    section_breadcrumb = metadata.get("section_breadcrumb") or ""
    keywords = " ".join(metadata.get("keywords") or [])
    keyphrases = " ".join(metadata.get("keyphrases") or [])
    exact_terms = " ".join(metadata.get("exact_terms") or [])
    acronyms = " ".join(metadata.get("acronyms") or [])
    abbreviation_text = " ".join((metadata.get("abbreviation_terms") or []) + (metadata.get("abbreviation_expansions") or []))
    keyword_identifiers = " ".join(metadata.get("keyword_identifiers") or [])
    keyword_standards = " ".join(metadata.get("keyword_standards") or [])
    table_keyword_terms = " ".join(metadata.get("table_keyword_terms") or [])
    section_keyword_terms = " ".join(metadata.get("section_keyword_terms") or [])
    overlap_context = metadata.get("overlap_context") or ""
    table_title = metadata.get("table_title") or ""
    table_columns = " ".join(metadata.get("table_columns") or [])
    table_rows = " ".join(metadata.get("table_rows") or [])
    table_records = " ".join(" ".join(record.values()) for record in (metadata.get("table_records") or []) if isinstance(record, dict))
    table_notes = " ".join(metadata.get("table_notes") or [])
    table_terms = " ".join(metadata.get("table_terms") or [])
    layout_regions = " ".join(metadata.get("layout_regions") or [])
    page_regions = " ".join(metadata.get("page_regions") or [])
    horizontal_regions = " ".join(metadata.get("horizontal_regions") or [])
    columns = " ".join(f"column {column}" for column in (metadata.get("columns") or []))
    chunk_strategy = metadata.get("chunk_strategy") or ""
    boundary_reason = metadata.get("chunk_boundary_reason") or ""
    domain_terms = " ".join(metadata.get("domain_terms") or [])
    entities = " ".join(metadata.get("engineering_entities") or [])
    canonical_entities = " ".join(metadata.get("engineering_canonical_entities") or [])
    primary_entities = " ".join(metadata.get("primary_entities") or [])
    entity_details = entity_metadata_text(metadata)
    identifiers = " ".join(metadata.get("technical_identifiers") or [])
    language_text = " ".join(
        str(metadata.get(key) or "")
        for key in ["language_code", "language_name", "translation_status", "translation_method", "translation_retrieval_text", "access_classification"]
    )
    access_tags = " ".join(metadata.get("access_control_tags") or [])
    access_policy = metadata.get("access_policy_action") or ""
    access_roles = " ".join(metadata.get("access_allowed_roles") or [])
    standards = " ".join(metadata.get("standards") or [])
    modalities = " ".join(metadata.get("requirement_modalities") or [])
    numeric_constraints = " ".join(
        " ".join(str(item.get(key, "")) for key in ["value", "unit", "context"])
        for item in (metadata.get("numeric_constraints") or [])
        if isinstance(item, dict)
    )
    unit_text = hybrid_numeric_text(metadata)
    formula_text = " ".join(
        str(item)
        for item in (
            [metadata.get("formula_retrieval_text") or ""]
            + (metadata.get("formula_equations") or [])
            + (metadata.get("formula_ratios") or [])
            + (metadata.get("formula_variables") or [])
            + (metadata.get("formula_units") or [])
        )
    )
    code_text = " ".join(
        str(item)
        for item in (
            [
                metadata.get("code_language") or "",
                metadata.get("code_parse_status") or "",
                metadata.get("code_retrieval_text") or "",
            ]
            + (metadata.get("code_functions") or [])
            + (metadata.get("code_classes") or [])
            + (metadata.get("code_imports") or [])
            + (metadata.get("code_calls") or [])
            + (metadata.get("code_assignments") or [])
            + (metadata.get("code_endpoints") or [])
            + (metadata.get("code_identifiers") or [])
        )
    )
    schema_text = schema_index_text(metadata)
    knowledge_graph_text = knowledge_graph_index_text(metadata)
    ontology_text = ontology_index_text(metadata)
    safety_flags = " ".join(metadata.get("safety_flags") or [])
    ingestion_signals = " ".join(
        str(item)
        for item in (
            (metadata.get("semantic_labels") or [])
            + (metadata.get("document_class_retrieval_tags") or [])
            + (metadata.get("query_expansion_terms") or [])
            + (metadata.get("figure_references") or [])
            + (metadata.get("figure_captions") or [])
            + (metadata.get("modalities") or [])
            + (metadata.get("reference_section_ids") or [])
            + (metadata.get("reference_standards") or [])
            + (metadata.get("relationship_types") or [])
            + (metadata.get("dependency_terms") or [])
            + (metadata.get("safety_terms") or [])
            + (metadata.get("hazard_terms") or [])
            + (metadata.get("numeric_units") or [])
            + (metadata.get("vector_dense_terms") or [])
            + (metadata.get("vector_sparse_terms") or [])
        )
    )
    compliance_flags = " ".join(metadata.get("compliance_flags") or [])
    retrieval_tags = " ".join(metadata.get("retrieval_tags") or [])
    answer_scope_hint = metadata.get("answer_scope_hint") or ""
    link_text = " ".join(
        str(item)
        for item in (
            (metadata.get("referenced_section_ids") or [])
            + (metadata.get("referenced_document_ids") or [])
            + (metadata.get("referenced_standards") or [])
            + (metadata.get("resolved_section_chunk_ids") or [])
            + (metadata.get("outbound_link_chunk_ids") or [])
        )
    )
    return f"{document_title}\n{section}\n{section_ids}\n{section_titles}\n{section_breadcrumb}\n{keywords}\n{keyphrases}\n{exact_terms}\n{acronyms}\n{abbreviation_text}\n{keyword_identifiers}\n{keyword_standards}\n{table_keyword_terms}\n{section_keyword_terms}\n{table_title}\n{table_columns}\n{table_rows}\n{table_records}\n{table_notes}\n{table_terms}\n{layout_regions}\n{page_regions}\n{horizontal_regions}\n{columns}\n{chunk_strategy}\n{boundary_reason}\n{domain_terms}\n{entities}\n{canonical_entities}\n{primary_entities}\n{entity_details}\n{identifiers}\n{language_text}\n{access_tags}\n{access_policy}\n{access_roles}\n{standards}\n{modalities}\n{numeric_constraints}\n{unit_text}\n{formula_text}\n{code_text}\n{schema_text}\n{knowledge_graph_text}\n{ontology_text}\n{safety_flags}\n{ingestion_signals}\n{compliance_flags}\n{retrieval_tags}\n{answer_scope_hint}\n{link_text}\n{overlap_context}\n{text}"


def hybrid_haystack(text: str, metadata: dict, surface: str = "sparse") -> str:
    if surface == "dense" and metadata.get("hybrid_dense_text"):
        return str(metadata.get("hybrid_dense_text") or "")
    if surface == "table" and metadata.get("hybrid_table_text"):
        return str(metadata.get("hybrid_table_text") or "")
    if surface == "metadata" and metadata.get("hybrid_metadata_text"):
        return str(metadata.get("hybrid_metadata_text") or "")
    if surface == "numeric" and metadata.get("hybrid_numeric_text"):
        return str(metadata.get("hybrid_numeric_text") or "")
    if metadata.get("hybrid_sparse_text"):
        return str(metadata.get("hybrid_sparse_text") or "")
    return searchable_text(text, metadata)


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
    haystack = hybrid_haystack(text, metadata, "sparse").lower()
    exact = sum(1 for term in query_terms if term in haystack)
    phrase_bonus = 1 if " ".join(query_terms) in haystack else 0
    sparse_terms = set(metadata.get("hybrid_sparse_terms") or [])
    sparse_hits = sum(1 for term in query_terms if normalize_sparse_term(term) in sparse_terms)
    return min(1.0, (exact + sparse_hits * 0.7 + phrase_bonus) / max(1, len(query_terms)))


def phrase_match_score(query_phrases: list[str], text: str, metadata: dict) -> float:
    if not query_phrases:
        return 0.0
    haystack = hybrid_haystack(text, metadata, "sparse").lower()
    prepared_phrases = set(metadata.get("hybrid_exact_phrases") or [])
    hits = sum(1 for phrase in query_phrases if phrase in haystack or phrase in prepared_phrases)
    return hits / len(query_phrases)


def rerank(query_terms: list[str], text: str, metadata: dict, dense_score: float = 0.0) -> float:
    section = " ".join(metadata.get("section_path") or []).lower()
    content_type_bonus = 0.16 if metadata.get("contains_table") and is_table_query(query_terms) else 0
    section_hits = sum(1 for term in query_terms if term in section)
    early_text = hybrid_haystack(text, metadata, "sparse")[:900].lower()
    early_hits = sum(1 for term in query_terms if term in early_text)
    lexical = ((section_hits * 1.4) + early_hits) / max(1, len(query_terms))
    dense = (dense_score + 1.0) / 2.0
    return min(1.0, (lexical * 0.55) + (dense * 0.45) + content_type_bonus)


def metadata_match_score(query_terms: list[str], query_phrases: list[str], metadata: dict) -> float:
    metadata_text = (metadata.get("hybrid_metadata_text") or " ".join(
        [
            " ".join(metadata.get("section_path") or []),
            metadata.get("section_title") or "",
            metadata.get("parent_section") or "",
            metadata.get("table_title") or "",
            " ".join(metadata.get("keywords") or []),
            " ".join(metadata.get("keyphrases") or []),
            " ".join(metadata.get("exact_terms") or []),
            " ".join(metadata.get("acronyms") or []),
            " ".join(metadata.get("keyword_identifiers") or []),
            " ".join(metadata.get("keyword_standards") or []),
            " ".join(metadata.get("table_keyword_terms") or []),
            " ".join(metadata.get("section_keyword_terms") or []),
            metadata.get("revision") or "",
            metadata.get("document_identifier") or "",
            metadata.get("validity_status") or "",
            metadata.get("filename") or "",
            " ".join(metadata.get("layout_regions") or []),
            " ".join(metadata.get("page_regions") or []),
            " ".join(metadata.get("horizontal_regions") or []),
            " ".join(metadata.get("domain_terms") or []),
            " ".join(metadata.get("engineering_entities") or []),
            " ".join(metadata.get("technical_identifiers") or []),
            " ".join(metadata.get("standards") or []),
            " ".join(metadata.get("requirement_modalities") or []),
            " ".join(metadata.get("safety_flags") or []),
            " ".join(metadata.get("compliance_flags") or []),
            " ".join(metadata.get("retrieval_tags") or []),
        metadata.get("answer_scope_hint") or "",
        entity_metadata_text(metadata),
        " ".join(metadata.get("referenced_section_ids") or []),
            " ".join(metadata.get("referenced_document_ids") or []),
            " ".join(metadata.get("referenced_standards") or []),
            " ".join(metadata.get("resolved_section_chunk_ids") or []),
            " ".join(metadata.get("outbound_link_chunk_ids") or []),
        ]
    )).lower()
    if not metadata_text:
        return 0.0
    term_hits = sum(1 for term in query_terms if term in metadata_text)
    phrase_hits = sum(1 for phrase in query_phrases if phrase in metadata_text)
    return min(1.0, (term_hits / max(1, len(query_terms))) * 0.75 + (phrase_hits / max(1, len(query_phrases) or 1)) * 0.25)


def parse_self_query_filters(query: str) -> dict:
    normalized = re.sub(r"\s+", " ", str(query or "")).strip()
    lower = normalized.lower()
    quoted = [item.strip() for item in re.findall(r"['\"]([^'\"]{2,120})['\"]", normalized)]
    section_refs = unique_preserve(
        re.findall(r"\b(?:section|sec\.?|clause|paragraph|para\.?)\s*((?:\d+\.)*\d+[A-Z]?)\b", normalized, flags=re.I)
        + re.findall(r"\b((?:\d+\.)+\d+)\b", normalized)
    )
    page_refs = unique_preserve(re.findall(r"\b(?:page|pg\.?)\s*(\d+)\b", normalized, flags=re.I))
    revision_refs = unique_preserve(re.findall(r"\b(?:rev(?:ision)?\.?|version|issue)\s*[:#]?\s*([A-Z]?\d+[A-Z]?|[A-Z])\b", normalized, flags=re.I))
    document_refs = unique_preserve(
        quoted
        + re.findall(r"\b(?:document|doc(?:ument)? id|file|filename)\s*(?:named|called|id|no\.?|number|#|:)?\s*([A-Z0-9][A-Z0-9_. -]{4,120})", normalized, flags=re.I)
    )
    standard_refs = unique_preserve(re.findall(r"\b(?:ASME\s+[A-Z]\d+(?:\.\d+)*[A-Z0-9.-]*|NORSOK\s+[A-Z]-\d+[A-Z0-9.-]*|API\s+\d+[A-Z0-9.-]*|ISO\s+\d+[A-Z0-9:.-]*|IEC\s+\d+[A-Z0-9:.-]*)\b", normalized, flags=re.I))
    identifier_refs = unique_preserve(re.findall(r"\b[A-Z]{2,}[A-Z0-9_-]*(?:-[A-Z0-9]+){1,}\b", normalized))
    content_types = []
    if re.search(r"\b(table|row|column|matrix)\b", lower):
        content_types.append("table")
    if re.search(r"\b(image|figure|diagram|drawing|p&id|layout)\b", lower):
        content_types.append("image")
    flags = []
    for name, pattern in {
        "numeric": r"\b(value|dimension|size|slope|rating|mm|bar|psi|1:\d+|numeric|number)\b",
        "safety": r"\b(safety|hazard|fire|explosion|emergency|shutdown|relief|critical)\b",
        "compliance": r"\b(compliance|standard|code|regulation|asme|norsok|iso|api|iec)\b",
        "requirement": r"\b(shall|must|required|requirement|mandatory|not allowed|shall not)\b",
    }.items():
        if re.search(pattern, lower):
            flags.append(name)
    active = bool(section_refs or page_refs or revision_refs or document_refs or standard_refs or identifier_refs or content_types or flags)
    return {
        "schema": SELF_QUERY_SCHEMA_VERSION,
        "active": active,
        "section_refs": section_refs[:20],
        "page_refs": page_refs[:20],
        "revision_refs": [item.upper() for item in revision_refs[:20]],
        "document_refs": [clean_self_query_value(item) for item in document_refs[:12] if clean_self_query_value(item)],
        "standard_refs": [item.upper() for item in standard_refs[:20]],
        "identifier_refs": identifier_refs[:20],
        "content_types": unique_preserve(content_types),
        "flags": unique_preserve(flags),
        "filter_count": sum(len(group) for group in [section_refs, page_refs, revision_refs, document_refs, standard_refs, identifier_refs, content_types, flags]),
    }


def clean_self_query_value(value: str) -> str:
    value = normalize_link_key(value)
    value = re.sub(r"\b(?:where|with|that|which|contains|has|from|in|section|page)\b.*$", "", value).strip(" .,:;-")
    return value


def self_query_filter_score(self_query: dict, metadata: dict, text: str) -> tuple[float, dict]:
    if not self_query.get("active"):
        return 0.0, {"active": False, "matched": [], "missed": [], "hard_filter_missed": False}
    matched: list[str] = []
    missed: list[str] = []
    hard_missed = False

    def check_list(label: str, expected: list[str], haystack: str, hard: bool = False) -> None:
        nonlocal hard_missed
        if not expected:
            return
        if any(normalize_link_key(item) and normalize_link_key(item) in haystack for item in expected):
            matched.append(label)
        else:
            missed.append(label)
            hard_missed = hard_missed or hard

    metadata_text = normalize_link_key(
        " ".join(
            str(part)
            for part in [
                metadata.get("filename") or "",
                metadata.get("document_title") or "",
                metadata.get("document_identifier") or "",
                metadata.get("revision") or "",
                metadata.get("validity_status") or "",
                metadata.get("section_breadcrumb") or "",
                metadata.get("current_section_id") or "",
                metadata.get("current_section_title") or "",
                metadata.get("parent_section_id") or "",
                metadata.get("parent_section_title") or "",
                metadata.get("table_title") or "",
                " ".join(metadata.get("section_ids") or []),
                " ".join(metadata.get("section_titles") or []),
                " ".join(metadata.get("standards") or []),
                " ".join(metadata.get("technical_identifiers") or []),
                " ".join(metadata.get("engineering_entities") or []),
                " ".join(metadata.get("semantic_labels") or []),
                " ".join(metadata.get("modalities") or []),
                text[:1200],
            ]
            if part
        )
    )
    check_list("section", self_query.get("section_refs") or [], metadata_text, hard=True)
    check_list("document", self_query.get("document_refs") or [], metadata_text, hard=True)
    check_list("standard", self_query.get("standard_refs") or [], metadata_text, hard=True)
    check_list("identifier", self_query.get("identifier_refs") or [], metadata_text, hard=True)

    if self_query.get("page_refs"):
        pages = {str(metadata.get("page_start") or ""), str(metadata.get("page_end") or ""), str(metadata.get("page_label_start") or ""), str(metadata.get("page_label_end") or "")}
        if any(page in pages for page in self_query.get("page_refs") or []):
            matched.append("page")
        else:
            missed.append("page")
            hard_missed = True
    if self_query.get("revision_refs"):
        revision_haystack = normalize_link_key(" ".join([metadata.get("revision") or "", " ".join(metadata.get("revision_candidates") or [])]))
        if any(normalize_link_key(revision) in revision_haystack for revision in self_query.get("revision_refs") or []):
            matched.append("revision")
        else:
            missed.append("revision")
            hard_missed = True
    for content_type in self_query.get("content_types") or []:
        if content_type == "table" and metadata.get("contains_table"):
            matched.append("content:table")
        elif content_type == "image" and (metadata.get("has_images") or metadata.get("image_block_count") or "image" in (metadata.get("modalities") or [])):
            matched.append("content:image")
        else:
            missed.append(f"content:{content_type}")
    flag_checks = {
        "numeric": bool(metadata.get("has_numeric_constraints") or metadata.get("numeric_constraint_count")),
        "safety": bool(metadata.get("safety_critical") or metadata.get("safety_score")),
        "compliance": bool(metadata.get("compliance_related") or metadata.get("standards")),
        "requirement": bool(metadata.get("has_requirement") or metadata.get("requirement_modalities")),
    }
    for flag in self_query.get("flags") or []:
        if flag_checks.get(flag):
            matched.append(f"flag:{flag}")
        else:
            missed.append(f"flag:{flag}")
    total = max(1, int(self_query.get("filter_count") or len(matched) + len(missed)))
    coverage = len(set(matched)) / total
    score = min(0.22, coverage * 0.18 + (0.04 if matched and not hard_missed else 0))
    return round(score, 5), {
        "active": True,
        "matched": sorted(set(matched)),
        "missed": sorted(set(missed)),
        "coverage": round(coverage, 3),
        "hard_filter_missed": hard_missed,
    }


def entity_match_score(query_terms: list[str], query_phrases: list[str], metadata: dict) -> float:
    entity_text = entity_metadata_text(metadata).lower()
    if not entity_text:
        return 0.0
    terms = [normalize_sparse_term(term) for term in query_terms if normalize_sparse_term(term)]
    if not terms and not query_phrases:
        return 0.0
    exact_hits = sum(1 for term in terms if term in entity_text)
    phrase_hits = sum(1 for phrase in query_phrases if phrase and phrase.lower() in entity_text)
    primary = " ".join(metadata.get("primary_entities") or []).lower()
    primary_hits = sum(1 for term in terms if term in primary)
    type_text = " ".join(metadata.get("engineering_entity_types") or []).lower()
    type_hits = sum(1 for term in terms if term in type_text)
    quality = metadata.get("entity_extraction_quality") or {}
    quality_score = float(quality.get("score") or 0.0) if isinstance(quality, dict) else 0.0
    raw = (
        (exact_hits / max(1, len(terms))) * 0.52
        + (phrase_hits / max(1, len(query_phrases) or 1)) * 0.22
        + (primary_hits / max(1, len(terms))) * 0.18
        + (type_hits / max(1, len(terms))) * 0.08
    )
    return round(min(1.0, raw * (0.85 + quality_score * 0.15)), 4)


def multi_index_score_from_row(row: dict) -> float:
    sparse = float(row.get("pgvector_sparse_score") or 0)
    table = float(row.get("table_index_score") or 0)
    numeric = float(row.get("numeric_index_score") or 0)
    entity = float(row.get("entity_index_score") or 0)
    citation = float(row.get("citation_index_score") or 0)
    raw = sparse * 0.015 + table * 0.025 + numeric * 0.022 + entity * 0.018 + citation * 0.012
    return round(min(0.12, raw), 5)


def route_score(
    route: RetrieverRoute,
    vector_score: float,
    keyword_score: float,
    phrase_score: float,
    rerank_score: float,
    table_score: float,
    metadata_score: float,
    entity_score: float,
) -> float:
    base = (
        route.semantic_weight * vector_score
        + route.keyword_weight * keyword_score
        + route.phrase_weight * phrase_score
        + route.rerank_weight * rerank_score
        + route.table_weight * table_score
        + route.metadata_weight * metadata_score
    )
    return base + entity_score * 0.08


def hybrid_score_breakdown(
    route: RetrieverRoute,
    vector_score: float,
    keyword_score: float,
    phrase_score: float,
    rerank_score: float,
    table_score: float,
    metadata_score: float,
    entity_score: float,
    capability_score: float,
    metadata: dict,
) -> dict:
    return {
        "schema": HYBRID_SCHEMA_VERSION,
        "prepared": bool(metadata.get("hybrid_prepared")),
        "dense": round(route.semantic_weight * vector_score, 5),
        "keyword": round(route.keyword_weight * keyword_score, 5),
        "phrase": round(route.phrase_weight * phrase_score, 5),
        "rerank": round(route.rerank_weight * rerank_score, 5),
        "table": round(route.table_weight * table_score, 5),
        "metadata": round(route.metadata_weight * metadata_score, 5),
        "entity": round(entity_score * 0.08, 5),
        "capability": round(capability_score, 5),
        "facets": metadata.get("hybrid_facets") or {},
    }


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
    records = " ".join(" ".join(record.values()) for record in (metadata.get("table_records") or []) if isinstance(record, dict)).lower()
    notes = " ".join(metadata.get("table_notes") or []).lower()
    table_haystack = (metadata.get("hybrid_table_text") or " ".join(
        [
            metadata.get("table_title") or "",
            " ".join(metadata.get("table_columns") or []),
            " ".join(metadata.get("table_rows") or []),
            " ".join(" ".join(record.values()) for record in (metadata.get("table_records") or []) if isinstance(record, dict)),
            " ".join(metadata.get("table_notes") or []),
            " ".join(metadata.get("table_terms") or []),
            text,
        ]
    )).lower()
    exact_hits = sum(1 for term in query_terms if term in table_haystack)
    column_hits = sum(1 for column in metadata.get("table_columns") or [] if any(term in column.lower() for term in query_terms))
    title_hits = sum(1 for term in query_terms if term in title)
    row_hits = sum(1 for term in query_terms if term in rows)
    record_hits = sum(1 for term in query_terms if term in records)
    note_hits = sum(1 for term in query_terms if term in notes)
    phrase_hits = sum(1 for phrase in query_phrases if phrase in title or phrase in rows)
    row_density = min(0.18, float(metadata.get("table_row_count") or 0) * 0.03)
    quality_bonus = min(0.1, float(metadata.get("table_quality_score") or 0) * 0.1)
    return min(
        1.0,
        (exact_hits / max(1, len(query_terms))) * 0.35
        + (row_hits / max(1, len(query_terms))) * 0.28
        + (record_hits / max(1, len(query_terms))) * 0.12
        + (note_hits / max(1, len(query_terms))) * 0.06
        + (title_hits / max(1, len(query_terms))) * 0.18
        + (phrase_hits / max(1, len(query_phrases) or 1)) * 0.18
        + (column_hits * 0.08)
        + row_density
        + quality_bonus,
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


def merge_sql_rows(rows: list, sql_rows: list[dict]) -> list:
    if not sql_rows:
        return rows
    merged = list(rows)
    seen_ids = {str(dict(row).get("id")) for row in rows}
    for row in sql_rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            continue
        merged.append(row)
        seen_ids.add(row_id)
    return merged


def merge_api_rows(rows: list, api_rows: list[dict]) -> list:
    if not api_rows:
        return rows
    merged = list(rows)
    seen_ids = {str(dict(row).get("id")) for row in rows}
    for row in api_rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            continue
        merged.append(row)
        seen_ids.add(row_id)
    return merged


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


def expand_linked_candidates(ranked: list[dict], all_scored: list[dict], limit: int) -> list[dict]:
    if not ranked:
        return ranked
    by_chunk_id: dict[str, dict] = {}
    for item in all_scored:
        chunk_id = str((item.get("metadata") or {}).get("chunk_id") or "")
        if chunk_id:
            by_chunk_id[chunk_id] = item
    selected = list(ranked)
    seen_ids = {item["id"] for item in selected}
    for item in ranked[: max(1, min(4, len(ranked)))]:
        metadata = item.get("metadata") or {}
        linked_ids = (
            (metadata.get("outbound_link_chunk_ids") or [])
            + (metadata.get("inbound_link_chunk_ids") or [])
            + (metadata.get("same_section_chunk_ids") or [])[:3]
        )
        for linked_id in linked_ids[:12]:
            linked = by_chunk_id.get(str(linked_id))
            if not linked or linked["id"] in seen_ids:
                continue
            linked = dict(linked)
            linked["score"] = max(float(linked.get("score", 0)), float(item.get("score", 0)) * 0.78)
            linked["link_expanded_from"] = metadata.get("chunk_id") or ""
            selected.append(linked)
            seen_ids.add(linked["id"])
            if len(selected) >= limit * 2:
                break
    return sorted(selected, key=lambda row: float(row.get("score", 0)), reverse=True)[: max(limit, len(ranked))]


def profile_score_boost(profile: QuestionProfile, route: RetrieverRoute, query_terms: list[str], query_phrases: list[str], text: str, metadata: dict) -> float:
    haystack = hybrid_haystack(text, metadata, "sparse").lower()
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
    return dedupe_fingerprint(text, {})["prefix_signature"]


def dedupe_fingerprint(text: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    normalized = normalize_for_dedupe(text)
    words = normalized.split()
    prefix_seed = " ".join(words[:100])
    numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)?\b", text.lower())))
    table_key = " ".join((metadata.get("table_columns") or [])[:12]) if metadata.get("contains_table") else ""
    section_key = metadata.get("current_section_id") or metadata.get("section_title") or ""
    return {
        "normalized": normalized,
        "words": words,
        "word_set": set(words),
        "shingles": word_shingles(words),
        "numbers": numbers,
        "table_key": normalize_for_dedupe(table_key),
        "section_key": normalize_for_dedupe(str(section_key)),
        "exact_signature": hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20],
        "prefix_signature": hashlib.sha1(prefix_seed.encode("utf-8")).hexdigest()[:20],
        "word_count": len(words),
        "unique_word_count": len(set(words)),
        "contains_table": bool(metadata.get("contains_table")),
        "has_numeric_constraints": bool(metadata.get("has_numeric_constraints") or numbers),
    }


def find_near_duplicate(candidate: dict, existing: list[dict]) -> dict | None:
    for item in existing:
        if not enough_dedupe_words(candidate, item):
            continue
        if dedupe_pair_protected(candidate, item):
            continue
        jaccard = set_similarity(candidate["word_set"], item["word_set"])
        containment = containment_similarity(candidate["word_set"], item["word_set"])
        shingle = set_similarity(candidate["shingles"], item["shingles"])
        if jaccard >= DEDUPE_JACCARD_THRESHOLD:
            return {"reason": "near_duplicate_jaccard", "score": jaccard}
        if containment >= DEDUPE_CONTAINMENT_THRESHOLD and shingle >= 0.72:
            return {"reason": "near_duplicate_containment", "score": containment}
        if shingle >= DEDUPE_SHINGLE_THRESHOLD:
            return {"reason": "near_duplicate_shingle", "score": shingle}
    return None


def enough_dedupe_words(left: dict, right: dict) -> bool:
    return int(left.get("word_count") or 0) >= DEDUPE_MIN_WORDS and int(right.get("word_count") or 0) >= DEDUPE_MIN_WORDS


def dedupe_protected(metadata: dict) -> bool:
    return bool(metadata.get("contains_table") or metadata.get("has_numeric_constraints") or metadata.get("preserve_together"))


def dedupe_pair_protected(left: dict, right: dict) -> bool:
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


def chat_session_title(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return "New chat"
    return compact_preview(clean, limit=56)


def compact_preview(text: str, limit: int = 88) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit].rstrip()}..."
