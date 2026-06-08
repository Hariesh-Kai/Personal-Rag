from __future__ import annotations

import re
from collections import Counter
from typing import Any

MEMORY_RETRIEVER_SCHEMA_VERSION = "engineering-memory-retriever-v1"
MEMORY_STOPWORDS = {
    "about",
    "above",
    "also",
    "and",
    "answer",
    "are",
    "does",
    "each",
    "explain",
    "from",
    "give",
    "have",
    "more",
    "previous",
    "question",
    "same",
    "section",
    "show",
    "that",
    "the",
    "this",
    "those",
    "what",
    "where",
    "which",
    "with",
}
FOLLOW_UP_PATTERNS = (
    r"^\s*(what about|and|also|then|that|this|those|it)\b",
    r"\b(previous|above|same section|that one|those sources|continue|more detail|details)\b",
)


def memory_retrieval_plan(question: str, route: Any, turns: list[dict[str, Any]]) -> dict[str, Any]:
    active = "memory" in set(getattr(route, "retrievers", ())) or looks_like_follow_up(question)
    selected_turns = turns[:6] if active else []
    source_ids = []
    source_filenames = []
    source_sections = []
    source_entities = []
    source_standards = []
    previous_questions = []
    previous_answers = []

    for turn in selected_turns:
        previous_questions.append(str(turn.get("resolved_question") or turn.get("question") or ""))
        previous_answers.append(str(turn.get("answer") or ""))
        source_ids.extend(safe_int(item) for item in turn.get("source_ids") or [] if safe_int(item))
        payload = turn.get("memory_payload") or {}
        for source in payload.get("sources") or []:
            if not isinstance(source, dict):
                continue
            source_filenames.append(str(source.get("filename") or ""))
            source_sections.append(str(source.get("section") or source.get("section_title") or ""))
            source_entities.extend(str(item) for item in source.get("entities") or [])
            source_standards.extend(str(item) for item in source.get("standards") or [])

    current_terms = memory_terms(question)
    history_terms = ranked_terms(" ".join(previous_questions + previous_answers))[:24]
    memory_terms_combined = unique_preserve(current_terms + history_terms + memory_terms(" ".join(source_sections + source_entities + source_standards)))[:40]
    resolved_query = build_memory_query(question, previous_questions, memory_terms_combined)
    return {
        "schema": MEMORY_RETRIEVER_SCHEMA_VERSION,
        "active": bool(active and selected_turns),
        "turn_count": len(selected_turns),
        "source_ids": unique_preserve(source_ids)[:40],
        "source_filenames": unique_preserve(clean_value(item) for item in source_filenames if clean_value(item))[:20],
        "source_sections": unique_preserve(clean_value(item) for item in source_sections if clean_value(item))[:20],
        "source_entities": unique_preserve(clean_value(item) for item in source_entities if clean_value(item))[:30],
        "source_standards": unique_preserve(clean_value(item).upper() for item in source_standards if clean_value(item))[:20],
        "current_terms": current_terms,
        "history_terms": history_terms,
        "memory_terms": memory_terms_combined,
        "resolved_query": resolved_query,
        "strategy": "recent_turn_source_ids_sections_entities_standards_followup_expansion",
    }


def memory_expanded_queries(question: str, seed_queries: list[str], plan: dict[str, Any]) -> list[str]:
    queries = list(seed_queries)
    if not plan.get("active"):
        return unique_preserve(queries)[:8]
    if plan.get("resolved_query"):
        queries.append(plan["resolved_query"])
    if plan.get("source_sections"):
        queries.append(f"same section {' '.join(plan['source_sections'][:6])} {question}")
    if plan.get("source_entities"):
        queries.append(f"same topic entities {' '.join(plan['source_entities'][:10])} {question}")
    if plan.get("source_standards"):
        queries.append(f"same standards {' '.join(plan['source_standards'][:8])} {question}")
    if plan.get("memory_terms"):
        queries.append(f"conversation memory {' '.join(plan['memory_terms'][:16])} {question}")
    return unique_preserve(query for query in queries if str(query).strip())[:8]


def memory_candidate_score(plan: dict[str, Any], row_id: Any, metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched": []}
    matched = []
    score = 0.0
    source_ids = {int(item) for item in plan.get("source_ids") or [] if safe_int(item)}
    if safe_int(row_id) in source_ids:
        score += 0.14
        matched.append("previous_source_chunk")

    haystack = memory_haystack(metadata, text)
    filename = clean_value(metadata.get("filename") or "")
    if filename and filename in {clean_value(item) for item in plan.get("source_filenames") or []}:
        score += 0.035
        matched.append("same_file")

    section_text = clean_value(" ".join([metadata.get("section_title") or "", metadata.get("current_section_title") or "", " ".join(metadata.get("section_path") or [])]))
    section_hits = sum(1 for section in plan.get("source_sections") or [] if clean_value(section) and clean_value(section) in section_text)
    if section_hits:
        score += min(0.08, section_hits * 0.035)
        matched.append("same_section")

    entity_hits = sum(1 for entity in plan.get("source_entities") or [] if clean_value(entity) and clean_value(entity) in haystack)
    standard_hits = sum(1 for standard in plan.get("source_standards") or [] if clean_value(standard) and clean_value(standard) in haystack)
    term_hits = sum(1 for term in plan.get("memory_terms") or [] if clean_value(term) and clean_value(term) in haystack)
    if entity_hits:
        score += min(0.07, entity_hits * 0.012)
        matched.append("same_entities")
    if standard_hits:
        score += min(0.06, standard_hits * 0.02)
        matched.append("same_standards")
    if term_hits:
        score += min(0.05, term_hits * 0.006)
        matched.append("memory_terms")

    return round(min(0.20, score), 5), {
        "schema": MEMORY_RETRIEVER_SCHEMA_VERSION,
        "active": True,
        "matched": unique_preserve(matched),
        "entity_hits": entity_hits,
        "standard_hits": standard_hits,
        "term_hits": term_hits,
        "source_id_hit": safe_int(row_id) in source_ids,
    }


def build_chat_memory_payload(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    sources = []
    for item in contexts[:12]:
        metadata = item.get("metadata") or {}
        sources.append(
            {
                "id": item.get("id"),
                "filename": item.get("filename") or metadata.get("filename") or "",
                "chunk_index": item.get("chunk_index"),
                "section": metadata.get("section_title") or metadata.get("current_section_title") or "",
                "section_id": metadata.get("current_section_id") or "",
                "page_start": metadata.get("page_start"),
                "entities": unique_preserve(
                    (metadata.get("primary_entities") or [])
                    + (metadata.get("engineering_entities") or [])
                    + (metadata.get("engineering_canonical_entities") or [])
                )[:20],
                "standards": unique_preserve(metadata.get("standards") or [])[:12],
                "table_title": metadata.get("table_title") or "",
                "contains_table": bool(metadata.get("contains_table")),
            }
        )
    return {
        "schema": MEMORY_RETRIEVER_SCHEMA_VERSION,
        "sources": sources,
        "source_count": len(sources),
    }


def build_memory_query(question: str, previous_questions: list[str], terms: list[str]) -> str:
    if not previous_questions:
        return question
    previous = previous_questions[0]
    if not looks_like_follow_up(question):
        return f"{question}\nConversation topic terms: {' '.join(terms[:18])}"
    return f"Previous question/topic: {previous}\nFollow-up question: {question}\nConversation topic terms: {' '.join(terms[:18])}"


def looks_like_follow_up(question: str) -> bool:
    normalized = str(question or "").lower().strip()
    return any(re.search(pattern, normalized) for pattern in FOLLOW_UP_PATTERNS) or normalized in {"why", "how", "explain", "more", "details"}


def memory_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", str(text or "")):
        value = clean_value(token)
        if value and value not in MEMORY_STOPWORDS:
            terms.append(value)
    return unique_preserve(terms)[:32]


def ranked_terms(text: str) -> list[str]:
    counts = Counter(memory_terms(text))
    return [term for term, _ in counts.most_common(40)]


def memory_haystack(metadata: dict[str, Any], text: str) -> str:
    return clean_value(
        " ".join(
            [
                text,
                metadata.get("filename") or "",
                metadata.get("section_title") or "",
                metadata.get("current_section_title") or "",
                " ".join(metadata.get("section_path") or []),
                metadata.get("table_title") or "",
                " ".join(metadata.get("standards") or []),
                " ".join(metadata.get("primary_entities") or []),
                " ".join(metadata.get("engineering_entities") or []),
                " ".join(metadata.get("engineering_canonical_entities") or []),
                " ".join(metadata.get("keywords") or []),
                " ".join(metadata.get("technical_identifiers") or []),
            ]
        )
    )


def clean_value(value: Any) -> str:
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


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
