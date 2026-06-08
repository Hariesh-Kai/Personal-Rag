from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

AGENTIC_RETRIEVER_SCHEMA_VERSION = "engineering-agentic-crewai-v1"
AGENTIC_ENABLE_VALUES = {"1", "true", "yes", "on"}
AGENTIC_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "between",
    "does",
    "each",
    "explain",
    "from",
    "have",
    "into",
    "need",
    "shall",
    "should",
    "show",
    "that",
    "the",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def agentic_retrieval_plan(question: str, profile: Any, route: Any, seed_queries: list[str]) -> dict[str, Any]:
    if "agentic" not in set(getattr(route, "retrievers", ())):
        return inactive_agentic_plan(question, profile, route, seed_queries)

    fallback = deterministic_agentic_plan(question, profile, route, seed_queries)
    if not crewai_enabled():
        return fallback

    try:
        crew_plan = crewai_plan(question, profile, route, seed_queries)
    except Exception as exc:
        fallback["backend"] = "deterministic_fallback_after_crewai_error"
        fallback["crewai_error"] = str(exc)[:500]
        return fallback

    if not crew_plan.get("queries"):
        return fallback
    crew_plan["queries"] = unique_preserve((crew_plan.get("queries") or []) + fallback["queries"])[:8]
    crew_plan["fallback_available"] = True
    return crew_plan


def agentic_candidate_score(plan: dict[str, Any], text: str, metadata: dict[str, Any]) -> float:
    if not plan.get("active"):
        return 0.0
    haystack = agentic_haystack(text, metadata)
    focus_terms = [normalize_term(term) for term in plan.get("focus_terms") or []]
    required_terms = [normalize_term(term) for term in plan.get("required_terms") or []]
    focus_terms = [term for term in focus_terms if term]
    required_terms = [term for term in required_terms if term]
    if not focus_terms and not required_terms:
        return 0.0

    focus_hits = sum(1 for term in focus_terms if term in haystack)
    required_hits = sum(1 for term in required_terms if term in haystack)
    section_bonus = 0.0
    if metadata.get("section_title") or metadata.get("section_path"):
        section_bonus += 0.015
    if metadata.get("outbound_link_chunk_ids") or metadata.get("same_section_chunk_ids"):
        section_bonus += 0.015
    if metadata.get("contains_table") and any(term in haystack for term in ("table", "row", "value", "slope", "dimension")):
        section_bonus += 0.02
    if metadata.get("safety_critical") and any(term in haystack for term in ("safety", "shall", "must", "fire", "emergency")):
        section_bonus += 0.02

    focus_score = (focus_hits / max(1, len(focus_terms))) * 0.055
    required_score = (required_hits / max(1, len(required_terms))) * 0.05
    confidence_weight = min(1.0, float(plan.get("confidence") or 0.7))
    return round(min(0.14, (focus_score + required_score + section_bonus) * confidence_weight), 5)


def inactive_agentic_plan(question: str, profile: Any, route: Any, seed_queries: list[str]) -> dict[str, Any]:
    return {
        "schema": AGENTIC_RETRIEVER_SCHEMA_VERSION,
        "active": False,
        "backend": "inactive",
        "used_crewai": False,
        "crewai_enabled": crewai_enabled(),
        "crewai_available": crewai_available(),
        "profile_type": getattr(profile, "type_id", ""),
        "route_primary": getattr(route, "primary", ""),
        "queries": unique_preserve(seed_queries)[:5],
        "focus_terms": important_terms(question),
        "required_terms": [],
        "retrieval_actions": [],
        "reasoning_steps": [],
        "confidence": 0.0,
    }


def deterministic_agentic_plan(question: str, profile: Any, route: Any, seed_queries: list[str]) -> dict[str, Any]:
    terms = important_terms(question)
    subquestions = decompose_question(question)
    actions = [
        "run hybrid retrieval for the original engineering question",
        "decompose multi-part or troubleshooting wording into focused subqueries",
        "expand with nearby/linked chunks when the first evidence is incomplete",
        "prefer exact identifiers, section titles, numeric values, shall/must wording, and citations",
    ]
    query_variants = list(seed_queries)
    query_variants.extend(subquestions)
    query_variants.extend(
        [
            f"exact evidence page section {' '.join(terms[:8])}",
            f"requirements constraints shall must {' '.join(terms[:8])}",
            f"related equipment systems upstream downstream {' '.join(terms[:8])}",
        ]
    )
    if is_table_like(question):
        query_variants.append(f"table rows columns numeric values {' '.join(terms[:8])}")
    if is_safety_like(question):
        query_variants.append(f"safety emergency fire shutdown exception conditions {' '.join(terms[:8])}")
    if is_revision_like(question):
        query_variants.append(f"revision version date latest changed {' '.join(terms[:8])}")

    return {
        "schema": AGENTIC_RETRIEVER_SCHEMA_VERSION,
        "active": True,
        "backend": "deterministic",
        "used_crewai": False,
        "crewai_enabled": crewai_enabled(),
        "crewai_available": crewai_available(),
        "profile_type": getattr(profile, "type_id", ""),
        "route_primary": getattr(route, "primary", ""),
        "queries": unique_preserve(query.strip() for query in query_variants if query.strip())[:8],
        "focus_terms": terms[:16],
        "required_terms": required_terms(question, getattr(profile, "type_id", "")),
        "retrieval_actions": actions,
        "reasoning_steps": [
            "identify the user intent and risk level",
            "split the question into evidence targets",
            "retrieve exact, semantic, section, and surrounding context evidence",
            "rerank toward chunks that satisfy the evidence targets",
        ],
        "confidence": 0.78,
    }


def crewai_plan(question: str, profile: Any, route: Any, seed_queries: list[str]) -> dict[str, Any]:
    ensure_usable_temp_dir()
    try:
        from crewai import Agent, Crew, Process, Task
    except Exception as exc:
        raise RuntimeError(f"CrewAI is not installed or could not be imported: {exc}") from exc

    description = (
        "Create a bounded retrieval plan for an engineering-document RAG system. "
        "Do not answer the user question. Return compact JSON with keys: "
        "queries, focus_terms, required_terms, retrieval_actions, reasoning_steps, confidence. "
        "Use only retrieval planning behavior, no external web/API assumptions.\n\n"
        f"Question: {question}\n"
        f"Question type: {getattr(profile, 'type_id', '')}\n"
        f"Route: {getattr(route, 'primary', '')}\n"
        f"Seed queries: {seed_queries[:5]}"
    )
    agent = Agent(
        role="Engineering RAG Retrieval Planner",
        goal="Plan safe, grounded, evidence-first retrieval for engineering documents.",
        backstory="You are a retrieval engineer. You create search queries and context expansion steps, not final answers.",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=description,
        expected_output="A JSON object containing retrieval planning fields only.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    parsed = parse_crewai_result(str(result))
    return {
        "schema": AGENTIC_RETRIEVER_SCHEMA_VERSION,
        "active": True,
        "backend": "crewai",
        "used_crewai": True,
        "crewai_enabled": True,
        "crewai_available": True,
        "profile_type": getattr(profile, "type_id", ""),
        "route_primary": getattr(route, "primary", ""),
        "queries": unique_preserve(parsed.get("queries") or seed_queries)[:8],
        "focus_terms": unique_preserve(parsed.get("focus_terms") or important_terms(question))[:16],
        "required_terms": unique_preserve(parsed.get("required_terms") or required_terms(question, getattr(profile, "type_id", "")))[:12],
        "retrieval_actions": unique_preserve(parsed.get("retrieval_actions") or [])[:8],
        "reasoning_steps": unique_preserve(parsed.get("reasoning_steps") or [])[:8],
        "confidence": clamp_float(parsed.get("confidence"), default=0.82),
    }


def parse_crewai_result(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    return {
        "queries": [line for line in lines if "query" in line.lower()][:5],
        "focus_terms": important_terms(text)[:12],
        "required_terms": [],
        "retrieval_actions": lines[:6],
        "reasoning_steps": lines[:6],
        "confidence": 0.72,
    }


def crewai_enabled() -> bool:
    return os.getenv("RAG_CREWAI_ENABLED", "").strip().lower() in AGENTIC_ENABLE_VALUES


def crewai_available() -> bool:
    ensure_usable_temp_dir()
    try:
        import crewai  # noqa: F401
    except Exception:
        return False
    return True


def ensure_usable_temp_dir() -> None:
    runtime_root = Path.cwd() / ".runtime"
    temp_root = Path(os.getenv("RAG_TEMP_DIR") or runtime_root / "tmp")
    crewai_root = runtime_root / "crewai"
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        crewai_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TEMP", str(temp_root))
        os.environ.setdefault("TMP", str(temp_root))
        os.environ.setdefault("TMPDIR", str(temp_root))
        os.environ["LOCALAPPDATA"] = str(crewai_root)
        os.environ["APPDATA"] = str(crewai_root)
        os.environ.setdefault("CREWAI_STORAGE_DIR", "rag-chatbot-agentic")
        tempfile.tempdir = str(temp_root)
        patch_appdirs(crewai_root)
    except OSError:
        pass


def patch_appdirs(crewai_root: Path) -> None:
    try:
        import appdirs
    except Exception:
        return

    def project_user_data_dir(appname: str | None = None, appauthor: str | None = None, version: str | None = None, roaming: bool = False) -> str:
        parts = [crewai_root]
        if appauthor:
            parts.append(Path(str(appauthor)))
        if appname:
            parts.append(Path(str(appname)))
        if version:
            parts.append(Path(str(version)))
        path = Path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    appdirs.user_data_dir = project_user_data_dir


def decompose_question(question: str) -> list[str]:
    pieces = re.split(r"\b(?:and|or|also|with|between|versus|vs\.?|compared to)\b|[;?]", question, flags=re.I)
    return [piece.strip() for piece in pieces if len(piece.strip()) > 8][:6]


def important_terms(text: str) -> list[str]:
    terms = []
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,}", text):
        normalized = normalize_term(term)
        if normalized and normalized not in AGENTIC_STOPWORDS:
            terms.append(normalized)
    return unique_preserve(terms)


def required_terms(question: str, profile_type: str) -> list[str]:
    terms = []
    lowered = question.lower()
    if profile_type in {"safety_critical", "safety_interpretation", "negative", "exception"} or is_safety_like(question):
        terms.extend(["shall", "must", "required", "safety", "exception", "unless"])
    if profile_type in {"table_numeric", "calculation"} or is_table_like(question):
        terms.extend(["table", "row", "value", "numeric", "mm", "slope"])
    if profile_type in {"temporal_revision"} or is_revision_like(question):
        terms.extend(["revision", "version", "date", "changed"])
    if re.search(r"\b(section|where|location)\b", lowered):
        terms.extend(["section", "page"])
    return unique_preserve(terms)


def is_table_like(question: str) -> bool:
    return bool(re.search(r"\b(table|row|column|slope|dimension|value|rating|mm|matrix|calculate|calculation)\b", question, flags=re.I))


def is_safety_like(question: str) -> bool:
    return bool(re.search(r"\b(safety|emergency|shutdown|fire|explosion|hazard|relief|shall not|must not|prohibited)\b", question, flags=re.I))


def is_revision_like(question: str) -> bool:
    return bool(re.search(r"\b(revision|version|rev\.?|latest|changed|date|temporal)\b", question, flags=re.I))


def agentic_haystack(text: str, metadata: dict[str, Any]) -> str:
    values = [
        text,
        metadata.get("section_title") or "",
        " ".join(metadata.get("section_path") or []),
        metadata.get("table_title") or "",
        " ".join(metadata.get("table_columns") or []),
        " ".join(metadata.get("table_rows") or []),
        " ".join(metadata.get("keywords") or []),
        " ".join(metadata.get("engineering_entities") or []),
        metadata.get("document_identifier") or "",
        metadata.get("revision") or "",
        metadata.get("filename") or "",
    ]
    return re.sub(r"\s+", " ", " ".join(str(value) for value in values)).lower()


def normalize_term(term: Any) -> str:
    return re.sub(r"[^a-z0-9_.:/-]+", " ", str(term).lower()).strip()


def unique_preserve(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else str(value).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def clamp_float(value: Any, default: float = 0.75) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
