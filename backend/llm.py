from __future__ import annotations

import re
from pathlib import Path

from .answer_quality import evaluate_answer
from .enterprise_answers import confidence_summary, deterministic_answer
from .question_types import QuestionProfile, classify_question, profile_payload, prompt_block


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


SYSTEM_PROMPT = """You are a document-grounded engineering QA assistant.
Use only the retrieved sources. Do not add external knowledge, best practices, assumptions, or teaching-style background.
If the sources do not define or support the answer, say exactly: Not found in the retrieved document context.
Prefer a shorter fully grounded answer over a broad answer.
Preserve exact constraints, measurements, orientations, standards, exceptions, and negative requirements.
Use bullets for requirements/lists, a short paragraph for definitions, and yes/no plus evidence for yes/no questions.
Cite every factual sentence or bullet with source IDs like [S1]."""


class LocalLLM:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model_path = self._find_gguf_model()
        self._llm = None
        self._load_error = None
        self.status = "llama-cpp available, model loads on first chat" if self.model_path else "extractive"

    def answer(self, question: str, contexts: list[dict], profile: QuestionProfile | None = None, metadata: dict | None = None) -> dict:
        profile = profile or classify_question(question)
        contexts = select_context_window(question, contexts, profile)
        deterministic = deterministic_answer(question, contexts, profile, metadata)
        if deterministic:
            return {
                "answer": ensure_citation(deterministic, contexts),
                "mode": f"{self.status} + enterprise-guard",
                "sources": _sources(contexts),
                "quality": evaluate_answer(question, deterministic, contexts, profile),
                "question_profile": profile_payload(profile),
                "retriever_route": contexts[0].get("retriever_route") if contexts else None,
                "confidence": confidence_summary(contexts),
            }

        if not contexts or retrieval_is_weak(question, contexts):
            return self._not_found(question, contexts, profile)

        self._ensure_loaded()
        if self._llm is None:
            text = self._extractive_answer(question, contexts)
            return {
                "answer": text,
                "mode": self.status,
                "sources": _sources(contexts),
                "quality": evaluate_answer(question, text, contexts, profile),
                "question_profile": profile_payload(profile),
                "retriever_route": contexts[0].get("retriever_route") if contexts else None,
                "confidence": confidence_summary(contexts),
            }

        context_text = "\n\n".join(
            f"[S{index} | {source_label(item)}]\n{item['text']}" for index, item in enumerate(contexts, start=1)
        )
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nQuestion handling profile:\n{prompt_block(profile)}\n\nRetrieved sources:\n{context_text}\n\nQuestion: {question}\n\n"
            "Answer with only retrieved-source facts. If unsupported, use the not-found sentence exactly.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        result = self._llm(
            prompt,
            max_tokens=240,
            temperature=0.05,
            repeat_penalty=1.08,
            stop=["<|im_end|>"],
        )
        text = ensure_citation(result["choices"][0]["text"].strip(), contexts)
        return {
            "answer": text,
            "mode": self.status,
            "sources": _sources(contexts),
            "quality": evaluate_answer(question, text, contexts, profile),
            "question_profile": profile_payload(profile),
            "retriever_route": contexts[0].get("retriever_route") if contexts else None,
            "confidence": confidence_summary(contexts),
        }

    def _not_found(self, question: str, contexts: list[dict], profile: QuestionProfile | None = None) -> dict:
        profile = profile or classify_question(question)
        text = "Not found in the retrieved document context."
        return {
            "answer": text,
            "mode": self.status,
            "sources": _sources(contexts),
            "quality": evaluate_answer(question, text, contexts, profile),
            "question_profile": profile_payload(profile),
            "retriever_route": contexts[0].get("retriever_route") if contexts else None,
            "confidence": confidence_summary(contexts),
        }

    def _find_gguf_model(self) -> Path | None:
        preferred = [
            self.models_dir / "Qwen2.5-VL-7B-GGUF" / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            self.models_dir / "Qwen2.5-VL-7B-GGUF" / "Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf",
        ]
        for path in preferred:
            if path.exists():
                return path
        matches = sorted(self.models_dir.rglob("*.gguf"))
        return matches[0] if matches else None

    def _ensure_loaded(self) -> None:
        if self._llm is not None or self._load_error is not None or self.model_path is None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            self._load_error = "llama-cpp-python is not installed"
            self.status = f"extractive: install llama-cpp-python to use {self.model_path}"
            return
        try:
            self.status = f"loading local model: {self.model_path.name}"
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=3072,
                n_threads=8,
                n_gpu_layers=0,
                verbose=False,
            )
            self.status = f"llama-cpp: {self.model_path.name}"
        except Exception as exc:
            self._load_error = str(exc)
            self.status = f"extractive: could not load {self.model_path.name}: {exc}"

    def _extractive_answer(self, question: str, contexts: list[dict]) -> str:
        query_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9]{3,}", question.lower())
            if term not in {"what", "when", "where", "which", "about", "from", "this", "that", "with"}
        }
        candidates: list[tuple[int, str, dict]] = []
        for item in contexts[:4]:
            for sentence in _sentences(item["text"]):
                normalized = sentence.lower()
                score = sum(1 for term in query_terms if term in normalized)
                if score or len(candidates) < 3:
                    candidates.append((score, sentence, item))

        best = sorted(candidates, key=lambda row: row[0], reverse=True)[:4]
        if not best:
            best = [(0, contexts[0]["text"][:700], contexts[0])]

        lines = []
        used = set()
        for _, sentence, item in best:
            citation = f"[S{contexts.index(item) + 1}]"
            text = sentence.strip()
            if not text or text in used:
                continue
            used.add(text)
            lines.append(f"- {text} {citation}")
            if len(lines) == 3:
                break

        return "Based on the uploaded document:\n\n" + "\n".join(lines)


def _sources(contexts: list[dict]) -> list[dict]:
    sources = []
    for index, item in enumerate(contexts, start=1):
        metadata = item.get("metadata", {})
        preview_limit = 900 if metadata.get("contains_table") else 350
        sources.append(
            {
                "citation_id": f"S{index}",
                "filename": item["filename"],
                "chunk_index": item["chunk_index"],
                "score": round(float(item["score"]), 4),
                "metadata": metadata,
                "text": item["text"][:preview_limit],
            }
        )
    return sources


def source_label(item: dict) -> str:
    metadata = item.get("metadata") or {}
    section = metadata.get("section_title") or "no section"
    table_title = metadata.get("table_title") or ""
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    pages = f"p.{page_start}" if page_start == page_end else f"p.{page_start}-{page_end}"
    table = f", table: {table_title}" if table_title else (", table" if metadata.get("contains_table") else "")
    return f"{section}, {pages}{table}"


def select_context_window(question: str, contexts: list[dict], profile: QuestionProfile | None = None) -> list[dict]:
    if not contexts:
        return []
    query_terms = query_terms_for(question)
    top_score = float(contexts[0].get("score", 0))
    threshold = max(0.18, top_score * 0.68)
    candidates = [item for item in contexts if float(item.get("score", 0)) >= threshold]
    if not candidates:
        candidates = contexts[:2]

    section_scores: dict[str, float] = {}
    for item in candidates[:5]:
        section = (item.get("metadata") or {}).get("section_title") or "No section"
        section_hits = section_query_hits(section, query_terms)
        section_scores[section] = section_scores.get(section, 0.0) + float(item.get("score", 0)) + (section_hits * 0.22)
    dominant_section = max(section_scores, key=section_scores.get) if section_scores else ""
    dominant_has_query_terms = bool(section_query_hits(dominant_section, query_terms))

    table_query = bool(query_terms & TABLE_QUERY_TERMS)
    type_query = is_type_or_list_query(question)
    strict = []
    for item in candidates:
        metadata = item.get("metadata") or {}
        section = metadata.get("section_title") or "No section"
        is_table = bool(metadata.get("contains_table"))
        section_hits = section_query_hits(section, query_terms)
        item_has_query_terms = section_hits or text_query_hits(item.get("text", ""), query_terms)
        same_major = same_major_section(section, dominant_section)
        if type_query and dominant_has_query_terms and not same_major and not section_hits and not (table_query and is_table):
            continue
        if (
            dominant_has_query_terms
            and section != dominant_section
            and not same_major
            and not item_has_query_terms
            and not (table_query and is_table)
        ):
            continue
        strict.append(item)

    window_size = profile.context_limit if profile else (6 if type_query else 4)
    return (strict or candidates)[:window_size]


def retrieval_is_weak(question: str, contexts: list[dict]) -> bool:
    if not contexts:
        return True
    top_score = float(contexts[0].get("score", 0))
    if top_score < 0.16:
        return True
    query_terms = query_terms_for(question)
    combined = " ".join(
        " ".join(
            [
                item.get("text", ""),
                (item.get("metadata") or {}).get("table_title") or "",
                " ".join((item.get("metadata") or {}).get("table_columns") or []),
                " ".join((item.get("metadata") or {}).get("table_rows") or []),
                " ".join((item.get("metadata") or {}).get("table_terms") or []),
            ]
        )
        for item in contexts[:3]
    ).lower()
    query_hits = sum(1 for term in query_terms if term in combined)
    return bool(query_terms) and query_hits == 0


def ensure_citation(answer: str, contexts: list[dict]) -> str:
    if not contexts or answer == "Not found in the retrieved document context.":
        return answer
    if re.search(r"\[S\d+\]", answer):
        return answer
    return f"{answer} [S1]"


def query_terms_for(question: str) -> set[str]:
    stop = {
        "about",
        "are",
        "define",
        "different",
        "does",
        "each",
        "explain",
        "from",
        "need",
        "paragraph",
        "short",
        "that",
        "this",
        "types",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return {term for term in re.findall(r"[a-zA-Z0-9_.-]{3,}", question.lower()) if term not in stop}


def is_type_or_list_query(question: str) -> bool:
    return bool(re.search(r"\b(types?|different|each|list|explain)\b", question, flags=re.I))


def section_query_hits(section: str, query_terms: set[str]) -> int:
    normalized = section.lower()
    return sum(1 for term in query_terms if term in normalized or (term.endswith("s") and term[:-1] in normalized))


def text_query_hits(text: str, query_terms: set[str]) -> int:
    normalized = text[:900].lower()
    return sum(1 for term in query_terms if term in normalized or (term.endswith("s") and term[:-1] in normalized))


def same_major_section(left: str, right: str) -> bool:
    left_match = re.match(r"^(\d+)\.", left or "")
    right_match = re.match(r"^(\d+)\.", right or "")
    return bool(left_match and right_match and left_match.group(1) == right_match.group(1))


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    compact = re.sub(r"\s*(?:\u2022|\u25aa|\u25cf|\u00b7)\s*", " ", compact)
    raw_pieces = [piece.strip(" -") for piece in re.split(r"(?<=[.!?])\s+", compact)]
    pieces: list[str] = []
    carry = ""
    for piece in raw_pieces:
        if not piece:
            continue
        if carry:
            piece = f"{carry} {piece}"
            carry = ""
        if re.search(r"\b(to|for|with|and|or|including)$", piece, flags=re.IGNORECASE):
            carry = piece
            continue
        pieces.append(piece)
    if carry:
        pieces.append(carry)
    return [piece for piece in pieces if len(piece) > 30]
