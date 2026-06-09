from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .question_types import QuestionProfile


NOT_FOUND = "Not found in the retrieved document context."


def deterministic_answer(question: str, contexts: list[dict], profile: QuestionProfile, metadata: dict[str, Any] | None = None) -> str:
    if profile.type_id == "meta_document":
        return meta_document_answer(metadata or {}, contexts)
    if profile.type_id in {"document_coverage", "search_discovery", "multi_document"}:
        return coverage_answer(contexts, profile)
    if profile.type_id == "temporal_revision":
        return temporal_revision_answer(contexts)
    if profile.type_id == "conflict_detection":
        return conflict_answer(contexts)
    if profile.type_id == "calculation":
        return calculation_answer(question, contexts)
    if profile.type_id == "image_diagram":
        return image_diagram_answer(contexts)
    if profile.type_id in {"engineering_decision", "safety_interpretation"}:
        return safety_decision_guard(contexts, profile)
    if profile.type_id in {"out_of_document", "knowledge_gap"} and weak_context(contexts):
        return NOT_FOUND
    if profile.type_id == "regulation_compliance":
        return compliance_answer(contexts)
    if profile.type_id == "uncertainty":
        return uncertainty_answer(contexts)
    if profile.type_id == "false_assumption":
        return false_assumption_answer(question, contexts)
    if profile.type_id == "negative":
        return negative_answer(question, contexts)
    if profile.type_id == "audit_traceability":
        return audit_answer(contexts)
    return ""





def coverage_answer(contexts: list[dict], profile: QuestionProfile) -> str:
    if not contexts:
        return NOT_FOUND
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for item in contexts:
        by_doc[item.get("filename", "unknown")].append(item)
    lines = [f"{profile.label} results from retrieved context:"]
    for doc_name, items in by_doc.items():
        lines.append(f"\n{doc_name}:")
        seen_sections = set()
        for index, item in enumerate(contexts, start=1):
            if item not in items:
                continue
            metadata = item.get("metadata") or {}
            section = metadata.get("section_title") or "No section"
            if section in seen_sections:
                continue
            seen_sections.add(section)
            page = metadata.get("page_label_start") or metadata.get("page_start") or "?"
            lines.append(f"- {section} | page {page} [S{index}]")
    return "\n".join(lines)


def temporal_revision_answer(contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    rows = []
    for index, item in enumerate(contexts, start=1):
        metadata = item.get("metadata") or {}
        revision = metadata.get("revision") or ""
        doc_id = metadata.get("document_identifier") or ""
        status = metadata.get("validity_status") or ""
        text_hits = re.findall(r"\b(?:revision|rev\.?|validity status|document id)\b.{0,80}", item.get("text", ""), flags=re.I)
        if revision or doc_id or status or text_hits:
            rows.append((index, item, revision, doc_id, status, text_hits[:2]))
    if not rows:
        return "No revision/version/date evidence was found in the retrieved context. A revision-aware comparison cannot be made from the current retrieval. [S1]"
    lines = ["Revision/version evidence found:"]
    for index, item, revision, doc_id, status, hits in rows[:6]:
        parts = []
        if doc_id:
            parts.append(f"document id {doc_id}")
        if revision:
            parts.append(f"revision {revision}")
        if status:
            parts.append(f"status {status}")
        if hits:
            parts.append("; ".join(normalize_space(hit) for hit in hits))
        lines.append(f"- {', '.join(parts)} [S{index}]")
    lines.append("No revision priority or change conclusion is made unless the compared revisions are both retrieved.")
    return "\n".join(lines)


def meta_document_answer(metadata: dict[str, Any], contexts: list[dict]) -> str:
    if metadata:
        return (
            "Document/index metadata:\n\n"
            f"- Active index session: {metadata.get('index_session', 'unknown')} [metadata]\n"
            f"- Documents indexed: {metadata.get('documents', 0)} [metadata]\n"
            f"- Chunks indexed: {metadata.get('chunks', 0)} [metadata]\n"
            f"- Embedding model: {metadata.get('embedding_model', 'unknown')} ({metadata.get('embedding_dimensions', '?')} dimensions) [metadata]\n"
            f"- OCR backend: {metadata.get('ocr_backend', 'unknown')} [metadata]"
        )
    if not contexts:
        return NOT_FOUND
    filenames = sorted({item.get("filename", "unknown") for item in contexts})
    sections = sorted({(item.get("metadata") or {}).get("section_title") or "No section" for item in contexts})
    return "Retrieved metadata:\n\n" + "\n".join(
        [f"- Documents: {', '.join(filenames)} [S1]", f"- Sections: {', '.join(sections[:8])} [S1]"]
    )


def conflict_answer(contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    positive = []
    negative = []
    for index, item in enumerate(contexts, start=1):
        for sentence in sentences(item.get("text", "")):
            lowered = sentence.lower()
            if re.search(r"\b(shall not|not allowed|prohibited|forbidden|must not)\b", lowered):
                negative.append((index, sentence))
            elif re.search(r"\b(shall|must|required|allowed|may|should)\b", lowered):
                positive.append((index, sentence))
    if not positive and not negative:
        return "No explicit conflict pattern was found in the retrieved context. Review the listed sources for related requirements. [S1]"
    lines = ["Potential conflict scan from retrieved sources:"]
    if positive:
        lines.append("\nSupporting/allowing statements:")
        lines.extend(f"- {text} [S{idx}]" for idx, text in positive[:4])
    if negative:
        lines.append("\nRestricting/negative statements:")
        lines.extend(f"- {text} [S{idx}]" for idx, text in negative[:4])
    lines.append("\nNo conflict is resolved unless document hierarchy or revision priority is explicitly retrieved.")
    return "\n".join(lines)


def calculation_answer(question: str, contexts: list[dict]) -> str:
    context_text = "\n".join(item.get("text", "") for item in contexts)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|%|bar|barg|psi|kpa|mpa|degrees?|deg|°)?\b|\b\d+:\d+\b", context_text, flags=re.I)
    has_math_request = bool(re.search(r"\b(calculate|compute|sum|total|difference|ratio)\b", question, flags=re.I))
    if not has_math_request:
        return ""
    if len(numbers) < 2:
        return "Calculation cannot be performed from the retrieved document context because the required numeric inputs are not all present. Retrieved numeric evidence: " + (", ".join(numbers[:6]) or "none") + ". [S1]"
    return (
        "Calculation request detected. Retrieved numeric inputs are: "
        + ", ".join(dict.fromkeys(numbers[:10]))
        + ". I will not calculate a derived engineering value unless the formula/operation and all required inputs are explicitly supported by the retrieved context. [S1]"
    )


def image_diagram_answer(contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    ocr_contexts = [
        item
        for item in contexts
        if "ocr" in ((item.get("metadata") or {}).get("extractor", "") + " " + str((item.get("metadata") or {}).get("ocr", ""))).lower()
        or "[IMAGE]" in item.get("text", "")
    ]
    if not ocr_contexts:
        return "The retrieved context does not contain extracted diagram/image text. OCR is available for scanned pages, but no diagram-specific content was retrieved for this question. [S1]"
    return "Image/diagram-derived content was retrieved. Use only the OCR/extracted text shown in the cited sources; visual interpretation beyond extracted text is not supported. [S1]"


def safety_decision_guard(contexts: list[dict], profile: QuestionProfile) -> str:
    if not contexts:
        return NOT_FOUND
    requirements = []
    for index, item in enumerate(contexts, start=1):
        for sentence in sentences(item.get("text", "")):
            if re.search(r"\b(shall|must|required|not allowed|shall not|minimum|maximum|safety|emergency|relief|fire|explosion)\b", sentence, flags=re.I):
                requirements.append((index, sentence))
    if not requirements:
        return "The retrieved context does not provide enough documented requirements to support a safety or engineering decision. Do not treat this as approval. [S1]"
    intro = (
        "I cannot make or approve an engineering/safety decision from RAG output. "
        "The retrieved document requirements are:"
        if profile.type_id == "engineering_decision"
        else "Safety interpretation is limited to the retrieved document wording. Relevant retrieved requirements are:"
    )
    return intro + "\n\n" + "\n".join(f"- {text} [S{idx}]" for idx, text in requirements[:5])


def compliance_answer(contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    standards = []
    for index, item in enumerate(contexts, start=1):
        for sentence in sentences(item.get("text", "")):
            if re.search(r"\b(shall|must|required|compliance|comply|ASME|NORSOK|ISO|P&ID|standard|code)\b", sentence, flags=re.I):
                standards.append((index, sentence))
    if not standards:
        return "No explicit compliance requirement or standard mapping was retrieved. Compliance cannot be claimed from the current context. [S1]"
    return "Compliance cannot be certified by the chatbot. Retrieved compliance/standard evidence:\n\n" + "\n".join(
        f"- {text} [S{idx}]" for idx, text in standards[:6]
    )


def uncertainty_answer(contexts: list[dict]) -> str:
    summary = confidence_summary(contexts)
    if not contexts:
        return f"Confidence: none. {summary['reason']}"
    return (
        f"Confidence: {summary['confidence']} based on top retrieval score {summary['top_score']} "
        f"and dominant section '{summary['dominant_section']}'. Use the cited sources as evidence and treat missing retrieved details as unresolved. [S1]"
    )


def false_assumption_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    terms = important_terms(question)
    context_text = "\n".join(item.get("text", "") for item in contexts).lower()
    hits = sorted(term for term in terms if term in context_text)
    negative = bool(re.search(r"\b(shall not|not allowed|prohibited|forbidden|must not|except|unless)\b", context_text))
    if not hits:
        return "The assumption in the question is not supported by the retrieved context. I cannot confirm it from the uploaded document. [S1]"
    if negative:
        return "The retrieved context contains related terms and also restrictive/exception wording, so the assumption should not be accepted without reviewing the cited requirement. [S1]"
    return "The retrieved context contains related evidence, but the assumption should be treated only as supported to the extent stated in the cited sources. [S1]"


def negative_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    context_text = "\n".join(item.get("text", "") for item in contexts)
    negative_sentences = [
        (index, sentence)
        for index, item in enumerate(contexts, start=1)
        for sentence in sentences(item.get("text", ""))
        if re.search(r"\b(shall not|not allowed|must not|prohibited|forbidden|not required)\b", sentence, flags=re.I)
    ]
    if negative_sentences:
        return "The retrieved context contains explicit negative/prohibitive wording:\n\n" + "\n".join(
            f"- {text} [S{idx}]" for idx, text in negative_sentences[:5]
        )
    if any(term in context_text.lower() for term in important_terms(question)):
        return "The retrieved context mentions the topic, but it does not state an explicit prohibition or negative requirement. Absence of evidence is not a prohibition. [S1]"
    return NOT_FOUND


def audit_answer(contexts: list[dict]) -> str:
    if not contexts:
        return NOT_FOUND
    lines = ["Traceability evidence:"]
    for index, item in enumerate(contexts[:8], start=1):
        metadata = item.get("metadata") or {}
        section = metadata.get("section_title") or "No section"
        page = metadata.get("page_label_start") or metadata.get("page_start") or "?"
        score = round(float(item.get("score", 0)), 4)
        snippet = item.get("text", "").strip().replace("\n", " ")[:220]
        lines.append(f"- [S{index}] {item.get('filename')} | {section} | page {page} | score {score}: {snippet}")
    return "\n".join(lines)


def weak_context(contexts: list[dict]) -> bool:
    return not contexts or float(contexts[0].get("score", 0)) < 0.25


def important_terms(text: str) -> set[str]:
    stop = {"what", "where", "which", "does", "about", "from", "this", "that", "with", "there", "document"}
    return {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]{2,}", text.lower()) if term not in stop}


def sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    parts = [part.strip(" -") for part in re.split(r"(?<=[.!?])\s+", compact) if len(part.strip()) > 30]
    return parts


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def confidence_summary(contexts: list[dict]) -> dict[str, Any]:
    if not contexts:
        return {"confidence": "none", "reason": "No chunks retrieved."}
    top = float(contexts[0].get("score", 0))
    sections = Counter((item.get("metadata") or {}).get("section_title") or "No section" for item in contexts)
    filenames = defaultdict(int)
    for item in contexts:
        filenames[item.get("filename", "unknown")] += 1
    if top >= 0.55:
        level = "high"
    elif top >= 0.32:
        level = "medium"
    else:
        level = "low"
    return {
        "confidence": level,
        "top_score": round(top, 4),
        "dominant_section": sections.most_common(1)[0][0],
        "document_count": len(filenames),
    }
