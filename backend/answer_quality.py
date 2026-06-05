from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .question_types import QuestionProfile, classify_question, profile_payload

STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "from",
    "have",
    "into",
    "only",
    "shall",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "document",
    "section",
    "context",
    "detailed",
    "follows",
    "when",
    "where",
    "which",
    "with",
    "within",
}

TECH_PATTERN = re.compile(r"\b(?:ASME|NORSOK|B31\.3|P&ID|PIDs?|FPSO|GRE|AISI|BOP|L-\d+|[A-Z]{2,}[-A-Z0-9_/]{2,})\b")
NUMBER_PATTERN = re.compile(r"[0-9]+:[0-9]+|[0-9]+\u00b0|\b\d+(?:\.\d+)?\s*(?:deg|degree|degrees|mm|cm|m|%|bar|barg|psi|kpa|mpa)?\b")


def evaluate_answer(question: str, answer: str, contexts: list[dict[str, Any]], profile: QuestionProfile | None = None) -> dict[str, Any]:
    profile = profile or classify_question(question)
    context_text = "\n".join(item.get("text", "") for item in contexts)
    clean_answer = strip_citations(answer)
    answer_terms = meaningful_terms(clean_answer)
    context_terms = meaningful_terms(context_text)
    query_terms = meaningful_terms(question)

    grounding = grounding_score(answer_terms, context_terms)
    hallucination = hallucination_risk(answer_terms, context_terms, clean_answer)
    specificity = specificity_score(clean_answer, context_text)
    context = context_preservation_score(query_terms, answer_terms, contexts)
    style = style_score(question, clean_answer)
    over_generation = over_generation_risk(clean_answer, contexts, grounding)
    retrieval = retrieval_quality(contexts, query_terms)
    type_fit = question_type_fit(question, clean_answer, contexts, profile)

    checks = {
        "grounding": grounding,
        "hallucination_control": hallucination,
        "specificity": specificity,
        "context_preservation": context,
        "answer_style": style,
        "over_generation_control": over_generation,
        "retrieval_quality": retrieval,
        "question_type_fit": type_fit,
    }
    overall = round(sum(check["score"] for check in checks.values()) / len(checks), 3)
    return {
        "overall_score": overall,
        "grade": grade(overall),
        "checks": checks,
        "retrieval_evidence": retrieval_evidence(contexts),
        "answer_profile": answer_profile(question, clean_answer),
        "question_profile": profile_payload(profile),
    }


def grounding_score(answer_terms: set[str], context_terms: set[str]) -> dict[str, Any]:
    if not answer_terms:
        return check(0.0, "weak", "No answer terms found to ground.")
    overlap = len(answer_terms & context_terms) / len(answer_terms)
    label = label_for(overlap, 0.58, 0.38)
    return check(overlap, label, f"{percent(overlap)} of meaningful answer terms appear in retrieved chunks.")


def hallucination_risk(answer_terms: set[str], context_terms: set[str], answer: str) -> dict[str, Any]:
    unsupported = sorted(answer_terms - context_terms)
    technical = [term for term in unsupported if is_technicalish(term)]
    risk = min(1.0, (len(technical) * 0.18) + (len(unsupported) / max(1, len(answer_terms)) * 0.45))
    score = 1.0 - risk
    label = label_for(score, 0.72, 0.5)
    detail = "No major unsupported technical terms detected." if not technical else f"Unsupported technical terms: {', '.join(technical[:8])}."
    return check(score, label, detail, {"unsupported_terms": unsupported[:12], "unsupported_technical_terms": technical[:8]})


def specificity_score(answer: str, context_text: str) -> dict[str, Any]:
    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    context_numbers = set(NUMBER_PATTERN.findall(context_text))
    answer_tech = set(TECH_PATTERN.findall(answer))
    context_tech = set(TECH_PATTERN.findall(context_text))
    preserved_numbers = len(answer_numbers & context_numbers)
    preserved_tech = len(answer_tech & context_tech)
    score = min(1.0, (preserved_numbers * 0.18) + (preserved_tech * 0.14) + (0.2 if has_engineering_modals(answer) else 0))
    label = label_for(score, 0.55, 0.28)
    detail = f"Preserved {preserved_numbers} numeric constraints and {preserved_tech} technical identifiers from retrieved context."
    return check(score, label, detail, {"numbers": sorted(answer_numbers & context_numbers), "technical_terms": sorted(answer_tech & context_tech)})


def context_preservation_score(query_terms: set[str], answer_terms: set[str], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    section_counts = Counter((item.get("metadata") or {}).get("section_title") or "No section" for item in contexts)
    dominant_section, dominant_count = section_counts.most_common(1)[0] if section_counts else ("No section", 0)
    query_coverage = len(query_terms & answer_terms) / max(1, len(query_terms))
    section_focus = dominant_count / max(1, len(contexts))
    score = min(1.0, (query_coverage * 0.55) + (section_focus * 0.45))
    label = label_for(score, 0.68, 0.45)
    detail = f"Answer covers {percent(query_coverage)} of query terms; top retrieval section is '{dominant_section}' ({dominant_count}/{len(contexts)} chunks)."
    return check(score, label, detail, {"dominant_section": dominant_section, "section_focus": round(section_focus, 3)})


def style_score(question: str, answer: str) -> dict[str, Any]:
    desired = desired_style(question)
    has_bullets = bool(re.search(r"(^|\n)\s*[-*]\s+", answer))
    word_count = len(answer.split())
    concise = word_count <= 180
    matched = (desired == "bullet list" and has_bullets) or (desired != "bullet list" and concise)
    score = 0.85 if matched and concise else 0.58 if concise else 0.35
    label = label_for(score, 0.72, 0.5)
    detail = f"Suggested style: {desired}. Answer length: {word_count} words."
    return check(score, label, detail, {"suggested_style": desired, "word_count": word_count})


def over_generation_risk(answer: str, contexts: list[dict[str, Any]], grounding: dict[str, Any]) -> dict[str, Any]:
    context_words = sum(len((item.get("text") or "").split()) for item in contexts)
    answer_words = len(answer.split())
    length_ratio = answer_words / max(1, min(context_words, 900))
    risk = 0.0
    if length_ratio > 0.45:
        risk += 0.25
    if grounding["score"] < 0.42:
        risk += 0.35
    if re.search(r"\b(generally|typically|in industry|best practice|usually)\b", answer, flags=re.I):
        risk += 0.25
    score = 1.0 - min(1.0, risk)
    label = label_for(score, 0.72, 0.5)
    detail = "Answer stayed focused on retrieved material." if score >= 0.72 else "Answer may contain broad or weakly grounded wording."
    return check(score, label, detail, {"answer_to_context_ratio": round(length_ratio, 3)})


def retrieval_quality(contexts: list[dict[str, Any]], query_terms: set[str]) -> dict[str, Any]:
    if not contexts:
        return check(0.0, "weak", "No chunks retrieved.")
    top_score = float(contexts[0].get("score", 0))
    avg_top3 = sum(float(item.get("score", 0)) for item in contexts[:3]) / min(3, len(contexts))
    section_counts = Counter((item.get("metadata") or {}).get("section_title") or "No section" for item in contexts)
    section_focus = section_counts.most_common(1)[0][1] / len(contexts)
    table_bonus = 0.08 if any((item.get("metadata") or {}).get("contains_table") for item in contexts) and query_terms & {"table", "slope", "rating", "class", "size"} else 0
    score = min(1.0, (top_score * 0.55) + (avg_top3 * 0.25) + (section_focus * 0.2) + table_bonus)
    label = label_for(score, 0.55, 0.35)
    detail = f"Top score {top_score:.3f}, top-3 average {avg_top3:.3f}, section focus {percent(section_focus)}."
    return check(score, label, detail)


def question_type_fit(question: str, answer: str, contexts: list[dict[str, Any]], profile: QuestionProfile) -> dict[str, Any]:
    checks: list[float] = []
    details: list[str] = []
    normalized_answer = answer.lower().strip()

    if profile.require_citations:
        has_citation = bool(re.search(r"\[S\d+\]", answer)) or answer == "Not found in the retrieved document context."
        checks.append(1.0 if has_citation else 0.0)
        details.append("citations present" if has_citation else "missing citations")

    if profile.type_id == "yes_no":
        starts_yes_no = normalized_answer.startswith(("yes", "no", "not stated", "not found"))
        checks.append(1.0 if starts_yes_no else 0.25)
        details.append("yes/no first" if starts_yes_no else "does not start with yes/no/not stated")

    if profile.type_id == "comparison":
        has_compare_shape = bool(re.search(r"\b(compared|whereas|while|difference|versus)\b|(^|\n)\s*[-*]\s+", normalized_answer))
        checks.append(1.0 if has_compare_shape else 0.45)
        details.append("comparison format present" if has_compare_shape else "comparison format weak")

    if profile.type_id in {"table_numeric", "calculation"}:
        context_numbers = set(NUMBER_PATTERN.findall("\n".join(item.get("text", "") for item in contexts)))
        answer_numbers = set(NUMBER_PATTERN.findall(answer))
        exact_numbers = bool(answer_numbers) and answer_numbers <= context_numbers
        not_found = answer == "Not found in the retrieved document context."
        checks.append(1.0 if exact_numbers or not_found else 0.25)
        details.append("numeric values are sourced" if exact_numbers else "numeric values missing or not fully sourced")

    if profile.type_id in {"safety_critical", "safety_interpretation", "engineering_decision", "negative", "knowledge_gap", "adversarial_stress"}:
        risky_phrases = bool(re.search(r"\b(you should|recommended to|safe to|definitely|guaranteed|go ahead)\b", normalized_answer))
        checks.append(0.15 if risky_phrases else 1.0)
        details.append("conservative safety wording" if not risky_phrases else "contains risky recommendation wording")

    if profile.type_id in {"procedural", "workflow"}:
        preserves_modal = bool(re.search(r"\b(shall|must|required|ensure|provided)\b", normalized_answer))
        checks.append(1.0 if preserves_modal else 0.45)
        details.append("mandatory wording preserved" if preserves_modal else "mandatory wording weak")

    if profile.type_id in {"enumeration", "multi_part", "search_discovery"}:
        has_structure = bool(re.search(r"(^|\n)\s*[-*]\s+", answer)) or len(re.findall(r"\b\d+\.", answer)) >= 2
        checks.append(1.0 if has_structure else 0.45)
        details.append("structured answer" if has_structure else "answer structure weak")

    if profile.type_id == "out_of_document":
        external_claim = bool(re.search(r"\b(in general|industry|internet|outside the document)\b", normalized_answer))
        checks.append(0.25 if external_claim else 1.0)
        details.append("external knowledge avoided" if not external_claim else "external wording detected")

    if not checks:
        concise = len(answer.split()) <= 180
        checks.append(1.0 if concise else 0.6)
        details.append("concise profile fit" if concise else "answer may be too long")

    score = sum(checks) / len(checks)
    return check(score, label_for(score, 0.72, 0.5), f"{profile.label}: {'; '.join(details[:4])}.")


def retrieval_evidence(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for item in contexts:
        metadata = item.get("metadata") or {}
        evidence.append(
            {
                "filename": item.get("filename"),
                "chunk_index": item.get("chunk_index"),
                "score": round(float(item.get("score", 0)), 4),
                "section": metadata.get("section_title") or "No section",
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "contains_table": bool(metadata.get("contains_table")),
                "keywords": metadata.get("keywords", [])[:8],
            }
        )
    return evidence


def answer_profile(question: str, answer: str) -> dict[str, Any]:
    return {
        "recommended_style": desired_style(question),
        "word_count": len(answer.split()),
        "has_bullets": bool(re.search(r"(^|\n)\s*[-*]\s+", answer)),
    }


def meaningful_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.&/-]{2,}", text)
        if token.lower() not in STOPWORDS
    }


def strip_citations(text: str) -> str:
    text = re.sub(r"\[S\d+\]", " ", text)
    text = re.sub(r"\[[^\]]*#\d+[^\]]*\]", " ", text)
    text = re.sub(r"\b\S+\.(?:pdf|docx|xlsx|csv|txt)\b", " ", text, flags=re.I)
    return text


def is_technicalish(term: str) -> bool:
    return bool(re.search(r"\d|[-_/&.]|[A-Z]{2,}", term)) or term in {"valve", "piping", "slope", "drain", "flare", "hydrocarbon"}


def has_engineering_modals(answer: str) -> bool:
    return bool(re.search(r"\b(shall|must|required|minimum|maximum|not be|only be|provided|installed)\b", answer, re.I))


def desired_style(question: str) -> str:
    normalized = question.lower()
    if normalized.startswith(("is ", "are ", "does ", "do ", "can ", "shall ", "should ")):
        return "yes/no with explanation"
    if any(word in normalized for word in ("list", "requirements", "constraints", "steps", "points")):
        return "bullet list"
    if any(word in normalized for word in ("compare", "difference", "versus", "vs")):
        return "comparison"
    return "short engineering paragraph"


def check(score: float, label: str, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"score": round(max(0.0, min(1.0, score)), 3), "label": label, "detail": detail}
    if extra:
        payload.update(extra)
    return payload


def label_for(score: float, good: float, watch: float) -> str:
    if score >= good:
        return "good"
    if score >= watch:
        return "watch"
    return "weak"


def grade(score: float) -> str:
    if score >= 0.74:
        return "enterprise-ready"
    if score >= 0.58:
        return "usable-check"
    if score >= 0.42:
        return "needs-review"
    return "weak"


def percent(value: float) -> str:
    return f"{round(value * 100)}%"
