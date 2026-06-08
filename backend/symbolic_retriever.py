from __future__ import annotations

import re
from typing import Any

SYMBOLIC_SCHEMA_VERSION = "engineering-symbolic-retriever-v1"

RULE_PATTERNS: dict[str, dict[str, Any]] = {
    "obligation": {
        "patterns": [r"\bshall\b", r"\bmust\b", r"\brequired\b", r"\bmandatory\b", r"\bensure\b"],
        "query_terms": ["shall", "must", "required", "mandatory", "requirement"],
        "weight": 1.2,
    },
    "prohibition": {
        "patterns": [r"\bshall not\b", r"\bmust not\b", r"\bnot allowed\b", r"\bprohibited\b", r"\bforbidden\b"],
        "query_terms": ["shall not", "must not", "not allowed", "prohibited"],
        "weight": 1.35,
    },
    "condition": {
        "patterns": [r"\bif\b", r"\bwhen\b", r"\bwhere\b", r"\bprovided that\b", r"\bin case\b", r"\bsubject to\b"],
        "query_terms": ["if", "when", "where", "condition", "provided that"],
        "weight": 1.18,
    },
    "exception": {
        "patterns": [r"\bexcept\b", r"\bunless\b", r"\bexcept where\b", r"\bexcept when\b", r"\bexcept for\b"],
        "query_terms": ["except", "unless", "exception"],
        "weight": 1.3,
    },
    "permission": {
        "patterns": [r"\bmay\b", r"\bcan\b", r"\bpermitted\b", r"\ballowed\b"],
        "query_terms": ["may", "permitted", "allowed"],
        "weight": 0.82,
    },
    "recommendation": {
        "patterns": [r"\bshould\b", r"\bpreferably\b", r"\brecommended\b"],
        "query_terms": ["should", "preferably", "recommended"],
        "weight": 0.95,
    },
    "definition": {
        "patterns": [r"\bis defined as\b", r"\bmeans\b", r"\brefers to\b", r"\bshall mean\b"],
        "query_terms": ["defined as", "means", "refers to"],
        "weight": 0.9,
    },
    "numeric_constraint": {
        "patterns": [r"\b\d+(?:\.\d+)?\s*(?:mm|m|bar|psi|deg|degree|%)\b", r"\b1\s*:\s*\d+\b", r"\bminimum\b", r"\bmaximum\b", r"\bat least\b"],
        "query_terms": ["minimum", "maximum", "numeric", "constraint", "value"],
        "weight": 1.28,
    },
    "safety_critical": {
        "patterns": [r"\bsafety\b", r"\bemergency\b", r"\bshutdown\b", r"\bfire\b", r"\bexplosion\b", r"\bhazard\b", r"\brelief\b"],
        "query_terms": ["safety", "emergency", "shutdown", "fire", "explosion", "hazard"],
        "weight": 1.25,
    },
}


def symbolic_plan(query: str, route: Any, profile: Any) -> dict[str, Any]:
    normalized = normalize_text(query)
    route_retrievers = set(getattr(route, "retrievers", ()) or ())
    profile_type = getattr(profile, "type_id", "")
    requested_types = detect_requested_rule_types(normalized, profile_type)
    active = bool(requested_types) or "symbolic" in route_retrievers
    return {
        "schema": SYMBOLIC_SCHEMA_VERSION,
        "active": active,
        "route_primary": getattr(route, "primary", ""),
        "profile_type": profile_type,
        "requested_rule_types": requested_types,
        "query_terms": symbolic_query_terms(requested_types),
        "strategy": "local_rule_symbol_extraction_modal_condition_exception_prohibition_requirement_scoring",
    }


def symbolic_expanded_queries(seed_queries: list[str], plan: dict[str, Any]) -> list[str]:
    queries = list(seed_queries)
    if not plan.get("active"):
        return unique_preserve(queries)[:16]
    rule_types = plan.get("requested_rule_types") or []
    if rule_types:
        queries.append(f"symbolic rules {' '.join(rule_types)} {seed_queries[0] if seed_queries else ''}".strip())
    terms = plan.get("query_terms") or []
    if terms:
        queries.append(f"engineering rule words {' '.join(terms[:24])}")
    for rule_type in rule_types[:8]:
        queries.append(f"{rule_type} {' '.join((RULE_PATTERNS.get(rule_type) or {}).get('query_terms') or [])}".strip())
    return unique_preserve(query for query in queries if str(query).strip())[:18]


def symbolic_candidate_score(plan: dict[str, Any], metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched_rules": []}
    haystack = symbolic_haystack(metadata, text)
    extracted_rules = extract_symbolic_rules(text, metadata)
    requested = plan.get("requested_rule_types") or []
    matched = []
    score = 0.0
    for rule_type, rule_data in extracted_rules.items():
        rule_weight = float((RULE_PATTERNS.get(rule_type) or {}).get("weight") or 1.0)
        requested_bonus = 1.35 if rule_type in requested else 1.0
        evidence_count = len(rule_data.get("evidence") or [])
        local_score = min(0.045, evidence_count * 0.014 * rule_weight * requested_bonus)
        if local_score:
            matched.append({"rule_type": rule_type, "evidence": (rule_data.get("evidence") or [])[:5], "score": round(local_score, 5)})
            score += local_score

    query_term_hits = matching_terms(plan.get("query_terms") or [], haystack)
    score += min(0.04, len(query_term_hits) * 0.006)
    metadata_bonus = metadata_symbolic_bonus(requested, metadata)
    score += metadata_bonus
    return round(min(0.18, score), 5), {
        "schema": SYMBOLIC_SCHEMA_VERSION,
        "active": True,
        "requested_rule_types": requested,
        "matched_rules": matched,
        "query_term_hits": query_term_hits[:24],
        "metadata_bonus": round(metadata_bonus, 5),
        "rule_coverage": round(len({item["rule_type"] for item in matched} & set(requested)) / max(1, len(requested)), 4),
    }


def extract_symbolic_rules(text: str, metadata: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    metadata = metadata or {}
    source = symbolic_haystack(metadata, text)
    result: dict[str, dict[str, Any]] = {}
    for rule_type, data in RULE_PATTERNS.items():
        evidence = []
        for pattern in data.get("patterns") or []:
            for match in re.finditer(pattern, source, flags=re.I):
                evidence.append(context_window(source, match.start(), match.end()))
                if len(evidence) >= 8:
                    break
            if len(evidence) >= 8:
                break
        if evidence:
            result[rule_type] = {"type": rule_type, "evidence": unique_preserve(evidence)}
    return result


def detect_requested_rule_types(normalized_query: str, profile_type: str) -> list[str]:
    requested = []
    profile_map = {
        "procedural": ["obligation", "condition"],
        "workflow": ["obligation", "condition"],
        "yes_no": ["obligation", "prohibition"],
        "safety_critical": ["safety_critical", "obligation", "prohibition", "condition"],
        "safety_interpretation": ["safety_critical", "obligation", "exception", "prohibition"],
        "conditional": ["condition", "obligation"],
        "exception": ["exception", "prohibition", "condition"],
        "negative": ["prohibition", "exception"],
        "multi_constraint": ["numeric_constraint", "condition", "obligation"],
        "table_numeric": ["numeric_constraint"],
        "calculation": ["numeric_constraint"],
        "regulation_compliance": ["obligation", "prohibition", "safety_critical"],
        "engineering_decision": ["obligation", "prohibition", "recommendation"],
    }
    requested.extend(profile_map.get(profile_type, []))
    for rule_type, data in RULE_PATTERNS.items():
        if rule_type == "permission" and re.search(r"\bnot allowed\b", normalized_query):
            continue
        if any(re.search(pattern, normalized_query, flags=re.I) for pattern in data.get("patterns") or []):
            requested.append(rule_type)
    if re.search(r"\b(do we need|required|requirement|mandatory)\b", normalized_query):
        requested.append("obligation")
    if re.search(r"\b(can|may|permitted)\b", normalized_query) or (re.search(r"\ballowed\b", normalized_query) and not re.search(r"\bnot allowed\b", normalized_query)):
        requested.append("permission")
    if re.search(r"\b(exception|except|unless)\b", normalized_query):
        requested.append("exception")
    if re.search(r"\b(not|avoid|forbid|prohibit)\b", normalized_query):
        requested.append("prohibition")
    return unique_preserve(requested)[:10]


def symbolic_query_terms(rule_types: list[str]) -> list[str]:
    terms = []
    for rule_type in rule_types:
        terms.extend((RULE_PATTERNS.get(rule_type) or {}).get("query_terms") or [])
    return unique_preserve(normalize_text(term) for term in terms if normalize_text(term))[:40]


def metadata_symbolic_bonus(requested: list[str], metadata: dict[str, Any]) -> float:
    score = 0.0
    if metadata.get("has_requirement") and ("obligation" in requested or not requested):
        score += 0.018
    if metadata.get("safety_critical") and ("safety_critical" in requested or not requested):
        score += 0.025
    if metadata.get("has_numeric_constraints") and ("numeric_constraint" in requested or not requested):
        score += 0.02
    tags = normalize_text(
        " ".join(
            [
                flatten_values(metadata.get("safety_flags") or []),
                flatten_values(metadata.get("compliance_flags") or []),
                flatten_values(metadata.get("retrieval_tags") or []),
                flatten_values(metadata.get("semantic_labels") or []),
            ]
        )
    )
    if requested and any(rule_type.replace("_", " ") in tags or rule_type in tags for rule_type in requested):
        score += 0.018
    return min(0.06, score)


def matching_terms(terms: list[str], haystack: str) -> list[str]:
    hits = []
    for term in terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        if " " in normalized:
            matched = normalized in haystack
        else:
            matched = bool(re.search(rf"\b{re.escape(normalized)}\b", haystack))
        if matched:
            hits.append(normalized)
    return unique_preserve(hits)


def symbolic_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(
        " ".join(
            [
                text,
                metadata.get("section_title") or "",
                metadata.get("current_section_id") or "",
                flatten_values(metadata.get("semantic_labels") or []),
                flatten_values(metadata.get("safety_flags") or []),
                flatten_values(metadata.get("compliance_flags") or []),
                flatten_values(metadata.get("numeric_constraints") or []),
                flatten_values(metadata.get("retrieval_tags") or []),
                flatten_values(metadata.get("keywords") or []),
            ]
        )
    )


def context_window(text: str, start: int, end: int, radius: int = 70) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def flatten_values(values: Any) -> str:
    if isinstance(values, dict):
        return " ".join(flatten_values(value) for value in values.values())
    if isinstance(values, (list, tuple, set)):
        return " ".join(flatten_values(value) for value in values)
    return str(values or "")


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
