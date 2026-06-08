from __future__ import annotations

import hashlib
import os
import re
from typing import Any


ACCESS_CONTROL_SCHEMA_VERSION = "engineering-access-control-v2"

CLASSIFICATION_LEVELS = {
    "public": 0,
    "unclassified": 1,
    "internal": 2,
    "restricted": 3,
    "confidential": 4,
    "controlled": 5,
}

ACCESS_PATTERNS: dict[str, dict[str, Any]] = {
    "confidential": {
        "classification": "confidential",
        "terms": ["confidential", "commercial in confidence", "company confidential", "proprietary"],
        "roles": ["engineering_manager", "document_controller", "safety_engineer"],
    },
    "controlled_copy": {
        "classification": "controlled",
        "terms": ["controlled copy", "uncontrolled copy", "controlled document", "document control"],
        "roles": ["document_controller", "engineering_manager"],
    },
    "internal": {
        "classification": "internal",
        "terms": ["internal use", "company internal", "for internal use only", "not to be distributed"],
        "roles": ["employee", "engineer", "safety_engineer", "document_controller"],
    },
    "distribution_restricted": {
        "classification": "restricted",
        "terms": ["third parties", "not to be reproduced", "all rights reserved", "distribution restricted"],
        "roles": ["engineering_manager", "document_controller"],
    },
    "draft_review": {
        "classification": "internal",
        "terms": ["draft", "preliminary", "for review", "issued for review"],
        "roles": ["engineer", "engineering_manager", "document_controller"],
    },
    "safety_sensitive": {
        "classification": "restricted",
        "terms": ["safety critical", "emergency shutdown", "fire and explosion", "hazard", "relief system"],
        "roles": ["safety_engineer", "engineering_manager"],
    },
}


def access_control_metadata(text: str, doc_meta: dict[str, Any], extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    haystack = normalize(" ".join([text, str(doc_meta.get("source_name") or ""), str(doc_meta.get("filename") or "")])).lower()
    matches = access_matches(haystack)
    classification = highest_classification(matches)
    sensitivity = CLASSIFICATION_LEVELS.get(classification, 1)
    allowed_roles = access_allowed_roles(matches, classification)
    tags = sorted({match["tag"] for match in matches} | ({classification} if classification not in {"public", "unclassified"} else set()))
    policy_required = classification not in {"public", "unclassified"} or bool(matches)
    action = policy_action(classification, tags)
    redaction_fields = redaction_recommendations(haystack, classification)
    return {
        "access_control_schema_version": ACCESS_CONTROL_SCHEMA_VERSION,
        "access_control_ready": True,
        "access_classification": classification,
        "access_sensitivity_level": sensitivity,
        "access_control_tags": tags,
        "access_policy_required": policy_required,
        "access_policy_action": action,
        "access_allowed_roles": allowed_roles,
        "access_denied_roles": denied_roles(allowed_roles),
        "access_redaction_required": bool(redaction_fields),
        "access_redaction_fields": redaction_fields,
        "access_control_matches": matches,
        "access_control_enforced": bool(os.environ.get("RAG_ACCESS_CONTROL_ENFORCED", "").lower() in {"1", "true", "yes"}),
        "access_control_source": "ingestion_policy_metadata",
        "access_control_note": "ingestion policy metadata is ready; request-time enforcement depends on caller role configuration",
        "access_control_hash": hashlib.sha1(f"{classification}|{','.join(tags)}|{haystack[:500]}".encode("utf-8")).hexdigest()[:16],
        "access_control_decision": {
            "classification": classification,
            "policy_required": policy_required,
            "action": action,
            "default_allow_without_role": classification in {"public", "unclassified"},
        },
    }


def access_allowed(metadata: dict[str, Any], user_roles: list[str] | None = None) -> tuple[bool, dict[str, Any]]:
    if not metadata.get("access_policy_required"):
        return True, {"reason": "no_policy_required", "required_roles": []}
    allowed_roles = set(metadata.get("access_allowed_roles") or [])
    roles = set(user_roles or configured_roles())
    if not allowed_roles:
        return False, {"reason": "no_allowed_roles_configured", "required_roles": []}
    matched = sorted(roles & allowed_roles)
    return bool(matched), {
        "reason": "role_match" if matched else "missing_required_role",
        "matched_roles": matched,
        "required_roles": sorted(allowed_roles),
        "classification": metadata.get("access_classification") or "",
    }


def configured_roles() -> list[str]:
    raw = os.environ.get("RAG_USER_ROLES", "")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def access_matches(haystack: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for tag, rule in ACCESS_PATTERNS.items():
        matched_terms = [term for term in rule["terms"] if term in haystack]
        if matched_terms:
            matches.append(
                {
                    "tag": tag,
                    "classification": rule["classification"],
                    "matched_terms": matched_terms[:8],
                    "allowed_roles": rule["roles"],
                }
            )
    return matches


def highest_classification(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "unclassified"
    return max((match["classification"] for match in matches), key=lambda item: CLASSIFICATION_LEVELS.get(item, 0))


def access_allowed_roles(matches: list[dict[str, Any]], classification: str) -> list[str]:
    roles = set()
    max_level = CLASSIFICATION_LEVELS.get(classification, 0)
    controlling_matches = [
        match
        for match in matches
        if CLASSIFICATION_LEVELS.get(str(match.get("classification") or ""), 0) == max_level
    ]
    for match in controlling_matches:
        roles.update(str(role).lower() for role in match.get("allowed_roles") or [])
    if not roles:
        if classification == "unclassified":
            roles.update(["employee", "engineer", "safety_engineer", "document_controller", "engineering_manager"])
        elif classification == "internal":
            roles.update(["employee", "engineer", "safety_engineer", "document_controller", "engineering_manager"])
        elif classification in {"restricted", "confidential"}:
            roles.update(["engineering_manager", "document_controller", "safety_engineer"])
        elif classification == "controlled":
            roles.update(["document_controller", "engineering_manager"])
    return sorted(roles)


def denied_roles(allowed_roles: list[str]) -> list[str]:
    known = {"guest", "contractor", "employee", "engineer", "safety_engineer", "document_controller", "engineering_manager"}
    return sorted(known - set(allowed_roles))


def policy_action(classification: str, tags: list[str]) -> str:
    if classification in {"controlled", "confidential"}:
        return "require_authorized_role_and_show_citation_only"
    if classification == "restricted" or "distribution_restricted" in tags:
        return "require_authorized_role"
    if classification == "internal":
        return "allow_employee_roles"
    return "allow"


def redaction_recommendations(haystack: str, classification: str) -> list[str]:
    fields = []
    if classification in {"confidential", "controlled"}:
        fields.append("distribution_notice")
    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", haystack, flags=re.I):
        fields.append("email_address")
    if re.search(r"\b(?:phone|mobile|tel)\s*[:#]?\s*\+?\d[\d -]{6,}\b", haystack, flags=re.I):
        fields.append("phone_number")
    return sorted(set(fields))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
