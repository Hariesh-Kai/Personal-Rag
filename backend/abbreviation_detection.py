from __future__ import annotations

import hashlib
import re
from typing import Any


ABBREVIATION_SCHEMA_VERSION = "engineering-abbreviation-detection-v1"

KNOWN_ABBREVIATIONS = {
    "ASME": "American Society of Mechanical Engineers",
    "API": "American Petroleum Institute",
    "BDV": "blowdown valve",
    "CV": "control valve",
    "ESD": "emergency shutdown",
    "HVAC": "heating ventilation and air conditioning",
    "IEC": "International Electrotechnical Commission",
    "ISO": "International Organization for Standardization",
    "NORSOK": "Norwegian petroleum industry standard",
    "P&ID": "piping and instrumentation diagram",
    "PID": "piping and instrumentation diagram",
    "PIDS": "piping and instrumentation diagrams",
    "PRV": "pressure relief valve",
    "PSV": "pressure safety valve",
    "ROV": "remotely operated valve",
    "SDV": "shutdown valve",
}


def abbreviation_metadata(text: str, keyword_set: Any | None = None, engineering_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    engineering_meta = engineering_meta or {}
    explicit = explicit_abbreviations(text)
    known = known_abbreviation_records(text, keyword_set, engineering_meta)
    unknown = unknown_acronym_records(text, explicit, known, keyword_set)
    records = merge_records([*explicit, *known, *unknown])
    expansion_terms = abbreviation_expansion_terms(records)
    return {
        "abbreviation_schema_version": ABBREVIATION_SCHEMA_VERSION,
        "abbreviation_detection_ready": True,
        "abbreviation_records": records,
        "abbreviation_count": len(records),
        "known_abbreviations": [record for record in records if record.get("source") == "known_dictionary"],
        "defined_abbreviations": [record for record in records if record.get("source") == "explicit_definition"],
        "unknown_abbreviations": [record for record in records if record.get("source") == "unknown_acronym"],
        "abbreviation_terms": [record["abbr"] for record in records],
        "abbreviation_expansions": expansion_terms,
        "abbreviation_retrieval_text": " ".join(expansion_terms),
        "abbreviation_expansion_ready": bool(records),
        "abbreviation_hash": hashlib.sha1("|".join(expansion_terms).encode("utf-8")).hexdigest()[:16] if expansion_terms else "",
    }


def explicit_abbreviations(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    patterns = [
        re.compile(r"\b(?P<long>[A-Z][A-Za-z][A-Za-z0-9 /&,-]{3,90}?)\s*\((?P<abbr>[A-Z][A-Z0-9&/-]{1,12})\)"),
        re.compile(r"\b(?P<abbr>[A-Z][A-Z0-9&/-]{1,12})\s*\((?P<long>[A-Z][A-Za-z][A-Za-z0-9 /&,-]{3,90}?)\)"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            abbr = normalize_abbr(match.group("abbr"))
            expansion = normalize_phrase(match.group("long"))
            if valid_abbreviation(abbr, expansion):
                records.append(record(abbr, expansion, "explicit_definition", 0.94))
    return records


def known_abbreviation_records(text: str, keyword_set: Any | None, engineering_meta: dict[str, Any]) -> list[dict[str, Any]]:
    haystack_terms = set(extract_acronyms(text))
    haystack_terms.update(str(item).upper() for item in getattr(keyword_set, "acronyms", []) or [])
    haystack_terms.update(str(item).upper() for item in engineering_meta.get("engineering_entity_aliases") or [])
    records = []
    for abbr in sorted(haystack_terms):
        normalized = normalize_abbr(abbr)
        if normalized in KNOWN_ABBREVIATIONS:
            records.append(record(normalized, KNOWN_ABBREVIATIONS[normalized], "known_dictionary", 0.88))
    return records


def unknown_acronym_records(
    text: str,
    explicit: list[dict[str, Any]],
    known: list[dict[str, Any]],
    keyword_set: Any | None,
) -> list[dict[str, Any]]:
    known_abbrs = {item["abbr"] for item in explicit + known}
    candidates = set(extract_acronyms(text))
    candidates.update(str(item).upper() for item in getattr(keyword_set, "acronyms", []) or [])
    records = []
    for abbr in sorted(candidates):
        normalized = normalize_abbr(abbr)
        if normalized not in known_abbrs and len(normalized) >= 2:
            records.append(record(normalized, "", "unknown_acronym", 0.35))
    return records[:40]


def merge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    priority = {"explicit_definition": 3, "known_dictionary": 2, "unknown_acronym": 1}
    for item in records:
        abbr = item["abbr"]
        existing = merged.get(abbr)
        if not existing or priority.get(item["source"], 0) > priority.get(existing["source"], 0):
            merged[abbr] = item
        elif existing and item.get("expansion") and item["expansion"] not in existing.get("aliases", []):
            existing.setdefault("aliases", []).append(item["expansion"])
    return sorted(merged.values(), key=lambda item: (item["source"] != "explicit_definition", item["abbr"]))[:80]


def abbreviation_expansion_terms(records: list[dict[str, Any]]) -> list[str]:
    terms = []
    for item in records:
        terms.append(item["abbr"])
        if item.get("expansion"):
            terms.append(item["expansion"])
        terms.extend(item.get("aliases") or [])
    return unique_preserve(terms)


def extract_acronyms(text: str) -> list[str]:
    values = []
    for match in re.finditer(r"\b(?:[A-Z]{2,}(?:[&/-][A-Z0-9]+)*|P&ID)\b", text or ""):
        abbr = normalize_abbr(match.group(0))
        if abbr not in values and not abbr.isdigit():
            values.append(abbr)
    return values[:100]


def valid_abbreviation(abbr: str, expansion: str) -> bool:
    if not abbr or not expansion or len(abbr) < 2:
        return False
    letters = "".join(word[0].upper() for word in re.findall(r"[A-Za-z]+", expansion))
    compact = re.sub(r"[^A-Z0-9]", "", abbr.upper())
    return compact[:2] in letters or letters[: len(compact)] == compact or compact in KNOWN_ABBREVIATIONS


def record(abbr: str, expansion: str, source: str, confidence: float) -> dict[str, Any]:
    return {
        "abbr": normalize_abbr(abbr),
        "expansion": normalize_phrase(expansion),
        "source": source,
        "confidence": round(confidence, 3),
        "aliases": [],
    }


def normalize_abbr(value: str) -> str:
    return str(value or "").strip().upper().replace("PIDS", "PIDS")


def normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" -:;,."))


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
