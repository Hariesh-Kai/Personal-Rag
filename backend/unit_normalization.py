from __future__ import annotations

import hashlib
import re
from typing import Any


UNIT_NORMALIZATION_SCHEMA_VERSION = "engineering-unit-normalization-v1"

UNIT_ALIASES = {
    "": "",
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "in": "inch",
    "inch": "inch",
    "inches": "inch",
    "ft": "ft",
    "feet": "ft",
    "foot": "ft",
    "bar": "bar",
    "barg": "bar",
    "bara": "bar",
    "psi": "psi",
    "psig": "psi",
    "kpa": "kpa",
    "mpa": "mpa",
    "pa": "pa",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "c": "c",
    "degc": "c",
    "celsius": "c",
    "hz": "hz",
    "rpm": "rpm",
    "kg": "kg",
    "g": "g",
    "ton": "ton",
    "tons": "ton",
    "tonne": "ton",
    "tonnes": "ton",
}

UNIT_FAMILIES = {
    "mm": "length",
    "cm": "length",
    "m": "length",
    "inch": "length",
    "ft": "length",
    "bar": "pressure",
    "psi": "pressure",
    "kpa": "pressure",
    "mpa": "pressure",
    "pa": "pressure",
    "deg": "angle",
    "c": "temperature",
    "%": "percent",
    "hz": "frequency",
    "rpm": "frequency",
    "kg": "mass",
    "g": "mass",
    "ton": "mass",
}

SI_CONVERSIONS = {
    "mm": ("m", 0.001, 0.0),
    "cm": ("m", 0.01, 0.0),
    "m": ("m", 1.0, 0.0),
    "inch": ("m", 0.0254, 0.0),
    "ft": ("m", 0.3048, 0.0),
    "pa": ("pa", 1.0, 0.0),
    "kpa": ("pa", 1000.0, 0.0),
    "mpa": ("pa", 1000000.0, 0.0),
    "bar": ("pa", 100000.0, 0.0),
    "psi": ("pa", 6894.757293, 0.0),
    "c": ("c", 1.0, 0.0),
    "deg": ("deg", 1.0, 0.0),
    "%": ("ratio", 0.01, 0.0),
    "hz": ("hz", 1.0, 0.0),
    "rpm": ("hz", 1.0 / 60.0, 0.0),
    "g": ("kg", 0.001, 0.0),
    "kg": ("kg", 1.0, 0.0),
    "ton": ("kg", 1000.0, 0.0),
}


def unit_normalization_metadata(constraints: list[dict[str, Any]], text: str = "") -> dict[str, Any]:
    normalized_constraints = [normalize_numeric_constraint(item) for item in constraints if isinstance(item, dict)]
    normalized_constraints = [item for item in normalized_constraints if item.get("value")]
    units = sorted({item.get("normalized_unit") for item in normalized_constraints if item.get("normalized_unit")})
    families = sorted({item.get("unit_family") for item in normalized_constraints if item.get("unit_family")})
    retrieval_text = unit_retrieval_text(normalized_constraints, text)
    return {
        "unit_normalization_schema_version": UNIT_NORMALIZATION_SCHEMA_VERSION,
        "unit_normalization_ready": True,
        "unit_normalization_applied": bool(normalized_constraints),
        "normalized_numeric_constraints": normalized_constraints[:80],
        "numeric_units": units,
        "numeric_unit_families": families,
        "canonical_numeric_units": sorted({item.get("canonical_unit") for item in normalized_constraints if item.get("canonical_unit")}),
        "unit_normalized_values": [item for item in normalized_constraints[:80] if item.get("canonical_value") is not None],
        "unit_ratio_constraints": [item for item in normalized_constraints[:40] if item.get("unit_family") == "ratio"],
        "unit_normalization_retrieval_text": retrieval_text,
        "unit_normalization_hash": hashlib.sha1(retrieval_text.encode("utf-8")).hexdigest()[:16] if retrieval_text else "",
        "numeric_constraint_schema_version": "engineering-numeric-constraints-v1",
        "numeric_constraint_ingestion_ready": bool(normalized_constraints),
    }


def normalize_numeric_constraint(item: dict[str, Any]) -> dict[str, Any]:
    value = str(item.get("value") or "").strip()
    original_unit = str(item.get("unit") or "").strip()
    normalized_unit = normalize_unit(original_unit)
    parsed = parse_numeric_value(value)
    family = unit_family(normalized_unit, value)
    canonical_unit, canonical_value = canonicalize_value(parsed, normalized_unit, family)
    normalized = {
        **item,
        "value": value,
        "unit": original_unit,
        "normalized_unit": normalized_unit,
        "unit_family": family,
        "canonical_unit": canonical_unit,
        "canonical_value": canonical_value,
        "canonical_value_text": format_number(canonical_value) if canonical_value is not None else "",
        "value_kind": parsed.get("kind"),
        "ratio_numerator": parsed.get("numerator"),
        "ratio_denominator": parsed.get("denominator"),
        "ratio_decimal": parsed.get("decimal"),
        "search_token": build_search_token(value, normalized_unit, canonical_value, canonical_unit, item.get("context", "")),
        "unit_aliases": unit_aliases(normalized_unit),
    }
    return normalized


def parse_numeric_value(value: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", value)
    ratio = re.match(r"^(?P<num>\d+(?:\.\d+)?)[/:](?P<den>\d+(?:\.\d+)?)$", compact)
    if ratio:
        numerator = float(ratio.group("num"))
        denominator = float(ratio.group("den"))
        decimal = numerator / denominator if denominator else None
        return {"kind": "ratio", "number": None, "numerator": numerator, "denominator": denominator, "decimal": decimal}
    try:
        return {"kind": "number", "number": float(compact), "numerator": None, "denominator": None, "decimal": None}
    except ValueError:
        return {"kind": "text", "number": None, "numerator": None, "denominator": None, "decimal": None}


def normalize_unit(unit: str) -> str:
    normalized = re.sub(r"[^A-Za-z%]+", "", str(unit or "").lower())
    return UNIT_ALIASES.get(normalized, normalized)


def unit_family(unit: str, value: str) -> str:
    if ":" in value or "/" in value:
        return "ratio"
    return UNIT_FAMILIES.get(unit, "number" if not unit else "other")


def canonicalize_value(parsed: dict[str, Any], unit: str, family: str) -> tuple[str, float | None]:
    if family == "ratio":
        return "ratio", parsed.get("decimal")
    number = parsed.get("number")
    if number is None:
        return unit, None
    conversion = SI_CONVERSIONS.get(unit)
    if not conversion:
        return unit, number
    canonical_unit, multiplier, offset = conversion
    return canonical_unit, round(number * multiplier + offset, 8)


def unit_aliases(unit: str) -> list[str]:
    return sorted({alias for alias, normalized in UNIT_ALIASES.items() if normalized == unit and alias})


def build_search_token(value: str, unit: str, canonical_value: float | None, canonical_unit: str, context: Any) -> str:
    parts = [
        value,
        unit,
        f"{format_number(canonical_value)} {canonical_unit}" if canonical_value is not None and canonical_unit else "",
        str(context or ""),
    ]
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def unit_retrieval_text(records: list[dict[str, Any]], text: str) -> str:
    parts: list[str] = []
    for item in records:
        parts.append(
            " ".join(
                str(item.get(key) or "")
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
    parts.append(text[:1200])
    return "\n".join(part for part in parts if part)


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.8g}"
    return f"{value:.6f}".rstrip("0").rstrip(".")
