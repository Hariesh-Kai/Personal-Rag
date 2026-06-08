from __future__ import annotations

import hashlib
import re
from typing import Any


FORMULA_SCHEMA_VERSION = "engineering-formula-equation-ingestion-v1"

EQUATION_RE = re.compile(
    r"(?P<formula>\b[A-Za-z][A-Za-z0-9_./-]*\s*(?:=|<=|>=|<|>|:=)\s*[^.;\n]{1,180})"
)
RATIO_RE = re.compile(r"\b(?P<ratio>\d+(?:\.\d+)?\s*[:/]\s*\d+(?:\.\d+)?)\b")
OPERATOR_RE = re.compile(r"(?:\+|-|\*|/|\^|=|<=|>=|<|>|:=)")
VARIABLE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{0,12}\b")
UNIT_EXPR_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|bar|barg|psi|kpa|mpa|deg|degree|degrees|%|c|hz|kg|ton|inch|in)\b", re.I)


def formula_ingestion_metadata(text: str, numeric_meta: dict[str, Any] | None = None, table_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    formulas = extract_formulas(text)
    ratios = extract_ratios(text)
    unit_expressions = extract_unit_expressions(text)
    records = formula_records(formulas, ratios, unit_expressions, numeric_meta or {}, table_meta or {})
    retrieval_terms = formula_retrieval_terms(records)
    return {
        "formula_schema_version": FORMULA_SCHEMA_VERSION,
        "formula_ingestion_ready": True,
        "formula_records": records,
        "formula_count": len(records),
        "formula_equations": [record["expression"] for record in records if record["kind"] == "equation"],
        "formula_ratios": [record["expression"] for record in records if record["kind"] == "ratio"],
        "formula_unit_expressions": [record["expression"] for record in records if record["kind"] == "unit_expression"],
        "formula_variables": sorted({var for record in records for var in record.get("variables", [])})[:50],
        "formula_operators": sorted({op for record in records for op in record.get("operators", [])})[:20],
        "formula_units": sorted({unit for record in records for unit in record.get("units", [])})[:30],
        "formula_retrieval_text": " ".join(retrieval_terms),
        "formula_calculation_ready": any(record["kind"] == "equation" for record in records) or bool((numeric_meta or {}).get("numeric_constraint_ingestion_ready")),
        "formula_preserve_exact_text": bool(records),
        "formula_hash": hashlib.sha1("|".join(retrieval_terms).encode("utf-8")).hexdigest()[:16] if retrieval_terms else "",
    }


def extract_formulas(text: str) -> list[str]:
    formulas = []
    for match in EQUATION_RE.finditer(text or ""):
        expression = clean_expression(match.group("formula"))
        if valid_formula(expression):
            formulas.append(expression)
    return unique_preserve(formulas)[:40]


def extract_ratios(text: str) -> list[str]:
    return unique_preserve(clean_expression(match.group("ratio")) for match in RATIO_RE.finditer(text or ""))[:40]


def extract_unit_expressions(text: str) -> list[str]:
    return unique_preserve(clean_expression(match.group(0)) for match in UNIT_EXPR_RE.finditer(text or ""))[:60]


def formula_records(
    formulas: list[str],
    ratios: list[str],
    unit_expressions: list[str],
    numeric_meta: dict[str, Any],
    table_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for expression in formulas:
        records.append(
            {
                "kind": "equation",
                "expression": expression,
                "variables": formula_variables(expression),
                "operators": OPERATOR_RE.findall(expression),
                "units": expression_units(expression),
                "source": "equation_pattern",
                "confidence": 0.86,
            }
        )
    for ratio in ratios:
        records.append(
            {
                "kind": "ratio",
                "expression": ratio,
                "variables": [],
                "operators": [":" if ":" in ratio else "/"],
                "units": [],
                "source": "ratio_pattern",
                "confidence": 0.78,
            }
        )
    for expression in unit_expressions:
        records.append(
            {
                "kind": "unit_expression",
                "expression": expression,
                "variables": [],
                "operators": [],
                "units": expression_units(expression),
                "source": "unit_pattern",
                "confidence": 0.68,
            }
        )
    for item in numeric_meta.get("normalized_numeric_constraints") or []:
        if isinstance(item, dict):
            expression = " ".join(str(item.get(key, "")) for key in ["value", "normalized_unit", "context"] if item.get(key))
            if expression:
                records.append(
                    {
                        "kind": "numeric_constraint",
                        "expression": clean_expression(expression),
                        "variables": [],
                        "operators": [],
                        "units": [str(item.get("normalized_unit"))] if item.get("normalized_unit") else [],
                        "source": "numeric_constraint_metadata",
                        "confidence": 0.72,
                    }
                )
    if table_meta.get("contains_table") or table_meta.get("table_rows"):
        records.append(
            {
                "kind": "table_formula_context",
                "expression": clean_expression(" ".join([table_meta.get("table_title", ""), " ".join(table_meta.get("table_columns") or [])])),
                "variables": [],
                "operators": [],
                "units": [],
                "source": "table_metadata",
                "confidence": 0.52,
            }
        )
    return dedupe_records(records)[:120]


def formula_variables(expression: str) -> list[str]:
    variables = []
    for token in VARIABLE_RE.findall(expression):
        lower = token.lower()
        if lower not in {"shall", "must", "should", "and", "or", "the", "for", "with", "from"} and not token.isdigit():
            variables.append(token)
    return unique_preserve(variables)[:20]


def expression_units(expression: str) -> list[str]:
    return unique_preserve(re.findall(r"\b(mm|cm|m|bar|barg|psi|kpa|mpa|deg|degree|degrees|%|c|hz|kg|ton|inch|in)\b", expression or "", flags=re.I))[:20]


def formula_retrieval_terms(records: list[dict[str, Any]]) -> list[str]:
    terms = []
    for record in records:
        terms.extend([record.get("kind", ""), record.get("expression", "")])
        terms.extend(record.get("variables") or [])
        terms.extend(record.get("operators") or [])
        terms.extend(record.get("units") or [])
    return unique_preserve(str(term) for term in terms if term)


def valid_formula(expression: str) -> bool:
    if len(expression) < 4:
        return False
    return bool(OPERATOR_RE.search(expression)) and not expression.lower().startswith(("http", "table"))


def clean_expression(expression: str) -> str:
    return re.sub(r"\s+", " ", str(expression or "").strip(" .;:,"))


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for record in records:
        key = (record.get("kind"), str(record.get("expression", "")).lower())
        if key not in seen and record.get("expression"):
            seen.add(key)
            deduped.append(record)
    return deduped


def unique_preserve(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
