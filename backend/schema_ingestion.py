from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA_INGESTION_SCHEMA_VERSION = "engineering-schema-ingestion-v1"

FENCED_SCHEMA_RE = re.compile(r"```(?P<language>json|yaml|yml|sql|openapi|schema)?\n(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
JSON_SCHEMA_SIGNAL_RE = re.compile(r'"(?:\$schema|type|properties|required|items|definitions|\$defs|components|paths)"\s*:', re.IGNORECASE)
OPENAPI_SIGNAL_RE = re.compile(r"\b(?:openapi|swagger|paths|components|schemas)\b\s*[:{]", re.IGNORECASE)
CREATE_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<table>[A-Za-z0-9_.\"`\[\]]+)\s*\((?P<body>.*?)\)\s*;?", re.IGNORECASE | re.DOTALL)
FIELD_LINE_RE = re.compile(
    r"^\s*[\"']?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)[\"']?\s*[:|]\s*(?P<type>[A-Za-z0-9_./\[\]{}<> -]+)",
    re.MULTILINE,
)
SQL_COLUMN_RE = re.compile(
    r"^\s*[\"`\[]?(?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"`\]]?\s+"
    r"(?P<type>[A-Za-z][A-Za-z0-9_() ,]*)(?P<constraints>\s+.*)?$",
    re.IGNORECASE,
)
REQUIRED_TERMS = {"not null", "primary key", "required", "mandatory"}


def schema_ingestion_metadata(
    text: str,
    table_meta: dict[str, Any] | None = None,
    code_meta: dict[str, Any] | None = None,
    doc_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = str(text or "")
    table_meta = table_meta or {}
    code_meta = code_meta or {}
    doc_meta = doc_meta or {}
    candidate = extract_schema_candidate(raw)
    schema_type = detect_schema_type(candidate, table_meta, code_meta, doc_meta)
    parsed_json = parse_json_object(candidate) if schema_type in {"json_schema", "openapi", "json"} else None
    json_payload = extract_json_schema_payload(parsed_json) if parsed_json is not None else empty_json_schema_payload()
    sql_payload = extract_sql_schema_payload(candidate)
    table_payload = extract_table_schema_payload(table_meta)
    fallback_payload = extract_fallback_schema_payload(candidate)

    fields = merge_records(
        json_payload["fields"],
        sql_payload["fields"],
        table_payload["fields"],
        fallback_payload["fields"],
    )
    required_fields = unique(
        json_payload["required_fields"]
        + sql_payload["required_fields"]
        + table_payload["required_fields"]
        + fallback_payload["required_fields"]
    )
    schema_names = unique(
        json_payload["schema_names"]
        + sql_payload["schema_names"]
        + table_payload["schema_names"]
        + fallback_payload["schema_names"]
    )
    types = unique([record.get("type", "") for record in fields] + json_payload["types"] + sql_payload["types"])
    relations = unique(json_payload["relations"] + sql_payload["relations"] + fallback_payload["relations"])
    constraints = unique(json_payload["constraints"] + sql_payload["constraints"] + fallback_payload["constraints"])
    detected = bool(fields or schema_names or constraints or schema_type not in {"unknown", "none"})
    retrieval_text = build_schema_retrieval_text(
        schema_type=schema_type,
        schema_names=schema_names,
        fields=fields,
        required_fields=required_fields,
        types=types,
        relations=relations,
        constraints=constraints,
        text=candidate,
    )
    return {
        "schema_ingestion_schema_version": SCHEMA_INGESTION_SCHEMA_VERSION,
        "schema_ingestion_ready": True,
        "schema_detected": detected,
        "schema_type": schema_type,
        "schema_parse_status": json_payload["parse_status"] if parsed_json is not None else ("sql_parsed" if sql_payload["fields"] else ("table_parsed" if table_payload["fields"] else "heuristic")),
        "schema_names": schema_names,
        "schema_fields": fields,
        "schema_field_names": [record["name"] for record in fields],
        "schema_required_fields": required_fields,
        "schema_optional_fields": [record["name"] for record in fields if record["name"] not in required_fields],
        "schema_types": types,
        "schema_constraints": constraints,
        "schema_relations": relations,
        "schema_endpoint_paths": json_payload["endpoint_paths"],
        "schema_component_names": json_payload["component_names"],
        "schema_table_names": sql_payload["schema_names"],
        "schema_field_count": len(fields),
        "schema_required_count": len(required_fields),
        "schema_retrieval_text": retrieval_text,
        "schema_hash": hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:16] if candidate.strip() else "",
    }


def extract_schema_candidate(text: str) -> str:
    blocks = [match.group("body").strip() for match in FENCED_SCHEMA_RE.finditer(text) if match.group("body").strip()]
    return "\n\n".join(blocks) if blocks else text


def detect_schema_type(text: str, table_meta: dict[str, Any], code_meta: dict[str, Any], doc_meta: dict[str, Any]) -> str:
    filename = str(doc_meta.get("filename") or doc_meta.get("source_filename") or "").lower()
    if filename.endswith((".schema.json", "schema.json")):
        return "json_schema"
    if filename.endswith((".openapi.json", ".openapi.yaml", ".swagger.json", ".swagger.yaml")):
        return "openapi"
    if filename.endswith(".sql") or CREATE_TABLE_RE.search(text):
        return "sql_schema"
    if code_meta.get("code_language") in {"json_schema", "json"} and JSON_SCHEMA_SIGNAL_RE.search(text):
        return "json_schema"
    if OPENAPI_SIGNAL_RE.search(text) and re.search(r"\b(get|post|put|delete|patch)\b\s*[:{]", text, re.IGNORECASE):
        return "openapi"
    if JSON_SCHEMA_SIGNAL_RE.search(text):
        return "json_schema"
    if table_meta.get("table_columns"):
        return "table_schema"
    if FIELD_LINE_RE.search(text):
        return "field_list"
    return "unknown"


def parse_json_object(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def extract_json_schema_payload(obj: Any) -> dict[str, Any]:
    fields: list[dict[str, str]] = []
    required_fields: list[str] = []
    schema_names: list[str] = []
    component_names: list[str] = []
    endpoint_paths: list[str] = []
    relations: list[str] = []
    constraints: list[str] = []
    types: list[str] = []

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            title = str(value.get("title") or value.get("$id") or "").strip()
            if title:
                schema_names.append(title)
            value_type = value.get("type")
            if isinstance(value_type, str):
                types.append(value_type)
            for required in value.get("required") or []:
                if isinstance(required, str):
                    required_fields.append(required)
            properties = value.get("properties")
            if isinstance(properties, dict):
                for name, spec in properties.items():
                    field_type = schema_type_name(spec)
                    fields.append({"name": str(name), "type": field_type, "source": path or "properties", "required": str(name in (value.get("required") or [])).lower()})
                    walk(spec, f"{path}.{name}".strip("."))
            for key in ("$ref", "ref"):
                if key in value:
                    relations.append(str(value[key]))
            components = value.get("components", {}).get("schemas", {}) if isinstance(value.get("components"), dict) else {}
            if isinstance(components, dict):
                component_names.extend(str(name) for name in components)
            paths = value.get("paths")
            if isinstance(paths, dict):
                endpoint_paths.extend(str(name) for name in paths)
            for key, child in value.items():
                if key in {"properties", "required"}:
                    continue
                if key in {"minimum", "maximum", "minLength", "maxLength", "pattern", "enum", "format"}:
                    constraints.append(f"{key}={child}")
                walk(child, f"{path}.{key}".strip("."))
        elif isinstance(value, list):
            for child in value:
                walk(child, path)

    walk(obj)
    return {
        "parse_status": "json_parsed",
        "fields": normalize_field_records(fields),
        "required_fields": unique(required_fields),
        "schema_names": unique(schema_names + component_names),
        "component_names": unique(component_names),
        "endpoint_paths": unique(endpoint_paths),
        "relations": unique(relations),
        "constraints": unique(constraints),
        "types": unique(types),
    }


def empty_json_schema_payload() -> dict[str, Any]:
    return {
        "parse_status": "not_json",
        "fields": [],
        "required_fields": [],
        "schema_names": [],
        "component_names": [],
        "endpoint_paths": [],
        "relations": [],
        "constraints": [],
        "types": [],
    }


def schema_type_name(spec: Any) -> str:
    if isinstance(spec, dict):
        if "$ref" in spec:
            return str(spec["$ref"]).rsplit("/", 1)[-1]
        value = spec.get("type") or spec.get("format") or spec.get("enum")
        if isinstance(value, list):
            return "|".join(str(item) for item in value)
        if value:
            return str(value)
        if "properties" in spec:
            return "object"
        if "items" in spec:
            return "array"
    return "unknown"


def extract_sql_schema_payload(text: str) -> dict[str, Any]:
    fields: list[dict[str, str]] = []
    required_fields: list[str] = []
    schema_names: list[str] = []
    constraints: list[str] = []
    relations: list[str] = []
    types: list[str] = []
    for match in CREATE_TABLE_RE.finditer(text):
        table = clean_identifier(match.group("table"))
        schema_names.append(table)
        for line in split_sql_columns(match.group("body")):
            normalized = line.strip().rstrip(",")
            lowered = normalized.lower()
            if not normalized or lowered.startswith(("primary key", "foreign key", "constraint", "unique", "check")):
                constraints.append(normalized)
                if "references" in lowered:
                    relations.append(normalized)
                continue
            column = SQL_COLUMN_RE.match(normalized)
            if not column:
                continue
            name = clean_identifier(column.group("name"))
            column_type = re.sub(r"\s+", " ", column.group("type")).strip()
            constraint = re.sub(r"\s+", " ", column.group("constraints") or "").strip()
            fields.append({"name": name, "type": column_type, "source": table, "required": str(any(term in lowered for term in REQUIRED_TERMS)).lower()})
            types.append(column_type)
            if any(term in lowered for term in REQUIRED_TERMS):
                required_fields.append(name)
            if constraint:
                constraints.append(f"{name} {constraint}")
            if "references" in lowered:
                relations.append(normalized)
    return {
        "fields": normalize_field_records(fields),
        "required_fields": unique(required_fields),
        "schema_names": unique(schema_names),
        "types": unique(types),
        "constraints": unique(constraints),
        "relations": unique(relations),
    }


def split_sql_columns(body: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def extract_table_schema_payload(table_meta: dict[str, Any]) -> dict[str, Any]:
    columns = [str(item) for item in table_meta.get("table_columns") or [] if str(item).strip()]
    title = str(table_meta.get("table_title") or "").strip()
    fields = [{"name": column, "type": "table_column", "source": title or "table", "required": "false"} for column in columns]
    return {
        "fields": normalize_field_records(fields),
        "required_fields": [],
        "schema_names": [title] if title else [],
    }


def extract_fallback_schema_payload(text: str) -> dict[str, Any]:
    fields: list[dict[str, str]] = []
    required_fields: list[str] = []
    for match in FIELD_LINE_RE.finditer(text):
        name = match.group("name")
        field_type = re.sub(r"\s+", " ", match.group("type")).strip(" ,-")
        line = match.group(0).lower()
        fields.append({"name": name, "type": field_type or "unknown", "source": "field_list", "required": str(any(term in line for term in REQUIRED_TERMS)).lower()})
        if any(term in line for term in REQUIRED_TERMS):
            required_fields.append(name)
    return {
        "fields": normalize_field_records(fields),
        "required_fields": unique(required_fields),
        "schema_names": [],
        "relations": [],
        "constraints": [],
    }


def normalize_field_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        name = re.sub(r"\s+", " ", str(record.get("name") or "")).strip()
        field_type = re.sub(r"\s+", " ", str(record.get("type") or "unknown")).strip()
        source = re.sub(r"\s+", " ", str(record.get("source") or "")).strip()
        required = str(record.get("required") or "false").lower()
        key = (name.lower(), field_type.lower(), source.lower())
        if not name or key in seen:
            continue
        seen.add(key)
        normalized.append({"name": name, "type": field_type, "source": source, "required": "true" if required == "true" else "false"})
    return normalized[:160]


def merge_records(*record_groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for group in record_groups:
        merged.extend(group)
    return normalize_field_records(merged)


def build_schema_retrieval_text(
    *,
    schema_type: str,
    schema_names: list[str],
    fields: list[dict[str, str]],
    required_fields: list[str],
    types: list[str],
    relations: list[str],
    constraints: list[str],
    text: str,
) -> str:
    field_text = " ".join(f"{record.get('name')} {record.get('type')} {record.get('source')}" for record in fields)
    parts = [
        f"schema type {schema_type}",
        "schemas " + " ".join(schema_names) if schema_names else "",
        "fields " + field_text if field_text else "",
        "required " + " ".join(required_fields) if required_fields else "",
        "types " + " ".join(types) if types else "",
        "relations " + " ".join(relations) if relations else "",
        "constraints " + " ".join(constraints) if constraints else "",
        text[:2200],
    ]
    return "\n".join(part for part in parts if part)


def clean_identifier(value: str) -> str:
    return str(value or "").strip().strip('"`[]')


def unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = re.sub(r"\s+", " ", str(value or "")).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
