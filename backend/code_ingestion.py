from __future__ import annotations

import ast
import hashlib
import re
from typing import Any


CODE_INGESTION_SCHEMA_VERSION = "engineering-code-ast-ingestion-v1"

FENCED_CODE_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+.-]*)\n(?P<body>.*?)```", re.DOTALL)
FUNCTION_RE = re.compile(r"\b(?:def|function|async\s+function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)
CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b")
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_., ]+))", re.MULTILINE)
SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|ALTER\s+TABLE|JOIN|WHERE|GROUP\s+BY)\b", re.IGNORECASE)
JSON_SCHEMA_RE = re.compile(r'"(?:type|properties|required|items|definitions|\$schema)"\s*:', re.IGNORECASE)
CODE_SIGNAL_RE = re.compile(
    r"(^\s*(?:def |class |import |from .+ import |@[\w.]+|if __name__|return |try:|except )|[{}();]|=>|::)",
    re.MULTILINE,
)


def code_ingestion_metadata(text: str, doc_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = str(text or "")
    doc_meta = doc_meta or {}
    snippets = extract_code_snippets(raw)
    candidate_text = "\n\n".join(snippet["body"] for snippet in snippets) if snippets else raw
    language = detect_code_language(candidate_text, snippets, doc_meta)
    code_like = is_code_like(candidate_text, language)

    ast_payload = parse_python_ast(candidate_text) if language == "python" and code_like else empty_ast_payload()
    fallback_symbols = fallback_code_symbols(candidate_text)
    functions = unique(ast_payload["functions"] + fallback_symbols["functions"])
    classes = unique(ast_payload["classes"] + fallback_symbols["classes"])
    imports = unique(ast_payload["imports"] + fallback_symbols["imports"])
    decorators = unique(ast_payload["decorators"])
    calls = unique(ast_payload["calls"])
    assignments = unique(ast_payload["assignments"])
    constants = unique(ast_payload["constants"])
    endpoints = extract_endpoints(candidate_text)
    sql_terms = unique(match.group(1).upper() for match in SQL_RE.finditer(candidate_text))
    schema_terms = unique(match.group(0).split(":", 1)[0].strip('"').strip() for match in JSON_SCHEMA_RE.finditer(candidate_text))
    identifiers = unique(
        functions
        + classes
        + imports
        + assignments
        + endpoints
        + re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", candidate_text)
    )[:120]

    parse_status = ast_payload["parse_status"] if language == "python" and code_like else ("not_python" if code_like else "not_code_like")
    code_retrieval_text = build_code_retrieval_text(
        language=language,
        functions=functions,
        classes=classes,
        imports=imports,
        decorators=decorators,
        calls=calls,
        assignments=assignments,
        endpoints=endpoints,
        sql_terms=sql_terms,
        schema_terms=schema_terms,
        snippets=snippets,
        text=candidate_text,
    )
    return {
        "code_ingestion_schema_version": CODE_INGESTION_SCHEMA_VERSION,
        "code_ingestion_ready": True,
        "code_detected": bool(code_like or snippets),
        "code_language": language,
        "code_parse_status": parse_status,
        "code_ast_available": bool(ast_payload["ast_available"]),
        "code_ast_error": ast_payload["ast_error"],
        "code_snippet_count": len(snippets),
        "code_line_count": count_code_lines(candidate_text) if code_like or snippets else 0,
        "code_functions": functions,
        "code_classes": classes,
        "code_imports": imports,
        "code_decorators": decorators,
        "code_calls": calls[:80],
        "code_assignments": assignments,
        "code_constants": constants[:80],
        "code_endpoints": endpoints,
        "code_sql_terms": sql_terms,
        "code_schema_terms": schema_terms,
        "code_identifiers": identifiers,
        "code_retrieval_text": code_retrieval_text,
        "code_symbol_count": len(unique(functions + classes + imports + assignments + endpoints)),
        "code_ast_node_count": ast_payload["node_count"],
        "code_ast_depth": ast_payload["max_depth"],
        "code_hash": hashlib.sha1(candidate_text.encode("utf-8")).hexdigest()[:16] if candidate_text.strip() else "",
    }


def extract_code_snippets(text: str) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    for match in FENCED_CODE_RE.finditer(text):
        body = match.group("body").strip()
        if not body:
            continue
        snippets.append({"language": match.group("language").lower(), "body": body})
    return snippets


def detect_code_language(text: str, snippets: list[dict[str, str]], doc_meta: dict[str, Any]) -> str:
    declared = next((snippet["language"] for snippet in snippets if snippet.get("language")), "")
    if declared:
        return normalize_language(declared)
    filename = str(doc_meta.get("filename") or doc_meta.get("source_filename") or "").lower()
    suffix_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    for suffix, language in suffix_map.items():
        if filename.endswith(suffix):
            return language
    stripped = text.strip()
    if re.search(r"^\s*(def|class|from\s+\w+|import\s+\w+|@[\w.]+)", stripped, re.MULTILINE):
        return "python"
    if SQL_RE.search(stripped):
        return "sql"
    if JSON_SCHEMA_RE.search(stripped):
        return "json_schema"
    if re.search(r"\b(function|const|let|var|=>|interface|type)\b", stripped):
        return "typescript" if re.search(r"\binterface\b|\btype\s+\w+\s*=", stripped) else "javascript"
    return "unknown"


def normalize_language(language: str) -> str:
    value = language.lower().strip()
    aliases = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "postgres": "sql",
        "pgsql": "sql",
        "jsonschema": "json_schema",
    }
    return aliases.get(value, value or "unknown")


def is_code_like(text: str, language: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if language not in {"unknown", ""}:
        return True
    if len(stripped.splitlines()) < 2 and not CODE_SIGNAL_RE.search(stripped):
        return False
    return bool(CODE_SIGNAL_RE.search(stripped))


def parse_python_ast(text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            **empty_ast_payload(),
            "parse_status": "syntax_error",
            "ast_error": f"{exc.__class__.__name__}: {exc.msg}",
        }
    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []
    decorators: list[str] = []
    calls: list[str] = []
    assignments: list[str] = []
    constants: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            decorators.extend(ast_name(item) for item in node.decorator_list if ast_name(item))
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            decorators.extend(ast_name(item) for item in node.decorator_list if ast_name(item))
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names)
        elif isinstance(node, ast.Call):
            name = ast_name(node.func)
            if name:
                calls.append(name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assignments.extend(assignment_names(node))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
            value = str(node.value)
            if 1 <= len(value) <= 80:
                constants.append(value)
    return {
        "ast_available": True,
        "parse_status": "parsed",
        "ast_error": "",
        "functions": unique(functions),
        "classes": unique(classes),
        "imports": unique(imports),
        "decorators": unique(decorators),
        "calls": unique(calls),
        "assignments": unique(assignments),
        "constants": unique(constants),
        "node_count": sum(1 for _ in ast.walk(tree)),
        "max_depth": ast_depth(tree),
    }


def empty_ast_payload() -> dict[str, Any]:
    return {
        "ast_available": False,
        "parse_status": "not_parsed",
        "ast_error": "",
        "functions": [],
        "classes": [],
        "imports": [],
        "decorators": [],
        "calls": [],
        "assignments": [],
        "constants": [],
        "node_count": 0,
        "max_depth": 0,
    }


def fallback_code_symbols(text: str) -> dict[str, list[str]]:
    imports: list[str] = []
    for match in IMPORT_RE.finditer(text):
        imports.extend(re.split(r"\s*,\s*", (match.group(1) or match.group(2) or "").strip()))
    return {
        "functions": unique(FUNCTION_RE.findall(text)),
        "classes": unique(CLASS_RE.findall(text)),
        "imports": unique(item for item in imports if item),
    }


def assignment_names(node: ast.AST) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        names.extend(target_names(target))
    return names


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in node.elts:
            names.extend(target_names(item))
        return names
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return ast_name(node.func)
    return ""


def ast_depth(node: ast.AST, depth: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(ast_depth(child, depth + 1) for child in children)


def extract_endpoints(text: str) -> list[str]:
    endpoint_patterns = [
        r"['\"](/api/[A-Za-z0-9_./{}:-]+)['\"]",
        r"['\"](/[A-Za-z0-9_./{}:-]{2,})['\"]",
        r"@\w+\.(?:get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]",
    ]
    values: list[str] = []
    for pattern in endpoint_patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return unique(values)[:50]


def count_code_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def build_code_retrieval_text(
    *,
    language: str,
    functions: list[str],
    classes: list[str],
    imports: list[str],
    decorators: list[str],
    calls: list[str],
    assignments: list[str],
    endpoints: list[str],
    sql_terms: list[str],
    schema_terms: list[str],
    snippets: list[dict[str, str]],
    text: str,
) -> str:
    parts = [
        f"code language {language}" if language else "",
        "functions " + " ".join(functions) if functions else "",
        "classes " + " ".join(classes) if classes else "",
        "imports " + " ".join(imports) if imports else "",
        "decorators " + " ".join(decorators) if decorators else "",
        "calls " + " ".join(calls[:40]) if calls else "",
        "assignments " + " ".join(assignments) if assignments else "",
        "endpoints " + " ".join(endpoints) if endpoints else "",
        "sql " + " ".join(sql_terms) if sql_terms else "",
        "schema " + " ".join(schema_terms) if schema_terms else "",
        "fenced code" if snippets else "",
        text[:2000],
    ]
    return "\n".join(part for part in parts if part)


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
