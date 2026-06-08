from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_RETRIEVER_SCHEMA_VERSION = "engineering-api-retriever-v1"
DEFAULT_TIMEOUT_SECONDS = 4.0
API_QUERY_STOPWORDS = {
    "about",
    "and",
    "are",
    "does",
    "from",
    "give",
    "how",
    "live",
    "show",
    "the",
    "this",
    "what",
    "where",
    "which",
    "with",
}


def api_retrieval_plan(query: str, route: Any, profile: Any) -> dict[str, Any]:
    endpoints = configured_endpoints()
    route_retrievers = set(getattr(route, "retrievers", ()) or ())
    live_intent = is_api_query(query)
    active = "api" in route_retrievers or live_intent
    return {
        "schema": API_RETRIEVER_SCHEMA_VERSION,
        "active": active,
        "configured": bool(endpoints),
        "route_primary": getattr(route, "primary", ""),
        "profile_type": getattr(profile, "type_id", ""),
        "query_terms": api_terms(query),
        "endpoint_count": len(endpoints),
        "endpoints": [safe_endpoint_summary(endpoint) for endpoint in endpoints],
        "live_intent": live_intent,
        "strategy": "safe_allowlisted_http_api_retrieval_normalized_into_evidence_rows",
    }


def run_api_retrieval(plan: dict[str, Any], query: str, query_vector: list[float], limit: int = 20) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    if not plan.get("active"):
        return [], {}, {**plan, "called": False, "reason": "api_not_requested", "results": []}
    endpoints = configured_endpoints()
    if not endpoints:
        return [], {}, {**plan, "called": False, "reason": "no_configured_api_endpoints", "results": []}

    rows: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    call_results = []
    for endpoint in endpoints[:6]:
        started = time.perf_counter()
        try:
            payload = call_endpoint(endpoint, query, plan, limit=limit)
            records = normalize_api_payload(payload, endpoint, limit=limit)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            call_results.append({"endpoint": endpoint.get("name"), "ok": True, "record_count": len(records), "elapsed_ms": elapsed_ms})
            for record in records:
                row = api_record_to_row(record, query_vector)
                rows.append(row)
                scores[str(row["id"])] = api_record_score(record, plan)
                if len(rows) >= limit:
                    break
        except Exception as exc:  # bounded, reported in detail; retrieval continues for other endpoints
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            call_results.append({"endpoint": endpoint.get("name"), "ok": False, "error": str(exc)[:240], "elapsed_ms": elapsed_ms})
        if len(rows) >= limit:
            break
    return rows, scores, {
        **plan,
        "called": True,
        "results": call_results,
        "matched_row_count": len(rows),
        "chunk_scores": scores,
    }


def api_candidate_score(plan: dict[str, Any], detail: dict[str, Any], row_id: Any, metadata: dict[str, Any], text: str) -> tuple[float, dict[str, Any]]:
    if not plan.get("active"):
        return 0.0, {"active": False, "matched": []}
    if not plan.get("configured") and metadata.get("source_type") != "api":
        return 0.0, {
            "schema": API_RETRIEVER_SCHEMA_VERSION,
            "active": True,
            "configured": False,
            "called": False,
            "matched": [],
            "term_hits": 0,
            "endpoint": "",
            "reason": "no_configured_api_endpoints",
        }
    chunk_scores = detail.get("chunk_scores") or {}
    score = float(chunk_scores.get(str(row_id)) or 0.0)
    haystack = api_haystack(metadata, text)
    matched = []
    term_hits = sum(1 for term in plan.get("query_terms") or [] if term in haystack)
    if term_hits:
        score += min(0.055, term_hits * 0.008)
        matched.append("query_terms")
    if metadata.get("source_type") == "api":
        score += 0.035
        matched.append("api_source")
    if metadata.get("api_endpoint"):
        score += 0.015
        matched.append("endpoint")
    return round(min(0.18, score), 5), {
        "schema": API_RETRIEVER_SCHEMA_VERSION,
        "active": True,
        "configured": bool(plan.get("configured")),
        "called": bool(detail.get("called")),
        "matched": matched,
        "term_hits": term_hits,
        "endpoint": metadata.get("api_endpoint") or "",
        "reason": detail.get("reason") or "",
    }


def configured_endpoints() -> list[dict[str, Any]]:
    raw = os.environ.get("RAG_API_RETRIEVER_ENDPOINTS", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    endpoints = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or not endpoint_allowed(url):
            continue
        endpoints.append(
            {
                "name": str(item.get("name") or urllib.parse.urlparse(url).netloc or "api"),
                "url": url,
                "method": str(item.get("method") or "GET").upper(),
                "headers": {str(k): str(v) for k, v in (item.get("headers") or {}).items()} if isinstance(item.get("headers"), dict) else {},
                "text_path": item.get("text_path") or "",
                "results_path": item.get("results_path") or "",
            }
        )
    return endpoints[:12]


def endpoint_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    allowed = [host.strip().lower() for host in os.environ.get("RAG_API_ALLOWED_HOSTS", "").split(",") if host.strip()]
    if not allowed:
        allowed = ["localhost", "127.0.0.1", "::1"]
    host = (parsed.hostname or "").lower()
    return host in allowed


def call_endpoint(endpoint: dict[str, Any], query: str, plan: dict[str, Any], limit: int) -> Any:
    method = str(endpoint.get("method") or "GET").upper()
    url = str(endpoint.get("url") or "")
    headers = {"Accept": "application/json", **(endpoint.get("headers") or {})}
    body = None
    if method == "POST":
        body = json.dumps({"query": query, "terms": plan.get("query_terms") or [], "limit": limit}).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        separator = "&" if urllib.parse.urlparse(url).query else "?"
        url = f"{url}{separator}{urllib.parse.urlencode({'q': query, 'limit': limit})}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    timeout = safe_float(os.environ.get("RAG_API_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS)
    with urllib.request.urlopen(request, timeout=max(0.5, min(timeout, 15.0))) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(512_000)
    if "json" in content_type.lower():
        return json.loads(raw.decode("utf-8", errors="replace"))
    return {"results": [{"text": raw.decode("utf-8", errors="replace")[:12000]}]}


def normalize_api_payload(payload: Any, endpoint: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    records = extract_records(payload, endpoint.get("results_path") or "")
    normalized = []
    for index, record in enumerate(records[:limit]):
        if isinstance(record, str):
            item = {"text": record}
        elif isinstance(record, dict):
            item = dict(record)
        else:
            item = {"text": json.dumps(record, ensure_ascii=False)[:12000]}
        text = extract_text(item, endpoint.get("text_path") or "")
        if not text:
            continue
        normalized.append(
            {
                "text": text[:12000],
                "title": str(item.get("title") or item.get("name") or item.get("id") or f"API result {index + 1}"),
                "url": str(item.get("url") or item.get("source") or endpoint.get("url") or ""),
                "endpoint": endpoint.get("name") or "",
                "raw": item,
                "rank": index + 1,
            }
        )
    return normalized


def extract_records(payload: Any, path: str) -> list[Any]:
    if path:
        payload = follow_path(payload, path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "items", "data", "records", "documents", "chunks"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return [payload]


def extract_text(item: dict[str, Any], path: str) -> str:
    if path:
        value = follow_path(item, path)
        if value:
            return stringify(value)
    for key in ("text", "content", "summary", "answer", "description", "body", "snippet"):
        if item.get(key):
            return stringify(item.get(key))
    return stringify(item)


def follow_path(payload: Any, path: str) -> Any:
    value = payload
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        else:
            return None
    return value


def api_record_to_row(record: dict[str, Any], query_vector: list[float]) -> dict[str, Any]:
    text = str(record.get("text") or "")
    endpoint = str(record.get("endpoint") or "api")
    signature = hashlib.sha1(f"{endpoint}\n{record.get('url')}\n{text[:500]}".encode("utf-8")).hexdigest()[:12]
    row_id = -int(signature[:10], 16)
    metadata = {
        "source_type": "api",
        "api_endpoint": endpoint,
        "api_url": record.get("url") or "",
        "api_title": record.get("title") or "",
        "api_rank": record.get("rank") or 0,
        "filename": f"API:{endpoint}",
        "section_title": record.get("title") or "API result",
        "keywords": api_terms(text)[:20],
        "retrieval_tags": ["api", "live", "external"],
    }
    return {
        "id": row_id,
        "document_id": 0,
        "filename": f"API:{endpoint}",
        "chunk_index": int(record.get("rank") or 0),
        "text": text,
        "embedding": json.dumps(query_vector),
        "metadata": json.dumps(metadata),
        "vector_score": 0.36,
        "api_row": True,
    }


def api_record_score(record: dict[str, Any], plan: dict[str, Any]) -> float:
    haystack = normalize_text(" ".join([record.get("text") or "", record.get("title") or "", record.get("endpoint") or ""]))
    hits = sum(1 for term in plan.get("query_terms") or [] if term in haystack)
    rank_bonus = max(0.0, 0.035 - (float(record.get("rank") or 1) - 1) * 0.004)
    return round(min(0.14, 0.045 + hits * 0.01 + rank_bonus), 5)


def safe_endpoint_summary(endpoint: dict[str, Any]) -> dict[str, str]:
    parsed = urllib.parse.urlparse(str(endpoint.get("url") or ""))
    return {"name": str(endpoint.get("name") or ""), "host": parsed.hostname or "", "method": str(endpoint.get("method") or "GET")}


def is_api_query(query: str) -> bool:
    return bool(re.search(r"\b(api|live|external system|monitoring|current sensor|real[- ]?time|from system)\b", str(query or ""), flags=re.I))


def api_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/&+-]{2,}", str(text or "")):
        normalized = normalize_text(token)
        if normalized and normalized not in API_QUERY_STOPWORDS:
            terms.append(normalized)
    return unique_preserve(terms)[:30]


def api_haystack(metadata: dict[str, Any], text: str) -> str:
    return normalize_text(" ".join([text, metadata.get("api_endpoint") or "", metadata.get("api_title") or "", metadata.get("filename") or "", flatten_values(metadata.get("keywords") or [])]))


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_.:/&+-]+", " ", str(value or "").lower())).strip()


def flatten_values(values: Any) -> str:
    if isinstance(values, dict):
        return " ".join(flatten_values(value) for value in values.values())
    if isinstance(values, (list, tuple, set)):
        return " ".join(flatten_values(value) for value in values)
    return str(values or "")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
