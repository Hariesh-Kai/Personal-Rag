from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, HttpRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from backend.llm import LocalLLM
from backend.ocr import ocr_status
from backend.question_types import classify_question, profile_payload
from backend.rag_store import RagStore
from backend.streaming_ingestion import stream_ingestion_events
from backend.structure_chunker import process_document_detailed


settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

store = RagStore(settings.DATA_DIR / "rag.sqlite3", settings.DATA_DIR / "chunks.jsonl")
llm = LocalLLM(settings.MODELS_DIR)
jobs: dict[str, dict] = {}


@require_GET
def health(_: HttpRequest) -> JsonResponse:
    index_status = store.index_status()
    return JsonResponse(
        {
            "ok": True,
            "llm": llm.status,
            "model_path": str(llm.model_path) if llm.model_path else None,
            **ocr_status(),
            **index_status,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def upload_document(request: HttpRequest) -> JsonResponse:
    upload = request.FILES.get("file")
    if upload is None:
        return JsonResponse({"detail": "File is required"}, status=400)

    job_id = str(uuid.uuid4())
    safe_name = Path(upload.name).name
    new_session = str(request.POST.get("new_session", "true")).lower() not in {"0", "false", "no"}
    target = settings.UPLOAD_DIR / f"{job_id}-{safe_name}"
    with target.open("wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)

    jobs[job_id] = {
        "id": job_id,
        "filename": safe_name,
        "progress": 5,
        "stage": "saved upload",
        "status": "running",
        "new_session": new_session,
    }

    worker = threading.Thread(
        target=process_upload,
        args=(job_id, target, safe_name, upload.content_type, new_session),
        daemon=True,
    )
    worker.start()
    return JsonResponse({"job_id": job_id})


@csrf_exempt
@require_http_methods(["POST"])
def upload_document_stream(request: HttpRequest) -> StreamingHttpResponse:
    upload = request.FILES.get("file")
    if upload is None:
        return StreamingHttpResponse(iter([json.dumps({"error": "File is required"}) + "\n"]), status=400, content_type="application/x-ndjson")

    safe_name = Path(upload.name).name
    new_session = str(request.POST.get("new_session", "true")).lower() not in {"0", "false", "no"}
    target = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}-{safe_name}"
    with target.open("wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    if new_session:
        store.reset()

    def store_result(result: object) -> dict:
        chunks = getattr(result, "chunks", []) or []
        if not chunks:
            raise RuntimeError("Extraction worked, but no structure-aware chunks were produced.")
        document_id = store.add_document(safe_name, upload.content_type, chunks)
        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "stored_chunk_count": store.last_add_document_stats.get("stored_chunks", len(chunks)),
            "index_session": store.current_index_session(),
            "storage": store.last_add_document_stats,
        }

    def event_lines() -> object:
        for event in stream_ingestion_events(target, process_document_detailed, result_callback=store_result):
            yield json.dumps(event, default=str) + "\n"

    return StreamingHttpResponse(event_lines(), content_type="application/x-ndjson")


@require_GET
def progress(_: HttpRequest, job_id: str) -> JsonResponse:
    job = jobs.get(job_id)
    if job is None:
        return JsonResponse({"detail": "Unknown upload job"}, status=404)
    return JsonResponse(job)


@csrf_exempt
@require_http_methods(["POST"])
def chat(request: HttpRequest) -> JsonResponse:
    started_at = time.perf_counter()
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"detail": "Question is required"}, status=400)
    session = store.ensure_chat_session(str(payload.get("session_id") or "").strip() or None, question)
    session_id = session["id"]
    store.add_chat_message(session_id, "user", question, {"index_session": store.current_index_session()})

    profile = classify_question(question)
    resolved_question = resolve_follow_up_question(question, profile.type_id)
    profile = classify_question(resolved_question)
    contexts = store.search(resolved_question, limit=8, profile=profile)
    retrieved_at = time.perf_counter()
    result = llm.answer(resolved_question, contexts, profile=profile, metadata={**store.index_status(), **ocr_status()})
    result["retrieval"] = retrieval_transparency(
        question=question,
        resolved_question=resolved_question,
        profile=profile,
        contexts=contexts,
        result=result,
        started_at=started_at,
        retrieved_at=retrieved_at,
    )
    store.log_retrieval(question, result, contexts)
    store.add_chat_turn(question, resolved_question, result, contexts)
    store.add_chat_message(
        session_id,
        "assistant",
        result.get("answer", ""),
        {
            "sources": result.get("sources") or [],
            "quality": result.get("quality") or {},
            "confidence": result.get("confidence") or {},
            "retriever_route": result.get("retriever_route") or {},
            "retrieval": result.get("retrieval") or {},
            "mode": result.get("mode") or "",
            "question_profile": result.get("question_profile") or {},
        },
    )
    result["session_id"] = session_id
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def chat_sessions(request: HttpRequest) -> JsonResponse:
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)
        return JsonResponse(store.create_chat_session(str(payload.get("title") or "")))
    limit = int(request.GET.get("limit", "40"))
    return JsonResponse(store.list_chat_sessions(limit=limit), safe=False)


@require_GET
def chat_session_detail(_: HttpRequest, session_id: str) -> JsonResponse:
    session = store.get_chat_session(session_id)
    if session is None:
        return JsonResponse({"detail": "Chat session not found"}, status=404)
    return JsonResponse(session)


@require_GET
def documents(_: HttpRequest) -> JsonResponse:
    return JsonResponse(store.list_documents(), safe=False)


@require_GET
def chunks(request: HttpRequest) -> JsonResponse:
    document_id = request.GET.get("document_id")
    parsed_id = int(document_id) if document_id else None
    return JsonResponse(store.list_chunks(document_id=parsed_id), safe=False)


@require_GET
def chunks_file(_: HttpRequest) -> FileResponse:
    store.export_chunks_log()
    return FileResponse(
        store.chunks_log_path.open("rb"),
        as_attachment=True,
        filename="chunks.jsonl",
        content_type="application/jsonl",
    )


@require_GET
def retrieval_logs(request: HttpRequest) -> JsonResponse:
    limit = int(request.GET.get("limit", "25"))
    return JsonResponse(store.retrieval_logs(limit=max(1, min(100, limit))), safe=False)


def process_upload(job_id: str, path: Path, filename: str, content_type: str | None, new_session: bool = True) -> None:
    try:
        if new_session:
            index_session = store.reset()
            jobs[job_id].update(index_session=index_session, stage="created new index session")

        def update_pipeline_progress(progress: int, stage: str, detail: str = "") -> None:
            jobs[job_id].update(progress=progress, stage=stage, detail=detail)

        result = process_document_detailed(path, progress_callback=update_pipeline_progress)
        chunks = result.chunks

        if not chunks:
            raise RuntimeError("Extraction worked, but no structure-aware chunks were produced.")

        jobs[job_id].update(progress=88, stage="embedding and storing prepared chunks", pipeline=result.metadata)
        document_id = store.add_document(filename, content_type, chunks)

        jobs[job_id].update(
            progress=100,
            stage="stored in database",
            status="complete",
            document_id=document_id,
            chunk_count=len(chunks),
            stored_chunk_count=store.last_add_document_stats.get("stored_chunks", len(chunks)),
            index_session=store.current_index_session(),
            pipeline=result.metadata,
            stages=result.stages,
            storage=store.last_add_document_stats,
        )
    except Exception as exc:
        jobs[job_id].update(progress=100, stage="failed", status="error", error=str(exc))


def resolve_follow_up_question(question: str, type_id: str) -> str:
    if type_id != "follow_up" and not looks_like_follow_up(question):
        return question
    recent = store.recent_chat_turns(limit=2)
    if not recent:
        return question
    previous = recent[0]
    return (
        f"Previous question: {previous.get('resolved_question') or previous.get('question')}\n"
        f"Previous answer: {previous.get('answer', '')[:700]}\n"
        f"Follow-up question: {question}"
    )


def looks_like_follow_up(question: str) -> bool:
    normalized = question.lower().strip()
    return bool(
        normalized.startswith(("what about", "and ", "also ", "then ", "it ", "that ", "this "))
        or normalized in {"why", "how", "explain", "more", "details"}
        or any(phrase in normalized for phrase in ("same section", "previous", "above", "that one", "those"))
    )


def retrieval_transparency(
    question: str,
    resolved_question: str,
    profile: object,
    contexts: list[dict],
    result: dict,
    started_at: float,
    retrieved_at: float,
) -> dict:
    finished_at = time.perf_counter()
    quality = result.get("quality") or {}
    confidence = result.get("confidence") or {}
    route = result.get("retriever_route") or (contexts[0].get("retriever_route") if contexts else None) or {}
    scores = [safe_float(item.get("score")) for item in contexts]
    top_score = max(scores) if scores else 0.0
    average_score = sum(scores) / len(scores) if scores else 0.0
    source_sections = source_section_summary(contexts)
    source_signals = source_signal_summary(contexts)
    quality_score = safe_float(quality.get("overall_score"))
    status_key, status_label = evidence_status(top_score, len(contexts), quality_score, result.get("answer", ""))

    return {
        "evidence_status": status_key,
        "evidence_label": status_label,
        "question_type": result.get("question_profile") or profile_payload(profile),
        "resolved_follow_up": question != resolved_question,
        "retrieved_query": resolved_question,
        "source_count": len(contexts),
        "top_score": round(top_score, 4),
        "average_score": round(average_score, 4),
        "dominant_section": confidence.get("dominant_section") or (source_sections[0]["section"] if source_sections else ""),
        "source_sections": source_sections,
        "source_signals": source_signals,
        "route": route,
        "primary_route": route.get("primary") if isinstance(route, dict) else None,
        "active_retrievers": route.get("retrievers", []) if isinstance(route, dict) else [],
        "answer_quality": {
            "grade": quality.get("grade"),
            "overall_score": round(quality_score, 4),
        },
        "timing": {
            "retrieval_ms": round((retrieved_at - started_at) * 1000),
            "total_ms": round((finished_at - started_at) * 1000),
        },
    }


def evidence_status(top_score: float, source_count: int, quality_score: float, answer: str = "") -> tuple[str, str]:
    if is_not_found_answer(answer):
        return "not_enough", "Insufficient evidence"
    if source_count == 0:
        return "not_enough", "Not enough evidence"
    if quality_score >= 0.82 and top_score >= 0.45:
        return "strong", "Strong evidence"
    if quality_score >= 0.6 or top_score >= 0.25:
        return "usable", "Usable evidence"
    if quality_score >= 0.35 or top_score > 0:
        return "weak", "Weak evidence"
    return "not_enough", "Not enough evidence"


def is_not_found_answer(answer: str) -> bool:
    normalized = " ".join(str(answer or "").lower().split())
    return normalized in {
        "not found",
        "not found.",
        "not found in the retrieved document context.",
    }


def source_section_summary(contexts: list[dict]) -> list[dict]:
    sections: dict[str, dict] = {}
    for item in contexts:
        metadata = item.get("metadata") or {}
        section = metadata.get("section_title") or "No section"
        current = sections.setdefault(section, {"section": section, "count": 0, "best_score": 0.0})
        current["count"] += 1
        current["best_score"] = max(current["best_score"], safe_float(item.get("score")))
    ranked = sorted(sections.values(), key=lambda row: (row["count"], row["best_score"]), reverse=True)
    for row in ranked:
        row["best_score"] = round(row["best_score"], 4)
    return ranked[:5]


def source_signal_summary(contexts: list[dict]) -> dict:
    signals = {
        "tables": 0,
        "numeric_constraints": 0,
        "safety": 0,
        "figures": 0,
        "code": 0,
        "schemas": 0,
        "knowledge_graph": 0,
        "ontology": 0,
        "non_english": 0,
    }
    for item in contexts:
        metadata = item.get("metadata") or {}
        if metadata.get("contains_table"):
            signals["tables"] += 1
        if metadata.get("has_numeric_constraints") or metadata.get("numeric_constraints"):
            signals["numeric_constraints"] += 1
        if metadata.get("safety_critical") or metadata.get("safety_flags"):
            signals["safety"] += 1
        if metadata.get("images") or metadata.get("figure_regions") or metadata.get("layout_images"):
            signals["figures"] += 1
        if metadata.get("code_detected") or metadata.get("code_symbols"):
            signals["code"] += 1
        if metadata.get("schema_detected") or metadata.get("schema_tables"):
            signals["schemas"] += 1
        if metadata.get("knowledge_graph_ready") or metadata.get("graph_entities"):
            signals["knowledge_graph"] += 1
        if metadata.get("ontology_concepts"):
            signals["ontology"] += 1
        language = str(metadata.get("language_code") or metadata.get("detected_language") or "").lower()
        if language and language not in {"en", "eng", "english"}:
            signals["non_english"] += 1
    return signals


def safe_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
