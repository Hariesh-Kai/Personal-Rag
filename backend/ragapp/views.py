from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from backend.llm import LocalLLM
from backend.ocr import ocr_status
from backend.question_types import classify_question
from backend.rag_store import RagStore
from backend.structure_chunker import process_document


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


@require_GET
def progress(_: HttpRequest, job_id: str) -> JsonResponse:
    job = jobs.get(job_id)
    if job is None:
        return JsonResponse({"detail": "Unknown upload job"}, status=404)
    return JsonResponse(job)


@csrf_exempt
@require_http_methods(["POST"])
def chat(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"detail": "Question is required"}, status=400)

    profile = classify_question(question)
    resolved_question = resolve_follow_up_question(question, profile.type_id)
    profile = classify_question(resolved_question)
    contexts = store.search(resolved_question, limit=8, profile=profile)
    result = llm.answer(resolved_question, contexts, profile=profile, metadata={**store.index_status(), **ocr_status()})
    store.log_retrieval(question, result, contexts)
    store.add_chat_turn(question, resolved_question, result, contexts)
    return JsonResponse(result)


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
        jobs[job_id].update(progress=18, stage="extracting text, blocks, tables, images")
        chunks = process_document(path)

        if not chunks:
            raise RuntimeError("Extraction worked, but no structure-aware chunks were produced.")

        jobs[job_id].update(progress=55, stage="cleaning, detecting hierarchy, preserving tables")

        jobs[job_id].update(progress=78, stage="embedding enriched chunks")
        document_id = store.add_document(filename, content_type, chunks)

        jobs[job_id].update(
            progress=100,
            stage="stored in database",
            status="complete",
            document_id=document_id,
            chunk_count=len(chunks),
            index_session=store.current_index_session(),
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
