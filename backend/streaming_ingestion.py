from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


STREAMING_INGESTION_SCHEMA_VERSION = "engineering-streaming-ingestion-v2"
STREAM_SENTINEL = "__rag_stream_done__"


@dataclass
class StreamingIngestionState:
    source_name: str
    started_at: float = field(default_factory=time.perf_counter)
    last_progress: int = 0
    last_stage: str = ""
    event_count: int = 0
    error: str = ""
    complete: bool = False

    def update(self, event: dict[str, Any]) -> None:
        self.event_count += 1
        self.last_progress = int(event.get("progress") or self.last_progress)
        self.last_stage = str(event.get("stage") or self.last_stage)
        if event.get("type") in {"stage_error", "error"}:
            self.error = str(event.get("detail") or event.get("payload", {}).get("error") or "")
        if event.get("type") in {"complete", "chunks_ready", "stream_complete"}:
            self.complete = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "streaming_schema_version": STREAMING_INGESTION_SCHEMA_VERSION,
            "streaming_source": self.source_name,
            "streaming_event_count": self.event_count,
            "streaming_last_progress": self.last_progress,
            "streaming_last_stage": self.last_stage,
            "streaming_complete": self.complete,
            "streaming_error": self.error,
            "streaming_elapsed_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
        }


def streaming_ingestion_metadata(events: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    state = StreamingIngestionState(source_name=source_name)
    event_types: dict[str, int] = {}
    for event in events:
        state.update(event)
        event_type = str(event.get("type") or "event")
        event_types[event_type] = event_types.get(event_type, 0) + 1
    return {
        **state.snapshot(),
        "streaming_supported": True,
        "streaming_mode": "threaded_queue_event_stream",
        "streaming_backpressure": "bounded_queue",
        "streaming_resumable": False,
        "streaming_event_types": event_types,
        "streaming_first_event": events[0] if events else {},
        "streaming_last_event": events[-1] if events else {},
    }


def stream_ingestion_events(
    path: Path,
    processor: Callable[..., Any],
    *,
    max_queue_size: int = 256,
    result_callback: Callable[[Any], dict[str, Any] | None] | None = None,
) -> Iterator[dict[str, Any]]:
    event_queue: queue.Queue[dict[str, Any] | str] = queue.Queue(maxsize=max_queue_size)
    state = StreamingIngestionState(source_name=path.name)
    emitted: list[dict[str, Any]] = []

    def put_event(event: dict[str, Any]) -> None:
        event = normalize_stream_event(event, source_name=path.name)
        state.update(event)
        emitted.append(event)
        event_queue.put(event)

    def worker() -> None:
        try:
            put_event(stream_event("stream_start", "document_loading_extraction", 0, f"source={path.name}", source=path.name))
            result = processor(path, event_callback=put_event)
            result_payload = result_callback(result) if result_callback else {}
            payload = {
                "chunk_count": len(getattr(result, "chunks", []) or []),
                "metadata": getattr(result, "metadata", {}) or {},
                "stages": getattr(result, "stages", []) or [],
                "result": result_payload or {},
                "streaming": streaming_ingestion_metadata(emitted, path.name),
            }
            put_event(stream_event("stream_complete", "ingestion_pipeline_ready", 100, "stream complete", payload, source=path.name))
        except Exception as exc:
            put_event(stream_event("stream_error", "ingestion_pipeline_failed", 100, str(exc), {"error": str(exc)}, source=path.name))
        finally:
            event_queue.put(STREAM_SENTINEL)

    thread = threading.Thread(target=worker, name=f"rag-ingest-stream-{path.name}", daemon=True)
    thread.start()
    while True:
        item = event_queue.get()
        if item == STREAM_SENTINEL:
            break
        yield item


def stream_event(
    event_type: str,
    stage: str,
    progress: int,
    detail: str = "",
    payload: dict[str, Any] | None = None,
    source: str = "",
) -> dict[str, Any]:
    return {
        "schema": STREAMING_INGESTION_SCHEMA_VERSION,
        "type": event_type,
        "stage": stage,
        "progress": max(0, min(100, int(progress))),
        "detail": detail,
        "payload": payload or {},
        "source": source,
        "timestamp_ms": int(time.time() * 1000),
    }


def normalize_stream_event(event: dict[str, Any], source_name: str) -> dict[str, Any]:
    normalized = dict(event or {})
    normalized["streaming_schema"] = STREAMING_INGESTION_SCHEMA_VERSION
    normalized.setdefault("source", source_name)
    normalized.setdefault("timestamp_ms", int(time.time() * 1000))
    normalized.setdefault("payload", {})
    normalized["progress"] = max(0, min(100, int(normalized.get("progress") or 0)))
    return normalized
