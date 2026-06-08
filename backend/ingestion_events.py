from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


EVENT_DRIVEN_INGESTION_SCHEMA_VERSION = "engineering-event-driven-ingestion-v1"
IngestionEventHandler = Callable[[dict[str, Any]], None]


@dataclass
class EventDrivenIngestionBus:
    """Small in-process event bus for ingestion.

    The app currently runs ingestion inside the Django process, so this bus is
    intentionally local. It gives the pipeline real publish/subscribe behavior
    without requiring Kafka/Celery/Redis just to make ingestion observable.
    """

    handlers: dict[str, list[tuple[str, IngestionEventHandler]]] = field(default_factory=lambda: defaultdict(list))
    history: list[dict[str, Any]] = field(default_factory=list)
    dead_letters: list[dict[str, Any]] = field(default_factory=list)

    def subscribe(self, handler_name: str, handler: IngestionEventHandler, event_type: str = "*") -> None:
        self.handlers[event_type].append((handler_name, handler))

    def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_event(event)
        self.history.append(normalized)
        for handler_name, handler in [*self.handlers.get("*", []), *self.handlers.get(normalized["type"], [])]:
            try:
                handler(normalized)
            except Exception as exc:
                self.dead_letters.append(
                    {
                        "handler": handler_name,
                        "event_id": normalized["event_id"],
                        "event_type": normalized["type"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return normalized

    def snapshot(self) -> dict[str, Any]:
        return event_driven_ingestion_metadata(self.history, self.dead_letters)


def make_ingestion_event(
    event_type: str,
    stage: str,
    progress: int,
    detail: str = "",
    payload: dict[str, Any] | None = None,
    source: str = "",
    sequence: int = 0,
) -> dict[str, Any]:
    event_id = f"{int(time.time() * 1000)}-{sequence:04d}-{event_type}-{stage}".replace(" ", "_")
    return {
        "schema": EVENT_DRIVEN_INGESTION_SCHEMA_VERSION,
        "event_id": event_id,
        "type": event_type,
        "stage": stage,
        "progress": max(0, min(100, int(progress))),
        "detail": detail,
        "payload": payload or {},
        "source": source,
        "timestamp_ms": int(time.time() * 1000),
        "sequence": sequence,
    }


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event or {})
    normalized.setdefault("schema", EVENT_DRIVEN_INGESTION_SCHEMA_VERSION)
    normalized.setdefault("event_id", f"{int(time.time() * 1000)}-event")
    normalized.setdefault("type", "event")
    normalized.setdefault("stage", "")
    normalized["progress"] = max(0, min(100, int(normalized.get("progress") or 0)))
    normalized.setdefault("detail", "")
    normalized.setdefault("payload", {})
    normalized.setdefault("source", "")
    normalized.setdefault("timestamp_ms", int(time.time() * 1000))
    normalized.setdefault("sequence", 0)
    return normalized


def publish_ingestion_event(
    bus: EventDrivenIngestionBus,
    event_type: str,
    stage: str,
    progress: int,
    detail: str = "",
    payload: dict[str, Any] | None = None,
    source: str = "",
    event_callback: Any | None = None,
) -> dict[str, Any]:
    event = make_ingestion_event(
        event_type=event_type,
        stage=stage,
        progress=progress,
        detail=detail,
        payload=payload,
        source=source,
        sequence=len(bus.history) + 1,
    )
    published = bus.publish(event)
    if event_callback:
        event_callback(published)
    return published


def event_driven_ingestion_metadata(
    events: list[dict[str, Any]],
    dead_letters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_events = [normalize_event(event) for event in events]
    type_counts = Counter(event["type"] for event in normalized_events)
    stage_counts = Counter(event["stage"] for event in normalized_events if event.get("stage"))
    failed_events = [
        event
        for event in normalized_events
        if event["type"] in {"stage_error", "error"} or str(event.get("payload", {}).get("status", "")).lower() == "error"
    ]
    return {
        "event_driven_ingestion_schema_version": EVENT_DRIVEN_INGESTION_SCHEMA_VERSION,
        "event_driven_ingestion_ready": True,
        "event_driven_ingestion_mode": "in_process_publish_subscribe_bus",
        "event_driven_ingestion_event_count": len(normalized_events),
        "event_driven_ingestion_event_types": dict(sorted(type_counts.items())),
        "event_driven_ingestion_stage_counts": dict(sorted(stage_counts.items())),
        "event_driven_ingestion_failed_events": failed_events[:20],
        "event_driven_ingestion_dead_letters": (dead_letters or [])[:20],
        "event_driven_ingestion_dead_letter_count": len(dead_letters or []),
        "event_driven_ingestion_complete": bool(normalized_events and normalized_events[-1]["type"] in {"chunks_ready", "complete"}),
        "event_driven_ingestion_observable": True,
        "event_driven_ingestion_subscriber_model": "named_handlers_by_event_type",
    }
