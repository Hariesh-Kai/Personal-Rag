from __future__ import annotations

import hashlib
import os
import re
from typing import Any


TRANSLATION_SCHEMA_VERSION = "engineering-translation-ingestion-v2"

ENGINEERING_TRANSLATION_GLOSSARY = {
    "वाल्व": "valve",
    "दबाव": "pressure",
    "सुरक्षा": "safety",
    "आग": "fire",
    "पाइप": "pipe",
    "ढलान": "slope",
    "அழுத்தம்": "pressure",
    "வால்வு": "valve",
    "பாதுகாப்பு": "safety",
    "குழாய்": "pipe",
    "సురక్ష": "safety",
    "వాల్వ్": "valve",
    "పీడనం": "pressure",
    "ಪೈಪ್": "pipe",
    "ವಾಲ್ವ್": "valve",
    "ضغط": "pressure",
    "صمام": "valve",
    "سلامة": "safety",
    "管": "pipe",
    "阀": "valve",
    "压力": "pressure",
}


def translation_ingestion_metadata(text: str, language_meta: dict[str, Any]) -> dict[str, Any]:
    source_language = str(language_meta.get("language_code") or "unknown")
    target_language = "en"
    required = source_language not in {"en", "unknown"} or bool(language_meta.get("multilingual"))
    glossary = glossary_translation_surface(text)
    model_translation = optional_local_translation(text, source_language, target_language) if required else ""
    translated_surface = model_translation or glossary or ""
    status = translation_status(required, model_translation, glossary)
    return {
        "translation_schema_version": TRANSLATION_SCHEMA_VERSION,
        "translation_ingestion_ready": True,
        "translation_required": required,
        "translation_available": bool(model_translation or glossary or not required),
        "translation_status": status,
        "translation_source_language": source_language,
        "translation_target_language": target_language,
        "translation_method": "local_model" if model_translation else ("engineering_glossary" if glossary else "not_required" if not required else "metadata_only"),
        "translation_original_text_preserved": True,
        "translation_text": translated_surface,
        "translation_retrieval_text": translation_retrieval_text(text, translated_surface, language_meta),
        "translation_glossary_terms": glossary_terms(text),
        "translation_confidence": translation_confidence(required, model_translation, glossary, language_meta),
        "translation_model_configured": bool(os.environ.get("RAG_TRANSLATION_MODEL")),
        "translation_safe_fallback": not bool(model_translation),
        "translation_hash": hashlib.sha1(f"{source_language}|{translated_surface[:500]}".encode("utf-8")).hexdigest()[:16],
    }


def glossary_translation_surface(text: str) -> str:
    hits = glossary_terms(text)
    if not hits:
        return ""
    return " ".join(f"{item['source']} means {item['target']}" for item in hits)


def glossary_terms(text: str) -> list[dict[str, str]]:
    normalized = str(text or "")
    terms = []
    for source, target in ENGINEERING_TRANSLATION_GLOSSARY.items():
        if source in normalized:
            terms.append({"source": source, "target": target})
    return terms[:40]


def optional_local_translation(text: str, source_language: str, target_language: str) -> str:
    model_ref = os.environ.get("RAG_TRANSLATION_MODEL", "").strip()
    if not model_ref:
        return ""
    try:
        from transformers import pipeline  # type: ignore

        translator = pipeline("translation", model=model_ref, tokenizer=model_ref, local_files_only=True)
        result = translator(text[:2500], max_length=2500)
        if isinstance(result, list) and result:
            return str(result[0].get("translation_text") or "").strip()
    except Exception:
        return ""
    return ""


def translation_retrieval_text(text: str, translated_surface: str, language_meta: dict[str, Any]) -> str:
    parts = [
        language_meta.get("language_code") or "",
        language_meta.get("language_name") or "",
        "multilingual" if language_meta.get("multilingual") else "",
        translated_surface,
    ]
    return normalize(" ".join(str(part) for part in parts if part))


def translation_status(required: bool, model_translation: str, glossary: str) -> str:
    if not required:
        return "not_required"
    if model_translation:
        return "translated_by_local_model"
    if glossary:
        return "partial_engineering_glossary_translation"
    return "translation_metadata_ready_no_engine"


def translation_confidence(required: bool, model_translation: str, glossary: str, language_meta: dict[str, Any]) -> float:
    if not required:
        return 1.0
    if model_translation:
        return round(min(0.9, 0.55 + float(language_meta.get("language_confidence") or 0) * 0.25), 3)
    if glossary:
        return 0.48
    return 0.2


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
