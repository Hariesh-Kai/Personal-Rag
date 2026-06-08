from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


E5_MODEL_NAME = "intfloat/e5-base-v2"
E5_VECTOR_SIZE = 768
HASH_VECTOR_SIZE = 384
EMBEDDING_SCHEMA_VERSION = "engineering-embeddings-v2"
MAX_EMBED_TEXT_CHARS = 12000
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
DEFAULT_HF_HOME = Path(r"D:\models\hf-cache")
DEFAULT_TEMP_DIR = Path(__file__).resolve().parent / "data" / "tmp"
FALLBACK_TEMP_DIR = Path(r"C:\tmp\rag-chatbot-tmp")
usable_temp_dir = next((path for path in [DEFAULT_TEMP_DIR, FALLBACK_TEMP_DIR] if ensure_directory(path)), DEFAULT_TEMP_DIR)
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TEMP", str(usable_temp_dir))
os.environ.setdefault("TMP", str(usable_temp_dir))
os.environ.setdefault("TMPDIR", str(usable_temp_dir))
tempfile.tempdir = str(usable_temp_dir)


def embed_query(text: str) -> list[float]:
    return embed_text(text, kind="query")["vector"]


def embed_passage(text: str) -> list[float]:
    return embed_text(text, kind="passage")["vector"]


def embed_query_with_metadata(text: str) -> dict[str, Any]:
    return embed_text(text, kind="query")


def embed_passage_with_metadata(text: str) -> dict[str, Any]:
    return embed_text(text, kind="passage")


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [item["vector"] for item in embed_texts(texts, kind="passage")]


def embed_text(text: str, kind: str = "passage") -> dict[str, Any]:
    return embed_texts([text], kind=kind)[0]


def embed_texts(texts: list[str], kind: str = "passage") -> list[dict[str, Any]]:
    prepared = [prepare_embedding_text(text, kind) for text in texts]
    started = time.perf_counter()
    try:
        model = _model()
        vectors = model.encode(prepared, batch_size=DEFAULT_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
        backend = "sentence-transformers"
        dimensions = embedding_dimension(model)
        error = ""
    except Exception as exc:
        vectors = [_hash_embed(text) for text in prepared]
        backend = "fallback-hash"
        dimensions = HASH_VECTOR_SIZE
        error = str(exc)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    results = []
    for source_text, prepared_text, vector in zip(texts, prepared, vectors):
        normalized_vector = validate_vector([float(value) for value in vector], dimensions)
        results.append(
            {
                "vector": normalized_vector,
                "metadata": embedding_metadata(
                    source_text=source_text,
                    prepared_text=prepared_text,
                    kind=kind,
                    backend=backend,
                    dimensions=len(normalized_vector),
                    vector=normalized_vector,
                    elapsed_ms=elapsed_ms,
                    error=error,
                ),
            }
        )
    return results


def embedding_status() -> dict:
    try:
        model = _model()
    except Exception as exc:
        return {
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "embedding_model": E5_MODEL_NAME,
            "embedding_backend": f"fallback-hash: {exc}",
            "embedding_dimensions": HASH_VECTOR_SIZE,
            "embedding_batch_size": DEFAULT_BATCH_SIZE,
            "embedding_text_limit": MAX_EMBED_TEXT_CHARS,
            "embedding_normalized": True,
        }
    return {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "embedding_model": E5_MODEL_NAME,
        "embedding_backend": "sentence-transformers",
        "embedding_dimensions": embedding_dimension(model),
        "embedding_batch_size": DEFAULT_BATCH_SIZE,
        "embedding_text_limit": MAX_EMBED_TEXT_CHARS,
        "embedding_normalized": True,
    }


def _embed(text: str) -> list[float]:
    return embed_text(text, kind="raw")["vector"]


def prepare_embedding_text(text: str, kind: str) -> str:
    cleaned = normalize_embedding_text(text)
    if len(cleaned) > MAX_EMBED_TEXT_CHARS:
        cleaned = cleaned[:MAX_EMBED_TEXT_CHARS].rsplit(" ", 1)[0] or cleaned[:MAX_EMBED_TEXT_CHARS]
    if kind == "query":
        return f"query: {cleaned}"
    if kind == "passage":
        return f"passage: {cleaned}"
    return cleaned


def normalize_embedding_text(text: str) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split())


def embedding_dimension(model: Any) -> int:
    return int(model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension())


def validate_vector(vector: list[float], expected_dimensions: int) -> list[float]:
    cleaned = [0.0 if not math.isfinite(value) else float(value) for value in vector]
    if expected_dimensions and len(cleaned) != expected_dimensions:
        cleaned = cleaned[:expected_dimensions] + [0.0] * max(0, expected_dimensions - len(cleaned))
    norm = math.sqrt(sum(value * value for value in cleaned))
    if norm == 0:
        return cleaned
    return [value / norm for value in cleaned]


def embedding_metadata(
    *,
    source_text: str,
    prepared_text: str,
    kind: str,
    backend: str,
    dimensions: int,
    vector: list[float],
    elapsed_ms: float,
    error: str,
) -> dict[str, Any]:
    norm = math.sqrt(sum(value * value for value in vector))
    return {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "embedding_model": E5_MODEL_NAME,
        "embedding_backend": backend,
        "embedding_kind": kind,
        "embedding_dimensions": dimensions,
        "embedding_normalized": True,
        "embedding_vector_norm": round(norm, 6),
        "embedding_vector_valid": bool(vector and all(math.isfinite(value) for value in vector)),
        "embedding_fallback": backend == "fallback-hash",
        "embedding_input_chars": len(str(source_text or "")),
        "embedding_prepared_chars": len(prepared_text),
        "embedding_truncated": len(normalize_embedding_text(source_text)) > MAX_EMBED_TEXT_CHARS,
        "embedding_batch_size": DEFAULT_BATCH_SIZE,
        "embedding_elapsed_ms": elapsed_ms,
        "embedding_error": error,
        "embedding_text_hash": hashlib.sha1(prepared_text.encode("utf-8")).hexdigest()[:16],
    }


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    local_path = os.environ.get("E5_MODEL_PATH")
    model_ref = local_path if local_path and Path(local_path).exists() else local_e5_path()
    if model_ref:
        return SentenceTransformer(str(model_ref), local_files_only=True)
    return SentenceTransformer(E5_MODEL_NAME, local_files_only=True)


def local_e5_path() -> Path | None:
    candidates = [
        DEFAULT_HF_HOME / "hub" / "models--intfloat--e5-base-v2" / "snapshots",
        Path(r"D:\models\intfloat\e5-base-v2"),
        Path(r"D:\models\e5-base-v2"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            if candidate.name == "snapshots":
                snapshots = sorted([path for path in candidate.iterdir() if path.is_dir()], reverse=True)
                if snapshots:
                    return snapshots[0]
            return candidate
    return None


def _hash_embed(text: str) -> list[float]:
    vector = [0.0] * HASH_VECTOR_SIZE
    tokens = [token.strip(".,;:!?()[]{}\"'").lower() for token in text.split()]
    for token in tokens:
        if not token:
            continue
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % HASH_VECTOR_SIZE
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
