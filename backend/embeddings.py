from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path


E5_MODEL_NAME = "intfloat/e5-base-v2"
E5_VECTOR_SIZE = 768
HASH_VECTOR_SIZE = 384
DEFAULT_HF_HOME = Path(r"D:\models\hf-cache")
DEFAULT_TEMP_DIR = Path(__file__).resolve().parent / "data" / "tmp"
DEFAULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("TEMP", str(DEFAULT_TEMP_DIR))
os.environ.setdefault("TMP", str(DEFAULT_TEMP_DIR))
os.environ.setdefault("TMPDIR", str(DEFAULT_TEMP_DIR))


def embed_query(text: str) -> list[float]:
    return _embed(f"query: {text}")


def embed_passage(text: str) -> list[float]:
    return _embed(f"passage: {text}")


def embedding_status() -> dict:
    try:
        model = _model()
    except Exception as exc:
        return {
            "embedding_model": E5_MODEL_NAME,
            "embedding_backend": f"fallback-hash: {exc}",
            "embedding_dimensions": HASH_VECTOR_SIZE,
        }
    return {
        "embedding_model": E5_MODEL_NAME,
        "embedding_backend": "sentence-transformers",
        "embedding_dimensions": int(
            model.get_embedding_dimension()
            if hasattr(model, "get_embedding_dimension")
            else model.get_sentence_embedding_dimension()
        ),
    }


def _embed(text: str) -> list[float]:
    try:
        model = _model()
        vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return [float(value) for value in vector]
    except Exception:
        return _hash_embed(text)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    local_path = os.environ.get("E5_MODEL_PATH")
    model_ref = local_path if local_path and Path(local_path).exists() else E5_MODEL_NAME
    return SentenceTransformer(model_ref)


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
