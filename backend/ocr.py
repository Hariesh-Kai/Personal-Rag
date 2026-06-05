from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TEMP_DIR = Path(os.environ.get("RAG_TEMP_DIR", DEFAULT_DATA_DIR / "tmp"))
DEFAULT_OCR_MODEL_DIR = Path(os.environ.get("EASYOCR_MODEL_DIR", DEFAULT_DATA_DIR / "easyocr"))
DEFAULT_TORCH_CACHE_DIR = DEFAULT_TEMP_DIR / "torch-cache"
for directory in (DEFAULT_TEMP_DIR, DEFAULT_OCR_MODEL_DIR, DEFAULT_TORCH_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TEMP", str(DEFAULT_TEMP_DIR))
os.environ.setdefault("TMP", str(DEFAULT_TEMP_DIR))
os.environ.setdefault("TMPDIR", str(DEFAULT_TEMP_DIR))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_CACHE_DIR))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(DEFAULT_TORCH_CACHE_DIR / "inductor"))


def ocr_status() -> dict:
    has_easyocr = easyocr_available()
    return {
        "ocr_available": bool(has_easyocr),
        "ocr_backend": "easyocr" if has_easyocr else "not_available",
        "ocr_detail": (
            "EasyOCR will run only for scanned/image-only PDF pages; OCR models load on first scanned page."
            if has_easyocr
            else "Install easyocr to enable scanned/image-only PDF OCR."
        ),
    }


def easyocr_available() -> bool:
    if importlib.util.find_spec("easyocr") is None:
        return False
    try:
        import easyocr  # noqa: F401
    except Exception:
        return False
    return True


def read_pdf_page_image(image: Any) -> list[dict[str, Any]]:
    """Run EasyOCR on an RGB page image and return positioned text blocks."""
    if not easyocr_available():
        return []
    reader = _reader()
    results = reader.readtext(image, detail=1, paragraph=False)
    blocks: list[dict[str, Any]] = []
    for box, text, confidence in results:
        normalized = " ".join(str(text).split())
        if not normalized or float(confidence or 0) < 0.25:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        blocks.append(
            {
                "text": normalized,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "confidence": float(confidence or 0),
            }
        )
    return blocks


@lru_cache(maxsize=1)
def _reader():
    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
        model_storage_directory=str(DEFAULT_OCR_MODEL_DIR),
        user_network_directory=str(DEFAULT_OCR_MODEL_DIR / "user_network"),
    )
