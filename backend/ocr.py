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
DEFAULT_OCR_LANGUAGES = tuple(
    language.strip()
    for language in os.environ.get("EASYOCR_LANGUAGES", "en").split(",")
    if language.strip()
)
MIN_OCR_CONFIDENCE = float(os.environ.get("EASYOCR_MIN_CONFIDENCE", "0.25"))
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
            "EasyOCR runs for scanned/image-heavy PDF pages and standalone image uploads; OCR models load on first OCR use."
            if has_easyocr
            else "Install easyocr to enable scanned/image-only PDF OCR."
        ),
        "ocr_languages": list(DEFAULT_OCR_LANGUAGES),
        "ocr_min_confidence": MIN_OCR_CONFIDENCE,
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
    return read_image_array(image)


def read_image_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run EasyOCR on a standalone image file and return OCR blocks plus image metadata."""
    if not easyocr_available():
        return [], {"ocr": "easyocr_unavailable"}
    image = load_image_array(path)
    if image is None:
        return [], {"ocr": "image_load_failed"}
    blocks = read_image_array(image)
    confidences = [float(block["confidence"]) for block in blocks]
    return blocks, {
        "ocr": "easyocr",
        "ocr_block_count": len(blocks),
        "ocr_confidence_avg": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }


def read_image_array(image: Any) -> list[dict[str, Any]]:
    """Run EasyOCR on an image-like array and return positioned text blocks."""
    if not easyocr_available():
        return []
    reader = _reader()
    prepared = prepare_image_for_ocr(image)
    results = reader.readtext(prepared, detail=1, paragraph=False)
    blocks: list[dict[str, Any]] = []
    for box, text, confidence in results:
        normalized = " ".join(str(text).split())
        if not normalized or float(confidence or 0) < MIN_OCR_CONFIDENCE:
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


def load_image_array(path: Path) -> Any | None:
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(image)
    except Exception:
        return None


def prepare_image_for_ocr(image: Any) -> Any:
    """Light preprocessing for scanned engineering pages without making OCR slow."""
    try:
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError:
        return image
    try:
        array = np.asarray(image)
        pil_image = Image.fromarray(array).convert("RGB")
        width, height = pil_image.size
        if max(width, height) < 1600:
            scale = min(2.0, 1600 / max(1, max(width, height)))
            pil_image = pil_image.resize((int(width * scale), int(height * scale)))
        grayscale = ImageOps.grayscale(pil_image)
        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.4)
        grayscale = grayscale.filter(ImageFilter.SHARPEN)
        return np.array(grayscale)
    except Exception:
        return image


@lru_cache(maxsize=1)
def _reader():
    import easyocr

    return easyocr.Reader(
        list(DEFAULT_OCR_LANGUAGES or ("en",)),
        gpu=False,
        verbose=False,
        model_storage_directory=str(DEFAULT_OCR_MODEL_DIR),
        user_network_directory=str(DEFAULT_OCR_MODEL_DIR / "user_network"),
    )
