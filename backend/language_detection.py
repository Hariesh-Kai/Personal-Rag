from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any


LANGUAGE_DETECTION_SCHEMA_VERSION = "engineering-language-detection-v2"

LANGUAGE_BY_SCRIPT = {
    "latin": ("en", "English"),
    "devanagari": ("hi", "Hindi / Devanagari"),
    "bengali": ("bn", "Bengali"),
    "tamil": ("ta", "Tamil"),
    "telugu": ("te", "Telugu"),
    "kannada": ("kn", "Kannada"),
    "malayalam": ("ml", "Malayalam"),
    "gujarati": ("gu", "Gujarati"),
    "gurmukhi": ("pa", "Punjabi / Gurmukhi"),
    "arabic": ("ar", "Arabic"),
    "cyrillic": ("ru", "Cyrillic / Russian-like"),
    "cjk": ("zh", "Chinese / CJK"),
}

SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "arabic": (0x0600, 0x06FF),
    "cyrillic": (0x0400, 0x04FF),
    "cjk": (0x4E00, 0x9FFF),
}

ENGLISH_STOPWORDS = {
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "shall",
    "should",
    "the",
    "to",
    "with",
}

ENGINEERING_LATIN_TERMS = {
    "valve",
    "pressure",
    "pipe",
    "piping",
    "safety",
    "drain",
    "vent",
    "flare",
    "slope",
    "support",
    "equipment",
}


def language_detection_metadata(text: str) -> dict[str, Any]:
    normalized = normalize(text)
    script_counts = count_scripts(normalized)
    total_script_chars = sum(script_counts.values())
    scripts = [script for script, _ in script_counts.most_common()]
    script_ratios = {
        script: round(count / max(1, total_script_chars), 4)
        for script, count in script_counts.items()
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}", normalized.lower())
    english_score = english_confidence(tokens, script_ratios)
    primary_script = scripts[0] if scripts else ""
    language_code, language_name, confidence = choose_language(primary_script, script_ratios, english_score, bool(normalized))
    multilingual = len([ratio for ratio in script_ratios.values() if ratio >= 0.12]) > 1
    if multilingual and language_code != "unknown":
        language_name = f"Mixed including {language_name}"
        confidence = min(confidence, 0.72)
    retrieval_text = language_retrieval_text(language_code, language_name, scripts, script_ratios, tokens)
    return {
        "language_schema_version": LANGUAGE_DETECTION_SCHEMA_VERSION,
        "language_detection_ready": True,
        "language_code": language_code,
        "language_name": language_name,
        "language_confidence": round(confidence, 3),
        "language_detection_method": "script_ratio_latin_stopword_engineering_terms",
        "detected_scripts": scripts,
        "language_script_counts": dict(script_counts),
        "language_script_ratios": script_ratios,
        "primary_script": primary_script,
        "multilingual": multilingual,
        "non_ascii_ratio": non_ascii_ratio(normalized),
        "english_signal_score": round(english_score, 4),
        "language_retrieval_text": retrieval_text,
        "language_hash": hashlib.sha1(retrieval_text.encode("utf-8")).hexdigest()[:16] if retrieval_text else "",
    }


def choose_language(primary_script: str, script_ratios: dict[str, float], english_score: float, has_text: bool) -> tuple[str, str, float]:
    if not has_text:
        return "unknown", "Unknown", 0.0
    latin_ratio = script_ratios.get("latin", 0.0)
    if primary_script == "latin" and (english_score >= 0.24 or latin_ratio >= 0.82):
        return "en", "English", min(0.96, 0.58 + english_score + latin_ratio * 0.18)
    if primary_script in LANGUAGE_BY_SCRIPT:
        code, name = LANGUAGE_BY_SCRIPT[primary_script]
        confidence = min(0.94, 0.55 + script_ratios.get(primary_script, 0.0) * 0.38)
        return code, name, confidence
    if latin_ratio >= 0.5:
        return "en", "English-like Latin", min(0.7, 0.42 + english_score)
    return "unknown", "Unknown", 0.2


def english_confidence(tokens: list[str], script_ratios: dict[str, float]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    stopword_hits = sum(counts[token] for token in ENGLISH_STOPWORDS)
    engineering_hits = sum(counts[token] for token in ENGINEERING_LATIN_TERMS)
    token_score = (stopword_hits * 0.75 + engineering_hits * 0.45) / max(1, len(tokens))
    latin_bonus = min(0.18, script_ratios.get("latin", 0.0) * 0.18)
    return min(0.65, token_score + latin_bonus)


def count_scripts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        code = ord(char)
        if "A" <= char <= "Z" or "a" <= char <= "z":
            counts["latin"] += 1
            continue
        for script, (start, end) in SCRIPT_RANGES.items():
            if start <= code <= end:
                counts[script] += 1
                break
    return counts


def detected_scripts(text: str) -> list[str]:
    return [script for script, _ in count_scripts(text).most_common()]


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(1 for char in text if ord(char) > 127) / len(text), 4)


def language_retrieval_text(language_code: str, language_name: str, scripts: list[str], ratios: dict[str, float], tokens: list[str]) -> str:
    top_tokens = [token for token, _ in Counter(tokens).most_common(24)]
    parts = [
        language_code,
        language_name,
        " ".join(scripts),
        " ".join(f"{script}:{ratio}" for script, ratio in ratios.items()),
        " ".join(top_tokens),
        "multilingual" if len(scripts) > 1 else "",
    ]
    return normalize(" ".join(part for part in parts if part))


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\x00", " ")).strip()
