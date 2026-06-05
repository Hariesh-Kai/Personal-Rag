from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .ocr import easyocr_available, read_pdf_page_image
from .document_loader import _read_docx, _read_json, _read_text


HEADING_RE = re.compile(r"^((?:\d+\.)+\d*|[A-Z]\.|\d+)\s+[A-Z][A-Za-z0-9 /,&()'_.:-]{3,}$")
SECTION_NUMBER_RE = re.compile(r"^(?P<num>(?:\d+\.)*\d+)\s+(?P<title>.+)$")
INLINE_SECTION_RE = re.compile(r"(?=\b\d+\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 /,&()'_.:-]{3,})")
MAX_CHUNK_CHARS = 1700
TABLE_CONTEXT_CHARS = 700
TABLE_HEADER_RE = re.compile(r"^TABLE:\n(?P<header>.+?)(?:\n|$)", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^[^\n|]+(?:\|[^\n|]+)+$", re.MULTILINE)
CONTAMINATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwhat to tell management\b",
        r"\bimmediate rule for debugging\b",
        r"\bstage-level failure record\b",
        r"\brag frameworks\b",
        r"\bupload lanes to ingest\b",
        r"\bretrieval grounded\b",
        r"\bhallucination control\b",
    ]
]
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "being",
    "can",
    "for",
    "from",
    "has",
    "have",
    "into",
    "may",
    "not",
    "shall",
    "should",
    "that",
    "the",
    "their",
    "this",
    "with",
    "will",
}


@dataclass
class ContentBlock:
    text: str
    page: int
    kind: str = "text"
    bbox: tuple[float, float, float, float] | None = None
    font_size: float = 0.0
    is_heading: bool = False
    heading_level: int = 0
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RichChunk:
    text: str
    metadata: dict[str, Any]


def process_document(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        blocks, doc_meta = extract_pdf_blocks(path)
    else:
        blocks, doc_meta = extract_generic_blocks(path)

    cleaned = remove_noise(blocks)
    structured = detect_structure(cleaned)
    chunks = dedupe_chunks(semantic_chunk(structured, doc_meta))
    return [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]


def extract_pdf_blocks(path: Path) -> tuple[list[ContentBlock], dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Structure-aware PDF extraction needs PyMuPDF.") from exc

    blocks: list[ContentBlock] = []
    ocr_pages = 0
    ocr_attempted = False
    with fitz.open(str(path)) as document:
        font_sizes: list[float] = []
        page_labels: dict[int, str] = {}
        for page_index, page in enumerate(document, start=1):
            page_blocks: list[ContentBlock] = []
            page_rect = page.rect
            table_blocks = extract_pdf_tables(page, page_index)
            table_bboxes = [block.bbox for block in table_blocks if block.bbox]
            page_text = page.get_text("dict")
            for raw_block in page_text.get("blocks", []):
                raw_bbox = tuple(raw_block.get("bbox", (0, 0, 0, 0)))
                if raw_block.get("type") == 1:
                    page_blocks.append(
                        ContentBlock(
                            text="[IMAGE]",
                            page=page_index,
                            kind="image",
                            bbox=raw_bbox,
                            metadata={"image": True},
                        )
                    )
                    continue
                if any(overlap_ratio(raw_bbox, table_bbox) > 0.45 for table_bbox in table_bboxes):
                    continue
                lines = []
                sizes = []
                for line in raw_block.get("lines", []):
                    spans = line.get("spans", [])
                    text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                    if text:
                        lines.append(text)
                        sizes.extend(float(span.get("size", 0)) for span in spans)
                text = normalize_line(" ".join(lines))
                if text:
                    size = max(sizes) if sizes else 0.0
                    font_sizes.append(size)
                    page_blocks.append(
                        ContentBlock(
                            text=text,
                            page=page_index,
                            kind="text",
                            bbox=raw_bbox,
                            font_size=size,
                            metadata={"page_width": page_rect.width, "page_height": page_rect.height},
                        )
                    )
            page_blocks.extend(table_blocks)
            page_blocks.sort(key=lambda block: ((block.bbox or (0, 0, 0, 0))[1], (block.bbox or (0, 0, 0, 0))[0]))
            label = detect_printed_page_label(page_blocks)
            if label:
                page_labels[page_index] = label
            if should_ocr_page(page_blocks):
                ocr_attempted = True
                ocr_blocks = extract_ocr_blocks(page, page_index)
                if ocr_blocks:
                    page_blocks.extend(ocr_blocks)
                    page_blocks.sort(key=lambda block: ((block.bbox or (0, 0, 0, 0))[1], (block.bbox or (0, 0, 0, 0))[0]))
                    ocr_pages += 1
            blocks.extend(page_blocks)

        doc_meta = {
            "extractor": "PyMuPDF",
            "pages": len(document),
            "page_labels": page_labels,
            "median_font_size": median(font_sizes) if font_sizes else 0,
            "ocr": f"easyocr_pages:{ocr_pages}" if ocr_pages else ("easyocr_unavailable" if ocr_attempted else "not_needed"),
            "note": "PyMuPDF extracts native PDF text, page geometry, images, and detected tables. EasyOCR is used only for scanned/image-only pages.",
        }
    return blocks, doc_meta


def should_ocr_page(page_blocks: list[ContentBlock]) -> bool:
    if not easyocr_available():
        return False
    text_chars = sum(len(block.text.strip()) for block in page_blocks if block.kind in {"text", "table"})
    image_blocks = sum(1 for block in page_blocks if block.kind == "image")
    return text_chars < 80 and image_blocks > 0


def extract_ocr_blocks(page: Any, page_number: int) -> list[ContentBlock]:
    try:
        import numpy as np
    except ImportError:
        return []
    try:
        pixmap = page.get_pixmap(matrix=page.parent.Matrix(2, 2), alpha=False)
    except Exception:
        try:
            import fitz

            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        except Exception:
            return []
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    scale_x = page.rect.width / max(1, pixmap.width)
    scale_y = page.rect.height / max(1, pixmap.height)
    blocks = []
    for item in read_pdf_page_image(image):
        x0, y0, x1, y1 = item["bbox"]
        blocks.append(
            ContentBlock(
                text=item["text"],
                page=page_number,
                kind="text",
                bbox=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                metadata={"ocr": "easyocr", "ocr_confidence": item["confidence"], "page_width": page.rect.width, "page_height": page.rect.height},
            )
        )
    return blocks


def extract_pdf_tables(page: Any, page_number: int) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return blocks
    try:
        tables = finder()
    except Exception:
        return blocks
    for table_index, table in enumerate(getattr(tables, "tables", []), start=1):
        try:
            rows = table.extract()
        except Exception:
            continue
        rendered = render_table(rows)
        if rendered and not is_page_furniture_table(rendered):
            blocks.append(
                ContentBlock(
                    text=rendered,
                    page=page_number,
                    kind="table",
                    bbox=tuple(getattr(table, "bbox", (0, 0, 0, 0))),
                    metadata={"table_index": table_index, "preserve_together": True},
                )
            )
    return blocks


def extract_generic_blocks(path: Path) -> tuple[list[ContentBlock], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = _read_docx(path)
        extractor = "python-docx"
    elif suffix == ".json":
        text = _read_json(path)
        extractor = "json"
    elif suffix == ".csv":
        text = render_table(list(csv.reader(io.StringIO(_read_text(path)))))
        extractor = "csv"
    else:
        text = _read_text(path)
        extractor = "plain-text"
    blocks = [ContentBlock(text=normalize_line(line), page=1) for line in text.splitlines() if normalize_line(line)]
    return blocks, {"extractor": extractor, "pages": 1, "ocr": "not_applicable"}


def remove_noise(blocks: list[ContentBlock]) -> list[ContentBlock]:
    line_counts = Counter(block.text for block in blocks if block.kind == "text")
    pages_by_line: dict[str, set[int]] = defaultdict(set)
    page_count = max((block.page for block in blocks), default=1)
    for block in blocks:
        pages_by_line[block.text].add(block.page)

    cleaned = []
    for block in blocks:
        text = normalize_table_text(block.text) if block.kind == "table" else normalize_line(block.text)
        if not text:
            continue
        if is_generated_or_debug_text(text):
            continue
        if block.kind == "image":
            cleaned.append(block)
            continue
        if block.kind == "table" and is_page_furniture_table(text):
            continue
        repeated_pages = len(pages_by_line[text])
        y0 = block.bbox[1] if block.bbox else 0
        page_height = float(block.metadata.get("page_height", 0) or 0)
        near_edge = page_height and (y0 < page_height * 0.09 or y0 > page_height * 0.90)
        if repeated_pages >= max(2, int(page_count * 0.35)) and (near_edge or len(text) < 120):
            continue
        if is_page_number(text) or is_watermark(text) or is_page_furniture(text) or line_counts[text] > max(4, page_count):
            continue
        block.text = text
        cleaned.append(block)
    return remove_front_matter(split_inline_sections(merge_wrapped_lines(cleaned)))


def detect_structure(blocks: list[ContentBlock]) -> list[ContentBlock]:
    body_size = median([block.font_size for block in blocks if block.font_size > 0] or [0])
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind == "table":
            block.section_path = [title for _, title in stack]
            continue
        block.is_heading = looks_like_heading(block.text, block.font_size, body_size)
        if block.is_heading:
            level = heading_level(block.text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text))
            block.heading_level = level
        block.section_path = [title for _, title in stack]
    return blocks


def semantic_chunk(blocks: list[ContentBlock], doc_meta: dict[str, Any]) -> list[RichChunk]:
    chunks: list[RichChunk] = []
    current: list[ContentBlock] = []
    current_topic = ""
    previous_context = ""

    def flush() -> None:
        nonlocal current, previous_context
        if not current:
            return
        text = "\n".join(block.text for block in current if block.text != "[IMAGE]").strip()
        if not text:
            current = []
            return
        meta = build_metadata(current, doc_meta)
        if previous_context:
            meta["overlap_context"] = previous_context
        chunks.append(RichChunk(text=text, metadata=meta))
        previous_context = tail_overlap(text)
        current = []

    consumed_table_indexes: set[int] = set()
    for index, block in enumerate(blocks):
        if block.kind == "image":
            continue
        if index in consumed_table_indexes:
            continue
        if block.kind == "table":
            flush()
            table_indexes = related_table_indexes(blocks, index)
            consumed_table_indexes.update(table_indexes)
            table_blocks = table_context_blocks(blocks, table_indexes)
            chunks.append(RichChunk(text=table_chunk_text(table_blocks), metadata=build_metadata(table_blocks, doc_meta)))
            previous_context = tail_overlap(block.text)
            continue

        topic = " > ".join(block.section_path)
        new_topic = topic and topic != current_topic
        hard_subsection = block.is_heading and bool(SECTION_NUMBER_RE.match(block.text))
        too_large = current_char_count(current) + len(block.text) > MAX_CHUNK_CHARS
        concept_break = current and (block.is_heading or new_topic or too_large)
        if current and hard_subsection:
            concept_break = True
        if concept_break:
            flush()
        current.append(block)
        current_topic = topic
    flush()
    return [chunk for chunk in chunks if len(chunk.text.strip()) >= 80]


def build_metadata(blocks: list[ContentBlock], doc_meta: dict[str, Any]) -> dict[str, Any]:
    pages = sorted({block.page for block in blocks})
    page_labels = doc_meta.get("page_labels") or {}
    section_path = next((block.section_path for block in reversed(blocks) if block.section_path), [])
    content_types = sorted({block.kind for block in blocks})
    text = "\n".join(block.text for block in blocks)
    contains_table = "table" in content_types or looks_like_table_text(text)
    table_meta = table_metadata(text) if contains_table else {}
    revision_meta = revision_metadata(text)
    return {
        "content_types": content_types,
        "section_path": section_path,
        "section_title": section_path[-1] if section_path else "",
        "parent_section": section_path[-2] if len(section_path) > 1 else "",
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
        "page_label_start": page_labels.get(pages[0]) if pages else None,
        "page_label_end": page_labels.get(pages[-1]) if pages else None,
        "contains_table": contains_table,
        "table_index": next((block.metadata.get("table_index") for block in blocks if block.metadata.get("table_index")), None),
        **table_meta,
        **revision_meta,
        "keywords": keywords(text),
        "extractor": doc_meta.get("extractor"),
        "ocr": doc_meta.get("ocr"),
    }


def revision_metadata(text: str) -> dict[str, Any]:
    compact = normalize_line(text)
    revision = ""
    document_id = ""
    status = ""
    revision_match = re.search(r"\b(?:revision|rev\.?|revision number)\s*[:#]?\s*([A-Z]?\d+[A-Z]?|[A-Z]\d?)\b", compact, flags=re.I)
    if revision_match:
        revision = revision_match.group(1)
    document_match = re.search(r"\b(?:document id|document n\.?|contractor document id|company document id)\s*[:#]?\s*([A-Z0-9_-]{6,})\b", compact, flags=re.I)
    if document_match:
        document_id = document_match.group(1)
    status_match = re.search(r"\b(?:validity status|status)\s*[:#]?\s*([A-Z][A-Z0-9 -]{1,40})\b", compact, flags=re.I)
    if status_match:
        status = status_match.group(1).strip()
    return {
        "revision": revision,
        "document_identifier": document_id,
        "validity_status": status,
    }


def looks_like_heading(text: str, font_size: float, body_size: float) -> bool:
    if len(text) > 140 or text.endswith("."):
        return False
    if SECTION_NUMBER_RE.match(text) or HEADING_RE.match(text):
        return True
    alpha = [char for char in text if char.isalpha()]
    uppercase_ratio = sum(1 for char in alpha if char.isupper()) / max(1, len(alpha))
    return bool(alpha) and uppercase_ratio > 0.72 and len(text.split()) <= 12 and (not body_size or font_size >= body_size)


def heading_level(text: str) -> int:
    match = SECTION_NUMBER_RE.match(text)
    if not match:
        return 1
    return min(6, match.group("num").count(".") + 1)


def merge_wrapped_lines(blocks: list[ContentBlock]) -> list[ContentBlock]:
    merged: list[ContentBlock] = []
    for block in blocks:
        if block.kind != "text" or not merged or merged[-1].kind != "text" or merged[-1].page != block.page:
            merged.append(block)
            continue
        previous = merged[-1]
        starts_new_subsection = bool(SECTION_NUMBER_RE.match(block.text))
        same_sectionish = (
            not starts_new_subsection
            and not previous.text.endswith((".", ":", ";", "?", "!"))
            and not looks_like_heading(block.text, block.font_size, 0)
        )
        if same_sectionish and len(previous.text) < 180 and len(block.text) < 180:
            previous.text = f"{previous.text} {block.text}"
        else:
            merged.append(block)
    return merged


def split_inline_sections(blocks: list[ContentBlock]) -> list[ContentBlock]:
    split_blocks: list[ContentBlock] = []
    for block in blocks:
        if block.kind != "text":
            split_blocks.append(block)
            continue
        if len(block.text) < 80:
            split_blocks.extend(split_numbered_heading_body(block))
            continue
        parts = [part.strip() for part in INLINE_SECTION_RE.split(block.text) if part.strip()]
        if len(parts) <= 1:
            split_blocks.extend(split_numbered_heading_body(block))
            continue
        for part in parts:
            split_blocks.extend(
                split_numbered_heading_body(
                    ContentBlock(
                        text=part,
                        page=block.page,
                        kind=block.kind,
                        bbox=block.bbox,
                        font_size=block.font_size,
                        metadata=dict(block.metadata),
                    )
                )
            )
    return split_blocks


def split_numbered_heading_body(block: ContentBlock) -> list[ContentBlock]:
    match = SECTION_NUMBER_RE.match(block.text)
    if not match:
        return [block]

    number = match.group("num")
    remainder = match.group("title").strip()
    tokens = remainder.split()
    if len(tokens) < 4:
        return [block]

    title_tokens: list[str] = []
    for token in tokens:
        clean = token.strip(",;:()[]")
        connector = clean.lower() in {"and", "or", "for", "of", "to", "with", "&"}
        title_word = bool(clean) and (clean[0].isupper() or clean.isupper() or connector)
        if not title_word:
            break
        title_tokens.append(token)
        if len(title_tokens) >= 8:
            break

    while title_tokens and title_tokens[-1].strip(",;:()[]").lower() in {"all", "the", "a", "an"}:
        title_tokens.pop()

    if len(title_tokens) < 2 or len(title_tokens) >= len(tokens):
        return [block]

    heading = f"{number} {' '.join(title_tokens)}"
    body = " ".join(tokens[len(title_tokens) :]).strip()
    if len(body) < 20:
        return [block]

    heading_block = ContentBlock(
        text=heading,
        page=block.page,
        kind=block.kind,
        bbox=block.bbox,
        font_size=block.font_size,
        metadata=dict(block.metadata),
    )
    body_block = ContentBlock(
        text=body,
        page=block.page,
        kind=block.kind,
        bbox=block.bbox,
        font_size=block.font_size,
        metadata=dict(block.metadata),
    )
    return [heading_block, body_block]


def render_table(rows: list[list[Any]]) -> str:
    normalized_rows = [[normalize_line(str(cell or "")) for cell in row] for row in rows if any(str(cell or "").strip() for cell in row)]
    if not normalized_rows:
        return ""
    column_count = max(len(row) for row in normalized_rows)
    padded = [row + [""] * (column_count - len(row)) for row in normalized_rows]
    widths = [max(len(row[index]) for row in padded) for index in range(column_count)]
    rendered = ["TABLE:"]
    for row_index, row in enumerate(padded):
        rendered.append(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())
        if row_index == 0 and len(padded) > 1:
            rendered.append(" | ".join("-" * min(widths[index], 32) for index in range(column_count)))
    return "\n".join(rendered)


def overlap_ratio(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    overlap_x = max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0))
    overlap_y = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    overlap_area = overlap_x * overlap_y
    left_area = max(1.0, (left_x1 - left_x0) * (left_y1 - left_y0))
    return overlap_area / left_area


def tail_overlap(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n+", text) if paragraph.strip()]
    if not paragraphs:
        return ""
    candidates = []
    for paragraph in reversed(paragraphs):
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if 45 <= len(sentence) <= 260:
                candidates.append(sentence)
                break
        if candidates:
            break
    overlap = candidates[0] if candidates else paragraphs[-1]
    words = overlap.split()
    if len(words) < 8:
        return ""
    return " ".join(words[:40])


def current_char_count(blocks: list[ContentBlock]) -> int:
    return sum(len(block.text) for block in blocks)


def related_table_indexes(blocks: list[ContentBlock], table_index: int) -> list[int]:
    table = blocks[table_index]
    indexes = [table_index]
    scan_start = table_index + 1
    for index in range(scan_start, min(len(blocks), table_index + 8)):
        block = blocks[index]
        if block.kind == "text" and block.is_heading:
            break
        if block.page != table.page or block.section_path != table.section_path:
            break
        if block.kind == "table":
            between = blocks[indexes[-1] + 1 : index]
            if any(is_table_title_text(item.text) for item in between if item.kind == "text"):
                break
            if any(item.kind == "text" and looks_related_to_table(item.text, table.text) for item in between) or not between:
                indexes.append(index)
                continue
        if block.kind == "text" and not looks_related_to_table(block.text, table.text):
            break
    return indexes


def table_context_blocks(blocks: list[ContentBlock], table_indexes: list[int]) -> list[ContentBlock]:
    first_index = table_indexes[0]
    last_index = table_indexes[-1]
    table = blocks[first_index]
    grouped: list[ContentBlock] = []
    before_chars = 0
    for block in reversed(blocks[max(0, first_index - 5) : first_index]):
        if block.kind == "table":
            break
        if block.kind != "text" or block.page != table.page or block.section_path != table.section_path:
            continue
        if block.is_heading or looks_related_to_table(block.text, table.text):
            grouped.insert(0, block)
            before_chars += len(block.text)
        if is_table_title_text(block.text):
            break
        if before_chars >= TABLE_CONTEXT_CHARS:
            break
    grouped.extend(
        block
        for block in blocks[first_index : last_index + 1]
        if block.kind == "table" or (block.kind == "text" and looks_related_to_table(block.text, table.text))
    )
    after_chars = 0
    for block in blocks[last_index + 1 : last_index + 5]:
        if block.kind == "table" or is_table_title_text(block.text):
            break
        if block.kind != "text" or block.page != table.page or block.section_path != table.section_path:
            continue
        if looks_related_to_table(block.text, table.text):
            grouped.append(block)
            after_chars += len(block.text)
        if after_chars >= 360:
            break
    return grouped


def table_chunk_text(blocks: list[ContentBlock]) -> str:
    lines: list[str] = []
    previous = ""
    for block in blocks:
        for line in block.text.splitlines():
            normalized = normalize_line(line)
            if not normalized or normalized == previous:
                continue
            previous = normalized
            lines.append(line.rstrip())
    return "\n".join(lines).strip()


def looks_related_to_table(text: str, table_text: str) -> bool:
    terms = set(keywords(table_text)[:10])
    text_terms = set(keywords(text)[:12])
    return bool(terms & text_terms) or len(text) < 180 or text.endswith(":")


def is_table_title_text(text: str) -> bool:
    normalized = normalize_line(text).lower()
    return bool(
        re.match(r"^(table\s*)?[\w .-]*requirement\s*:", normalized)
        or re.match(r"^[\w .-]*\btable\s*:", normalized)
    )


def detect_printed_page_label(blocks: list[ContentBlock]) -> str | None:
    page_text = "\n".join(block.text for block in blocks)
    match = re.search(r"sheet\s+of\s+sheets\s+(\d+\s*/\s*\d+)", page_text, flags=re.IGNORECASE)
    if match:
        return normalize_line(match.group(1))
    match = re.search(r"\bpage\s+(\d+\s*(?:of|/)\s*\d+)\b", page_text, flags=re.IGNORECASE)
    if match:
        return normalize_line(match.group(1))
    return None


def table_metadata(text: str) -> dict[str, Any]:
    title = table_title(text)
    columns = table_columns(text)
    rows = table_rows(text)
    table_terms = sorted(set(keywords(" ".join([title, " ".join(columns), " ".join(rows)]))))
    return {
        "table_title": title,
        "table_columns": columns,
        "table_rows": rows[:80],
        "table_row_count": len(rows),
        "table_terms": table_terms[:30],
    }


def table_title(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    for index, line in enumerate(lines):
        if line == "TABLE:" and index > 0:
            title_lines = []
            for candidate in reversed(lines[:index]):
                if candidate.startswith(("Note:", "TABLE:")):
                    continue
                if len(candidate) <= 180:
                    title_lines.insert(0, candidate)
                if len(title_lines) >= 2:
                    break
            return " ".join(title_lines).strip()
    return ""


def table_columns(text: str) -> list[str]:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    for index, line in enumerate(lines):
        if line == "TABLE:" and index + 1 < len(lines):
            return [cell.strip() for cell in lines[index + 1].split("|") if cell.strip()]
    compact = normalize_line(text)
    known_headers = [
        ["Pipe Direction", "Slope", "Remarks"],
        ["Direction", "Slope", "Remarks"],
        ["Size", "Rating", "Class"],
    ]
    for headers in known_headers:
        if all(header.lower() in compact.lower() for header in headers):
            return headers
    return []


def table_rows(text: str) -> list[str]:
    rows = []
    columns = [column.lower() for column in table_columns(text)]
    for line in text.splitlines():
        row = normalize_line(line)
        if "|" not in row:
            continue
        row_cells = [cell.strip().lower() for cell in row.split("|") if cell.strip()]
        is_separator = bool(re.fullmatch(r"[-|\s]+", row))
        is_header = bool(columns) and row_cells == columns
        if row and not is_separator and not is_header:
            rows.append(row)
    if not rows and looks_like_table_text(text):
        compact = normalize_line(text)
        for marker in ["Aft to Forward", "Forward to Aft", "Transverse"]:
            if marker.lower() in compact.lower():
                rows.append(marker)
    return rows


def looks_like_table_text(text: str) -> bool:
    normalized = normalize_line(text).lower()
    if "table:" in normalized:
        return True
    header_sets = [
        {"pipe direction", "slope", "remarks"},
        {"direction", "slope", "remarks"},
        {"size", "rating", "class"},
    ]
    return any(all(header in normalized for header in headers) for headers in header_sets)


def dedupe_chunks(chunks: list[RichChunk]) -> list[RichChunk]:
    kept: list[RichChunk] = []
    signatures: set[str] = set()
    for chunk in chunks:
        normalized = normalize_for_dedupe(chunk.text)
        if len(normalized) < 40:
            continue
        signature = " ".join(normalized.split()[:90])
        if signature in signatures:
            continue
        if any(jaccard_words(normalized, normalize_for_dedupe(existing.text)) > 0.9 for existing in kept):
            continue
        signatures.add(signature)
        kept.append(chunk)
    return kept


def normalize_for_dedupe(text: str) -> str:
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text.lower())
    text = re.sub(r"\W+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jaccard_words(left: str, right: str) -> float:
    left_words = set(left.split())
    right_words = set(right.split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def keywords(text: str) -> list[str]:
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", text)]
    counts = Counter(term for term in terms if term not in STOPWORDS)
    return [term for term, _ in counts.most_common(18)]


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def normalize_table_text(text: str) -> str:
    lines = [normalize_line(line) for line in text.replace("\x00", " ").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def is_page_number(text: str) -> bool:
    return bool(re.match(r"^(page\s*)?\d+\s*(of|/)\s*\d+$", text, flags=re.IGNORECASE) or re.match(r"^\d+$", text))


def is_watermark(text: str) -> bool:
    normalized = text.lower()
    return (
        normalized in {"confidential", "draft", "controlled copy"}
        or "this document is property" in normalized
        or "this document is the property" in normalized
        or "shall neither be shown to third parties" in normalized
    )


def is_page_furniture(text: str) -> bool:
    normalized = text.lower()
    patterns = [
        "company logo",
        "contractor logo",
        "business name",
        "revision index",
        "issued for tender",
        "contractor prepared",
        "contractor verified",
        "company approved",
        "document n.",
        "job n.",
    ]
    return len(text) < 260 and any(pattern in normalized for pattern in patterns)


def is_generated_or_debug_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in CONTAMINATION_PATTERNS)


def is_page_furniture_table(text: str) -> bool:
    normalized = text.lower()
    markers = [
        "company document id",
        "contractor document id",
        "sheet of sheets",
        "validity status",
        "revision number",
    ]
    return sum(1 for marker in markers if marker in normalized) >= 2


def remove_front_matter(blocks: list[ContentBlock]) -> list[ContentBlock]:
    first_real_index = 0
    for index, block in enumerate(blocks):
        if block.kind == "table" and block.page <= 2:
            continue
        if block.is_heading or SECTION_NUMBER_RE.match(block.text) or re.match(r"^(purpose|scope|introduction)\b", block.text, re.I):
            first_real_index = index
            break
    if first_real_index > 0:
        return blocks[first_real_index:]
    return blocks
