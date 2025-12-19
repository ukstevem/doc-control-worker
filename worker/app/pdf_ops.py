# app/pdf_ops.py

import sys
from pathlib import Path
from typing import Dict, Optional

import fitz
from pdf2image import convert_from_path, pdfinfo_from_path

from app.config import (
    MAX_PAGES_TO_ANALYSE,
    TEXT_RATIO_THRESHOLD,
    TEXT_CHAR_THRESHOLD,
)
from app.path_utils import ensure_parent_dir


# -------------------------------------------------------------------------
# PDF page count + rendering
# -------------------------------------------------------------------------


def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Return the number of pages in a PDF using pdfinfo_from_path."""
    try:
        info = pdfinfo_from_path(str(pdf_path), userpw=None)
    except Exception as exc:
        print(f"[error] Failed to read PDF info for {pdf_path}: {exc}", file=sys.stderr)
        return None

    pages = info.get("Pages")
    if isinstance(pages, int) and pages > 0:
        return pages

    print(f"[error] Unexpected or missing page count for {pdf_path}: {pages}", file=sys.stderr)
    return None


def render_page(pdf_path: Path, page_number: int, output_path: Path) -> bool:
    """
    Render a specific page of a PDF to a PNG at output_path.

    Returns True on success, False otherwise.
    """
    print(f"[info] Rendering page {page_number} of {pdf_path} -> {output_path}")

    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_number,
            last_page=page_number,
            fmt="png",
        )
    except Exception as exc:
        print(f"[error] Failed to convert {pdf_path} page {page_number}: {exc}", file=sys.stderr)
        return False

    if not images:
        print(
            f"[error] No pages returned when converting {pdf_path} page {page_number}",
            file=sys.stderr,
        )
        return False

    image = images[0]
    try:
        ensure_parent_dir(output_path)
        image.save(str(output_path), "PNG")
    except Exception as exc:
        print(
            f"[error] Failed to save preview for {pdf_path} page {page_number}: {exc}",
            file=sys.stderr,
        )
        return False

    return True


# -------------------------------------------------------------------------
# PDF classification (reference vs unknown)
# -------------------------------------------------------------------------


def is_a4_or_letter(page_rect: fitz.Rect, tolerance: float = 0.15) -> bool:
    """
    Roughly detect A4 or Letter in PDF points.

    A4 ~ 595 x 842 pt, Letter ~ 612 x 792 pt.
    We normalise orientation (portrait/landscape) and allow some tolerance.
    """
    w = float(page_rect.width)
    h = float(page_rect.height)
    short_side = min(w, h)
    long_side = max(w, h)

    # A4
    a4_short, a4_long = 595.0, 842.0
    # Letter
    lt_short, lt_long = 612.0, 792.0

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tolerance * b

    is_a4 = close(short_side, a4_short) and close(long_side, a4_long)
    is_letter = close(short_side, lt_short) and close(long_side, lt_long)
    return is_a4 or is_letter


def analyse_page_text_density(page: fitz.Page) -> Dict[str, float]:
    """
    Compute simple text-density metrics for a page:

      - text_area_ratio: sum(area of text blocks) / area(page)
      - text_char_count: total characters in text blocks
    """
    rect = page.rect
    page_area = float(rect.width) * float(rect.height)
    if page_area <= 0.0:
        return {"text_area_ratio": 0.0, "text_char_count": 0.0}

    try:
        blocks = page.get_text("blocks")
    except Exception:
        return {"text_area_ratio": 0.0, "text_char_count": 0.0}

    text_area = 0.0
    text_chars = 0

    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not isinstance(text, str):
            continue
        text_stripped = text.strip()
        if not text_stripped:
            continue

        try:
            b_rect = fitz.Rect(x0, y0, x1, y1)
        except Exception:
            continue

        area = b_rect.get_area()
        if area <= 0.0:
            continue

        text_area += area
        text_chars += len(text_stripped)

    text_area_ratio = text_area / page_area if page_area > 0.0 else 0.0
    return {
        "text_area_ratio": float(text_area_ratio),
        "text_char_count": float(text_chars),
    }


def classify_pdf_kind(pdf_path: Path) -> str:
    """
    Classify a PDF as:

      - 'reference' if it looks like a text-heavy A4/Letter document
      - 'unknown' otherwise.

    We do NOT try to label 'drawing_pack' here; we only avoid mislabelling
    drawings as reference.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"[error] Failed to open PDF {pdf_path} for classification: {exc}", file=sys.stderr)
        return "unknown"

    try:
        page_count = doc.page_count
    except Exception:
        page_count = 0

    if page_count <= 0:
        doc.close()
        return "unknown"

    pages_to_check = min(page_count, MAX_PAGES_TO_ANALYSE)
    text_like_pages = 0

    for i in range(pages_to_check):
        try:
            page = doc[i]
        except Exception:
            continue

        metrics = analyse_page_text_density(page)
        text_ratio = metrics["text_area_ratio"]
        text_chars = metrics["text_char_count"]
        small_sheet = is_a4_or_letter(page.rect)

        print(
            f"[debug] classify_pdf_kind {pdf_path.name} page={i+1} "
            f"a4_or_letter={small_sheet} text_ratio={text_ratio:.3f} "
            f"text_chars={text_chars:.0f}"
        )

        # Treat any page with enough text density as "text-like",
        # regardless of exact page size. We keep is_a4_or_letter
        # in the debug log for visibility but don't gate on it.
        if (
            text_ratio >= TEXT_RATIO_THRESHOLD
            and text_chars >= TEXT_CHAR_THRESHOLD
        ):
            text_like_pages += 1


    doc.close()

    if text_like_pages >= max(1, pages_to_check // 2):
        return "reference"

    return "unknown"
