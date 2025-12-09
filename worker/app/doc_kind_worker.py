# worker/app/doc_kind_worker.py

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from supabase import Client, create_client

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_DOCS_PER_RUN = 10          # bound how many documents we classify per run
MAX_PAGES_TO_ANALYSE = 10      # bound how many pages we scan per document

DOC_NAS_ROOT_ENV = "DOC_NAS_ROOT"

# Heuristic thresholds
TEXT_RATIO_THRESHOLD = 0.25    # fraction of page area covered by text
TEXT_CHAR_THRESHOLD = 200      # minimum chars on a page to be "text-heavy"


# ---------------------------------------------------------------------
# Supabase + env helpers
# ---------------------------------------------------------------------

def create_supabase_client() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY")

    if not url or not key:
        print("[error] SUPABASE_URL or service key not set", file=sys.stderr)
        return None

    try:
        client: Client = create_client(url, key)
    except Exception as exc:
        print(f"[error] Failed to create Supabase client: {exc}", file=sys.stderr)
        return None

    return client


def get_nas_root() -> Path:
    root_str = os.getenv(DOC_NAS_ROOT_ENV)
    if not root_str:
        print(f"[error] {DOC_NAS_ROOT_ENV} is not set", file=sys.stderr)
        sys.exit(1)

    root = Path(root_str)
    if not root.exists():
        print(f"[error] NAS root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    return root


# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------

def fetch_unclassified_documents(client: Client, limit: int = MAX_DOCS_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch a bounded set of processed documents where doc_kind is not set.

    We only look at documents with status='processed' so that page
    images and document_pages rows should already exist.
    """
    try:
        response = (
            client.table("document_files")
            .select("id, storage_object_path, doc_kind, status")
            .is_("doc_kind", None)
            .eq("status", "processed")
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch unclassified documents: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    print(f"[info] fetched {len(rows)} document_files row(s) for doc_kind classification")
    return rows


def update_document_kind(client: Client, document_id: Any, kind: str) -> None:
    try:
        (
            client.table("document_files")
            .update({"doc_kind": kind})
            .eq("id", document_id)
            .execute()
        )
        print(f"[info] document_id={document_id}: doc_kind set to {kind!r}")
    except Exception as exc:
        print(
            f"[error] Failed to update document_files.doc_kind for {document_id}: {exc}",
            file=sys.stderr,
        )


def mark_pages_non_drawing(client: Client, document_id: Any) -> None:
    """
    For a reference document, we can mark pages as 'non_drawing' so they
    won't be repeatedly fed into the titleblock matching pipeline.
    """
    try:
        (
            client.table("document_pages")
            .update({"status": "non_drawing"})
            .eq("document_id", document_id)
            .in_("status", ["rendered", "match_failed"])
            .execute()
        )
        print(f"[info] document_id={document_id}: pages marked as non_drawing (where applicable)")
    except Exception as exc:
        print(
            f"[error] Failed to mark document_pages as non_drawing for {document_id}: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------
# PDF analysis helpers
# ---------------------------------------------------------------------

def is_a4_or_letter(page_rect: fitz.Rect, tolerance: float = 0.15) -> bool:
    """
    Roughly detect A4 or Letter in PDF points.

    A4 ~ 595 x 842 pt
    Letter ~ 612 x 792 pt

    We allow some tolerance because PDFs may have minor scaling.
    """
    w = float(page_rect.width)
    h = float(page_rect.height)

    # Normalise orientation (portrait vs landscape)
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

      - text_area_ratio: area(text blocks) / area(page)
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

      - 'reference' if it looks like a multi-page text-heavy A4/Letter doc
      - 'unknown' otherwise

    We do NOT forcibly label 'drawing_pack' here; we just avoid calling
    something a reference unless we're confident.
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

        rect = page.rect
        is_small_sheet = is_a4_or_letter(rect)
        metrics = analyse_page_text_density(page)
        text_ratio = metrics["text_area_ratio"]
        text_chars = metrics["text_char_count"]

        print(
            f"[debug] {pdf_path.name} page={i+1} "
            f"a4_or_letter={is_small_sheet} "
            f"text_ratio={text_ratio:.3f} text_chars={text_chars:.0f}"
        )

        if is_small_sheet and text_ratio >= TEXT_RATIO_THRESHOLD and text_chars >= TEXT_CHAR_THRESHOLD:
            text_like_pages += 1

    doc.close()

    if text_like_pages >= max(1, pages_to_check // 2):
        return "reference"

    return "unknown"


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def process_document_row(client: Client, nas_root: Path, row: Dict[str, Any]) -> None:
    document_id = row.get("id")
    storage_path = row.get("storage_object_path")

    if document_id is None or not storage_path:
        print(
            f"[error] document_files row missing id or storage_object_path: {row}",
            file=sys.stderr,
        )
        return

    pdf_path = nas_root / storage_path
    if not pdf_path.is_file():
        print(
            f"[error] document_id={document_id}: PDF file not found at {pdf_path}",
            file=sys.stderr,
        )
        return

    kind = classify_pdf_kind(pdf_path)
    if kind == "reference":
        update_document_kind(client, document_id, "reference")
        # Optional: mark pages as non_drawing so they are not reprocessed by the
        # titleblock matching worker.
        mark_pages_non_drawing(client, document_id)
    else:
        # For now, leave doc_kind as NULL for non-reference docs
        print(
            f"[info] document_id={document_id}: classification={kind!r}; leaving doc_kind unchanged"
        )


def run_once() -> int:
    client = create_supabase_client()
    if client is None:
        return 1

    nas_root = get_nas_root()
    print(f"[info] Using NAS root: {nas_root}")

    docs = fetch_unclassified_documents(client, limit=MAX_DOCS_PER_RUN)
    if not docs:
        print("[info] No unclassified processed documents found")
        return 0

    for row in docs:
        process_document_row(client, nas_root, row)

    return 0


def main() -> int:
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
