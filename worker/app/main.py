import os
import sys
import subprocess
import time
import fitz

from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf2image import convert_from_path, pdfinfo_from_path
from supabase import Client, create_client

# ----------------------------------------------------------------------------- 
# Configuration constants (Power of 10: explicit, bounded work)
# ----------------------------------------------------------------------------- 

MAX_DOCS_PER_RUN = 10  # Upper bound on documents per run
DOC_NAS_ROOT_ENV = "DOC_NAS_ROOT"
DERIVED_BUCKET_ENV = "DOC_DERIVED_BUCKET"

# Loop / orchestration configuration
WORKER_MODE_ENV = "WORKER_MODE"              # "once" or "loop"
WORKER_LOOP_SLEEP_ENV = "WORKER_LOOP_SLEEP"  # seconds between cycles
WORKER_MAX_CYCLES_ENV = "WORKER_MAX_CYCLES"  # 0 = run forever

# Titleblock-related subworkers to invoke after PDF/page work.
# These must be runnable as: python -m <module_name>
TITLEBLOCK_WORKER_MODULES = (
    "app.titleblock_match_worker",
    "app.titleblock_extract_worker",
)

# Bound how many pages we render per document (you may already have this)
MAX_PAGES_PER_DOC = 50

# How we analyse for reference docs
MAX_PAGES_TO_ANALYSE = 10
TEXT_RATIO_THRESHOLD = 0.25   # fraction of page area covered by text
TEXT_CHAR_THRESHOLD = 200     # minimum chars per page to call it "text-heavy"


# ----------------------------------------------------------------------------- 
# Supabase helpers
# ----------------------------------------------------------------------------- 


def create_supabase_client() -> Optional[Client]:
    """
    Create a Supabase client if env vars are set, otherwise return None.

    We support both SUPABASE_SERVICE_ROLE_KEY and SUPABASE_SECRET_KEY
    for compatibility. This worker only ever runs on the server.
    """
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


def ping_document_files_table(client: Client) -> None:
    """Print the number of rows in document_files (bounded sanity check)."""
    try:
        response = (
            client.table("document_files")
            .select("id", count="exact")
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Supabase ping failed: {exc}", file=sys.stderr)
        return

    count = getattr(response, "count", None)
    print(f"[info] document_files table reachable; count={count}")


def update_document_kind(client: Client, document_id: Any, kind: str) -> None:
    """Set document_files.doc_kind, e.g. 'reference'."""
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
    For a reference document, mark rendered/match_failed pages as non_drawing
    so they don't re-enter the titleblock matching pipeline.
    """
    try:
        (
            client.table("document_pages")
            .update({"status": "non_drawing"})
            .eq("document_id", document_id)
            .gt("page_number", 1) 
            .in_("status", ["rendered", "match_failed"])
            .execute()
        )
        print(f"[info] document_id={document_id}: pages marked non_drawing")
    except Exception as exc:
        print(
            f"[error] Failed to mark document_pages as non_drawing for {document_id}: {exc}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------------------- 
# Environment / path helpers
# ----------------------------------------------------------------------------- 


def get_nas_root() -> Path:
    """
    Resolve the NAS root path for raw + derived files.

    DOC_NAS_ROOT should point at the root described in the
    doc-control-storage-layout-v1 spec (e.g. /data/doc_control).
    """
    root_str = os.getenv(DOC_NAS_ROOT_ENV)
    if not root_str:
        print(f"[error] {DOC_NAS_ROOT_ENV} is not set", file=sys.stderr)
        sys.exit(1)

    root = Path(root_str)
    if not root.exists():
        print(f"[error] NAS root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    return root


def build_raw_pdf_path(nas_root: Path, row: Dict[str, Any]) -> Optional[Path]:
    """
    Build the absolute path to the raw PDF on NAS from a document_files row.

    We expect storage_object_path to contain a relative path such as:
      raw/enquiries/ENQ-1234/<id>_name.pdf
    """
    storage_path = row.get("storage_object_path")
    if not storage_path:
        print("[error] document_files.storage_object_path is empty", file=sys.stderr)
        return None

    pdf_path = nas_root / storage_path
    return pdf_path


def build_page_image_rel_path(row: Dict[str, Any], page_number: int) -> Optional[str]:
    """
    Build the relative path for the derived page image, following:

      derived/pages/enquiries/{enquirynumber}/{document_id}/p{page}.png
      derived/pages/projects/{projectnumber}/{document_id}/p{page}.png

    Returns a POSIX-style relative path or None on error.
    """
    document_id = row.get("id")
    enquirynumber = row.get("enquirynumber")
    projectnumber = row.get("projectnumber")

    if document_id is None:
        print("[error] document_files row missing id", file=sys.stderr)
        return None

    if projectnumber:
        stage = "projects"
        parent = projectnumber
    elif enquirynumber:
        stage = "enquiries"
        parent = enquirynumber
    else:
        print(
            f"[error] document_id={document_id} has neither enquirynumber nor projectnumber",
            file=sys.stderr,
        )
        return None

    rel = f"derived/pages/{stage}/{parent}/{document_id}/p{page_number}.png"
    return rel


def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory for a file path exists."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[error] Failed to create directory {path.parent}: {exc}", file=sys.stderr)
        raise


# ----------------------------------------------------------------------------- 
# Database operations
# ----------------------------------------------------------------------------- 


def fetch_uploaded_pdfs(client: Client, limit: int = MAX_DOCS_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch a bounded set of PDF documents with status='uploaded' from document_files.

    We filter by status in SQL and filter extensions in Python for robustness.
    """
    try:
        response = (
            client.table("document_files")
            .select(
                "id,enquirynumber,projectnumber,"
                "original_filename,file_ext,storage_bucket,storage_object_path,"
                "status"
            )
            .eq("status", "uploaded")
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch uploaded documents: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    results: List[Dict[str, Any]] = []

    for row in rows:
        ext = str(row.get("file_ext") or "").lower().lstrip(".")
        if ext == "pdf":
            results.append(row)

    if not results:
        print("[info] No 'uploaded' PDF rows found in document_files")

    return results


def update_document_status(
    client: Client,
    document_id: Any,
    status: str,
    page_count: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update document_files.status (and optionally page_count / processing_error)."""
    update_data: Dict[str, Any] = {"status": status}
    if page_count is not None:
        update_data["page_count"] = page_count
    if error_message is not None:
        update_data["processing_error"] = error_message[:500]

    try:
        (
            client.table("document_files")
            .update(update_data)
            .eq("id", document_id)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to update document_files.status for {document_id}: {exc}",
            file=sys.stderr,
        )


def upsert_document_page(
    client: Client,
    document_id: Any,
    page_number: int,
    image_bucket: str,
    image_object_path: str,
    status: str,
) -> None:
    """
    Upsert a single document_pages row for a given document + page.

    Schema summary:
      - document_id (uuid)
      - page_number (int)
      - image_bucket (text)
      - image_object_path (text)
      - status (text)
    """
    row: Dict[str, Any] = {
        "document_id": document_id,
        "page_number": page_number,
        "image_bucket": image_bucket,
        "image_object_path": image_object_path,
        "status": status,
        "processing_error": None,
    }

    try:
        (
            client.table("document_pages")
            .upsert(row, on_conflict="document_id,page_number")
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to upsert document_pages for document_id={document_id}, "
            f"page={page_number}: {exc}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------------------- 
# PDF processing
# ----------------------------------------------------------------------------- 

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

        if small_sheet and text_ratio >= TEXT_RATIO_THRESHOLD and text_chars >= TEXT_CHAR_THRESHOLD:
            text_like_pages += 1

    doc.close()

    if text_like_pages >= max(1, pages_to_check // 2):
        return "reference"

    return "unknown"


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

# ----------------------------------------------------------------------------- 
# Orchestrator
# ----------------------------------------------------------------------------- 


def process_document_row(
    client: Client,
    nas_root: Path,
    derived_bucket: str,
    row: Dict[str, Any],
) -> None:
    """
    Process a single document_files row:

      - Resolve raw PDF path from storage_object_path
      - Count pages
      - For pages 1..N (bounded by MAX_PAGES_PER_DOC):
          * render PNG to derived/pages/.../p{n}.png
          * upsert a document_pages row with status='rendered'
      - Update document_files.status + page_count

    This is page-agnostic; titleblock matching / OCR happens later.
    """
    document_id = row.get("id")
    if document_id is None:
        print("[error] document_files row missing id; skipping", file=sys.stderr)
        return

    pdf_path = build_raw_pdf_path(nas_root, row)
    if pdf_path is None:
        update_document_status(client, document_id, "error", error_message="Missing storage_object_path")
        return

    if not pdf_path.is_file():
        message = f"PDF file not found at {pdf_path}"
        print(f"[error] {message} (document_id={document_id})", file=sys.stderr)
        update_document_status(client, document_id, "error", error_message=message)
        return

    print(f"[info] Processing document_id={document_id}, file={pdf_path}")

    # Mark as processing at document level
    update_document_status(client, document_id, "processing")

    page_count = get_pdf_page_count(pdf_path)
    if page_count is None:
        update_document_status(client, document_id, "error", error_message="Unable to determine page count")
        return

    max_pages = min(page_count, MAX_PAGES_PER_DOC)
    if max_pages <= 0:
        update_document_status(client, document_id, "error", error_message="No pages to render")
        return

    first_error: Optional[str] = None

    for page_number in range(1, max_pages + 1):
        image_rel = build_page_image_rel_path(row, page_number=page_number)
        if image_rel is None:
            msg = f"Cannot determine image path for page {page_number}"
            print(f"[error] document_id={document_id}: {msg}", file=sys.stderr)
            if first_error is None:
                first_error = msg
            continue

        image_abs = nas_root / image_rel

        ok = render_page(pdf_path, page_number, image_abs)
        if not ok:
            msg = f"Failed to render page {page_number}"
            print(f"[error] document_id={document_id}: {msg}", file=sys.stderr)
            if first_error is None:
                first_error = msg
            # Keep going – partial output is better than none
            continue

        # One row per (document_id, page_number)
        upsert_document_page(
            client=client,
            document_id=document_id,
            page_number=page_number,
            image_bucket=derived_bucket,
            image_object_path=image_rel,
            status="rendered",
        )

    # Final document-level status
    if first_error is not None:
        update_document_status(
            client,
            document_id,
            "processed",
            page_count=page_count,
            error_message=first_error,
        )
        print(
            f"[info] document_id={document_id} processed with issues "
            f"(pages={page_count}, first_error={first_error!r})"
        )
    else:
        update_document_status(
            client,
            document_id,
            "processed",
            page_count=page_count,
        )
        print(
            f"[info] document_id={document_id} processed successfully "
            f"(pages={page_count}, rendered_pages={max_pages})"
        )

    # ------------------------------------------------------------------
    # Classify document kind (reference vs unknown) based on PDF content.
    # Only sets 'reference' when we're confident; else leaves it alone.
    # ------------------------------------------------------------------
    kind = classify_pdf_kind(pdf_path)
    if kind == "reference":
        update_document_kind(client, document_id, "reference")
        # For reference docs, we don't want to waste CPU on titleblock matching.
        mark_pages_non_drawing(client, document_id)


def run_once() -> int:
    """Run one bounded batch of work and exit."""
    client = create_supabase_client()
    if client is None:
        return 1

    nas_root = get_nas_root()
    derived_bucket = os.getenv(DERIVED_BUCKET_ENV, "doc_nas_derived")

    ping_document_files_table(client)

    print(f"[info] Using NAS root: {nas_root}")
    print(f"[info] Using derived bucket: {derived_bucket}")

    rows = fetch_uploaded_pdfs(client, limit=MAX_DOCS_PER_RUN)
    if not rows:
        return 0

    for row in rows:
        process_document_row(client, nas_root, derived_bucket, row)

    return 0


# ----------------------------------------------------------------------------- 
# Subworker orchestration
# ----------------------------------------------------------------------------- 


def run_subworker_module(module_name: str) -> None:
    """
    Run a secondary worker module via `python -m <module_name>`.

    This lets main orchestrate the titleblock workers without depending
    on their internal implementation details.
    """
    print(f"[info] Running subworker module: {module_name}", flush=True)

    try:
        completed = subprocess.run(
            [sys.executable, "-m", module_name],
            check=False,
        )
    except Exception as exc:
        print(
            f"[error] Exception while running {module_name}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return

    if completed.returncode != 0:
        print(
            f"[warn] {module_name} exited with code {completed.returncode}",
            file=sys.stderr,
            flush=True,
        )


def run_loop() -> int:
    """
    Run the main PDF worker and the titleblock workers in a simple loop.

    Controlled by env vars:
      - WORKER_LOOP_SLEEP: seconds to sleep between cycles (default 10)
      - WORKER_MAX_CYCLES: if > 0, stop after this many cycles (default 0 = forever)
    """
    sleep_str = os.getenv(WORKER_LOOP_SLEEP_ENV, "10")
    max_cycles_str = os.getenv(WORKER_MAX_CYCLES_ENV, "0")

    try:
        sleep_seconds = int(sleep_str)
    except ValueError:
        sleep_seconds = 10

    try:
        max_cycles = int(max_cycles_str)
    except ValueError:
        max_cycles = 0

    cycle = 0

    print(
        f"[info] Starting worker loop: sleep={sleep_seconds}s max_cycles={max_cycles}",
        flush=True,
    )

    while True:
        exit_code = run_once()
        if exit_code != 0:
            print(
                f"[warn] run_once() returned non-zero exit code {exit_code}",
                file=sys.stderr,
                flush=True,
            )

        for module_name in TITLEBLOCK_WORKER_MODULES:
            run_subworker_module(module_name)

        cycle += 1
        if 0 < max_cycles <= cycle:
            print("[info] Max cycles reached; exiting loop.", flush=True)
            break

        time.sleep(sleep_seconds)

    return 0


def main() -> int:
    """Entry point. Chooses between one-shot and loop modes."""
    mode = os.getenv(WORKER_MODE_ENV, "once").lower()
    if mode == "loop":
        return run_loop()
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
