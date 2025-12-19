# app/document_worker.py

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from supabase import Client

from app.config import MAX_PAGES_PER_DOC
from app.db_utils import (
    update_document_status,
    upsert_document_page,
    update_document_kind,
    mark_pages_non_drawing,
)
from app.path_utils import build_page_image_rel_path, build_raw_pdf_path
from app.pdf_ops import get_pdf_page_count, render_page, classify_pdf_kind
from app.hash_utils import compute_file_sha256


def render_all_pages_for_document(
    client: Client,
    nas_root: Path,
    derived_bucket: str,
    document_row: Dict[str, Any],
    pdf_path: Path,
    page_count: int,
) -> Dict[str, Any]:
    """
    Render all pages for a document (bounded by MAX_PAGES_PER_DOC) and upsert
    document_pages rows.

    Returns:
      {
        "first_error": Optional[str],
        "rendered_pages": int,
    }
    """
    document_id = document_row.get("id")
    if document_id is None:
        return {"first_error": "document_files row missing id", "rendered_pages": 0}

    max_pages = min(page_count, MAX_PAGES_PER_DOC)
    if max_pages <= 0:
        return {"first_error": "No pages to render", "rendered_pages": 0}

    first_error: Optional[str] = None
    rendered_pages = 0

    for page_number in range(1, max_pages + 1):
        image_rel = build_page_image_rel_path(document_row, page_number=page_number)
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

        page_hash = compute_file_sha256(image_abs)

        upsert_document_page(
            client=client,
            document_id=document_id,
            page_number=page_number,
            image_bucket=derived_bucket,
            image_object_path=image_rel,
            status="rendered",
            page_sha256=page_hash,
        )
        rendered_pages += 1

    return {"first_error": first_error, "rendered_pages": rendered_pages}


def classify_and_mark_document(
    client: Client,
    document_id: Any,
    pdf_path: Path,
) -> None:
    """
    Classify a PDF and update doc_kind / page statuses as needed.

    Currently:
      - If classified as 'reference', set document_files.doc_kind to 'reference'
        and mark pages >1 as non_drawing so titleblock matching skips them.
    """
    kind = classify_pdf_kind(pdf_path)
    if kind != "reference":
        return

    update_document_kind(client, document_id, "reference")
    mark_pages_non_drawing(client, document_id)


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
      - Render pages 1..N (bounded by MAX_PAGES_PER_DOC)
      - Upsert document_pages rows with status='rendered'
      - Update document_files.status + page_count
      - Classify PDF kind and mark reference docs
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

    render_result = render_all_pages_for_document(
        client=client,
        nas_root=nas_root,
        derived_bucket=derived_bucket,
        document_row=row,
        pdf_path=pdf_path,
        page_count=page_count,
    )

    first_error = render_result["first_error"]
    rendered_pages = render_result["rendered_pages"]

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
            f"(pages={page_count}, rendered_pages={rendered_pages}, first_error={first_error!r})"
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
            f"(pages={page_count}, rendered_pages={rendered_pages})"
        )

    # Classify and mark reference docs (does nothing if not classified as reference)
    classify_and_mark_document(client, document_id, pdf_path)
