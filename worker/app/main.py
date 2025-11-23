import os
import sys
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


def render_first_page(pdf_path: Path, output_path: Path) -> bool:
    """
    Render the first page of a PDF to a PNG at output_path.

    Returns True on success, False otherwise.
    """
    print(f"[info] Rendering first page of {pdf_path} -> {output_path}")

    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=1,
            last_page=1,
            fmt="png",
        )
    except Exception as exc:
        print(f"[error] Failed to convert {pdf_path}: {exc}", file=sys.stderr)
        return False

    if not images:
        print(f"[error] No pages returned when converting {pdf_path}", file=sys.stderr)
        return False

    image = images[0]
    try:
        ensure_parent_dir(output_path)
        image.save(str(output_path), "PNG")
    except Exception as exc:
        print(f"[error] Failed to save preview for {pdf_path}: {exc}", file=sys.stderr)
        return False

    return True


# ----------------------------------------------------------------------------- 
# Orchestrator
# ----------------------------------------------------------------------------- 


def process_document_row(client: Client, nas_root: Path, derived_bucket: str, row: Dict[str, Any]) -> None:
    """
    Process a single document_files row:

      - Resolve raw PDF path from storage_object_path
      - Count pages
      - Render page 1 to derived/pages/.../p1.png
      - Upsert document_pages row for page 1
      - Update document_files.status + page_count
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

    # Mark as processing
    update_document_status(client, document_id, "processing")

    page_count = get_pdf_page_count(pdf_path)
    if page_count is None:
        update_document_status(client, document_id, "error", error_message="Unable to determine page count")
        return

    image_rel = build_page_image_rel_path(row, page_number=1)
    if image_rel is None:
        update_document_status(client, document_id, "error", error_message="Cannot determine image path")
        return

    image_abs = nas_root / image_rel

    ok = render_first_page(pdf_path, image_abs)
    if not ok:
        update_document_status(client, document_id, "error", error_message="Failed to render first page")
        return

    upsert_document_page(
        client=client,
        document_id=document_id,
        page_number=1,
        image_bucket=derived_bucket,
        image_object_path=image_rel,
        status="rendered",
    )

    update_document_status(client, document_id, "processed", page_count=page_count)

    print(
        f"[info] document_id={document_id} processed successfully "
        f"(pages={page_count}, image={image_rel})"
    )


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


def main() -> int:
    """Entry point. Kept small for testability."""
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
