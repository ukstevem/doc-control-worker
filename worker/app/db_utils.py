# app/db_utils.py

import sys
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from app.config import MAX_DOCS_PER_RUN


# -------------------------------------------------------------------------
# Supabase client helpers
# -------------------------------------------------------------------------


def create_supabase_client() -> Optional[Client]:
    """
    Create a Supabase client if env vars are set, otherwise return None.

    We support both SUPABASE_SERVICE_ROLE_KEY and SUPABASE_SECRET_KEY.
    """
    import os

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


# -------------------------------------------------------------------------
# document_files operations
# -------------------------------------------------------------------------


def fetch_uploaded_pdfs(
    client: Client,
    limit: int = MAX_DOCS_PER_RUN,
) -> List[Dict[str, Any]]:
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


# -------------------------------------------------------------------------
# document_pages operations
# -------------------------------------------------------------------------


def upsert_document_page(
    client: Client,
    document_id: Any,
    page_number: int,
    image_bucket: str,
    image_object_path: str,
    status: str,
    page_sha256: Optional[str] = None,
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

    if page_sha256 is not None:
        row["page_sha256"] = page_sha256

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
