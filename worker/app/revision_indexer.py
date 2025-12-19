# app/revision_indexer.py

import sys
from typing import Any, Dict, List, Optional

from supabase import Client


def index_revision_for_page(client: Client, page_id: Any) -> None:
    """
    Update document_state and linking fields for a single document_pages row,
    based purely on other rows in document_pages.

    Rules:
      - Group by (projectnumber OR enquirynumber, drawing_number).
      - First time we see this drawing_number in a scope:
          -> document_state = 'current'
      - Same number + same revision:
          -> if same page_sha256 as an existing page => duplicate_of_page_id
             and document_state = same as that page
          -> if different hash => content_mismatch = true, document_state = 'pending'
      - Same number, different revision:
          -> document_state = 'pending'
          -> supersedes_page_id points at current version if present
    """
    if page_id is None:
        return

    # Fetch the current page row with the fields we care about
    try:
        response = (
            client.table("document_pages")
            .select(
                "id, projectnumber, enquirynumber, drawing_number, "
                "revision, page_sha256, document_state"
            )
            .eq("id", page_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] index_revision_for_page: failed to load page_id={page_id}: {exc}",
            file=sys.stderr,
        )
        return

    data = getattr(response, "data", None) or []
    if not data:
        print(
            f"[warn] index_revision_for_page: no document_pages row found for id={page_id}",
            file=sys.stderr,
        )
        return

    page_row: Dict[str, Any] = data[0]
    doc_number = page_row.get("drawing_number")
    revision = page_row.get("revision")
    page_hash = page_row.get("page_sha256")
    projectnumber = page_row.get("projectnumber")
    enquirynumber = page_row.get("enquirynumber")

    if not doc_number:
        # No drawing/document number → nothing to index
        return

    # Derive a scope: projectnumber preferred, else enquirynumber
    scope_key = projectnumber or enquirynumber
    scope_filter: Optional[str] = scope_key if scope_key else None

    # Fetch other pages with same drawing_number within scope (if any)
    try:
        query = (
            client.table("document_pages")
            .select(
                "id, projectnumber, enquirynumber, drawing_number, revision, "
                "document_state, page_sha256"
            )
            .eq("drawing_number", doc_number)
        )
        if scope_filter:
            if projectnumber:
                query = query.eq("projectnumber", projectnumber)
            else:
                query = query.eq("enquirynumber", enquirynumber)

        response = query.execute()
    except Exception as exc:
        print(
            f"[error] index_revision_for_page: query failed for drawing_number={doc_number!r}: {exc}",
            file=sys.stderr,
        )
        return

    rows = getattr(response, "data", None) or []
    others: List[Dict[str, Any]] = [r for r in rows if r.get("id") != page_id]

    # Case 1: no other pages with this number in this scope → first version
    if not others:
        try:
            (
                client.table("document_pages")
                .update(
                    {
                        "document_state": "current",
                        "duplicate_of_page_id": None,
                        "supersedes_page_id": None,
                        "superseded_by_page_id": None,
                        "content_mismatch": False,
                    }
                )
                .eq("id", page_id)
                .execute()
            )
            print(
                f"[info] page_id={page_id}: first version of {doc_number!r} "
                f"in scope={scope_filter!r}, marked as current"
            )
        except Exception as exc:
            print(
                f"[error] Failed to set document_state=current for page_id={page_id}: {exc}",
                file=sys.stderr,
            )
        return

    revision_str = revision or ""
    same_rev = [r for r in others if (r.get("revision") or "") == revision_str]
    diff_rev = [r for r in others if (r.get("revision") or "") != revision_str]

    # Case 2: same doc number + same revision already exists
    if same_rev:
        dup = None
        if page_hash:
            for r in same_rev:
                if r.get("page_sha256") and r["page_sha256"] == page_hash:
                    dup = r
                    break

        if dup:
            parent_state = dup.get("document_state") or "current"
            try:
                (
                    client.table("document_pages")
                    .update(
                        {
                            "document_state": parent_state,
                            "duplicate_of_page_id": dup.get("id"),
                            "content_mismatch": False,
                        }
                    )
                    .eq("id", page_id)
                    .execute()
                )
                print(
                    f"[info] page_id={page_id}: duplicate of page_id={dup.get('id')} "
                    f"for drawing_number={doc_number!r}, revision={revision!r}"
                )
            except Exception as exc:
                print(
                    f"[error] Failed to mark page_id={page_id} as duplicate: {exc}",
                    file=sys.stderr,
                )
            return

        # Same revision but different hash → questionable
        try:
            (
                client.table("document_pages")
                .update(
                    {
                        "document_state": "pending",
                        "duplicate_of_page_id": None,
                        "content_mismatch": True,
                    }
                )
                .eq("id", page_id)
                .execute()
            )
            print(
                f"[info] page_id={page_id}: same revision as existing pages for "
                f"{doc_number!r} but different content; marked pending + mismatch"
            )
        except Exception as exc:
            print(
                f"[error] Failed to mark page_id={page_id} as pending/mismatch: {exc}",
                file=sys.stderr,
            )
        return

    # Case 3: same doc number but different revision only
    current_candidates = [r for r in diff_rev if r.get("document_state") == "current"]
    parent = current_candidates[0] if current_candidates else None
    parent_id = parent.get("id") if parent else None

    try:
        (
            client.table("document_pages")
            .update(
                {
                    "document_state": "pending",
                    "supersedes_page_id": parent_id,
                    "content_mismatch": False,
                }
            )
            .eq("id", page_id)
            .execute()
        )
        print(
            f"[info] page_id={page_id}: new revision candidate for {doc_number!r}, "
            f"supersedes_page_id={parent_id}"
        )
    except Exception as exc:
        print(
            f"[error] Failed to set document_state=pending for page_id={page_id}: {exc}",
            file=sys.stderr,
        )
