# app/path_utils.py

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DOC_NAS_ROOT_ENV


def get_nas_root() -> Path:
    """
    Resolve the NAS root path for raw + derived files.

    DOC_NAS_ROOT should point at the root described in the
    doc-control-storage-layout-v1 spec (e.g. /data/doc_control).
    """
    import os

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

    return nas_root / storage_path


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

    return f"derived/pages/{stage}/{parent}/{document_id}/p{page_number}.png"


def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory for a file path exists."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[error] Failed to create directory {path.parent}: {exc}", file=sys.stderr)
        raise
