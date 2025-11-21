import os
import sys
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from supabase import create_client, Client

def create_supabase_client() -> Client | None:
    """Create a Supabase client if env vars are set, otherwise return None."""
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        print("[info] SUPABASE_URL or SUPABASE_SECRET_KEY not set; skipping DB work")
        return None

    try:
        client: Client = create_client(url, secret_key)
    except Exception as exc:
        print(f"[error] Failed to create Supabase client: {exc}", file=sys.stderr)
        return None

    return client

def ping_document_files_table(client: Client) -> None:
    """Print the number of rows in document_files (bounded)."""
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

    # supabase-py v2 returns a struct with .data and .count
    count = getattr(response, "count", None)
    print(f"[info] document_files table reachable; count={count}")


def get_env(name: str, default: str | None = None) -> str:
    """Get an env var or exit with a clear error."""
    value = os.getenv(name, default)
    if value is None:
        print(f"[error] Environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return value


def list_pdf_files(root: Path, limit: int = 10) -> List[Path]:
    """Return a bounded list of PDF files from the root (non-recursive)."""
    if not root.exists():
        print(f"[error] DOC_NAS_ROOT does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    files: List[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".pdf":
            files.append(entry)
        if len(files) >= limit:
            # Power-of-10 style: always bound loops
            break
    return files


def ensure_preview_dir(base: Path) -> Path:
    """Ensure the preview directory exists and return it."""
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[error] Failed to create preview directory {base}: {exc}", file=sys.stderr)
        sys.exit(1)
    return base


def render_first_page(pdf_path: Path, preview_dir: Path) -> Path | None:
    """
    Render the first page of a PDF to a JPEG file in preview_dir.

    Returns the path to the rendered image, or None on failure.
    """
    output_filename = pdf_path.stem + "_p1.jpg"
    output_path = preview_dir / output_filename

    print(f"[info] Rendering first page of {pdf_path.name} -> {output_path}")

    try:
        # Only render page 1 to keep work bounded
        images = convert_from_path(
            pdf_path,
            first_page=1,
            last_page=1,
            fmt="jpeg",
        )
    except Exception as exc:
        print(f"[error] Failed to convert {pdf_path}: {exc}", file=sys.stderr)
        return None

    if not images:
        print(f"[error] No pages returned when converting {pdf_path}", file=sys.stderr)
        return None

    image = images[0]
    try:
        image.save(output_path, "JPEG")
    except Exception as exc:
        print(f"[error] Failed to save preview for {pdf_path}: {exc}", file=sys.stderr)
        return None

    return output_path


def main() -> int:
    """Small, testable main function."""
    doc_root_str = os.getenv("DOC_NAS_ROOT", "/data/input")
    preview_root_str = os.getenv("PREVIEW_ROOT", "/data/state/previews")

    doc_root = Path(doc_root_str)
    preview_root = ensure_preview_dir(Path(preview_root_str))

    # Supabase connectivity check (optional)
    supabase = create_supabase_client()
    if supabase is not None:
        ping_document_files_table(supabase)
        
    print(f"[info] Using DOC_NAS_ROOT={doc_root}")
    print(f"[info] Using PREVIEW_ROOT={preview_root}")

    pdf_files = list_pdf_files(doc_root)
    if not pdf_files:
        print("[info] No PDF files found in DOC_NAS_ROOT")
        return 0

    print("[info] Found PDF files:")
    for f in pdf_files:
        print(f"  - {f.name}")

    # Render previews for a small, bounded number of PDFs
    for pdf in pdf_files:
        preview = render_first_page(pdf, preview_root)
        if preview is not None:
            print(f"[info] Preview written to: {preview}")
        else:
            print(f"[warn] Preview not created for: {pdf.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
