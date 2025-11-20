import os
import sys
from pathlib import Path
from typing import List


def get_env(name: str, default: str | None = None) -> str:
    """Get an env var or exit with a clear error."""
    value = os.getenv(name, default)
    if value is None:
        print(f"[error] Environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return value


def list_pdf_files(root: Path) -> List[Path]:
    """Return a bounded list of PDF files from the NAS root (non-recursive)."""
    if not root.exists():
        print(f"[error] DOC_NAS_ROOT does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    files: List[Path] = []
    for entry in root.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".pdf":
            files.append(entry)
        if len(files) >= 100:
            # Power-of-10 style: always bound loops
            break
    return files


def main() -> int:
    """Small, testable main function."""
    doc_root_str = os.getenv("DOC_NAS_ROOT", "/data/input")
    doc_root = Path(doc_root_str)

    print(f"[info] Using DOC_NAS_ROOT={doc_root}")

    pdf_files = list_pdf_files(doc_root)
    if not pdf_files:
        print("[info] No PDF files found in DOC_NAS_ROOT")
        return 0

    print("[info] Found PDF files:")
    for f in pdf_files:
        print(f"  - {f.name}")

    # TODO: next steps (separate chat):
    #  - connect to Supabase using SUPABASE_URL + SUPABASE_SERVICE_KEY
    #  - render page 1 using pdf2image
    #  - insert rows into document_files / document_pages

    return 0


if __name__ == "__main__":
    # Single, simple control flow entrypoint
    sys.exit(main())
