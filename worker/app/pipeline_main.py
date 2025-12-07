"""
Simple test-time orchestrator to run the end-to-end title-block pipeline:

1) Render first-page PNGs for any 'uploaded' PDFs (app.main.run_once).
2) Run grid detection on one tagged page (titleblock_grid_worker.run_from_supabase).
3) Run OCR + template creation for tagged pages (titleblock_extract_worker.run_once).
4) Run template matching for remaining pages (titleblock_match_worker.run_once).

This is deliberately small and thin; each worker still owns its own bounds
and error handling.
"""

import sys

# Local-package imports (this file lives in the same 'app' package)
from . import main as upload_worker
from . import titleblock_grid_worker
from . import titleblock_extract_worker
from . import titleblock_match_worker


def run_pipeline() -> int:
    """
    Run the full pipeline once.

    Returns an OR-ed exit status from the individual workers
    (0 = everything OK, non-zero if any stage signalled an error).
    """
    status = 0

    # ------------------------------------------------------------------
    # Step 1: render first-page PNGs for new PDFs
    # ------------------------------------------------------------------
    print("[PIPELINE] Step 1: render uploaded PDFs → first-page PNGs")
    try:
        # main.run_once() already prints its own logs and returns an int.
        s = upload_worker.run_once()
        if s is None:
            s = 0
        status |= int(s)
    except Exception as exc:
        print(f"[PIPELINE] Step 1 failed: {exc}", file=sys.stderr)
        status |= 1

    # ------------------------------------------------------------------
    # Step 2: grid detection on tagged pages
    # ------------------------------------------------------------------
    print("[PIPELINE] Step 2: grid detection for tagged pages")
    try:
        # This processes at most one tagged page per call.
        titleblock_grid_worker.run_from_supabase()
    except Exception as exc:
        print(f"[PIPELINE] Step 2 failed: {exc}", file=sys.stderr)
        status |= 1

    # ------------------------------------------------------------------
    # Step 3: OCR + template creation
    # ------------------------------------------------------------------
    print("[PIPELINE] Step 3: OCR + template creation for tagged pages")
    try:
        s = titleblock_extract_worker.run_once()
        if s is None:
            s = 0
        status |= int(s)
    except Exception as exc:
        print(f"[PIPELINE] Step 3 failed: {exc}", file=sys.stderr)
        status |= 1

    # ------------------------------------------------------------------
    # Step 4: template matching for remaining pages
    # ------------------------------------------------------------------
    print("[PIPELINE] Step 4: template matching for remaining pages")
    try:
        s = titleblock_match_worker.run_once()
        if s is None:
            s = 0
        status |= int(s)
    except Exception as exc:
        print(f"[PIPELINE] Step 4 failed: {exc}", file=sys.stderr)
        status |= 1

    return status


def main() -> int:
    """CLI entry point for `python -m app.pipeline_main`."""
    return run_pipeline()


if __name__ == "__main__":
    sys.exit(main())
