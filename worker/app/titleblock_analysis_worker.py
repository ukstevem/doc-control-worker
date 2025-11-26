import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from supabase import Client, create_client

# ---------------------------------------------------------------------
# Configuration (Power of 10: explicit bounds)
# ---------------------------------------------------------------------

MAX_PAGES_PER_RUN = 20          # Max pages to process in one run
MAX_LINES_PER_PAGE = 200        # Max Hough lines to consider per page
FINGERPRINT_BINS = 12           # Histogram bins for horizontal/vertical lines

DOC_NAS_ROOT_ENV = "DOC_NAS_ROOT"


# ---------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------

def create_supabase_client() -> Optional[Client]:
    """Create a Supabase client or return None on failure."""
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


# ---------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------

def get_nas_root() -> Path:
    """Resolve and validate the NAS root path used for images."""
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
# DB access
# ---------------------------------------------------------------------

def fetch_pages_with_titleblock(client: Client, limit: int = MAX_PAGES_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch a bounded list of document_pages rows that:

      - have image_object_path
      - have titleblock_x/y/width/height set
      - have no titleblock_fingerprint yet
    """
    try:
        response = (
            client.table("document_pages")
            .select(
                "id, document_id, page_number, image_object_path, "
                "titleblock_x, titleblock_y, titleblock_width, titleblock_height, "
                "titleblock_fingerprint"
            )
            .is_("titleblock_fingerprint", "null")
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch pages for titleblock analysis: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    results: List[Dict[str, Any]] = []

    for row in rows:
        # Filter in Python to avoid 'not null' complexity in SQL builder
        if (
            row.get("image_object_path") and
            row.get("titleblock_x") is not None and
            row.get("titleblock_y") is not None and
            row.get("titleblock_width") is not None and
            row.get("titleblock_height") is not None
        ):
            results.append(row)

    if not results:
        print("[info] No pages with titleblock needing fingerprint")
    else:
        print(f"[info] Fetched {len(results)} page(s) for titleblock fingerprinting")

    return results


def update_page_fingerprint(
    client: Client,
    page_id: Any,
    fingerprint: Dict[str, Any],
    version: int,
) -> None:
    """Store the computed fingerprint on document_pages."""
    update_data: Dict[str, Any] = {
        "titleblock_fingerprint": fingerprint,
        "titleblock_fingerprint_version": version,
        "processing_error": None,
    }

    try:
        (
            client.table("document_pages")
            .update(update_data)
            .eq("id", page_id)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to update document_pages(id={page_id}) fingerprint: {exc}",
            file=sys.stderr,
        )


def update_page_error(client: Client, page_id: Any, message: str) -> None:
    """Record an error on document_pages for this analysis step."""
    update_data: Dict[str, Any] = {
        "processing_error": message[:500],
    }

    try:
        (
            client.table("document_pages")
            .update(update_data)
            .eq("id", page_id)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to update document_pages(id={page_id}) on error: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------
# Hough-based fingerprint
# ---------------------------------------------------------------------

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load a grayscale image from disk."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[error] Failed to load image: {image_path}", file=sys.stderr)
        return None
    return img


def compute_titleblock_fingerprint(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> Dict[str, Any]:
    """
    Compute a simple Hough-line fingerprint for the titleblock region.

    Returns a JSON-serializable dict with:
      - version
      - width, height
      - horizontal_bins, vertical_bins
    """
    # Clamp ROI to image bounds
    h_img, w_img = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    roi_w = max(0, x1 - x0)
    roi_h = max(0, y1 - y0)

    if roi_w == 0 or roi_h == 0:
        return {
            "version": 1,
            "width": roi_w,
            "height": roi_h,
            "horizontal_bins": [0] * FINGERPRINT_BINS,
            "vertical_bins": [0] * FINGERPRINT_BINS,
        }

    roi = img[y0:y1, x0:x1]

    # Edge detection
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)

    # Probabilistic Hough transform for line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=50,
        minLineLength=max(roi_w, roi_h) * 0.1,
        maxLineGap=max(roi_w, roi_h) * 0.05,
    )

    horiz_bins = [0] * FINGERPRINT_BINS
    vert_bins = [0] * FINGERPRINT_BINS

    if lines is not None:
        # Limit the number of lines considered to keep work bounded
        limited = lines[:MAX_LINES_PER_PAGE]
        for line in limited:
            x1_line, y1_line, x2_line, y2_line = line[0]
            dx = abs(x2_line - x1_line)
            dy = abs(y2_line - y1_line)

            if dx + dy == 0:
                continue

            if dx >= dy:
                # Treat as horizontal
                y_mid = (y1_line + y2_line) / 2.0
                y_norm = y_mid / float(roi_h)
                bin_index = int(y_norm * FINGERPRINT_BINS)
                bin_index = max(0, min(FINGERPRINT_BINS - 1, bin_index))
                horiz_bins[bin_index] += 1
            else:
                # Treat as vertical
                x_mid = (x1_line + x2_line) / 2.0
                x_norm = x_mid / float(roi_w)
                bin_index = int(x_norm * FINGERPRINT_BINS)
                bin_index = max(0, min(FINGERPRINT_BINS - 1, bin_index))
                vert_bins[bin_index] += 1

    fingerprint: Dict[str, Any] = {
        "version": 1,
        "width": roi_w,
        "height": roi_h,
        "horizontal_bins": horiz_bins,
        "vertical_bins": vert_bins,
    }
    return fingerprint


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def process_page(client: Client, nas_root: Path, row: Dict[str, Any]) -> None:
    """
    Process a single document_pages row:

      - Load image from image_object_path
      - Crop to titleblock
      - Compute Hough-based fingerprint
      - Store fingerprint and version
    """
    page_id = row.get("id")
    image_rel = row.get("image_object_path")
    x = row.get("titleblock_x")
    y = row.get("titleblock_y")
    w = row.get("titleblock_width")
    h = row.get("titleblock_height")

    if page_id is None or image_rel is None:
        print(f"[error] Page row missing id or image_object_path: {row}", file=sys.stderr)
        return

    if None in (x, y, w, h):
        print(f"[error] Page id={page_id} missing titleblock bbox; skipping", file=sys.stderr)
        return

    image_path = nas_root / image_rel
    if not image_path.is_file():
        message = f"Image file not found at {image_path}"
        print(f"[error] {message} (page_id={page_id})", file=sys.stderr)
        update_page_error(client, page_id, message)
        return

    img = load_image(image_path)
    if img is None:
        update_page_error(client, page_id, "Failed to load image")
        return

    fingerprint = compute_titleblock_fingerprint(img, int(x), int(y), int(w), int(h))
    update_page_fingerprint(client, page_id, fingerprint, version=1)

    print(f"[info] Page id={page_id} fingerprint computed")


def run_once() -> int:
    """Run one bounded batch of titleblock analysis and exit."""
    client = create_supabase_client()
    if client is None:
        return 1

    nas_root = get_nas_root()
    print(f"[info] Using NAS root: {nas_root}")

    pages = fetch_pages_with_titleblock(client, limit=MAX_PAGES_PER_RUN)
    if not pages:
        return 0

    for row in pages:
        process_page(client, nas_root, row)

    return 0


def main() -> int:
    """Entry point. Kept small for testability."""
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
