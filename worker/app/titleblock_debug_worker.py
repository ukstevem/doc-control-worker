import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from supabase import Client, create_client

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_PAGES_PER_RUN = 5
MAX_LINES_PER_PAGE = 200
GRID_MERGE_TOLERANCE = 5.0

DOC_NAS_ROOT_ENV = "DOC_NAS_ROOT"
DEBUG_SUBDIR = "derived/debug"  # under NAS root


# ---------------------------------------------------------------------
# Supabase + env helpers
# ---------------------------------------------------------------------

def create_supabase_client() -> Optional[Client]:
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


def get_nas_root() -> Path:
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
# DB: which page to debug?
# ---------------------------------------------------------------------

def fetch_pages_with_titleblock(client: Client, limit: int = MAX_PAGES_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch document_pages rows we actually care about for Hough debugging:

      - status in ('tagged', 'Tagged')
      - has an image_object_path
      - has titleblock_x/y/width/height set
    """
    try:
        response = (
            client.table("document_pages")
            .select(
                "id, document_id, page_number, status, "
                "image_object_path, "
                "titleblock_x, titleblock_y, titleblock_width, titleblock_height"
            )
            .in_("status", ["tagged", "Tagged"])
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch document_pages: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    print(f"[info] fetched {len(rows)} tagged document_pages row(s) from Supabase")

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        page_id = row.get("id")

        if not row.get("image_object_path"):
            print(f"[debug] page_id={page_id} skipped: no image_object_path")
            continue

        if any(
            row.get(name) is None
            for name in ("titleblock_x", "titleblock_y", "titleblock_width", "titleblock_height")
        ):
            print(f"[debug] page_id={page_id} skipped: missing titleblock bbox")
            continue

        print(f"[debug] page_id={page_id} is a candidate (status={row.get('status')})")
        candidates.append(row)

    print(f"[info] {len(candidates)} tagged page(s) have image + titleblock bbox")
    return candidates

# ---------------------------------------------------------------------
# Hough helpers
# ---------------------------------------------------------------------

def dedup_positions(values: List[float], tolerance: float) -> List[float]:
    """Merge near-duplicate positions into a single sorted list."""
    if not values:
        return []

    sorted_vals = sorted(values)
    merged: List[float] = [sorted_vals[0]]

    for v in sorted_vals[1:]:
        if abs(v - merged[-1]) <= tolerance:
            merged[-1] = 0.5 * (merged[-1] + v)
        else:
            merged.append(v)

    return merged


def detect_hough_lines(roi_gray: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[float]]:
    """
    Run Canny + HoughLinesP on the ROI and keep only long, nearly-horizontal /
    nearly-vertical lines.

    Horizontals must span most of the width (to avoid text baselines).
    Verticals can be shorter (so we keep internal column dividers).
    """
    h, w = roi_gray.shape[:2]
    if h == 0 or w == 0:
        return [], [], []

    # Blur to suppress fine text detail
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # Strong Canny so we only see strong edges
    edges = cv2.Canny(roi_blur, 120, 250, apertureSize=3)

    # Base minimum length to even consider from Hough
    min_span = min(w, h)
    base_min_len = 0.25 * min_span   # 25% of smaller dimension

    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=90,
        minLineLength=base_min_len,
        maxLineGap=0.03 * max(w, h),
    )

    segments: List[Tuple[int, int, int, int]] = []
    x_positions: List[float] = []
    y_positions: List[float] = []

    if lines is not None:
        for line in lines[:MAX_LINES_PER_PAGE]:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = (dx * dx + dy * dy) ** 0.5
            if length < base_min_len:
                continue

            abs_dx = abs(dx)
            abs_dy = abs(dy)
            if abs_dx + abs_dy == 0:
                continue

            # Only lines within ~10° of horizontal/vertical
            max_slope = 0.18  # tan(10°) ≈ 0.176

            if abs_dx >= abs_dy:
                # Candidate horizontal
                if abs_dy / max(abs_dx, 1.0) > max_slope:
                    continue
                # Must span *most* of the width to avoid text baselines
                if length < 0.8 * w:
                    continue
                segments.append((x1, y1, x2, y2))
                y_mid = 0.5 * (y1 + y2)
                y_positions.append(y_mid)
            else:
                # Candidate vertical
                if abs_dx / max(abs_dy, 1.0) > max_slope:
                    continue
                # Allow shorter verticals so internal columns survive
                if length < 0.05 * h:
                    continue
                segments.append((x1, y1, x2, y2))
                x_mid = 0.5 * (x1 + x2)
                x_positions.append(x_mid)

    # Merge nearby positions (20px) so each real line shows once
    x_lines = dedup_positions(x_positions, tolerance=20.0)
    y_lines = dedup_positions(y_positions, tolerance=20.0)

    print(f"[info] Hough kept {len(segments)} long grid-like segment(s) in ROI")
    print(f"[info] x_lines={x_lines}")
    print(f"[info] y_lines={y_lines}")

    return segments, x_lines, y_lines



# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------

def ensure_debug_dir(nas_root: Path) -> Path:
    debug_dir = nas_root / DEBUG_SUBDIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def draw_debug_images(
    nas_root: Path,
    page_row: Dict[str, Any],
    x_lines: List[float],
    y_lines: List[float],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    """
    Draw:
      - full page image with ROI rectangle
      - ROI image with synthetic grid lines from x_lines / y_lines

    Save both into NAS_ROOT/derived/debug.
    """
    page_id = page_row.get("id")
    image_rel = page_row.get("image_object_path")

    image_path = nas_root / image_rel
    img_color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"[error] Failed to load image {image_path}", file=sys.stderr)
        return

    # 1) Full image with ROI rectangle (green)
    full = img_color.copy()
    cv2.rectangle(full, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # 2) ROI with idealised grid lines (red)
    roi_color = img_color[y0:y1, x0:x1].copy()
    roi_h, roi_w = roi_color.shape[:2]

    # Vertical lines (x_lines)
    for x in x_lines:
        x_i = int(round(x))
        if 0 <= x_i < roi_w:
            cv2.line(roi_color, (x_i, 0), (x_i, roi_h - 1), (0, 0, 255), 1)

    # Horizontal lines (y_lines)
    for y in y_lines:
        y_i = int(round(y))
        if 0 <= y_i < roi_h:
            cv2.line(roi_color, (0, y_i), (roi_w - 1, y_i), (0, 0, 255), 1)

    debug_dir = ensure_debug_dir(nas_root)

    full_out = debug_dir / f"titleblock_full_{page_id}.png"
    roi_out = debug_dir / f"titleblock_roi_grid_{page_id}.png"

    cv2.imwrite(str(full_out), full)
    cv2.imwrite(str(roi_out), roi_color)

    print(f"[info] Wrote full debug image: {full_out}")
    print(f"[info] Wrote ROI grid image:  {roi_out}")


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def process_page(nas_root: Path, page_row: Dict[str, Any]) -> None:
    page_id = page_row.get("id")
    image_rel = page_row.get("image_object_path")

    image_path = nas_root / image_rel
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"[error] Failed to load image {image_path}", file=sys.stderr)
        return

    img_h, img_w = img_gray.shape[:2]

    tb_x_rel = page_row.get("titleblock_x")
    tb_y_rel = page_row.get("titleblock_y")
    tb_w_rel = page_row.get("titleblock_width")
    tb_h_rel = page_row.get("titleblock_height")

    if None in (tb_x_rel, tb_y_rel, tb_w_rel, tb_h_rel):
        print(f"[error] page_id={page_id} missing titleblock rel bbox", file=sys.stderr)
        return

    # Treat them as 0–1
    x0 = int(round(float(tb_x_rel) * img_w))
    y0 = int(round(float(tb_y_rel) * img_h))
    x1 = int(round((float(tb_x_rel) + float(tb_w_rel)) * img_w))
    y1 = int(round((float(tb_y_rel) + float(tb_h_rel)) * img_h))

    # Clamp
    x0 = max(0, min(img_w, x0))
    y0 = max(0, min(img_h, y0))
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))

    if x1 <= x0 or y1 <= y0:
        print(f"[error] ROI is empty after clamping (page_id={page_id})", file=sys.stderr)
        return

    roi_gray = img_gray[y0:y1, x0:x1]

    _segments, x_lines, y_lines = detect_hough_lines(roi_gray)

    if not _segments:
        print(f"[warn] No Hough line segments found for page_id={page_id}")

    draw_debug_images(nas_root, page_row, x_lines, y_lines, x0, y0, x1, y1)

def run_once() -> int:
    client = create_supabase_client()
    if client is None:
        print("[error] Supabase client not created", file=sys.stderr)
        return 1

    nas_root = get_nas_root()
    print(f"[info] Using NAS root: {nas_root}")

    pages = fetch_pages_with_titleblock(client, limit=MAX_PAGES_PER_RUN)
    if not pages:
        print("[info] No pages with titleblock bbox found")
        return 0

    # For now, just process the first page for clarity
    page = pages[0]
    print(f"[info] Debugging page_id={page.get('id')} status={page.get('status')}")

    process_page(nas_root, page)
    return 0


def main() -> int:
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
