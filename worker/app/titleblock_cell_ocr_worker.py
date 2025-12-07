"""
titleblock_cell_ocr_worker.py

Given:
- A page image (PNG/JPEG)
- A grid JSON from titleblock_grid_worker.py
- One or more click points inside the title-block

This worker:
1. Crops the title-block from the page using the grid JSON crop_offset / crop_size.
2. For each click (in page-normalised coords, 0–1):
   - Converts to title-block pixels.
   - Finds the horizontal band (row) the click is in.
   - Detects vertical lines *within that band* only.
   - Computes a cell bounding box that contains the click.
3. Writes debug images of each cell for visual inspection.
4. (Optional) Runs OCR on the cell and prints the text.

Supabase integration is kept minimal:
- Uses supabase_client.fetch_next_tagged_page() to find a 'tagged' page.
- A placeholder fetch_titleblock_clicks_for_page() shows how to fetch clicks;
  you should adapt the table / column names to your actual schema.

CLI usage (manual test mode):

    python -m app.titleblock_cell_ocr_worker \
      --input /data/input/.../p1.png \
      --grid-json /data/input/.../p1_grid.json \
      --click-x 0.5 --click-y 0.9 \
      --debug-dir /data/input/.../debug_cells

Supabase mode (no args):

    python -m app.titleblock_cell_ocr_worker

This will:
- Get the next 'tagged' page via Supabase.
- Infer the grid JSON path from the image path.
- Load click points from Supabase (see fetch_titleblock_clicks_for_page()).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.supabase_client import (
    load_supabase_config_from_env,
    create_supabase_client,
    fetch_next_tagged_page,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClickInTitleblock:
    click: ClickPoint
    x_tb: float
    y_tb: float

@dataclass
class GridInfo:
    crop_offset_x: int
    crop_offset_y: int
    crop_width: int
    crop_height: int
    vertical_lines: List[int]
    horizontal_lines: List[int]

@dataclass
class AreaBox:
    field_name: str
    x_rel: float
    y_rel: float
    width_rel: float
    height_rel: float

@dataclass
class ClickPoint:
    """
    Represents a single click inside the full page image.

    click_x_norm / click_y_norm are 0–1 normalised coordinates
    relative to the full page width/height.
    """
    id: str
    field_name: str
    click_x_norm: float
    click_y_norm: float


@dataclass
class CellExtractionConfig:
    input_path: Path
    grid_json_path: Path
    click_x_norm: float
    click_y_norm: float
    debug_dir: Optional[Path]


# ---------------------------------------------------------------------------
# Basic image helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to load image at {path}"
        raise RuntimeError(msg)
    return image


def prepare_binary(titleblock_img: np.ndarray) -> np.ndarray:
    """Convert title-block crop to binary (lines as white)."""
    if titleblock_img.ndim == 3:
        gray = cv2.cvtColor(titleblock_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = titleblock_img

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def load_grid_json(path: Path) -> GridInfo:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    offset = data.get("crop_offset", {})
    size = data.get("crop_size", {})

    verticals = data.get("vertical_lines", []) or []
    horizontals = data.get("horizontal_lines", []) or []

    return GridInfo(
        crop_offset_x=int(offset.get("x", 0)),
        crop_offset_y=int(offset.get("y", 0)),
        crop_width=int(size.get("width", 0)),
        crop_height=int(size.get("height", 0)),
        vertical_lines=[int(v) for v in verticals],
        horizontal_lines=[int(h) for h in horizontals],
    )


# ---------------------------------------------------------------------------
# Band / cell geometry
# ---------------------------------------------------------------------------

def build_bands(
    horizontals: List[int],
    total_height: int,
) -> List[Tuple[int, int]]:
    """
    Build (y0, y1) bands from a list of horizontal line positions.
    Bands cover [0, total_height-1].
    """
    if not horizontals:
        return [(0, total_height - 1)]

    h_sorted = sorted(horizontals)
    edges = [0] + h_sorted + [total_height - 1]

    bands: List[Tuple[int, int]] = []
    for i in range(len(edges) - 1):
        y0 = edges[i]
        y1 = edges[i + 1]
        if y1 <= y0:
            continue
        bands.append((y0, y1))
    return bands


def find_band_for_y(
    bands: List[Tuple[int, int]],
    y: float,
) -> Optional[Tuple[int, int]]:
    """Find the band (y0, y1) that contains y."""
    for y0, y1 in bands:
        if y0 <= y <= y1:
            return y0, y1
    return None


def detect_band_verticals(
    binary: np.ndarray,
    band_y0: int,
    band_y1: int,
    min_band_fraction: float = 0.6,
) -> List[int]:
    """
    Detect vertical lines within a single band [band_y0, band_y1].

    - Looks only at that vertical strip.
    - Counts 'ink' per column.
    - Requires at least min_band_fraction * band_height pixels set to treat
      a column as vertical.
    - Groups contiguous columns into line centres.
    """
    height, width = binary.shape

    y0 = max(0, min(band_y0, height - 1))
    y1 = max(0, min(band_y1, height))
    if y1 <= y0:
        return []

    # Small margin to avoid sitting exactly on the horizontal line
    band_margin = 2
    y0_m = max(0, y0 + band_margin)
    y1_m = max(y0_m, y1 - band_margin)
    if y1_m <= y0_m:
        return []

    band = binary[y0_m:y1_m, :]
    band_height = y1_m - y0_m

    col_sum = (band > 0).sum(axis=0).astype(np.float32)
    threshold = band_height * min_band_fraction
    indices = np.where(col_sum >= threshold)[0]

    if indices.size == 0:
        return []

    lines: List[int] = []
    start = prev = int(indices[0])

    for idx in indices[1:]:
        idx_int = int(idx)
        if idx_int == prev + 1:
            prev = idx_int
            continue
        center = (start + prev) // 2
        lines.append(center)
        start = prev = idx_int

    center_last = (start + prev) // 2
    lines.append(center_last)
    return lines


def compute_cell_bbox_for_click(
    binary_tb: np.ndarray,
    horizontals: List[int],
    verticals: List[int],
    click_x_tb: float,
    click_y_tb: float,
    margin: int = 3,
) -> Tuple[int, int, int, int]:
    """
    Given a click in title-block coords (x_tb, y_tb), compute a cell bbox
    using the grid lines directly:

    - Use horizontal_lines to find the band (top/bottom) around the click.
    - Use vertical_lines to find nearest left/right boundaries around the click.
    - Apply a small margin inward.

    All coordinates are in title-block pixel space.
    """
    tb_height, tb_width = binary_tb.shape

    h_sorted = sorted(horizontals)
    v_sorted = sorted(verticals)

    # --- Top / bottom: choose band from horizontals ---
    y_top = 0
    y_bottom = tb_height - 1

    for y in h_sorted:
        if y <= click_y_tb:
            y_top = y
        elif y > click_y_tb and y_bottom == tb_height - 1:
            y_bottom = y
            break

    # --- Left / right: choose boundaries from verticals ---
    x_left = 0
    x_right = tb_width - 1

    for x in v_sorted:
        if x <= click_x_tb:
            x_left = x
        elif x > click_x_tb and x_right == tb_width - 1:
            x_right = x
            break

    # --- Apply a small margin and clamp ---
    x0 = max(0, int(round(x_left)) + margin)
    x1 = min(tb_width - 1, int(round(x_right)) - margin)
    y0 = max(0, int(round(y_top)) + margin)
    y1 = min(tb_height - 1, int(round(y_bottom)) - margin)

    # Fallback if something collapses
    if x1 <= x0:
        x0 = max(0, int(round(x_left)))
        x1 = min(tb_width - 1, int(round(x_right)))
    if y1 <= y0:
        y0 = max(0, int(round(y_top)))
        y1 = min(tb_height - 1, int(round(y_bottom)))

    return x0, y0, x1, y1


def group_clicks_into_rows(
    clicks_tb: List[ClickInTitleblock],
    tb_height: int,
    row_merge_ratio: float = 0.05,
) -> List[List[ClickInTitleblock]]:
    """
    Group clicks into rows based on their y position in title-block coords.

    row_merge_ratio:
        Two clicks are considered in the same row if their y differs by less than
        row_merge_ratio * tb_height.
    """
    if not clicks_tb:
        return []

    clicks_sorted = sorted(clicks_tb, key=lambda c: c.y_tb)
    rows: List[List[ClickInTitleblock]] = []

    for c in clicks_sorted:
        if not rows:
            rows.append([c])
            continue

        last_row = rows[-1]
        mean_y = sum(x.y_tb for x in last_row) / float(len(last_row))
        if abs(c.y_tb - mean_y) <= row_merge_ratio * tb_height:
            last_row.append(c)
        else:
            rows.append([c])

    return rows


def compute_row_boundaries(
    rows: List[List[ClickInTitleblock]],
    tb_height: int,
) -> List[Tuple[int, int]]:
    """
    Given rows (each a list of ClickInTitleblock), compute (y0, y1) for each row.

    Row boundaries are midpoints between row centroids.
    """
    if not rows:
        return []

    row_means = [
        sum(c.y_tb for c in row) / float(len(row)) for row in rows
    ]

    row_bounds: List[Tuple[int, int]] = []
    for i, mean_y in enumerate(row_means):
        if i == 0:
            y0 = 0
        else:
            y0 = int(round(0.5 * (row_means[i - 1] + mean_y)))

        if i == len(row_means) - 1:
            y1 = tb_height - 1
        else:
            y1 = int(round(0.5 * (mean_y + row_means[i + 1])))

        y0 = max(0, min(y0, tb_height - 1))
        y1 = max(0, min(y1, tb_height - 1))
        if y1 <= y0:
            y1 = min(tb_height - 1, y0 + 1)

        row_bounds.append((y0, y1))

    return row_bounds

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def compute_column_boundaries_for_row(
    row_clicks: List[ClickInTitleblock],
    tb_width: int,
    margin: int = 3,
) -> dict[str, Tuple[int, int]]:
    """
    For a given row, compute left/right x boundaries for each field
    based on click x positions.

    - Sort clicks by x_tb.
    - Use midpoints between consecutive clicks as vertical cut lines.
    - First field: from left edge (0) to first cut.
    - Last field: from last cut to right edge (tb_width - 1).
    """
    if not row_clicks:
        return {}

    # Sort by x
    row_sorted = sorted(row_clicks, key=lambda c: c.x_tb)
    centers = [c.x_tb for c in row_sorted]

    # Compute split positions (midpoints between centers)
    splits: List[float] = []
    for i in range(len(centers) - 1):
        splits.append(0.5 * (centers[i] + centers[i + 1]))

    field_to_bounds: dict[str, Tuple[int, int]] = {}

    for idx, c in enumerate(row_sorted):
        if idx == 0:
            left = 0.0
        else:
            left = splits[idx - 1]

        if idx == len(row_sorted) - 1:
            right = float(tb_width - 1)
        else:
            right = splits[idx]

        x0 = max(0, int(round(left)) + margin)
        x1 = min(tb_width - 1, int(round(right)) - margin)
        if x1 <= x0:
            # Fallback: no margin
            x0 = max(0, int(round(left)))
            x1 = min(tb_width - 1, int(round(right)))

        field_name = c.click.field_name or c.click.id
        field_to_bounds[field_name] = (x0, x1)

    return field_to_bounds


# ---------------------------------------------------------------------------
# OCR (optional)
# ---------------------------------------------------------------------------

def ocr_cell(cell_img: np.ndarray) -> str:
    """
    Run OCR on a cell image using pytesseract if available.

    Returns stripped text, or "" if OCR is not available.
    """
    try:
        import pytesseract
    except ImportError:
        print("[warn] pytesseract not installed; skipping OCR for this cell")
        return ""

    config = "--psm 6"  # assume a block of text
    text = pytesseract.image_to_string(cell_img, config=config)
    return text.strip()


# ---------------------------------------------------------------------------
# Supabase helpers (clicks)
# ---------------------------------------------------------------------------

def fetch_titleblock_areas_for_page(
    client,
    page_id: str,
) -> List[AreaBox]:
    """
    Fetch boxed areas for a given page from document_pages.titleblock_fingerprint.

    Expects JSON like:

        {
          "areas": [
            {
              "field": "drawing_number",
              "x_rel": 0.10,
              "y_rel": 0.90,
              "width_rel": 0.62,
              "height_rel": 0.07
            },
            ...
          ],
          "version": 2
        }

    All *_rel values are 0–1 relative to the titleblock crop.
    """
    print(f"[info] fetching title-block areas for page_id={page_id}")

    response = (
        client.table("document_pages")
        .select("titleblock_fingerprint")
        .eq("id", page_id)
        .limit(1)
        .execute()
    )

    rows = getattr(response, "data", None) or []
    if not rows:
        print("[info] no document_pages row found for this page_id")
        return []

    row = rows[0]
    raw_fp = row.get("titleblock_fingerprint")

    if raw_fp is None:
        print("[info] titleblock_fingerprint is NULL; no areas stored yet")
        return []

    # raw_fp might be a JSON string or already a dict
    if isinstance(raw_fp, str):
        try:
            fp = json.loads(raw_fp)
        except json.JSONDecodeError as exc:
            print(f"[error] failed to parse titleblock_fingerprint JSON: {exc}")
            return []
    elif isinstance(raw_fp, dict):
        fp = raw_fp
    else:
        print(
            f"[error] titleblock_fingerprint has unexpected type {type(raw_fp)}; "
            "expected str or dict"
        )
        return []

    areas_data = fp.get("areas", [])
    if not isinstance(areas_data, list):
        print("[error] titleblock_fingerprint.areas is not a list")
        return []

    areas: List[AreaBox] = []
    for item in areas_data:
        try:
            field_name = str(item.get("field", ""))
            x_rel = float(item["x_rel"])
            y_rel = float(item["y_rel"])
            width_rel = float(item["width_rel"])
            height_rel = float(item["height_rel"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[warn] skipping malformed area entry {item!r}: {exc}")
            continue

        areas.append(
            AreaBox(
                field_name=field_name,
                x_rel=x_rel,
                y_rel=y_rel,
                width_rel=width_rel,
                height_rel=height_rel,
            )
        )

    print(f"[info] fetched {len(areas)} area box(es) from titleblock_fingerprint")
    return areas

def fetch_titleblock_clicks_for_page(
    client,
    page_id: str,
) -> List[ClickPoint]:
    """
    Fetch click points for a given page from document_pages.titleblock_fingerprint.

    Expects document_pages.titleblock_fingerprint to contain JSON like:

        {
          "clicks": [
            {"field": "drawing_number", "x_rel": 0.40, "y_rel": 0.95},
            {"field": "drawing_title",  "x_rel": 0.50, "y_rel": 0.68},
            ...
          ],
          "version": 1
        }

    x_rel / y_rel are 0–1 normalised to the full page.
    """
    print(f"[info] fetching title-block clicks for page_id={page_id}")

    response = (
        client.table("document_pages")
        .select("titleblock_fingerprint")
        .eq("id", page_id)
        .limit(1)
        .execute()
    )

    rows = getattr(response, "data", None) or []
    if not rows:
        print("[info] no document_pages row found for this page_id")
        return []

    row = rows[0]
    raw_fp = row.get("titleblock_fingerprint")

    if raw_fp is None:
        print("[info] titleblock_fingerprint is NULL; no clicks stored yet")
        return []

    # raw_fp might already be a dict (depending on Supabase client), or a JSON string
    if isinstance(raw_fp, str):
        try:
            fp = json.loads(raw_fp)
        except json.JSONDecodeError as exc:
            print(f"[error] failed to parse titleblock_fingerprint JSON: {exc}")
            return []
    elif isinstance(raw_fp, dict):
        fp = raw_fp
    else:
        print(
            f"[error] titleblock_fingerprint has unexpected type {type(raw_fp)}; "
            "expected str or dict"
        )
        return []

    clicks_data = fp.get("clicks", [])
    if not isinstance(clicks_data, list):
        print("[error] titleblock_fingerprint.clicks is not a list")
        return []

    clicks: List[ClickPoint] = []
    for item in clicks_data:
        try:
            field_name = str(item.get("field", ""))
            x_rel = float(item["x_rel"])
            y_rel = float(item["y_rel"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[warn] skipping malformed click entry {item!r}: {exc}")
            continue

        clicks.append(
            ClickPoint(
                id=f"{page_id}:{field_name}",
                field_name=field_name,
                click_x_norm=x_rel,
                click_y_norm=y_rel,
            )
        )

    print(f"[info] fetched {len(clicks)} click point(s) from titleblock_fingerprint")
    return clicks

# ---------------------------------------------------------------------------
# Debug output for cells
# ---------------------------------------------------------------------------

def save_cell_debug_image(
    debug_dir: Path,
    field_name: str,
    cell_img: np.ndarray,
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_name = field_name or "field"
    safe_name = safe_name.replace(" ", "_")
    out_path = debug_dir / f"cell_{safe_name}.png"
    cv2.imwrite(str(out_path), cell_img)
    return out_path


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> Optional[CellExtractionConfig]:
    parser = argparse.ArgumentParser(
        description="Extract a title-block cell around a click and OCR it.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to page image (PNG/JPEG).",
    )
    parser.add_argument(
        "--grid-json",
        type=Path,
        help="Path to grid JSON (from titleblock_grid_worker).",
    )
    parser.add_argument(
        "--click-x",
        type=float,
        help="Click X normalised to page width (0–1).",
    )
    parser.add_argument(
        "--click-y",
        type=float,
        help="Click Y normalised to page height (0–1).",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory to write debug cell crops.",
    )

    args = parser.parse_args()

    # If no --input, assume Supabase mode
    if args.input is None or args.grid_json is None or args.click_x is None or args.click_y is None:
        return None

    return CellExtractionConfig(
        input_path=args.input,
        grid_json_path=args.grid_json,
        click_x_norm=args.click_x,
        click_y_norm=args.click_y,
        debug_dir=args.debug_dir,
    )


# ---------------------------------------------------------------------------
# Core run functions
# ---------------------------------------------------------------------------

def run_single_click_extraction(config: CellExtractionConfig) -> None:
    """
    CLI mode: manual test for a single click.
    """
    page_img = load_image(config.input_path)
    page_h, page_w = page_img.shape[:2]

    grid = load_grid_json(config.grid_json_path)

    # Crop title-block from page
    y0 = grid.crop_offset_y
    x0 = grid.crop_offset_x
    y1 = y0 + grid.crop_height
    x1 = x0 + grid.crop_width

    titleblock_img = page_img[y0:y1, x0:x1]
    binary_tb = prepare_binary(titleblock_img)

    # Click in page-normalised coords → page pixels
    click_x_px = max(0.0, min(config.click_x_norm, 1.0)) * (page_w - 1)
    click_y_px = max(0.0, min(config.click_y_norm, 1.0)) * (page_h - 1)

    # Convert to title-block coords
    click_x_tb = click_x_px - x0
    click_y_tb = click_y_px - y0

    print(
        f"[info] click (page_norm=({config.click_x_norm:.3f}, {config.click_y_norm:.3f})) "
        f"→ page_px=({click_x_px:.1f}, {click_y_px:.1f}) "
        f"→ tb_px=({click_x_tb:.1f}, {click_y_tb:.1f})"
    )

    # Compute cell bbox in title-block coords
    x0_cell, y0_cell, x1_cell, y1_cell = compute_cell_bbox_for_click(
        binary_tb,
        grid.horizontal_lines,
        grid.vertical_lines,
        click_x_tb,
        click_y_tb,
    )

    print(
        f"[info] cell bbox in title-block coords: "
        f"x0={x0_cell}, y0={y0_cell}, x1={x1_cell}, y1={y1_cell}"
    )

    cell_img = titleblock_img[y0_cell:y1_cell, x0_cell:x1_cell]

    cell_text = ocr_cell(cell_img)
    print(f"[info] OCR text: {repr(cell_text)}")

    if config.debug_dir is not None:
        out_path = save_cell_debug_image(
            config.debug_dir,
            "manual_click",
            cell_img,
        )
        print(f"[info] debug cell written to {out_path}")


def run_from_supabase() -> None:
    """
    Supabase mode:

    - Get next 'tagged' page from document_pages.
    - Infer grid JSON path from the image path (for titleblock crop).
    - Read boxed areas from titleblock_fingerprint.areas (titleblock-relative 0–1).
    - For each area, crop the corresponding region, OCR it, and write a debug image.
    """
    print("[info] titleblock_cell_ocr_worker: running in Supabase mode")

    try:
        cfg = load_supabase_config_from_env()
    except Exception as exc:
        print(f"[error] failed to load Supabase config: {exc}", file=sys.stderr)
        return

    client = create_supabase_client(cfg)

    tagged_page = fetch_next_tagged_page(client, cfg, status_value="tagged")
    if tagged_page is None:
        print("[info] no tagged pages found in document_pages")
        return

    print(
        f"[info] using page id={tagged_page.id} "
        f"document_id={tagged_page.document_id} "
        f"page_number={tagged_page.page_number}"
    )
    print(f"[info] page image (rel): {tagged_page.page_image_rel_path}")
    print(f"[info] page image (abs): {tagged_page.page_image_abs_path}")

    page_path = Path(tagged_page.page_image_abs_path)
    if not page_path.exists():
        print(
            f"[error] page image does not exist at {page_path}",
            file=sys.stderr,
        )
        return

    # Infer grid JSON path: same dir, stem + "_grid.json"
    grid_json_path = page_path.with_name(page_path.stem + "_grid.json")
    if not grid_json_path.exists():
        print(
            f"[error] grid JSON not found at {grid_json_path} "
            f"(run titleblock_grid_worker first)",
            file=sys.stderr,
        )
        return

    grid = load_grid_json(grid_json_path)
    page_img = load_image(page_path)

    # Crop title-block using the same offset/size as the grid worker
    y0 = grid.crop_offset_y
    x0 = grid.crop_offset_x
    y1 = y0 + grid.crop_height
    x1 = x0 + grid.crop_width

    titleblock_img = page_img[y0:y1, x0:x1]
    tb_height, tb_width = titleblock_img.shape[:2]

    # Fetch boxed areas (version 2 fingerprint)
    areas = fetch_titleblock_areas_for_page(client, tagged_page.id)
    if not areas:
        print("[info] no areas found in titleblock_fingerprint; nothing to do")
        return

    debug_dir = page_path.parent / "debug_cells"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for area in areas:
        field_name = area.field_name or "field"

        # Clamp normalised values
        x_rel = max(0.0, min(area.x_rel, 1.0))
        y_rel = max(0.0, min(area.y_rel, 1.0))
        w_rel = max(0.0, min(area.width_rel, 1.0))
        h_rel = max(0.0, min(area.height_rel, 1.0))

        # Convert to [0,1] end coords, clipped to the titleblock
        x0_n = x_rel
        y0_n = y_rel
        x1_n = min(1.0, x_rel + w_rel)
        y1_n = min(1.0, y_rel + h_rel)

        # Scale to pixels
        x0_px = int(round(x0_n * (tb_width - 1)))
        y0_px = int(round(y0_n * (tb_height - 1)))
        x1_px = int(round(x1_n * (tb_width - 1)))
        y1_px = int(round(y1_n * (tb_height - 1)))

        # Apply a small inward margin to avoid box borders
        margin = 2
        x0_px = max(0, x0_px + margin)
        y0_px = max(0, y0_px + margin)
        x1_px = min(tb_width - 1, x1_px - margin)
        y1_px = min(tb_height - 1, y1_px - margin)

        if x1_px <= x0_px or y1_px <= y0_px:
            print(
                f"[warn] invalid or collapsed bbox for field={field_name!r} "
                f"(x0={x0_px}, y0={y0_px}, x1={x1_px}, y1={y1_px}); skipping"
            )
            continue

        print(
            f"[info] field={field_name!r} "
            f"box_rel=(x={x_rel:.3f}, y={y_rel:.3f}, w={w_rel:.3f}, h={h_rel:.3f}) "
            f"→ box_px=(x0={x0_px}, y0={y0_px}, x1={x1_px}, y1={y1_px})"
        )

        cell_img = titleblock_img[y0_px:y1_px, x0_px:x1_px]
        if cell_img.size == 0:
            print(
                f"[warn] empty cell crop for field={field_name!r}; skipping",
                file=sys.stderr,
            )
            continue

        cell_text = ocr_cell(cell_img)
        print(f"[info] OCR [{field_name}]: {repr(cell_text)}")

        out_path = save_cell_debug_image(
            debug_dir,
            field_name,
            cell_img,
        )
        print(f"[info] debug cell for {field_name!r} → {out_path}")

        # TODO: write OCR result back to Supabase if desired, e.g.:
        # client.table("document_pages").update(
        #     {f"{field_name}_ocr": cell_text}
        # ).eq("id", tagged_page.id).execute()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # If full CLI args provided, run manual test mode
    config = parse_args()
    if config is not None:
        run_single_click_extraction(config)
        return

    # Otherwise, Supabase mode
    run_from_supabase()


if __name__ == "__main__":
    main()
