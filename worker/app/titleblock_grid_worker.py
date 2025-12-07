"""
titleblock_grid_worker.py

Detects title-block grid lines (vertical and horizontal) in a drawing page image.

The worker:
1. Loads a page image (PNG or JPEG).
2. Crops the title-block using a normalised bbox (0–1 coordinates).
3. Detects grid lines using morphological operations and projection profiles.
4. Outputs:
   - A JSON file with vertical and horizontal line positions (in title-block pixels),
     plus the crop offsets.
   - An optional debug image showing the grid overlay.

Usage example:

    python -m app.titleblock_grid_worker \
        --input /data/input/pages/enquiries/55555/page_1.png \
        --titleblock-bbox 0.0 0.7 1.0 1.0 \
        --output-json /data/input/derived/pages/enquiries/55555/page_1_grid.json \
        --debug-image /data/input/derived/pages/enquiries/55555/page_1_grid_debug.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2, sys
import numpy as np

from app.supabase_client import (
    load_supabase_config_from_env,
    create_supabase_client,
    fetch_next_tagged_page,
)

@dataclass
class GridDetectionConfig:
    input_path: Path
    titleblock_bbox_norm: tuple[float, float, float, float]
    output_json: Path
    debug_image: Path | None
    min_vertical_fraction: float = 0.6   # used as min_band_fraction
    min_horizontal_fraction: float = 0.5
    margin_pixels: int = 2



# ---------------------------------------------------------------------------
# Small helpers (coordinate mapping, IO)
# ---------------------------------------------------------------------------

def parse_args() -> GridDetectionConfig:
    parser = argparse.ArgumentParser(
        description="Detect title-block grid lines in a drawing page."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the page image (PNG / JPEG).",
    )

    parser.add_argument(
        "--titleblock-bbox",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        required=True,
        help="Normalised title-block bbox (0–1) relative to full page: left top right bottom.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path to write grid lines JSON.",
    )

    parser.add_argument(
        "--debug-image",
        type=Path,
        default=None,
        help="Optional path to write a debug PNG with grid overlay.",
    )

    args = parser.parse_args()

    bbox_norm = _clamp_bbox_norm(
        args.titleblock_bbox[0],
        args.titleblock_bbox[1],
        args.titleblock_bbox[2],
        args.titleblock_bbox[3],
    )

    return GridDetectionConfig(
        input_path=args.input,
        titleblock_bbox_norm=bbox_norm,
        output_json=args.output_json,
        debug_image=args.debug_image,
    )


def _clamp_bbox_norm(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Tuple[float, float, float, float]:
    left_c = max(0.0, min(left, 1.0))
    top_c = max(0.0, min(top, 1.0))
    right_c = max(0.0, min(right, 1.0))
    bottom_c = max(0.0, min(bottom, 1.0))

    if right_c <= left_c:
        right_c = min(1.0, left_c + 0.01)

    if bottom_c <= top_c:
        bottom_c = min(1.0, top_c + 0.01)

    return left_c, top_c, right_c, bottom_c


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to load image at {path}"
        raise RuntimeError(msg)
    return image


def norm_to_px(
    norm_x: float,
    norm_y: float,
    width: int,
    height: int,
) -> Tuple[int, int]:
    nx = max(0.0, min(norm_x, 1.0))
    ny = max(0.0, min(norm_y, 1.0))
    x = int(round(nx * (width - 1)))
    y = int(round(ny * (height - 1)))
    return x, y


def crop_titleblock(
    page_img: np.ndarray,
    bbox_norm: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, int, int]:
    height, width = page_img.shape[:2]
    left_n, top_n, right_n, bottom_n = bbox_norm

    x0, y0 = norm_to_px(left_n, top_n, width, height)
    x1, y1 = norm_to_px(right_n, bottom_n, width, height)

    x0_c = max(0, min(x0, width - 1))
    x1_c = max(0, min(x1, width))
    y0_c = max(0, min(y0, height - 1))
    y1_c = max(0, min(y1, height))

    if x1_c <= x0_c or y1_c <= y0_c:
        msg = "Invalid title-block crop after clamping."
        raise RuntimeError(msg)

    crop = page_img[y0_c:y1_c, x0_c:x1_c]
    return crop, x0_c, y0_c


# ---------------------------------------------------------------------------
# Image processing for grid detection
# ---------------------------------------------------------------------------

def prepare_binary(titleblock_img: np.ndarray) -> np.ndarray:
    if titleblock_img.ndim == 3:
        gray = cv2.cvtColor(titleblock_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = titleblock_img

    # Otsu threshold, invert so black lines become 255
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary

def detect_vertical_lines_simple(binary: np.ndarray) -> List[int]:
    """
    Simple global detector used as fallback when there are no horizontals.

    Looks for columns with above-average ink and groups them into lines.
    """
    height, width = binary.shape
    col_sum = (binary > 0).sum(axis=0).astype(np.float32)

    mean_val = float(col_sum.mean())
    std_val = float(col_sum.std())
    threshold = mean_val + 0.5 * std_val

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

    lines.append((start + prev) // 2)
    return lines

def detect_vertical_lines(
    binary: np.ndarray,
    std_factor: float,
) -> list[int]:
    """
    Detect vertical grid lines using a simple projection + peak threshold:

    - Compute column-wise counts of 'ink' (white pixels in the inverted binary).
    - Smooth the 1D projection to reduce noise.
    - Mark columns where the smoothed value is >= mean + std_factor * std.
    - Group contiguous marked columns and take the centre of each group.

    std_factor controls how aggressive we are:
    - Higher → fewer, stronger lines.
    - Lower → more, weaker lines (may pick up some noise).
    """
    height, width = binary.shape

    # 1) Column-wise "ink" count (lines are 255 in our inverted binary)
    col_sum = (binary > 0).sum(axis=0).astype(np.float32)

    # 2) Smooth a little to reduce text noise
    window = max(3, width // 200)  # small window; e.g. 3–5 pixels
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        proj = np.convolve(col_sum, kernel, mode="same")
    else:
        proj = col_sum

    # 3) Threshold at mean + std_factor * std
    mean_val = float(proj.mean())
    std_val = float(proj.std())
    threshold = mean_val + std_factor * std_val

    # Debug prints (keep or remove as you like)
    print(
        f"[debug] vertical projection: mean={mean_val:.1f}, std={std_val:.1f}, "
        f"threshold={threshold:.1f}"
    )

    indices = np.where(proj >= threshold)[0]
    if indices.size == 0:
        print("[debug] no vertical columns above threshold")
        return []

    # 4) Group contiguous indices into lines
    lines: list[int] = []
    start = prev = int(indices[0])

    for idx in indices[1:]:
        idx_int = int(idx)
        if idx_int == prev + 1:
            prev = idx_int
            continue
        # end of group
        lines.append((start + prev) // 2)
        start = prev = idx_int

    # final group
    lines.append((start + prev) // 2)

    print(f"[debug] detected {len(lines)} vertical grid line(s): {lines}")
    return lines

def detect_horizontal_lines(
    binary: np.ndarray,
    min_fraction: float,
) -> List[int]:
    _, width = binary.shape
    kernel_width = max(1, int(width * 0.25))

    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )

    horizontal = cv2.erode(binary, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    row_sum = horizontal.sum(axis=1) / 255.0
    min_len = width * min_fraction
    return _find_line_positions(row_sum, min_len)

def detect_vertical_lines_from_bands(
    binary: np.ndarray,
    horizontals: List[int],
    min_band_fraction: float = 0.6,
    merge_distance: int = 3,
    strength_ratio: float = 0.6,
) -> List[int]:
    """
    Detect vertical grid lines using per-band projections, then filter by
    global column strength.

    Always returns a list (possibly empty).
    """
    height, width = binary.shape

    if not horizontals:
        # Fallback: no horizontals → simple global detector (edge case).
        return detect_vertical_lines_simple(binary)

    # Ensure sorted and build band edges
    h_sorted = sorted(horizontals)
    edges = [0] + h_sorted + [height - 1]

    # Each group tracks x_sum and count; we merge candidates across bands
    groups: List[dict] = []

    def _add_candidate(x_center: int) -> None:
        """Merge x_center into nearest group or start a new one."""
        for g in groups:
            current_center = g["x_sum"] / max(1, g["count"])
            if abs(x_center - current_center) <= merge_distance:
                g["x_sum"] += x_center
                g["count"] += 1
                return
        groups.append({"x_sum": float(x_center), "count": 1})

    for i in range(len(edges) - 1):
        y_start = edges[i]
        y_end = edges[i + 1]

        # Small margin to avoid sitting on the horizontal line itself
        band_margin = 2
        y0 = max(0, y_start + band_margin)
        y1 = min(height, y_end - band_margin)

        if y1 <= y0:
            continue

        band = binary[y0:y1, :]
        band_height = y1 - y0

        # Column-wise "ink" within this band
        col_sum_band = (band > 0).sum(axis=0).astype(np.float32)

        threshold = band_height * min_band_fraction
        band_indices = np.where(col_sum_band >= threshold)[0]

        if band_indices.size == 0:
            continue

        # Group contiguous columns into line centres for this band
        start = prev = int(band_indices[0])

        for idx in band_indices[1:]:
            idx_int = int(idx)
            if idx_int == prev + 1:
                prev = idx_int
                continue
            center = (start + prev) // 2
            _add_candidate(center)
            start = prev = idx_int

        # Final group in this band
        center_last = (start + prev) // 2
        _add_candidate(center_last)

    if not groups:
        print("[debug] no verticals found in any band; falling back to simple detector")
        return detect_vertical_lines_simple(binary)

    # Global column ink for the entire titleblock
    col_sum_total = (binary > 0).sum(axis=0).astype(np.float32)

    centers: List[int] = []
    strengths: List[float] = []

    for g in groups:
        center = int(round(g["x_sum"] / max(1, g["count"])))
        if 0 <= center < width:
            centers.append(center)
            strengths.append(float(col_sum_total[center]))

    if not centers:
        print("[debug] no valid centers after grouping; falling back to simple detector")
        return detect_vertical_lines_simple(binary)

    strengths_arr = np.array(strengths, dtype=np.float32)
    median_strength = float(np.median(strengths_arr))

    if median_strength <= 0.0:
        # Degenerate case: keep all
        print("[debug] median vertical strength <= 0; keeping all candidates")
        final_all = sorted(set(centers))
        print(f"[debug] band-based verticals (no strength filter): {final_all}")
        return final_all

    threshold_strength = median_strength * strength_ratio

    print(
        f"[debug] vertical strengths (centers → strength): "
        f"{list(zip(centers, strengths))}"
    )
    print(
        f"[debug] median_strength={median_strength:.1f}, "
        f"threshold_strength={threshold_strength:.1f}"
    )

    final_lines: List[int] = []
    for center, strength in zip(centers, strengths):
        if strength >= threshold_strength:
            final_lines.append(center)

    final_lines = sorted(set(final_lines))
    print(f"[debug] band-based verticals after strength filter: {final_lines}")
    return final_lines

def _find_line_positions(
    projection: np.ndarray,
    min_len: float,
) -> List[int]:
    threshold = max(1.0, min_len * 0.4)
    indices = np.where(projection >= threshold)[0]

    if indices.size == 0:
        return []

    lines: List[int] = []
    start = int(indices[0])
    prev = start

    for idx in indices[1:]:
        idx_int = int(idx)
        if idx_int == prev + 1:
            prev = idx_int
            continue
        center = (start + prev) // 2
        lines.append(center)
        start = idx_int
        prev = idx_int

    center_last = (start + prev) // 2
    lines.append(center_last)
    return lines


# ---------------------------------------------------------------------------
# Output helpers (JSON and debug overlay)
# ---------------------------------------------------------------------------

def save_grid_json(
    path: Path,
    verticals: List[int],
    horizontals: List[int],
    crop_width: int,
    crop_height: int,
    offset_x: int,
    offset_y: int,
) -> None:
    data = {
        "crop_offset": {"x": offset_x, "y": offset_y},
        "crop_size": {"width": crop_width, "height": crop_height},
        "vertical_lines": verticals,
        "horizontal_lines": horizontals,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def draw_debug_image(
    crop_img: np.ndarray,
    verticals: List[int],
    horizontals: List[int],
    path: Path,
) -> None:
    debug = crop_img.copy()
    height, width = debug.shape[:2]

    for x in verticals:
        if 0 <= x < width:
            cv2.line(
                debug,
                (x, 0),
                (x, height - 1),
                (0, 0, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    for y in horizontals:
        if 0 <= y < height:
            cv2.line(
                debug,
                (0, y),
                (width - 1, y),
                (0, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), debug)


# ---------------------------------------------------------------------------
# Main control flow
# ---------------------------------------------------------------------------

def run_grid_detection(config: GridDetectionConfig) -> None:
    page_img = load_image(config.input_path)

    titleblock_img, offset_x, offset_y = crop_titleblock(
        page_img,
        config.titleblock_bbox_norm,
    )

    binary = prepare_binary(titleblock_img)

    # First: horizontals (as before, these are working well)
    horizontals = detect_horizontal_lines(
        binary,
        config.min_horizontal_fraction,
    )

    # Then: verticals, using the bands between horizontals
    verticals = detect_vertical_lines_from_bands(
        binary,
        horizontals,
        min_band_fraction=0.6,   # tweakable
        merge_distance=3,        # tweakable
    )

    height, width = titleblock_img.shape[:2]

    save_grid_json(
        config.output_json,
        verticals,
        horizontals,
        width,
        height,
        offset_x,
        offset_y,
    )

    # Always produce a debug PNG
    if config.debug_image is not None:
        debug_path = config.debug_image
    else:
        debug_path = config.output_json.with_name(
            config.output_json.stem + "_debug.png"
        )

    draw_debug_image(
        titleblock_img,
        verticals,
        horizontals,
        debug_path,
    )

def run_from_supabase() -> None:
    """
    Fetch one 'Tagged' page from document_pages, run grid detection on its
    title-block, and write JSON + debug PNG next to the page image.
    """
    print("[info] titleblock_grid_worker: running in Supabase mode (no CLI args)")

    try:
        cfg = load_supabase_config_from_env()
    except Exception as exc:
        print(f"[error] failed to load Supabase config: {exc}", file=sys.stderr)
        return

    client = create_supabase_client(cfg)

    tagged_page = fetch_next_tagged_page(client, cfg, status_value="tagged")
    if tagged_page is None:
        print("[info] no Tagged pages found in document_pages")
        return

    print(
        f"[info] found Tagged page id={tagged_page.id} "
        f"document_id={tagged_page.document_id} "
        f"page_number={tagged_page.page_number}"
    )
    print(f"[info] image path (rel): {tagged_page.page_image_rel_path}")
    print(f"[info] image path (abs): {tagged_page.page_image_abs_path}")

    if not tagged_page.page_image_abs_path.exists():
        print(
            f"[error] image file does not exist at {tagged_page.page_image_abs_path}",
            file=sys.stderr,
        )
        return

    bbox_norm = (
        tagged_page.titleblock_x,
        tagged_page.titleblock_y,
        tagged_page.titleblock_x + tagged_page.titleblock_width,
        tagged_page.titleblock_y + tagged_page.titleblock_height,
    )

    out_json = tagged_page.page_image_abs_path.with_name(
        tagged_page.page_image_abs_path.stem + "_grid.json"
    )
    debug_png = tagged_page.page_image_abs_path.with_name(
        tagged_page.page_image_abs_path.stem + "_grid_debug.png"
    )

    grid_config = GridDetectionConfig(
        input_path=Path(tagged_page.page_image_abs_path),
        titleblock_bbox_norm=bbox_norm,
        output_json=out_json,
        debug_image=debug_png,
    )

    print(f"[info] running grid detection → {out_json}")
    run_grid_detection(grid_config)
    print(f"[info] grid detection complete; wrote {out_json} and {debug_png}")

def main() -> None:
    # If no extra CLI args are given, run in Supabase mode:
    if len(sys.argv) == 1:
        run_from_supabase()
        return

    # Otherwise, behave like the original CLI worker (manual paths).
    config = parse_args()
    run_grid_detection(config)


if __name__ == "__main__":
    main()
