# worker/app/titleblock_match_worker.py

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from supabase import Client, create_client

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_PAGES_PER_RUN = 10
MAX_TEMPLATES = 100

DOC_NAS_ROOT_ENV = "DOC_NAS_ROOT"
FINGERPRINT_SIZE = 64

MIN_SIMILARITY_THRESHOLD = 0.97
TRUST_FP_OVERRIDE = 0.95  

GRID_WEIGHT = 0.5
FP_WEIGHT = 0.5
MAX_TOTAL_SCORE = 0.30
GRID_ACCEPT_THRESHOLD = 0.9  # lower = stricter



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
# DB helpers
# ---------------------------------------------------------------------

def fetch_templates(client: Client) -> List[Dict[str, Any]]:
    """
    Fetch document_titleblock_templates rows with template JSON.
    We pre-extract:
      - _fingerprint: template["fingerprint"]
      - _field_boxes: template["field_boxes"]
      - _page_bbox_norm: template["page_bbox_norm"] (if present)
    """
    try:
        response = (
            client.table("document_titleblock_templates")
            .select("id, template")
            .limit(MAX_TEMPLATES)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch document_titleblock_templates: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    print(f"[info] fetched {len(rows)} titleblock_template row(s) from Supabase")

    valid: List[Dict[str, Any]] = []
    for row in rows:
        tpl = row.get("template") or {}
        if not isinstance(tpl, dict):
            continue

        fp = tpl.get("fingerprint")
        fb = tpl.get("field_boxes")

        if not fp or not fb:
            # No fingerprint or no field geometry → useless for matching
            continue

        row["_fingerprint"] = fp
        row["_field_boxes"] = fb

        bbox_norm = tpl.get("page_bbox_norm")
        if isinstance(bbox_norm, dict):
            # Expecting keys: x, y, w, h (all 0–1, full page)
            row["_page_bbox_norm"] = bbox_norm

        valid.append(row)

    print(f"[info] {len(valid)} template(s) have usable fingerprint + field_boxes")
    return valid

def fetch_pages_to_match(client: Client, limit: int = MAX_PAGES_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch candidate document_pages rows that:

      - have an image_object_path (PNG/JPEG already rendered),
      - do NOT already have matched_titleblock_template_id.

    We *do not* require per-page titleblock_x/y/width/height here;
    the template's page_bbox_norm is used as the default ROI.
    """
    try:
        response = (
            client.table("document_pages")
            .select(
                "id, document_id, page_number, image_object_path, "
                "titleblock_x, titleblock_y, titleblock_width, titleblock_height, "
                "matched_titleblock_template_id"
            )
            .limit(limit * 3)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch document_pages for matching: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    print(f"[info] fetched {len(rows)} document_pages row(s) from Supabase for matching")

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("matched_titleblock_template_id") is not None:
            # Already matched → skip
            continue
        if not row.get("image_object_path"):
            continue
        candidates.append(row)

    print(f"[info] {len(candidates)} page(s) are candidates for titleblock template matching")
    return candidates


def fetch_document_info(client: Client, document_id: Any) -> Optional[Dict[str, Any]]:
    try:
        response = (
            client.table("document_files")
            .select("id, storage_object_path")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch document_files for id={document_id}: {exc}", file=sys.stderr)
        return None

    rows = getattr(response, "data", None) or []
    if not rows:
        print(f"[error] No document_files row found for id={document_id}", file=sys.stderr)
        return None

    return rows[0]


def update_page_fields(
    client: Client,
    page_id: Any,
    fields: Dict[str, Optional[str]],
) -> None:
    update_data: Dict[str, Any] = {}
    for key in ("drawing_number", "drawing_title", "revision"):
        if key in fields:
            update_data[key] = fields[key]

    if not update_data:
        return

    try:
        (
            client.table("document_pages")
            .update(update_data)
            .eq("id", page_id)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to update document_pages(id={page_id}) fields: {exc}",
            file=sys.stderr,
        )


def update_page_template_link(
    client: Client,
    page_id: Any,
    template_id: Any,
    confidence: float,
) -> None:
    try:
        (
            client.table("document_pages")
            .update(
                {
                    "matched_titleblock_template_id": template_id,
                    "matched_titleblock_confidence": float(confidence),
                }
            )
            .eq("id", page_id)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to update template link on document_pages(id={page_id}): {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------
# Geometry + fingerprint helpers
# ---------------------------------------------------------------------

def compute_titleblock_fingerprint(
    img_gray: np.ndarray,
    x0_tb: int,
    y0_tb: int,
    x1_tb: int,
    y1_tb: int,
    size: int = FINGERPRINT_SIZE,
) -> List[int]:
    """
    Compute the same edge64 fingerprint used in document_titleblock_templates:
      - crop ROI
      - Canny edges
      - resize to size x size
      - flatten to list of 0–255 ints
    """
    roi = img_gray[y0_tb:y1_tb, x0_tb:x1_tb]
    if roi.size == 0:
        return []

    edges = cv2.Canny(roi, 80, 200)
    small = cv2.resize(edges, (size, size), interpolation=cv2.INTER_AREA)
    flat = small.astype("uint8").flatten().tolist()
    return flat

def detect_grid_lines(img_gray: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect approximate vertical and horizontal grid lines in a grayscale ROI.
    Returns two sorted lists of x and y positions in pixels.
    """
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=30,
        maxLineGap=10,
    )

    verticals: List[int] = []
    horizontals: List[int] = []
    if lines is None:
        return verticals, horizontals

    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) < 5:
            verticals.append((x1 + x2) // 2)
        elif abs(y2 - y1) < 5:
            horizontals.append((y1 + y2) // 2)

    verticals.sort()
    horizontals.sort()
    return verticals, horizontals


def mean_abs_diff(a: List[float], b: List[float]) -> float:
    """
    Mean absolute difference between two normalised lists in [0,1].
    If either list is empty, return 1.0 (max difference).
    """
    if not a or not b:
        return 1.0
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(n):
        total += abs(a[i] - b[i])
    return total / float(n)



def compute_similarity(vec1: List[int], vec2: List[int]) -> float:
    """
    Compute a similarity score in [0,1] between two fingerprint vectors.
    1.0 = identical, 0 ~ very different.
    """
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) != len(vec2):
        return 0.0

    a = np.asarray(vec1, dtype=np.float32) / 255.0
    b = np.asarray(vec2, dtype=np.float32) / 255.0

    diff = a - b
    # Normalised L2 distance
    dist = float(np.linalg.norm(diff) / np.sqrt(a.size))
    # Map to similarity: 1 at dist=0, tapering down as dist grows
    sim = 1.0 - dist
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return sim


def map_img_bbox_to_pdf_rect(
    img_width: int,
    img_height: int,
    bbox_img: Tuple[int, int, int, int],
    page_rect: fitz.Rect,
) -> fitz.Rect:
    x0_img, y0_img, x1_img, y1_img = bbox_img
    page_w = float(page_rect.width)
    page_h = float(page_rect.height)

    x0_pdf = (x0_img / float(img_width)) * page_w
    x1_pdf = (x1_img / float(img_width)) * page_w

    y0_pdf = ((img_height - y1_img) / float(img_height)) * page_h
    y1_pdf = ((img_height - y0_img) / float(img_height)) * page_h

    return fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)


# ---------------------------------------------------------------------
# Apply template to a page: extract text using field_boxes
# ---------------------------------------------------------------------

def extract_fields_using_template(
    pdf_path: Path,
    page_number: int,
    img_width: int,
    img_height: int,
    x0_tb: int,
    y0_tb: int,
    x1_tb: int,
    y1_tb: int,
    field_boxes: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    For a given page and titleblock bbox, use template field_boxes (normalised
    to the titleblock rect in PDF coords) to extract text for each field.
    """
    result: Dict[str, str] = {}

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"[error] Failed to open PDF {pdf_path}: {exc}", file=sys.stderr)
        return result

    try:
        if page_number < 1 or page_number > doc.page_count:
            print(
                f"[error] Page number {page_number} out of range for {pdf_path} "
                f"(page_count={doc.page_count})",
                file=sys.stderr,
            )
            return result

        page = doc[page_number - 1]
        page_rect = page.rect

        # Titleblock rect in PDF coords
        tb_rect_pdf = map_img_bbox_to_pdf_rect(
            img_width=img_width,
            img_height=img_height,
            bbox_img=(x0_tb, y0_tb, x1_tb, y1_tb),
            page_rect=page_rect,
        )

        tb_w = float(tb_rect_pdf.width)
        tb_h = float(tb_rect_pdf.height)
        if tb_w <= 0.0 or tb_h <= 0.0:
            return result

        for field_name, box in field_boxes.items():
            x0_rel = float(box.get("x0_rel", 0.0))
            y0_rel = float(box.get("y0_rel", 0.0))
            x1_rel = float(box.get("x1_rel", 0.0))
            y1_rel = float(box.get("y1_rel", 0.0))

            x0_pdf = tb_rect_pdf.x0 + x0_rel * tb_w
            x1_pdf = tb_rect_pdf.x0 + x1_rel * tb_w
            y0_pdf = tb_rect_pdf.y0 + y0_rel * tb_h
            y1_pdf = tb_rect_pdf.y0 + y1_rel * tb_h

            rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
            text = page.get_text("text", clip=rect) or ""
            text = text.strip()

            print(
                f"[info] auto-extract field={field_name!r} "
                f"rect_pdf=({x0_pdf:.1f}, {y0_pdf:.1f}, {x1_pdf:.1f}, {y1_pdf:.1f}) "
                f"text={text!r}"
            )

            if text:
                result[field_name] = text

    finally:
        doc.close()

    return result


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def process_page(
    client: Client,
    nas_root: Path,
    page_row: Dict[str, Any],
    templates: List[Dict[str, Any]],
) -> None:
    page_id = page_row.get("id")
    document_id = page_row.get("document_id")
    page_number = page_row.get("page_number")
    image_rel = page_row.get("image_object_path")

    if page_id is None or document_id is None or page_number is None:
        print(
            f"[error] Page row missing id/document_id/page_number: {page_row}",
            file=sys.stderr,
        )
        return

    if not image_rel:
        print(f"[error] page_id={page_id} missing image_object_path", file=sys.stderr)
        return

    # Optional per-page override of titleblock bbox (0–1, full page)
    tb_x_rel_page = page_row.get("titleblock_x")
    tb_y_rel_page = page_row.get("titleblock_y")
    tb_w_rel_page = page_row.get("titleblock_width")
    tb_h_rel_page = page_row.get("titleblock_height")

    doc_info = fetch_document_info(client, document_id)
    if doc_info is None:
        print(f"[error] Missing document_files row for page_id={page_id}", file=sys.stderr)
        return

    storage_path = doc_info.get("storage_object_path")
    if not storage_path:
        print(
            f"[error] document_files.storage_object_path is empty (page_id={page_id})",
            file=sys.stderr,
        )
        return

    pdf_path = nas_root / storage_path
    if not pdf_path.is_file():
        print(f"[error] PDF file not found at {pdf_path} (page_id={page_id})", file=sys.stderr)
        return

    image_path = nas_root / image_rel
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"[error] Failed to load image {image_path} (page_id={page_id})", file=sys.stderr)
        return

    img_h, img_w = img_gray.shape[:2]

    # ------------------------------------------------------------------
    # Find best template by combined grid + fingerprint score
    # ------------------------------------------------------------------
    best_template_id: Optional[Any] = None
    best_field_boxes: Optional[Dict[str, Dict[str, float]]] = None
    best_bbox_px: Optional[Tuple[int, int, int, int]] = None
    best_score: float = float("inf")
    best_fp_sim: float = 0.0
    best_grid_score: float = 1.0

    for tpl in templates:
        tpl_id = tpl.get("id")
        tpl_template = tpl.get("template") or tpl

        fp_tpl = tpl_template.get("fingerprint") or {}
        grid_tpl = tpl_template.get("grid") or {}
        fb = tpl_template.get("field_boxes") or {}

        if not fp_tpl or not fb:
            continue

        tpl_vn = grid_tpl.get("verticals_norm", []) or []
        tpl_hn = grid_tpl.get("horizontals_norm", []) or []

        # Decide bbox for THIS template on THIS page:
        # 1) per-page override if present
        # 2) otherwise template.page_bbox_norm
        bbox_norm = None
        if None not in (tb_x_rel_page, tb_y_rel_page, tb_w_rel_page, tb_h_rel_page):
            bbox_norm = {
                "x": float(tb_x_rel_page),
                "y": float(tb_y_rel_page),
                "w": float(tb_w_rel_page),
                "h": float(tb_h_rel_page),
            }
        else:
            bbox_norm = tpl_template.get("page_bbox_norm")

        if not isinstance(bbox_norm, dict):
            # No usable bbox for this template
            continue

        x_rel = float(bbox_norm.get("x", 0.0))
        y_rel = float(bbox_norm.get("y", 0.0))
        w_rel = float(bbox_norm.get("w", 0.0))
        h_rel = float(bbox_norm.get("h", 0.0))

        # Convert normalised bbox (0–1) → pixel ROI, clamp to image bounds
        x0_tb = int(round(x_rel * img_w))
        y0_tb = int(round(y_rel * img_h))
        x1_tb = int(round((x_rel + w_rel) * img_w))
        y1_tb = int(round((y_rel + h_rel) * img_h))

        x0_tb = max(0, min(img_w, x0_tb))
        y0_tb = max(0, min(img_h, y0_tb))
        x1_tb = max(0, min(img_w, x1_tb))
        y1_tb = max(0, min(img_h, y1_tb))

        if x1_tb <= x0_tb or y1_tb <= y0_tb:
            # Degenerate ROI
            continue

        roi = img_gray[y0_tb:y1_tb, x0_tb:x1_tb]
        if roi.size == 0:
            continue

        # --- Grid score: compare detected lines to template grid ---
        v_px, h_px = detect_grid_lines(roi)
        w_crop = max(1, roi.shape[1])
        h_crop = max(1, roi.shape[0])
        v_norm = [float(x) / float(w_crop) for x in v_px]
        h_norm = [float(y) / float(h_crop) for y in h_px]

        grid_v_score = mean_abs_diff(v_norm, tpl_vn)
        grid_h_score = mean_abs_diff(h_norm, tpl_hn)
        grid_score = 0.5 * (grid_v_score + grid_h_score)

        # --- Fingerprint similarity ---
        fp_vec = compute_titleblock_fingerprint(
            img_gray,
            x0_tb,
            y0_tb,
            x1_tb,
            y1_tb,
        )
        fp_tpl_data = fp_tpl.get("data")
        if not fp_vec or not fp_tpl_data:
            continue

        fp_sim = compute_similarity(fp_vec, fp_tpl_data)

        # Combined cost: smaller is better
        total_score = GRID_WEIGHT * grid_score + FP_WEIGHT * (1.0 - fp_sim)

        print(
            f"[debug] page_id={page_id} tpl={tpl_id} "
            f"grid_score={grid_score:.3f} fp_sim={fp_sim:.3f} "
            f"total_score={total_score:.3f}"
        )

        if total_score < best_score:
            best_score = total_score
            best_template_id = tpl_id
            best_field_boxes = fb
            best_bbox_px = (x0_tb, y0_tb, x1_tb, y1_tb)
            best_fp_sim = fp_sim
            best_grid_score = grid_score

    # ------------------------------------------------------------------
    # Decide if we trust the best match
    # ------------------------------------------------------------------
    if best_template_id is None or best_field_boxes is None or best_bbox_px is None:
        print(f"[info] page_id={page_id}: no usable template candidates")
        try:
            client.table("document_pages").update(
                {"status": "no live match, please add zones of interest"}
            ).eq("id", page_id).execute()
        except Exception as exc:
            print(
                f"[warn] page_id={page_id}: failed to update status after no-match: {exc}",
                file=sys.stderr,
            )
        return

    print(
        f"[info] page_id={page_id}: best template {best_template_id} "
        f"total_score={best_score:.3f} grid_score={best_grid_score:.3f} "
        f"fp_sim={best_fp_sim:.3f}"
    )

    # Decide if we trust this match
    trusted = False

    # 1) Very strong fp similarity: trust even if grid_score a bit noisy
    if best_fp_sim >= TRUST_FP_OVERRIDE:
        trusted = True
        print(
            f"[info] page_id={page_id}: trusting match via fp override "
            f"(fp_sim={best_fp_sim:.3f} ≥ {TRUST_FP_OVERRIDE:.3f})"
        )
    else:
        # 2) Otherwise require BOTH:
        #    - grid+fp combined score below threshold
        #    - fp similarity above the basic threshold
        if best_score <= MAX_TOTAL_SCORE and best_fp_sim >= MIN_SIMILARITY_THRESHOLD:
            trusted = True
            print(
                f"[info] page_id={page_id}: trusting match via combined score "
                f"(score={best_score:.3f} ≤ {MAX_TOTAL_SCORE:.3f}, "
                f"fp_sim={best_fp_sim:.3f} ≥ {MIN_SIMILARITY_THRESHOLD:.3f})"
            )

    if not trusted:
        print(
            f"[info] page_id={page_id}: match not trusted "
            f"(score={best_score:.3f}, fp_sim={best_fp_sim:.3f}); "
            f"marking as 'no live match, please add zones of interest'"
        )
        try:
            client.table("document_pages").update(
                {"status": "no live match, please add zones of interest"}
            ).eq("id", page_id).execute()
        except Exception as exc:
            print(
                f"[warn] page_id={page_id}: failed to update status after low-score: {exc}",
                file=sys.stderr,
            )
        return


    x0_tb, y0_tb, x1_tb, y1_tb = best_bbox_px

    # ------------------------------------------------------------------
    # SUCCESS: auto-tag this page to mimic the client workflow
    # ------------------------------------------------------------------

    # 1) Compute normalised titleblock bbox from the pixels we actually used
    tb_x_rel = float(x0_tb) / float(img_w)
    tb_y_rel = float(y0_tb) / float(img_h)
    tb_w_rel = float(x1_tb - x0_tb) / float(img_w)
    tb_h_rel = float(y1_tb - y0_tb) / float(img_h)

    # 2) Build "areas" list from the template's field_boxes so it looks
    #    like the client-provided zones-of-interest JSON.
    areas: List[Dict[str, Any]] = []
    for field_name, box in (best_field_boxes or {}).items():
        try:
            x0_rel = float(box["x0_rel"])
            x1_rel = float(box["x1_rel"])
            y0_rel = float(box["y0_rel"])
            y1_rel = float(box["y1_rel"])
        except (KeyError, TypeError, ValueError):
            continue

        width_rel = x1_rel - x0_rel
        height_rel = y1_rel - y0_rel
        if width_rel <= 0.0 or height_rel <= 0.0:
            continue

        areas.append(
            {
                "field": field_name,
                "x_rel": x0_rel,
                "y_rel": y0_rel,
                "width_rel": width_rel,
                "height_rel": height_rel,
            }
        )

    fingerprint_payload: Dict[str, Any] = {
        "version": 2,
        "areas": areas,
    }

    # 3) Update the document_pages row so it looks just like a manually
    #    tagged page (bbox + zones + template link).
    try:
        client.table("document_pages").update(
            {
                "matched_titleblock_template_id": best_template_id,
                "matched_titleblock_confidence": round(float(best_fp_sim), 3),
                "titleblock_x": round(tb_x_rel, 6),
                "titleblock_y": round(tb_y_rel, 6),
                "titleblock_width": round(tb_w_rel, 6),
                "titleblock_height": round(tb_h_rel, 6),
                "titleblock_fingerprint": fingerprint_payload,
                "status": "tagged",
            }
        ).eq("id", page_id).execute()
        print(
            f"[info] page_id={page_id}: auto-tagged from template "
            f"{best_template_id} with fp_sim={best_fp_sim:.3f} "
            f"score={best_score:.3f}"
        )
    except Exception as exc:
        print(
            f"[warn] page_id={page_id}: failed to write auto-tag data: {exc}",
            file=sys.stderr,
        )
        return

    # From here, titleblock_extract_worker can pick this page up in the
    # normal way (status='tagged', bbox set, fingerprint JSON present).


def run_once() -> int:
    client = create_supabase_client()
    if client is None:
        print("[error] Supabase client not created", file=sys.stderr)
        return 1

    nas_root = get_nas_root()
    print(f"[info] Using NAS root: {nas_root}")

    templates = fetch_templates(client)
    if not templates:
        print("[info] No usable document_titleblock_templates found; nothing to match")
        return 0

    pages = fetch_pages_to_match(client, limit=MAX_PAGES_PER_RUN)
    if not pages:
        print("[info] No pages eligible for template matching")
        return 0

    for row in pages:
        process_page(client, nas_root, row, templates)

    return 0


def main() -> int:
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
