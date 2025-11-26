import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Any, Dict, List

import cv2
import fitz  # PyMuPDF
import numpy as np
from supabase import Client, create_client

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]


DEBUG_OCR_CROPS = os.environ.get("DEBUG_OCR_CROPS", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_PAGES_PER_RUN = 10
DOC_NAS_ROOT_ENV = "DOC_NAS_ROOT"

FINGERPRINT_SIZE = 64  # edge thumbnail size for titleblock fingerprint


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

def parse_fingerprint(raw_fp: Any) -> Optional[Dict[str, Any]]:
    """
    Parse titleblock_fingerprint JSON and ensure it has a 'clicks' list.
    """
    if raw_fp is None:
        return None

    if isinstance(raw_fp, str):
        try:
            raw_fp = json.loads(raw_fp)
        except json.JSONDecodeError:
            return None

    if not isinstance(raw_fp, dict):
        return None

    clicks = raw_fp.get("clicks")
    if not isinstance(clicks, list) or not clicks:
        return None

    return raw_fp


def fetch_tagged_pages(client: Client, limit: int = MAX_PAGES_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch document_pages rows ready for titleblock extraction:

      - status in ('tagged', 'Tagged')
      - image_object_path set
      - titleblock_x/y/width/height set (0–1 fractions)
      - titleblock_fingerprint has clicks
    """
    try:
        response = (
            client.table("document_pages")
            .select(
                "id, document_id, page_number, status, "
                "image_object_path, "
                "titleblock_x, titleblock_y, titleblock_width, titleblock_height, "
                "titleblock_fingerprint, "
                "drawing_number, drawing_title, revision"
            )
            .in_("status", ["tagged", "Tagged"])
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch tagged pages: {exc}", file=sys.stderr)
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

        fp = parse_fingerprint(row.get("titleblock_fingerprint"))
        if fp is None:
            print(f"[debug] page_id={page_id} skipped: no usable fingerprint.clicks")
            continue

        row["_parsed_fingerprint"] = fp
        candidates.append(row)
        print(f"[debug] page_id={page_id} is a candidate for extraction")

    print(f"[info] {len(candidates)} tagged page(s) are extraction candidates")
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


def update_page_error(client: Client, page_id: Any, message: str) -> None:
    try:
        (
            client.table("document_pages")
            .update({"processing_error": message[:500]})
            .eq("id", page_id)
            .execute()
        )
    except Exception as exc:
        print(
            f"[error] Failed to update document_pages(id={page_id}) on error: {exc}",
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


def insert_titleblock_template(
    client: Client,
    sample_page_id: Any,
    template_json: Dict[str, Any],
    name: Optional[str] = None,
) -> Optional[Any]:
    """
    Insert a new titleblock_templates row and return its id.
    We keep this simple for now; matching/deduplication is a later concern.
    """
    payload: Dict[str, Any] = {
        "sample_document_page_id": sample_page_id,
        "template": template_json,
    }
    if name:
        payload["name"] = name

    try:
        response = client.table("titleblock_templates").insert(payload).execute()
    except Exception as exc:
        print(f"[error] Failed to insert titleblock_template: {exc}", file=sys.stderr)
        return None

    rows = getattr(response, "data", None) or []
    if not rows:
        print("[error] titleblock_templates insert returned no rows", file=sys.stderr)
        return None

    template_id = rows[0].get("id")
    print(f"[info] Created titleblock_template id={template_id}")
    return template_id


# ---------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------

def ocr_text_from_region(
    img_gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    psm: int = 7,
    debug_output_path: Optional[Path] = None,
) -> str:
    """
    Run Tesseract OCR on a grayscale region.
    Returns a single line of cleaned text (or empty string).

    If debug_output_path is not None, saves the binarised ROI as a PNG.
    """
    h, w = img_gray.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    if x1 <= x0 or y1 <= y0:
        return ""

    roi = img_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return ""

    try:
        _, roi_bin = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except Exception:
        roi_bin = roi

    # Optional debug crop
    if debug_output_path is not None:
        try:
            debug_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_output_path), roi_bin)
        except Exception as exc:
            print(
                f"[warn] Failed to write OCR debug crop to {debug_output_path}: {exc}",
                file=sys.stderr,
            )

    config = f"--psm {psm}"
    text = pytesseract.image_to_string(roi_bin, config=config)

    if not text:
        return ""

    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def compute_field_boxes_from_clicks(
    *,
    x0_tb: int,
    y0_tb: int,
    x1_tb: int,
    y1_tb: int,
    clicks: List[Dict[str, Any]],
    band_height_rel: float = 0.14,
) -> Dict[str, Dict[str, float]]:
    """
    Build field_boxes in *titleblock-relative* coordinates (0–1).

    We assume:
      - titleblock_x/y/width/height are 0–1 in page space
      - click x_rel, y_rel are 0–1 *within the titleblock*
    Strategy:
      - group clicks by field
      - for each field, we make a horizontal band across the whole TB width
        centred on the average click y_rel, with configurable height.
    """
    if x1_tb <= x0_tb or y1_tb <= y0_tb:
        return {}

    # Group clicks by field
    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for c in clicks:
        field = c.get("field")
        x_rel = c.get("x_rel")
        y_rel = c.get("y_rel")
        if (
            not field
            or not isinstance(x_rel, (float, int))
            or not isinstance(y_rel, (float, int))
        ):
            continue
        by_field.setdefault(field, []).append(c)

    field_boxes: Dict[str, Dict[str, float]] = {}

    half_band = band_height_rel / 2.0

    for field, pts in by_field.items():
        # Average Y of all clicks for that field
        ys = [float(p["y_rel"]) for p in pts]
        cy = sum(ys) / max(len(ys), 1)

        # Clamp centre
        cy = max(0.0, min(1.0, cy))

        y0_rel = max(0.0, cy - half_band)
        y1_rel = min(1.0, cy + half_band)

        # Full width of TB: x0_rel=0, x1_rel=1
        field_boxes[field] = {
            "x0_rel": 0.0,
            "x1_rel": 1.0,
            "y0_rel": y0_rel,
            "y1_rel": y1_rel,
        }

    return field_boxes


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


def map_img_point_to_pdf(
    x_img: float,
    y_img: float,
    img_width: int,
    img_height: int,
    page_rect: fitz.Rect,
) -> Tuple[float, float]:
    """
    Map a point (x_img, y_img) in image coordinates to a point in PDF page
    coordinates (bottom-left origin).
    """
    page_w = float(page_rect.width)
    page_h = float(page_rect.height)

    x_pdf = (x_img / float(img_width)) * page_w
    y_pdf = ((img_height - y_img) / float(img_height)) * page_h

    return x_pdf, y_pdf


def rects_intersect(a: fitz.Rect, b: fitz.Rect) -> bool:
    inter = a & b
    return inter.get_area() > 0.0


def choose_block_for_point(
    blocks: List[Tuple[Any, ...]],
    point_pdf: Tuple[float, float],
    tb_rect_pdf: fitz.Rect,
) -> Optional[Tuple[str, fitz.Rect]]:
    """
    Given a list of blocks from page.get_text('blocks'), choose the block that
    best corresponds to the click point *inside the titleblock*.

    Main pass:
      - Only consider blocks whose CENTER lies inside tb_rect_pdf.
      - Among those, prefer blocks that CONTAIN the click point.
      - If no block contains the point, choose the one whose center is
        closest to the click point.

    Fallback:
      - If no block has its center inside tb_rect_pdf, fall back to
        any block that intersects tb_rect_pdf (old behaviour),
        using the same priority/nearest-center logic.

    Returns (text, rect) or None.
    """
    px, py = point_pdf

    def rect_contains_point(r: fitz.Rect, x: float, y: float) -> bool:
        return (r.x0 <= x <= r.x1) and (r.y0 <= y <= r.y1)

    best_text: Optional[str] = None
    best_rect: Optional[fitz.Rect] = None
    best_priority = 999
    best_dist2 = float("inf")

    # ---------- FIRST PASS: blocks whose CENTER is inside the titleblock ----------
    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not text or not isinstance(text, str):
            continue

        rect = fitz.Rect(x0, y0, x1, y1)

        # Block centre must be inside the titleblock region
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        if not rect_contains_point(tb_rect_pdf, cx, cy):
            continue

        contains_click = rect_contains_point(rect, px, py)
        priority = 0 if contains_click else 1

        dx = cx - px
        dy = cy - py
        dist2 = dx * dx + dy * dy

        if priority < best_priority or (priority == best_priority and dist2 < best_dist2):
            best_priority = priority
            best_dist2 = dist2
            best_text = text.strip()
            best_rect = rect

    if best_text and best_rect:
        return best_text, best_rect

    # ---------- FALLBACK: any block that INTERSECTS the titleblock (old logic) ----------
    best_text = None
    best_rect = None
    best_priority = 999
    best_dist2 = float("inf")

    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not text or not isinstance(text, str):
            continue

        rect = fitz.Rect(x0, y0, x1, y1)
        # Old condition: any intersection at all
        if not rects_intersect(rect, tb_rect_pdf):
            continue

        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)

        contains_click = rect_contains_point(rect, px, py)
        priority = 0 if contains_click else 1

        dx = cx - px
        dy = cy - py
        dist2 = dx * dx + dy * dy

        if priority < best_priority or (priority == best_priority and dist2 < best_dist2):
            best_priority = priority
            best_dist2 = dist2
            best_text = text.strip()
            best_rect = rect

    if best_text and best_rect:
        # Optional debug:
        # print("[warn] choose_block_for_point: using INTERSECT fallback", file=sys.stderr)
        return best_text, best_rect

    return None


# ---------------------------------------------------------------------
# Fingerprint helpers (edge thumbnail)
# ---------------------------------------------------------------------

def compute_titleblock_fingerprint(
    img_gray: np.ndarray,
    x0_tb: int,
    y0_tb: int,
    x1_tb: int,
    y1_tb: int,
    size: int = FINGERPRINT_SIZE,
) -> Dict[str, Any]:
    """
    Compute a simple structural fingerprint of the titleblock region:

      - crop titleblock ROI
      - Canny edges
      - resize to size x size
      - store as flattened 0–255 list
    """
    roi = img_gray[y0_tb:y1_tb, x0_tb:x1_tb]
    if roi.size == 0:
        return {
            "type": "edge64",
            "width": size,
            "height": size,
            "data": [],
        }

    edges = cv2.Canny(roi, 80, 200)
    small = cv2.resize(edges, (size, size), interpolation=cv2.INTER_AREA)
    flat = small.astype("uint8").flatten().tolist()

    return {
        "type": "edge64",
        "width": size,
        "height": size,
        "data": flat,
    }


def compute_field_boxes_relative_to_titleblock(
    tb_rect_pdf: fitz.Rect,
    field_block_rects: Dict[str, fitz.Rect],
) -> Dict[str, Dict[str, float]]:
    """
    For each field, compute a normalised box (0–1) relative to the titleblock
    rect in PDF coordinates.
    """
    boxes: Dict[str, Dict[str, float]] = {}

    w_tb = float(tb_rect_pdf.width)
    h_tb = float(tb_rect_pdf.height)
    if w_tb <= 0.0 or h_tb <= 0.0:
        return boxes

    for name, rect in field_block_rects.items():
        # Intersection for robustness (in case block slightly overlaps)
        inter = rect & tb_rect_pdf
        if inter.get_area() <= 0.0:
            continue

        x0_rel = (inter.x0 - tb_rect_pdf.x0) / w_tb
        x1_rel = (inter.x1 - tb_rect_pdf.x0) / w_tb
        y0_rel = (inter.y0 - tb_rect_pdf.y0) / h_tb
        y1_rel = (inter.y1 - tb_rect_pdf.y0) / h_tb

        boxes[name] = {
            "x0_rel": float(max(0.0, min(1.0, x0_rel))),
            "y0_rel": float(max(0.0, min(1.0, y0_rel))),
            "x1_rel": float(max(0.0, min(1.0, x1_rel))),
            "y1_rel": float(max(0.0, min(1.0, y1_rel))),
        }

    return boxes


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

def process_page(
    client: Client,
    nas_root: Path,
    page_row: Dict[str, Any],
) -> None:
    page_id = page_row.get("id")
    document_id = page_row.get("document_id")
    page_number = page_row.get("page_number")
    image_rel = page_row.get("image_object_path")
    fp = page_row.get("_parsed_fingerprint")

    if page_id is None or document_id is None or page_number is None:
        print(
            f"[error] Page row missing id/document_id/page_number: {page_row}",
            file=sys.stderr,
        )
        return

    if not image_rel:
        print(f"[error] page_id={page_id} missing image_object_path", file=sys.stderr)
        return

    # Normalised titleblock coordinates from client (0–1, full page)
    tb_x_rel = page_row.get("titleblock_x")
    tb_y_rel = page_row.get("titleblock_y")
    tb_w_rel = page_row.get("titleblock_width")
    tb_h_rel = page_row.get("titleblock_height")

    if None in (tb_x_rel, tb_y_rel, tb_w_rel, tb_h_rel):
        print(f"[error] page_id={page_id} missing titleblock bbox", file=sys.stderr)
        return

    # Parsed fingerprint is expected to be a dict with "clicks"
    if not isinstance(fp, dict):
        fp = {}
    clicks = fp.get("clicks") or []
    if not isinstance(clicks, list):
        clicks = []

    if not clicks:
        print(
            f"[info] page_id={page_id}: no clicks in fingerprint; "
            f"skipping OCR + template creation"
        )
        return

    # Look up parent document_files row (still needed for consistency)
    doc_info = fetch_document_info(client, document_id)
    if doc_info is None:
        update_page_error(client, page_id, "Missing parent document_files row")
        return

    storage_path = doc_info.get("storage_object_path")
    if not storage_path:
        update_page_error(client, page_id, "document_files.storage_object_path is empty")
        return

    image_path = nas_root / image_rel
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        update_page_error(client, page_id, "Failed to load image for titleblock extraction")
        return

    img_h, img_w = img_gray.shape[:2]

    # Titleblock ROI in image pixels (from 0–1 fractions, full image)
    x0_tb = int(round(float(tb_x_rel) * img_w))
    y0_tb = int(round(float(tb_y_rel) * img_h))
    x1_tb = int(round((float(tb_x_rel) + float(tb_w_rel)) * img_w))
    y1_tb = int(round((float(tb_y_rel) + float(tb_h_rel)) * img_h))

    x0_tb = max(0, min(img_w, x0_tb))
    y0_tb = max(0, min(img_h, y0_tb))
    x1_tb = max(0, min(img_w, x1_tb))
    y1_tb = max(0, min(img_h, y1_tb))

    if x1_tb <= x0_tb or y1_tb <= y0_tb:
        update_page_error(client, page_id, "Titleblock ROI is empty after clamping")
        return

    # ---------- OCR debug root (NEW) ----------
    debug_root: Optional[Path] = None
    if DEBUG_OCR_CROPS:
        debug_root = image_path.parent / "debug_ocr"
        print(f"Debug Print sent to {debug_root}")
        try:
            debug_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(
                f"[warn] Could not create debug_ocr directory at {debug_root}: {exc}",
                file=sys.stderr,
            )
            debug_root = None

    # ---------- Build field_boxes from clicks ----------
    field_boxes = compute_field_boxes_from_clicks(
        x0_tb=x0_tb,
        y0_tb=y0_tb,
        x1_tb=x1_tb,
        y1_tb=y1_tb,
        clicks=clicks,
    )

    if not field_boxes:
        print(
            f"[info] page_id={page_id}: no field_boxes from clicks; "
            f"skipping OCR + template creation"
        )
        return

    field_texts: Dict[str, str] = {}

    tb_width = x1_tb - x0_tb
    tb_height = y1_tb - y0_tb

    for field_name, box in field_boxes.items():
        x0_rel = float(box["x0_rel"])
        x1_rel = float(box["x1_rel"])
        y0_rel = float(box["y0_rel"])
        y1_rel = float(box["y1_rel"])

        # Convert TB-relative (0–1) → image pixels
        fx0 = int(round(x0_tb + x0_rel * tb_width))
        fx1 = int(round(x0_tb + x1_rel * tb_width))
        fy0 = int(round(y0_tb + y0_rel * tb_height))
        fy1 = int(round(y1_tb + y1_rel * tb_height))

        if field_name == "revision":
            psm = 10  # single char
        else:
            psm = 7   # single line

        # ---------- Build debug path per field (NEW) ----------
        debug_path = None
        if DEBUG_OCR_CROPS and debug_root is not None:
            safe_field = "".join(ch if ch.isalnum() else "_" for ch in str(field_name))
            debug_path = debug_root / (
                f"page{page_number}_field-{safe_field}_pageid-{page_id}.png"
            )

        text = ocr_text_from_region(
            img_gray,
            fx0,
            fy0,
            fx1,
            fy1,
            psm=psm,
            debug_output_path=debug_path,
        )

        print(
            f"[info] page_id={page_id} field={field_name!r} "
            f"tb_box_rel=({x0_rel:.3f},{y0_rel:.3f},{x1_rel:.3f},{y1_rel:.3f}) "
            f"pix=({fx0},{fy0},{fx1},{fy1}) OCR={text!r}"
        )

        if text:
            field_texts[field_name] = text

    # Write extracted text back to document_pages
    updates: Dict[str, Optional[str]] = {}
    if "drawing_number" in field_texts:
        updates["drawing_number"] = field_texts["drawing_number"]
    if "drawing_title" in field_texts:
        updates["drawing_title"] = field_texts["drawing_title"]
    if "revision" in field_texts:
        updates["revision"] = field_texts["revision"]

    if updates:
        update_page_fields(client, page_id, updates)
    else:
        print(
            f"[info] page_id={page_id}: OCR found no text for any field"
        )

    # ----- Template JSON (fingerprint + geometry) -----

    fingerprint = compute_titleblock_fingerprint(
        img_gray,
        x0_tb,
        y0_tb,
        x1_tb,
        y1_tb,
    )

    required_fields = {"drawing_number", "drawing_title", "revision"}
    missing = required_fields.difference(field_boxes.keys())
    if missing:
        print(
            f"[warn] page_id={page_id}: missing field_boxes for {sorted(missing)}; "
            f"skipping titleblock_template creation"
        )
        return

    template_json: Dict[str, Any] = {
        "version": 1,
        "fingerprint": fingerprint,
        "field_boxes": field_boxes,
    }

    name = None
    if "drawing_number" in field_texts:
        name = f"titleblock_{field_texts['drawing_number']}"

    template_id = insert_titleblock_template(client, page_id, template_json, name=name)
    if template_id is not None:
        update_page_template_link(client, page_id, template_id, confidence=1.0)


def run_once() -> int:
    client = create_supabase_client()
    if client is None:
        print("[error] Supabase client not created", file=sys.stderr)
        return 1

    nas_root = get_nas_root()
    print(f"[info] Using NAS root: {nas_root}")

    pages = fetch_tagged_pages(client, limit=MAX_PAGES_PER_RUN)
    if not pages:
        print("[info] No tagged pages ready for extraction")
        return 0

    for row in pages:
        process_page(client, nas_root, row)

    return 0


def main() -> int:
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
