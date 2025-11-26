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

# Very conservative: only match when fingerprints are very close
MIN_SIMILARITY_THRESHOLD = 0.8


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
    Fetch titleblock_templates rows with template JSON.
    """
    try:
        response = (
            client.table("titleblock_templates")
            .select("id, template")
            .limit(MAX_TEMPLATES)
            .execute()
        )
    except Exception as exc:
        print(f"[error] Failed to fetch titleblock_templates: {exc}", file=sys.stderr)
        return []

    rows = getattr(response, "data", None) or []
    print(f"[info] fetched {len(rows)} titleblock_template row(s) from Supabase")

    valid: List[Dict[str, Any]] = []
    for row in rows:
        tpl = row.get("template")
        if not isinstance(tpl, dict):
            continue
        fp = tpl.get("fingerprint")
        fb = tpl.get("field_boxes")
        if not isinstance(fp, dict) or not isinstance(fb, dict):
            continue
        data = fp.get("data")
        if not isinstance(data, list) or not data:
            continue
        row["_fingerprint"] = fp
        row["_field_boxes"] = fb
        valid.append(row)

    print(f"[info] {len(valid)} template(s) have usable fingerprint + field_boxes")
    return valid


def fetch_pages_to_match(client: Client, limit: int = MAX_PAGES_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch document_pages that have a titleblock bbox + image, but no matched template yet.
    We keep the server-side filter simple and filter further in Python.
    """
    try:
        response = (
            client.table("document_pages")
            .select(
                "id, document_id, page_number, status, "
                "image_object_path, "
                "titleblock_x, titleblock_y, titleblock_width, titleblock_height, "
                "matched_titleblock_template_id"
            )
            .limit(limit)
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
            continue
        if not row.get("image_object_path"):
            continue
        if any(
            row.get(name) is None
            for name in ("titleblock_x", "titleblock_y", "titleblock_width", "titleblock_height")
        ):
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
    Compute the same edge64 fingerprint used in titleblock_templates:
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
        print(f"[error] Page row missing id/document_id/page_number: {page_row}", file=sys.stderr)
        return

    if not image_rel:
        print(f"[error] page_id={page_id} missing image_object_path", file=sys.stderr)
        return

    tb_x_rel = page_row.get("titleblock_x")
    tb_y_rel = page_row.get("titleblock_y")
    tb_w_rel = page_row.get("titleblock_width")
    tb_h_rel = page_row.get("titleblock_height")

    if None in (tb_x_rel, tb_y_rel, tb_w_rel, tb_h_rel):
        print(f"[error] page_id={page_id} missing titleblock bbox", file=sys.stderr)
        return

    doc_info = fetch_document_info(client, document_id)
    if doc_info is None:
        print(f"[error] Missing document_files row for page_id={page_id}", file=sys.stderr)
        return

    storage_path = doc_info.get("storage_object_path")
    if not storage_path:
        print(f"[error] document_files.storage_object_path is empty (page_id={page_id})", file=sys.stderr)
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

    # Titleblock ROI in image pixels (from 0–1 fractions)
    x0_tb = int(round(float(tb_x_rel) * img_w))
    y0_tb = int(round(float(tb_y_rel) * img_h))
    x1_tb = int(round((float(tb_x_rel) + float(tb_w_rel)) * img_w))
    y1_tb = int(round((float(tb_y_rel) + float(tb_h_rel)) * img_h))

    x0_tb = max(0, min(img_w, x0_tb))
    y0_tb = max(0, min(img_h, y0_tb))
    x1_tb = max(0, min(img_w, x1_tb))
    y1_tb = max(0, min(img_h, y1_tb))

    if x1_tb <= x0_tb or y1_tb <= y0_tb:
        print(f"[error] Titleblock ROI is empty after clamping (page_id={page_id})", file=sys.stderr)
        return

    # Compute fingerprint for this page's titleblock
    fp_vec = compute_titleblock_fingerprint(img_gray, x0_tb, y0_tb, x1_tb, y1_tb, size=FINGERPRINT_SIZE)
    if not fp_vec:
        print(f"[warn] page_id={page_id}: empty fingerprint; cannot match template")
        return

    # Compare against all templates
    best_template_id: Optional[Any] = None
    best_field_boxes: Optional[Dict[str, Dict[str, float]]] = None
    best_sim = 0.0

    for tpl in templates:
        tpl_id = tpl.get("id")
        fp = tpl.get("_fingerprint") or {}
        fb = tpl.get("_field_boxes") or {}
        if not fp or not fb:
            continue
        data = fp.get("data")
        if not isinstance(data, list) or not data:
            continue

        sim = compute_similarity(fp_vec, data)
        if sim > best_sim:
            best_sim = sim
            best_template_id = tpl_id
            best_field_boxes = fb

    if best_template_id is None or best_field_boxes is None:
        print(f"[info] page_id={page_id}: no template match candidates")
        return

    print(
        f"[info] page_id={page_id}: best template id={best_template_id} "
        f"similarity={best_sim:.3f}"
    )

    if best_sim < MIN_SIMILARITY_THRESHOLD:
        print(
            f"[info] page_id={page_id}: best similarity {best_sim:.3f} "
            f"below threshold {MIN_SIMILARITY_THRESHOLD:.3f}; not auto-linking"
        )
        return

    # Extract fields using the matched template's field_boxes
    field_texts = extract_fields_using_template(
        pdf_path=pdf_path,
        page_number=int(page_number),
        img_width=img_w,
        img_height=img_h,
        x0_tb=x0_tb,
        y0_tb=y0_tb,
        x1_tb=x1_tb,
        y1_tb=y1_tb,
        field_boxes=best_field_boxes,
    )

    if field_texts:
        update_page_fields(client, page_id, {
            # Template field names are expected to match these keys
            "drawing_number": field_texts.get("drawing_number"),
            "drawing_title": field_texts.get("drawing_title"),
            "revision": field_texts.get("revision"),
        })

    # Link page to template with similarity as confidence
    update_page_template_link(client, page_id, best_template_id, confidence=best_sim)


def run_once() -> int:
    client = create_supabase_client()
    if client is None:
        print("[error] Supabase client not created", file=sys.stderr)
        return 1

    nas_root = get_nas_root()
    print(f"[info] Using NAS root: {nas_root}")

    templates = fetch_templates(client)
    if not templates:
        print("[info] No usable titleblock_templates found; nothing to match")
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
