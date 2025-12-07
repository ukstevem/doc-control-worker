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
    Parse the JSON fingerprint into a Python dict.

    We now expect a v2-style object with an "areas" array containing
    boxed regions inside the title-block (all coords normalised 0–1
    relative to the title-block crop).
    """
    if raw_fp is None:
        return None

    if isinstance(raw_fp, str):
        try:
            fp = json.loads(raw_fp)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw_fp, dict):
        fp = raw_fp
    else:
        return None

    areas = fp.get("areas")
    if not isinstance(areas, list) or not areas:
        return None

    return fp


def fetch_tagged_pages(client: Client, limit: int = MAX_PAGES_PER_RUN) -> List[Dict[str, Any]]:
    """
    Fetch document_pages rows ready for titleblock extraction:

      - status in ('tagged', 'Tagged')
      - image_object_path set
      - titleblock_x/y/width/height set (0–1 fractions)
      - titleblock_fingerprint has areas
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
            print(
                f"[info] skipping page_id={page_id} because it has no fingerprint.areas"
            )
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
    Insert a new document_titleblock_templates row and return its id.
    We keep this simple for now; matching/deduplication is a later concern.
    """
    payload: Dict[str, Any] = {
        "sample_document_page_id": sample_page_id,
        "template": template_json,
    }
    if name:
        payload["name"] = name

    try:
        response = client.table("document_titleblock_templates").insert(payload).execute()
    except Exception as exc:
        print(f"[error] Failed to insert titleblock_template: {exc}", file=sys.stderr)
        return None

    rows = getattr(response, "data", None) or []
    if not rows:
        print("[error] document_titleblock_templates insert returned no rows", file=sys.stderr)
        return None

    template_id = rows[0].get("id")
    print(f"[info] Created titleblock_template id={template_id}")
    return template_id


# ---------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------

def ocr_vertical_text_labels(
    img_gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> List[str]:
    """
    OCR for labels that are rotated 90° (e.g. 'DWG NUM').

    We:
    - Crop ROI from img_gray
    - Binarise
    - Rotate 90° so vertical text becomes horizontal
    - Run Tesseract and keep mostly-letter tokens as "labels"
    """
    h, w = img_gray.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    if x1 <= x0 or y1 <= y0:
        return []

    roi = img_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    try:
        _, roi_bin = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except Exception:
        roi_bin = roi

    # Rotate 90° clockwise so vertical text becomes horizontal
    try:
        rot = cv2.rotate(roi_bin, cv2.ROTATE_90_CLOCKWISE)
    except Exception:
        rot = roi_bin

    try:
        rot = cv2.resize(
            rot,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
    except Exception:
        pass

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.:"
    text = pytesseract.image_to_string(rot, config=config)
    if not text:
        return []

    labels: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for tok in line.split():
            tok = tok.strip().upper()
            if not tok:
                continue
            digits = sum(ch.isdigit() for ch in tok)
            letters = sum("A" <= ch <= "Z" for ch in tok)
            # Treat mostly-letter tokens as labels (e.g. 'DWG', 'DWGNUM')
            if letters >= 2:
                labels.append(tok)

    # Deduplicate
    return sorted(set(labels))


def clean_drawing_number(raw_text: str) -> str:
    """
    Clean drawing_number OCR result.

    Handles:
      - leading junk digits like '23C25001-1-1042' → 'C25001-1-1042'
      - removal of obvious label words (DWG, NUM, DRAWING, TITLE, REV, ...)
    """
    if not raw_text:
        return ""

    t = raw_text.strip().upper()
    if not t:
        return ""

    # --------------------------------------------------
    # Step 1: character-level fix for leading junk digits
    # --------------------------------------------------
    # Find first alphabetic character (likely the 'C' in C25001-1-1042)
    first_letter_idx = -1
    for i, ch in enumerate(t):
        if "A" <= ch <= "Z":
            first_letter_idx = i
            break

    if first_letter_idx > 0:
        prefix = t[:first_letter_idx]
        suffix = t[first_letter_idx:]

        # Prefix is considered "junk" if:
        # - it is short (<= 2 chars)
        # - and contains only digits (optionally punctuation like -_/.:)
        if prefix and len(prefix) <= 2:
            if all((c.isdigit() or c in "-_.:/") for c in prefix):
                # Check that suffix looks like a real drawing number:
                digits_suffix = sum(c.isdigit() for c in suffix)
                letters_suffix = sum("A" <= c <= "Z" for c in suffix)
                if digits_suffix >= 3 and letters_suffix >= 1:
                    # Drop the numeric prefix (e.g. '23')
                    t = suffix

    # --------------------------------------------------
    # Step 2: token-level label removal & sanity
    # --------------------------------------------------
    LABEL_WORDS = {
        "DWG",
        "NUM",
        "NO",
        "NO.",
        "DRAWING",
        "TITLE",
        "REV",
        "REVISION",
    }

    tokens = t.split()
    if not tokens:
        return t

    cleaned_tokens: List[str] = []
    has_good_pattern = False

    for tok in tokens:
        tok_u = tok.upper().strip(":")
        if tok_u in LABEL_WORDS:
            # strip labels like 'DWG', 'NUM', etc
            continue

        digits = sum(ch.isdigit() for ch in tok_u)
        letters = sum("A" <= ch <= "Z" for ch in tok_u)

        # "Good" drawing-number-like token:
        if (digits >= 1 and letters >= 1 and len(tok_u) >= 3) or digits >= 3:
            cleaned_tokens.append(tok_u)
            has_good_pattern = True
        else:
            # ambiguous short tokens (e.g. '23') – we only keep them
            # if we don't find any better tokens
            cleaned_tokens.append(tok_u)

    if has_good_pattern and cleaned_tokens:
        return " ".join(cleaned_tokens).strip()

    if cleaned_tokens:
        return " ".join(cleaned_tokens).strip()

    return t

def ocr_text_from_region(
    img_gray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    psm: int = 7,
    whitelist: Optional[str] = None,
    multi_line: bool = False,   # kept for interface; we choose psm per field
    debug_output_path: Optional[Path] = None,
    field_name: Optional[str] = None,
) -> str:
    """
    OCR a region of the page.

    For drawing_number / revision we:
      - drop tall-skinny (rotated) words,
      - drop much smaller label text (smaller font),
      - drop known label words like DWG / NUM / DRAWING / TITLE / REV.

    For drawing_title we keep it simple:
      - just use Tesseract's line-based output so we don't chop words
        or scramble the order, and let normalise_ocr_for_field() do
        the label cleanup.
    """
    if pytesseract is None:
        return ""

    h_img, w_img = img_gray.shape[:2]
    x0 = max(0, min(w_img, x0))
    x1 = max(0, min(w_img, x1))
    y0 = max(0, min(h_img, y0))
    y1 = max(0, min(h_img, y1))

    if x1 <= x0 or y1 <= y0:
        return ""

    roi = img_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return ""

    # Upscale a bit to help OCR
    try:
        roi_big = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    except Exception:
        roi_big = roi

    # Light denoise + binarise
    roi_blur = cv2.GaussianBlur(roi_big, (3, 3), 0)
    _, roi_bin = cv2.threshold(
        roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Optional debug image
    if debug_output_path is not None:
        try:
            cv2.imwrite(str(debug_output_path), roi_bin)
        except Exception as exc:
            print(
                f"[warn] ocr_text_from_region: failed to write debug image: {exc}",
                file=sys.stderr,
            )

    # Build Tesseract config
    config = f"--psm {int(psm)} -c preserve_interword_spaces=1"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    # ------------------------------------------------------------------
    # SIMPLE PATH FOR DRAWING TITLE: don't overthink it, keep order.
    # ------------------------------------------------------------------
    if field_name == "drawing_title":
        try:
            txt = pytesseract.image_to_string(roi_bin, config=config)
        except Exception as exc:
            print(
                f"[warn] ocr_text_from_region(drawing_title): image_to_string failed: {exc}",
                file=sys.stderr,
            )
            return ""
        return txt.strip()

    # ------------------------------------------------------------------
    # TOKEN-BASED PATH for drawing_number, revision, and others
    # ------------------------------------------------------------------
    try:
        data = pytesseract.image_to_data(
            roi_bin,
            output_type=pytesseract.Output.DICT,
            config=config,
        )
    except Exception as exc:
        print(
            f"[warn] ocr_text_from_region: image_to_data failed: {exc}",
            file=sys.stderr,
        )
        try:
            txt = pytesseract.image_to_string(roi_bin, config=config)
        except Exception:
            return ""
        return txt.strip()

    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    tokens: List[Dict[str, Any]] = []
    for i in range(len(texts)):
        raw = texts[i] or ""
        text = raw.strip()
        if not text:
            continue

        try:
            conf = float(confs[i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 40.0:
            # Very low-confidence noise
            continue

        try:
            x = int(lefts[i])
            y = int(tops[i])
            w = int(widths[i])
            h = int(heights[i])
        except (TypeError, ValueError):
            continue

        tokens.append({"t": text, "x": x, "y": y, "w": w, "h": h})

    if not tokens:
        txt = pytesseract.image_to_string(roi_bin, config=config)
        return txt.strip()

    # --- Extra filtering for the key fields (NOT drawing_title now) ---
    if field_name in ("drawing_number", "revision"):
        # 1) Drop clearly vertical / tall-skinny words (rotated labels)
        non_vertical: List[Dict[str, Any]] = []
        for tok in tokens:
            w = max(1, tok["w"])
            h = tok["h"]
            # heuristic: rotated glyphs are much taller than they are wide
            if h > w * 1.5:
                continue
            non_vertical.append(tok)
        if non_vertical:
            tokens = non_vertical

        if tokens:
            # 2) Drop much smaller label text (based on height)
            heights_list = [t["h"] for t in tokens]
            max_h = max(heights_list)
            size_cutoff = max_h * 0.7  # keep only the big stuff

            big_tokens = [t for t in tokens if t["h"] >= size_cutoff]
            if big_tokens:
                tokens = big_tokens

        if tokens:
            # 3) Drop obvious label words
            LABEL_WORDS = {
                "DWG",
                "NUM",
                "NO",
                "NO.",
                "DRAWING",
                "TITLE",
                "REV",
                "REVISION",
            }
            cleaned: List[Dict[str, Any]] = []
            for tok in tokens:
                norm = tok["t"].upper().strip(":")
                if norm in LABEL_WORDS:
                    continue
                cleaned.append(tok)
            if cleaned:
                tokens = cleaned

    # If nothing survived filtering, fall back to plain OCR
    if not tokens:
        txt = pytesseract.image_to_string(roi_bin, config=config)
        return txt.strip()

    # --- Rebuild text in reading order (top-to-bottom, left-to-right) ---
    heights_list = [t["h"] for t in tokens]
    max_h = max(heights_list)
    line_gap = max_h * 0.6

    tokens.sort(key=lambda t: (t["y"], t["x"]))

    lines: List[str] = []
    current_y: Optional[int] = None
    current_words: List[str] = []

    for tok in tokens:
        if current_y is None:
            current_y = tok["y"]
            current_words = [tok["t"]]
            continue

        if abs(tok["y"] - current_y) <= line_gap:
            # same line
            current_words.append(tok["t"])
        else:
            # new line
            lines.append(" ".join(current_words))
            current_y = tok["y"]
            current_words = [tok["t"]]

    if current_words:
        lines.append(" ".join(current_words))

    result = "\n".join(lines).strip()
    if not result:
        result = pytesseract.image_to_string(roi_bin, config=config).strip()

    return result

def normalise_ocr_for_field(field_name: str, text: str) -> str:
    """
    Light, field-specific cleanup of OCR results.

    Drawing-number-specific logic is handled separately by clean_drawing_number().
    """
    if not text:
        return ""

    t = text.strip()

    if field_name == "revision":
        # Typical issues: stray spaces, lowercase, '|' instead of '1'
        t = t.replace(" ", "")
        t = t.replace("|", "1")
        t = t.upper()
        if len(t) > 4:
            t = t[:4]
        return t

    if field_name == "drawing_title":
        # We want to strip label-like prefixes such as:
        #   "DRAWING TITLE: ..."
        #   "Drg. Title: ..."
        #   "DRG TITLE ..."
        # while keeping the actual title text.
        lines = t.splitlines()
        cleaned_segments: List[str] = []

        LABEL_PREFIXES = [
            "DRAWING TITLE",
            "DRG. TITLE",
            "DRG TITLE",
            "DRG.",
            "DRG",
            "TITLE",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            up = stripped.upper()

            # Does this line start with any known label prefix?
            matched_prefix = None
            for pref in LABEL_PREFIXES:
                if up.startswith(pref):
                    matched_prefix = pref
                    break

            if matched_prefix is not None:
                # Try to keep only the part after ':' if present
                if ":" in stripped:
                    label_part, rest = stripped.split(":", 1)
                    rest = rest.strip()
                    if rest:
                        cleaned_segments.append(rest)
                    # If nothing after colon, treat as pure label and drop
                else:
                    # Line is just a label like "DRG TITLE" → drop it
                    pass
                continue

            # Non-label line: keep as-is
            cleaned_segments.append(stripped)

        if cleaned_segments:
            return " ".join(cleaned_segments).strip()

        # If we somehow stripped everything, fall back to original text
        return t

    # Fallback: minimal cleanup for other fields
    return t

def compute_field_boxes_from_clicks(
    *,
    img_width: int,
    img_height: int,
    x0_tb: int,
    y0_tb: int,
    x1_tb: int,
    y1_tb: int,
    clicks: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Build *titleblock-relative* boxes from TB-relative clicks.

    - We assume click x_rel / y_rel are already 0–1 inside the titleblock.
    - We create a small box around each click, expressed again 0–1 in TB space.
    """

    # How “wide” and “tall” each field box is, as a fraction of the TB size.
    # You can tweak these if the crops are still too big/small.
    BOX_HALF_WIDTH_TB  = 0.18   # 18% of TB width either side of the click
    BOX_HALF_HEIGHT_TB = 0.08   # 8% of TB height above/below the click

    field_boxes: Dict[str, Dict[str, float]] = {}

    for click in clicks:
        field_name = click.get("field")
        x_rel = click.get("x_rel")
        y_rel = click.get("y_rel")

        if (
            not field_name
            or not isinstance(x_rel, (float, int))
            or not isinstance(y_rel, (float, int))
        ):
            continue

        # Click is already TB-relative 0–1 → clamp for safety
        cx = max(0.0, min(1.0, float(x_rel)))
        cy = max(0.0, min(1.0, float(y_rel)))

        # Tight box around the click, in TB-relative coords
        x0_rel = max(0.0, cx - BOX_HALF_WIDTH_TB)
        x1_rel = min(1.0, cx + BOX_HALF_WIDTH_TB)
        y0_rel = max(0.0, cy - BOX_HALF_HEIGHT_TB)
        y1_rel = min(1.0, cy + BOX_HALF_HEIGHT_TB)

        field_boxes[field_name] = {
            "x0_rel": x0_rel,
            "x1_rel": x1_rel,
            "y0_rel": y0_rel,
            "y1_rel": y1_rel,
        }

    return field_boxes

def compute_field_boxes_from_areas(
    *,
    areas: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Build titleblock-relative field boxes from client-supplied areas.

    Each area is expected to have:
      - field: field name (e.g. 'drawing_number')
      - x_rel, y_rel: 0–1 top-left within the titleblock
      - width_rel, height_rel: 0–1 size within the titleblock

    Returns:
      {
        "drawing_number": {"x0_rel": ..., "y0_rel": ..., "x1_rel": ..., "y1_rel": ...},
        ...
      }
    """
    field_boxes: Dict[str, Dict[str, float]] = {}

    for item in areas:
        try:
            field_name = str(item.get("field", "")).strip()
            x_rel = float(item["x_rel"])
            y_rel = float(item["y_rel"])
            w_rel = float(item["width_rel"])
            h_rel = float(item["height_rel"])
        except (KeyError, TypeError, ValueError):
            continue

        if not field_name:
            continue

        # Clamp to [0, 1]
        x_rel = max(0.0, min(1.0, x_rel))
        y_rel = max(0.0, min(1.0, y_rel))
        w_rel = max(0.0, min(1.0, w_rel))
        h_rel = max(0.0, min(1.0, h_rel))

        x0_rel = x_rel
        y0_rel = y_rel
        x1_rel = min(1.0, x_rel + w_rel)
        y1_rel = min(1.0, y_rel + h_rel)

        if x1_rel <= x0_rel or y1_rel <= y0_rel:
            continue

        field_boxes[field_name] = {
            "x0_rel": x0_rel,
            "y0_rel": y0_rel,
            "x1_rel": x1_rel,
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

def mean_abs_diff(a, b):
    if not a or not b:
        return 1.0
    n = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(n)) / n

def detect_grid_lines(img_gray):
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=30, maxLineGap=10)
    v, h = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < 5:
                v.append((x1 + x2) // 2)
            elif abs(y2 - y1) < 5:
                h.append((y1 + y2) // 2)
    return sorted(v), sorted(h)

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

    # Parsed fingerprint is expected to be a dict with an "areas" list (v2 workflow)
    if not isinstance(fp, dict):
        fp = {}
    areas = fp.get("areas") or []
    if not isinstance(areas, list) or not areas:
        print(
            f"[info] page_id={page_id}: no 'areas' in fingerprint; "
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

    # ---------- OCR debug root ----------
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

    # ---------- Build field_boxes from areas ----------
    field_boxes = compute_field_boxes_from_areas(areas=areas)

    if not field_boxes:
        print(
            f"[info] page_id={page_id}: no field_boxes from areas; "
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
        fy1 = int(round(y0_tb + y1_rel * tb_height))

        # Small inward margin to avoid borders
        margin = 2
        fx0 = max(x0_tb, fx0 + margin)
        fy0 = max(y0_tb, fy0 + margin)
        fx1 = min(x1_tb, fx1 - margin)
        fy1 = min(y1_tb, fy1 - margin)

        # ---------- Per-field OCR config ----------
        if field_name == "revision":
            psm = 10
            whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            multi_line = False
        elif field_name == "drawing_number":
            psm = 7
            whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/ ."
            multi_line = False
        else:
            # drawing_title or any other texty field
            psm = 6  # block of text
            whitelist = None
            multi_line = True

        # ---------- Build debug path per field ----------
        debug_path = None
        if DEBUG_OCR_CROPS and debug_root is not None:
            safe_field = "".join(ch if ch.isalnum() else "_" for ch in str(field_name))
            debug_path = debug_root / (
                f"page{page_number}_field-{safe_field}_pageid-{page_id}.png"
            )

        # Field-aware OCR with label/size filtering
        raw_text = ocr_text_from_region(
            img_gray,
            fx0,
            fy0,
            fx1,
            fy1,
            psm=psm,
            whitelist=whitelist,
            multi_line=multi_line,
            debug_output_path=debug_path,
            field_name=field_name,
        )

        if field_name == "drawing_number":
            text = clean_drawing_number(raw_text)
        else:
            text = normalise_ocr_for_field(field_name, raw_text)

        print(
            f"[info] page_id={page_id} field={field_name!r} "
            f"tb_box_rel=({x0_rel:.3f},{y0_rel:.3f},{x1_rel:.3f},{y1_rel:.3f}) "
            f"pix=({fx0},{fy0},{fx1},{fy1}) OCR_raw={raw_text!r} OCR_clean={text!r}"
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

    # New: page bbox (0–1 on full page) and normalised grid from *_grid.json
    page_bbox_norm = {
        "x": float(tb_x_rel),
        "y": float(tb_y_rel),
        "w": float(tb_w_rel),
        "h": float(tb_h_rel),
    }

    grid_norm: Dict[str, Any] = {}
    try:
        grid_json_path = image_path.with_name(image_path.stem + "_grid.json")
        if grid_json_path.exists():
            with grid_json_path.open("r", encoding="utf-8") as f:
                g = json.load(f)
            w_crop = max(1, int(g["crop_size"]["width"]))
            h_crop = max(1, int(g["crop_size"]["height"]))
            v_px = g.get("vertical_lines", []) or []
            h_px = g.get("horizontal_lines", []) or []
            v_norm = [round(x / w_crop, 3) for x in v_px]
            h_norm = [round(y / h_crop, 3) for y in h_px]
            grid_norm = {
                "verticals_norm": v_norm,
                "horizontals_norm": h_norm,
            }
    except Exception as exc:
        print(f"[warn] page_id={page_id}: could not attach grid to template: {exc}")

    required_fields = {"drawing_number", "drawing_title", "revision"}
    missing = required_fields.difference(field_boxes.keys())
    if missing:
        print(
            f"[warn] page_id={page_id}: missing field_boxes for {sorted(missing)}; "
            f"skipping titleblock_template creation"
        )
        return

    template_json: Dict[str, Any] = {
        "version": 2,
        "page_bbox_norm": page_bbox_norm,
        "grid": grid_norm,
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
