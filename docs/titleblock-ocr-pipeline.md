# Title-block OCR pipeline (v2)

This document describes the end-to-end pipeline from **uploaded PDF** to
**page-level OCR of the title-block** (`drawing_number`, `drawing_title`,
`revision`).

It also lists the Docker commands to run each stage while we’re testing.

---

## High-level flow

1. **Upload PDF** → row in `document_files` with `status='uploaded'`.
2. **Render first page to PNG** → row in `document_pages` with `status='rendered'`.
3. **User tags title-block + field boxes** in the web client → `document_pages`
   updated with:
   - `titleblock_x`, `titleblock_y`, `titleblock_width`, `titleblock_height`
   - `titleblock_fingerprint` JSON (v2: `areas[...]`)
   - `status='tagged'`
4. **Grid worker** runs on the tagged page, crops the title-block and finds the
   line grid → writes `p1_grid.json` + `p1_grid_debug.png`.
5. **Extract worker**:
   - Reads the v2 `areas` fingerprint.
   - OCRs each textbox (drawing number, title, revision) from the title-block.
   - Writes text back to `document_pages`.
   - Builds a `document_titleblock_templates` row containing:
     `fingerprint` (edge64), `field_boxes`, `grid`, `page_bbox_norm`.
6. **Match worker**:
   - For other pages that have a title-block bbox but no template link,
     compares fingerprints and grid against existing templates.
   - If a match is trusted, auto-tags the page to look like a client-tagged
     page (bbox + v2 `areas`) and/or extracts OCR using the template boxes.

---

## Environment and storage

- **NAS root** for all raw + derived files:
  - `DOC_NAS_ROOT=/data/input` (inside the `pdf_worker` container).
- **Derived page images** (first-page PNGs) follow:

  ```text
  derived/pages/enquiries/{enquirynumber}/{document_id}/p{page}.png
  derived/pages/projects/{projectnumber}/{document_id}/p{page}.png
