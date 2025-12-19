# app/config.py

from typing import Tuple

# -------------------------------------------------------------------------
# Configuration constants (Power of 10: explicit, bounded work)
# -------------------------------------------------------------------------

MAX_DOCS_PER_RUN: int = 10  # Upper bound on documents per run
DOC_NAS_ROOT_ENV: str = "DOC_NAS_ROOT"
DERIVED_BUCKET_ENV: str = "DOC_DERIVED_BUCKET"

# Loop / orchestration configuration
WORKER_MODE_ENV: str = "WORKER_MODE"              # "once" or "loop"
WORKER_LOOP_SLEEP_ENV: str = "WORKER_LOOP_SLEEP"  # seconds between cycles
WORKER_MAX_CYCLES_ENV: str = "WORKER_MAX_CYCLES"  # 0 = run forever

# Titleblock-related subworkers to invoke after PDF/page work.
# These must be runnable as: python -m <module_name>
TITLEBLOCK_WORKER_MODULES: Tuple[str, ...] = (
    "app.titleblock_match_worker",
    "app.titleblock_extract_worker",
)

# Bound how many pages we render per document
MAX_PAGES_PER_DOC: int = 50

# How we analyse for reference docs
MAX_PAGES_TO_ANALYSE: int = 100
TEXT_RATIO_THRESHOLD: float = 0.12   # fraction of page area covered by text
TEXT_CHAR_THRESHOLD: int = 400       # minimum chars per page to call it "text-heavy"
