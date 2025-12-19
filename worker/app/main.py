# app/main.py

import os
import sys
import subprocess
import time
from pathlib import Path

from app.config import (
    DERIVED_BUCKET_ENV,
    WORKER_MODE_ENV,
    WORKER_LOOP_SLEEP_ENV,
    WORKER_MAX_CYCLES_ENV,
    TITLEBLOCK_WORKER_MODULES,
)
from app.db_utils import (
    create_supabase_client,
    ping_document_files_table,
    fetch_uploaded_pdfs,
)
from app.path_utils import get_nas_root
from app.document_worker import process_document_row


def run_once() -> int:
    """Run one bounded batch of work and exit."""
    client = create_supabase_client()
    if client is None:
        return 1

    nas_root: Path = get_nas_root()
    derived_bucket = os.getenv(DERIVED_BUCKET_ENV, "doc_nas_derived")

    ping_document_files_table(client)

    print(f"[info] Using NAS root: {nas_root}")
    print(f"[info] Using derived bucket: {derived_bucket}")

    rows = fetch_uploaded_pdfs(client)
    if not rows:
        return 0

    for row in rows:
        process_document_row(client, nas_root, derived_bucket, row)

    return 0


def run_subworker_module(module_name: str) -> None:
    """
    Run a secondary worker module via `python -m <module_name>`.

    This lets main orchestrate the titleblock workers without depending
    on their internal implementation details.
    """
    print(f"[info] Running subworker module: {module_name}", flush=True)

    try:
        completed = subprocess.run(
            [sys.executable, "-m", module_name],
            check=False,
        )
    except Exception as exc:
        print(
            f"[error] Exception while running {module_name}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return

    if completed.returncode != 0:
        print(
            f"[warn] {module_name} exited with code {completed.returncode}",
            file=sys.stderr,
            flush=True,
        )


def run_loop() -> int:
    """
    Run the main PDF worker and the titleblock workers in a simple loop.

    Controlled by env vars:
      - WORKER_LOOP_SLEEP: seconds to sleep between cycles (default 10)
      - WORKER_MAX_CYCLES: if > 0, stop after this many cycles (default 0 = forever)
    """
    sleep_str = os.getenv(WORKER_LOOP_SLEEP_ENV, "10")
    max_cycles_str = os.getenv(WORKER_MAX_CYCLES_ENV, "0")

    try:
        sleep_seconds = int(sleep_str)
    except ValueError:
        sleep_seconds = 10

    try:
        max_cycles = int(max_cycles_str)
    except ValueError:
        max_cycles = 0

    cycle = 0

    print(
        f"[info] Starting worker loop: sleep={sleep_seconds}s max_cycles={max_cycles}",
        flush=True,
    )

    while True:
        exit_code = run_once()
        if exit_code != 0:
            print(
                f"[warn] run_once() returned non-zero exit code {exit_code}",
                file=sys.stderr,
                flush=True,
            )

        for module_name in TITLEBLOCK_WORKER_MODULES:
            run_subworker_module(module_name)

        cycle += 1
        if 0 < max_cycles <= cycle:
            print("[info] Max cycles reached; exiting loop.", flush=True)
            break

        time.sleep(sleep_seconds)

    return 0


def main() -> int:
    """Entry point. Chooses between one-shot and loop modes."""
    mode = os.getenv(WORKER_MODE_ENV, "once").lower()
    if mode == "loop":
        return run_loop()
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
