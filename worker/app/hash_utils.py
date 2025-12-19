# app/hash_utils.py

import hashlib
import sys
from pathlib import Path
from typing import Optional


def compute_file_sha256(path: Path) -> Optional[str]:
    """
    Compute SHA-256 for a file in bounded-size chunks.

    Returns the hex digest string or None on error.
    """
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    except OSError as exc:
        print(f"[error] Failed to read file for hashing: {path} ({exc})", file=sys.stderr)
        return None

    return h.hexdigest()
