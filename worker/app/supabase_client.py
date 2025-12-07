from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass
import os

from supabase import Client, create_client


@dataclass
class SupabaseConfig:
    url: str
    key: str
    nas_root: Path


@dataclass
class TaggedPage:
    id: str
    document_id: str
    page_number: int
    status: str
    page_image_rel_path: str
    page_image_abs_path: Path
    titleblock_x: float
    titleblock_y: float
    titleblock_width: float
    titleblock_height: float

def load_supabase_config_from_env() -> SupabaseConfig:
    """Load Supabase + NAS config from environment variables."""

    url = os.environ.get("SUPABASE_URL", "").strip()
    if not url:
        raise RuntimeError("SUPABASE_URL is not set in the environment.")

    # Try several common key names so we don't depend on a single one.
    key_candidates = [
        "SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_SERVICE_KEY",
        "SUPABASE_KEY",
        "SUPABASE_SECRET",
        "SUPABASE_ANON_KEY"
    ]

    key: Optional[str] = None
    for name in key_candidates:
        value = os.environ.get(name, "").strip()
        if value:
            print(f"[info] using {name} for Supabase auth")
            key = value
            break

    if not key:
        raise RuntimeError(
            "No Supabase key found. Tried: " + ", ".join(key_candidates)
        )

    nas_root_str = os.environ.get("NAS_ROOT", "/data/input").strip()
    nas_root = Path(nas_root_str)

    return SupabaseConfig(url=url, key=key, nas_root=nas_root)


def create_supabase_client(config: SupabaseConfig) -> Client:
    return create_client(config.url, config.key)


def fetch_next_tagged_page(
    client: Client,
    config: SupabaseConfig,
    status_value: str = "tagged",
) -> Optional[TaggedPage]:
    """
    Fetch the next row from document_pages where status == status_value.

    We assume:
    - document_pages.status is the status column.
    - document_pages.image_object_path is the relative PNG path under NAS_ROOT.
    - document_pages.titleblock_x/y/width/height are 0–1 normalised coords.
    """

    print(f"[info] querying document_pages where status = '{status_value}'")

    response = (
        client.table("document_pages")
        .select(
            "id,document_id,page_number,status,"
            "image_object_path,"
            "titleblock_x,titleblock_y,titleblock_width,titleblock_height"
        )
        .eq("status", status_value)
        .limit(1)
        .execute()
    )

    rows = getattr(response, "data", None) or []
    print(f"[info] document_pages rows returned: {len(rows)}")

    if not rows:
        return None

    row = rows[0]

    rel_path = row.get("image_object_path")
    if not rel_path:
        raise RuntimeError(
            f"Row {row.get('id')} has no image_object_path; cannot process."
        )

    abs_path = (config.nas_root / rel_path).resolve()

    return TaggedPage(
        id=str(row["id"]),
        document_id=str(row["document_id"]),
        page_number=int(row["page_number"]),
        status=row.get("status", ""),
        page_image_rel_path=rel_path,
        page_image_abs_path=abs_path,
        titleblock_x=float(row["titleblock_x"]),
        titleblock_y=float(row["titleblock_y"]),
        titleblock_width=float(row["titleblock_width"]),
        titleblock_height=float(row["titleblock_height"]),
    )