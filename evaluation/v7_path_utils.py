"""V7 path helpers: JSON keys use split prefix (e.g. 3/foo.jpg), flat dirs do not."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional


def _dirs_to_search(primary_dir: str, extra_dirs: Optional[Iterable[str]] = None) -> List[str]:
    dirs: List[str] = []
    for d in [primary_dir, *(extra_dirs or [])]:
        d = (d or "").strip()
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def resolve_original_image_path(
    original_dir: str,
    json_key: str,
    extra_dirs: Optional[Iterable[str]] = None,
) -> str:
    """Resolve original image path across one or more flat original directories."""
    fallback = list(extra_dirs or [])
    env_extra = os.environ.get("EVAL_V7_ORIG_FALLBACK_DIRS", "")
    if env_extra:
        fallback.extend(p.strip() for p in env_extra.split(":") if p.strip())

    for base in _dirs_to_search(original_dir, fallback):
        direct = os.path.join(base, json_key)
        if os.path.exists(direct):
            return direct
        flat = os.path.join(base, os.path.basename(json_key))
        if os.path.exists(flat):
            return flat

    return os.path.join(original_dir, json_key)
