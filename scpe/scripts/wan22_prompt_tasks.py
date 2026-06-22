"""Load enhanced_prompt tasks and build Wan I2V prompts (shared by cloud/local generators)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ace_i2v_locale import load_wan22_wrap, normalize_lang


def build_final_prompt(enhanced_prompt: str, lang: Optional[str] = None) -> str:
    wrap = load_wan22_wrap(normalize_lang(lang or os.getenv("ACE_LANG")))
    return (wrap["prefix"] + enhanced_prompt.strip() + wrap["suffix"]).strip()


def load_enhanced_tasks(
    json_path: Path,
    *,
    split: Optional[str] = None,
) -> List[Dict[str, str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tasks: List[Dict[str, str]] = []

    if "L1L2" in data or "L3" in data:
        splits = ["L1L2", "L3"] if split is None else [split]
        for sp in splits:
            bucket = data.get(sp)
            if not isinstance(bucket, dict):
                continue
            for image_name, meta in bucket.items():
                if not isinstance(meta, dict):
                    continue
                ep = (meta.get("enhanced_prompt") or "").strip()
                if ep:
                    tasks.append(
                        {
                            "split": sp,
                            "image_name": image_name,
                            "instruction": (meta.get("instruction") or "").strip(),
                            "enhanced_prompt": ep,
                        }
                    )
    else:
        for image_name, meta in data.items():
            if image_name.startswith("_") or not isinstance(meta, dict):
                continue
            ep = (meta.get("enhanced_prompt") or "").strip()
            if ep:
                tasks.append(
                    {
                        "split": split or "all",
                        "image_name": image_name,
                        "instruction": (meta.get("instruction") or "").strip(),
                        "enhanced_prompt": ep,
                    }
                )

    return tasks


def has_existing_video(image_name: str, output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    stem = Path(image_name).stem
    return any(output_dir.glob(f"{stem}_*.mp4")) or any(output_dir.glob(f"{stem}.mp4"))


def tasks_for_split(all_tasks: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    return [t for t in all_tasks if t.get("split") == split]
