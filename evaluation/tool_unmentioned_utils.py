"""Helpers for L3 tool_unmentioned samples (optional tracking + QA yellow box)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union

TOOL_UNMENTIONED_TAG = "tool_unmentioned"


def resolve_hoi_tags(sample: dict) -> str:
    """HOI 路由用标签字符串，仅使用 tags_v6（辅以 tag 列表字段）。"""
    if not isinstance(sample, dict):
        return ""

    val = sample.get("tags_v6", "")
    if isinstance(val, list):
        parts = [str(t).strip() for t in val if str(t).strip()]
        if parts:
            return ", ".join(parts)
    elif isinstance(val, str) and val.strip():
        return val.strip()

    tag = sample.get("tag")
    if isinstance(tag, list):
        parts = [str(t).strip() for t in tag if str(t).strip()]
        if parts:
            return ", ".join(parts)
    elif isinstance(tag, str) and tag.strip():
        return tag.strip()

    return ""


def is_tool_unmentioned_sample(sample: dict) -> bool:
    """True if sample is tagged as L3-reasoning-tool_unmentioned."""
    if not isinstance(sample, dict):
        return False

    tags_v6 = resolve_hoi_tags(sample)
    if TOOL_UNMENTIONED_TAG in tags_v6:
        return True

    tag = sample.get("tag")
    if isinstance(tag, list) and any(TOOL_UNMENTIONED_TAG in str(t) for t in tag):
        return True
    if isinstance(tag, str) and TOOL_UNMENTIONED_TAG in tag:
        return True
    return False


def question_mentions_yellow_box(question: str, *, legacy_only: bool = False) -> bool:
    """Detect yellow-box references in VQA questions."""
    if not question:
        return False
    q = question.lower()
    if legacy_only:
        return "yellow box" in q
    return "yellow box" in q or "yellow bounding box" in q


def has_valid_tool_bboxes(tool_bboxes: Any) -> bool:
    if not tool_bboxes:
        return False
    if isinstance(tool_bboxes, list):
        return len(tool_bboxes) >= 4
    if isinstance(tool_bboxes, str):
        return bool(tool_bboxes.strip())
    return False


def load_tool_tracked_bbox(
    track_dir: Optional[str],
    image_name: str,
    frame_key: str = "frame_00001",
) -> Optional[List[int]]:
    """Load SAM2 tool bbox JSON from {track_dir}/tool_bboxes/{stem}.json."""
    if not track_dir:
        return None

    stem = Path(image_name).stem
    json_path = Path(track_dir) / "tool_bboxes" / f"{stem}.json"
    if not json_path.is_file():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracked = data.get("tracked_bboxes") or {}
        box = tracked.get(frame_key)
        if box and len(box) >= 4:
            return [int(round(float(x))) for x in box[:4]]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def resolve_tool_bbox_for_qa(
    tool_bboxes: Any,
    *,
    parse_tool_bboxes_fn,
    is_edited: bool,
    tool_track_dir: Optional[str],
    image_name: str,
    frame_key: str = "frame_00001",
) -> Optional[List[int]]:
    """Prefer tracked tool box on edited frames when available."""
    if is_edited and tool_track_dir:
        tracked = load_tool_tracked_bbox(tool_track_dir, image_name, frame_key=frame_key)
        if tracked:
            return tracked
    return parse_tool_bboxes_fn(tool_bboxes)
