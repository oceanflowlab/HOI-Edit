"""Load cn/en playbook seeds, ACE role prompts, and Wan2.2 wrap strings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def normalize_lang(lang: str | None) -> str:
    v = (lang or os.getenv("ACE_LANG", "cn")).strip().lower()
    if v in ("en", "english"):
        return "en"
    return "cn"


@lru_cache(maxsize=4)
def load_playbook_seed(lang: str = "cn") -> Dict[str, Any]:
    path = DATA_DIR / f"playbook_seed_{normalize_lang(lang)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Playbook seed not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_ace_prompts(lang: str = "cn") -> Dict[str, str]:
    path = DATA_DIR / f"ace_prompts_{normalize_lang(lang)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"ACE prompts not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_wan22_wrap(lang: str = "cn") -> Dict[str, str]:
    path = DATA_DIR / f"wan22_wrap_{normalize_lang(lang)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Wan22 wrap not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_qa2_prompts(lang: str = "cn") -> Dict[str, str]:
    path = DATA_DIR / f"qa2_prompts_{normalize_lang(lang)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"QA2 prompts not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_qa2_prompt(role: str, lang: str = "cn", **fields: str) -> str:
    prompts = load_qa2_prompts(lang)
    if role not in prompts:
        raise KeyError(f"Unknown QA2 prompt role: {role}")
    text = prompts[role]
    if role == "call1" and "fs_rules" not in fields:
        fields = {**fields, "fs_rules": prompts.get("fs_rules", "")}
    for key, value in fields.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def qa2_quality_line(action_type: str, lang: str = "cn") -> str:
    prompts = load_qa2_prompts(lang)
    if action_type == "dynamic":
        return prompts.get("quality_dynamic", "")
    return prompts.get("quality_static", "")


def format_ace_prompt(role: str, lang: str = "cn", **fields: str) -> str:
    prompts = load_ace_prompts(lang)
    if role not in prompts:
        raise KeyError(f"Unknown ACE prompt role: {role}")
    text = prompts[role]
    for key, value in fields.items():
        text = text.replace(f"{{{key}}}", value)
    return text
