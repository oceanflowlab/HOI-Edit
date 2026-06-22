#!/usr/bin/env python3
"""Compute L1/L2/L3 scores from scoring_final JSONs + HOI + question_v6 QA.

Pools (5models table):
  L1/L2 averages -> L1L2 scoring_final only (n=499 header)
  L3 average     -> L3 scoring_final only (n=136 header)

Usage:
  python evaluation/compute_scoring_final_scores.py --model YOUR_NAME --decimals 4
  python evaluation/compute_scoring_final_scores.py --model name_a,name_b --decimals 4
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

POS_INITIAL = "L2-location_understanding-specific_pos_initial"

TAGS_L1L2 = [
    "L1-interaction-relation_change_only",
    "L1-interaction-relation_not_occur",
    "L1-interaction-relation_occur",
    "L2-location_understanding-specific_pos_end",
    "L2-location_understanding-specific_pos_initial",
]

TAGS_L3 = [
    "L3-non_rigid_change",
    "L3-reasoning_process",
]
def spec_for_model(model_name: str) -> dict:
    """Build lookup spec from the same name used in MODELS= / --model for QA+HOI."""
    return {
        "label": model_name,
        "model_dir": model_name,
        "qa_key": model_name,
        "legacy_hoi": [],
        "legacy_qa": [],
    }


def workspace_hoi_paths(ws: Path, model_dir: str) -> list[Path]:
    paths = []
    for split in ("L1L2", "L3"):
        p = (
            ws
            / "eval_runs"
            / f"{model_dir}_{split}_full"
            / f"{model_dir}_{split}"
            / f"results_{model_dir}_{split}_google_full.json"
        )
        if p.exists():
            paths.append(p)
    return paths


def workspace_qa_paths(ws: Path, model_dir: str) -> list[Path]:
    paths = []
    for split in ("L1L2", "L3"):
        p = ws / "eval_runs" / f"qa_results_v6_{split}_{model_dir}.json"
        if p.exists():
            paths.append(p)
    return paths


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], dict):
        return data["results"]
    return data if isinstance(data, dict) else {}


def merge_paths(paths: list[Path]) -> dict:
    merged = {}
    for path in paths:
        for key, row in load_json(path).items():
            if isinstance(row, dict):
                merged[key] = row
    return merged


def parse_tags(s) -> list[str]:
    return [t.strip() for t in re.split(r"[,，]", s or "") if t.strip()]


def is_l1(tag: str) -> bool:
    return tag.startswith("L1-")


def other_tags(tags_v6) -> list[str]:
    return [t for t in parse_tags(tags_v6) if not is_l1(t) and t != POS_INITIAL]


def normalize_questions(q) -> list[str]:
    if q is None:
        return []
    if isinstance(q, list):
        return [str(x).strip() for x in q if str(x).strip()]
    s = str(q).strip()
    return [s] if s else []


def yes_score(ans) -> float | None:
    if not ans:
        return None
    answer = (ans.get("answer") or "").strip().lower()
    conf = ans.get("confidence")
    if conf is None:
        return None
    conf = float(conf)
    if answer.startswith("yes"):
        return conf
    if answer.startswith("no"):
        return 0.0
    return None


def hoi_fail(row) -> bool:
    if not isinstance(row, dict):
        return True
    if row.get("has_error"):
        return True
    for key in (
        "vqa_error",
        "subject_similarity_error",
        "object_similarity_error",
        "processing_error",
    ):
        if row.get(key):
            return True
    return False


def metric(row, key: str) -> float | None:
    if hoi_fail(row):
        return None
    value = row.get(key)
    return None if value is None else float(value)


def tag_qa_score(key, entry, tag, qa, qa_key) -> float | None:
    others = other_tags(entry.get("tags_v6"))
    qv6 = normalize_questions(entry.get("question_v6"))
    if tag not in others:
        return None
    idx = others.index(tag)
    if idx >= len(qv6):
        return None
    question = qv6[idx]
    qa_row = qa.get(key)
    if not qa_row:
        return None
    qmap = {(q.get("question") or "").strip(): q for q in qa_row.get("questions") or []}
    q = qmap.get(question)
    if not q:
        return None
    return yes_score((q.get("answers") or {}).get(qa_key))


def sample_iqa(i, key, entry, tag, qa, qa_key):
    if i is None:
        return None, False
    if is_l1(tag) or tag == POS_INITIAL:
        return i, True
    qa_val = tag_qa_score(key, entry, tag, qa, qa_key)
    if qa_val is None:
        return None, False
    return min(i, qa_val), True


def round_mean(values: list[float], decimals: int) -> float | None:
    return round(mean(values), decimals) if values else None


def stats_for_tag(tag, keys, anno, hoi, qa, qa_key, decimals):
    keys = sorted(set(keys))
    is_vals, ss, os, iqas = [], [], [], []
    for key in keys:
        row = hoi.get(key)
        entry = anno[key]
        i = metric(row, "max_yes_confidence")
        s = metric(row, "subject_similarity")
        o = metric(row, "object_similarity")
        if i is not None:
            is_vals.append(i)
        if s is not None:
            ss.append(s)
        if o is not None:
            os.append(o)
        iqa, ok = sample_iqa(i, key, entry, tag, qa, qa_key)
        if ok and iqa is not None:
            iqas.append(iqa)
    return {
        "tag": tag,
        "n": len(keys),
        "n_hoi": len(is_vals),
        "I": round_mean(is_vals, decimals),
        "S": round_mean(ss, decimals),
        "O": round_mean(os, decimals),
        "IQA": round_mean(iqas, decimals),
        "IQA_n": len(iqas),
    }


def stats_for_avg(label, pred, anno, hoi, qa, qa_key, decimals):
    pairs = []
    for key, entry in anno.items():
        for tag in parse_tags(entry.get("tags_v6")):
            if pred(tag):
                pairs.append((key, tag))
    keys = sorted({k for k, _ in pairs})
    is_vals, ss, os, iqas = [], [], [], []
    seen_i, seen_s, seen_o = set(), set(), set()
    for key, tag in pairs:
        row = hoi.get(key)
        entry = anno[key]
        i = metric(row, "max_yes_confidence")
        s = metric(row, "subject_similarity")
        o = metric(row, "object_similarity")
        if i is not None and key not in seen_i:
            is_vals.append(i)
            seen_i.add(key)
        if s is not None and key not in seen_s:
            ss.append(s)
            seen_s.add(key)
        if o is not None and key not in seen_o:
            os.append(o)
            seen_o.add(key)
        iqa, ok = sample_iqa(i, key, entry, tag, qa, qa_key)
        if ok and iqa is not None:
            iqas.append(iqa)
    return {
        "tag": label,
        "n": len(keys),
        "pairs": len(pairs),
        "n_hoi": len(is_vals),
        "I": round_mean(is_vals, decimals),
        "S": round_mean(ss, decimals),
        "O": round_mean(os, decimals),
        "IQA": round_mean(iqas, decimals),
        "IQA_n": len(iqas),
    }


def compute_section(anno, per_tags, hoi, qa, qa_key, decimals, include_l3_avg=False):
    by_tag = defaultdict(list)
    for key, entry in anno.items():
        for tag in parse_tags(entry.get("tags_v6")):
            by_tag[tag].append(key)

    rows = []
    for tag in per_tags:
        if by_tag.get(tag):
            rows.append(stats_for_tag(tag, by_tag[tag], anno, hoi, qa, qa_key, decimals))

    rows.append(stats_for_avg("L1_average", lambda t: t.startswith("L1-"), anno, hoi, qa, qa_key, decimals))
    rows.append(stats_for_avg("L2_average", lambda t: t.startswith("L2-"), anno, hoi, qa, qa_key, decimals))
    if include_l3_avg:
        rows.append(
            stats_for_avg("L3_average", lambda t: t.startswith("L3-"), anno, hoi, qa, qa_key, decimals)
        )
    return rows


def format_row(row, label_prefix: str) -> dict:
    out = {"tag": row["tag"], "n": row["n"]}
    for m in ("I", "IQA", "S", "O"):
        v = row.get(m)
        if v is not None:
            out[f"{label_prefix}_{m}"] = v
    return out


def build_model_bundle(ws: Path, legacy_root: Path | None, spec: dict) -> tuple[dict, dict, str]:
    model_dir = spec["model_dir"]
    hoi_paths = workspace_hoi_paths(ws, model_dir)
    qa_paths = workspace_qa_paths(ws, model_dir)
    if legacy_root and legacy_root.exists():
        for rel in spec.get("legacy_hoi", []):
            p = legacy_root / rel
            if p.exists():
                hoi_paths.append(p)
        for rel in spec.get("legacy_qa", []):
            p = legacy_root / rel
            if p.exists():
                qa_paths.append(p)
    hoi = merge_paths(hoi_paths)
    qa = merge_paths(qa_paths)
    return hoi, qa, spec["qa_key"]


def main():
    parser = argparse.ArgumentParser(description="Score scoring_final pools (L1L2 n=499, L3 n=136)")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="cr_eval_workspace root (annotations)",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=None,
        help="Optional legacy repo root (e.g. parent gjy/) to merge historical HOI/QA",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Same name passed to MODELS= / --model when running QA and HOI",
    )
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON summary")
    args = parser.parse_args()

    ws = args.workspace.resolve()
    legacy = args.legacy_root.resolve() if args.legacy_root else None
    cr = Path(os.environ.get("HOI_EDIT_DATA_DIR", str(ws / "data"))).resolve()
    if not cr.exists():
        legacy_cr = ws / "data_v7/CR"
        cr = legacy_cr if legacy_cr.exists() else cr

    l1l2 = load_json(cr / "collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json")
    l3 = load_json(cr / "collected_annotations_bboxes_v7_L3_questions_scoring_final.json")
    if not l1l2 or not l3:
        raise SystemExit(f"Missing scoring_final JSON under {cr}")

    model_names = [m.strip() for m in args.model.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("--model must not be empty")

    summary = {
        "annotations": {
            "L1L2": str(cr / "collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json"),
            "L3": str(cr / "collected_annotations_bboxes_v7_L3_questions_scoring_final.json"),
        },
        "shared_root": str(legacy or ws),
        "workspace_eval_runs": str(ws / "eval_runs"),
        "decimals": args.decimals,
        "L1L2_final": {},
        "L3_final": {},
    }

    for model_name in model_names:
        spec = spec_for_model(model_name)
        hoi, qa, qa_key = build_model_bundle(ws, legacy, spec)
        prefix = spec["label"]

        l1l2_rows = compute_section(l1l2, TAGS_L1L2, hoi, qa, qa_key, args.decimals, include_l3_avg=False)
        l3_rows = compute_section(l3, TAGS_L3, hoi, qa, qa_key, args.decimals, include_l3_avg=True)

        summary["L1L2_final"][model_name] = [format_row(r, prefix) for r in l1l2_rows]
        summary["L3_final"][model_name] = [format_row(r, prefix) for r in l3_rows]

        print(f"\n=== {prefix} ===")
        print("[L1L2 pool]")
        for r in l1l2_rows:
            if r["tag"].endswith("average") or r["tag"] in TAGS_L1L2:
                print(f"  {r['tag']:45s} n={r['n']:3d}  I={r['I']}  IQA={r['IQA']}")
        print("[L3 pool]")
        for r in l3_rows:
            if r["tag"].endswith("average") or r["tag"] in TAGS_L3:
                print(f"  {r['tag']:45s} n={r['n']:3d}  I={r['I']}  IQA={r['IQA']}")

    out_path = args.out or (ws / "eval_runs" / f"scoring_final_scores_{args.decimals}dp.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
