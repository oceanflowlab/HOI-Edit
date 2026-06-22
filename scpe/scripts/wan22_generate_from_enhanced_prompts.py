#!/usr/bin/env python3
"""
用 DashScope wan2.2-i2v-flash，根据原图 + enhanced_prompts JSON 生成视频。
不调用 Gemini、不更新 Playbook。

支持 enhanced_prompts 的 by_split 格式（_meta + L1L2 + L3）或扁平
{image_name: {instruction, enhanced_prompt, ...}} 格式。

环境变量：DASHSCOPE_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dashscope
import requests
from dashscope import VideoSynthesis
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCPE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

PROMPT_PREFIX = "The camera is fixed and the person is the only one in the frame. "
PROMPT_SUFFIX = " With no other person's additional movements."


def encode_file_base64(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image: {file_path}")
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def download_video(video_url: str, save_path: Path) -> bool:
    try:
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  ✗ download failed: {e}")
        return False


def build_final_prompt(enhanced_prompt: str) -> str:
    return (
        PROMPT_PREFIX
        + enhanced_prompt.strip()
        + PROMPT_SUFFIX
    ).strip()


def load_enhanced_tasks(
    json_path: Path,
    *,
    split: Optional[str] = None,
) -> List[Dict[str, str]]:
    """从 by_split 或扁平 JSON 加载任务。"""
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


def run_one_generation(
    *,
    image_path: Path,
    enhanced_prompt: str,
    output_dir: Path,
    model: str,
    resolution: str,
    duration: int,
    prompt_extend: bool,
    watermark: bool,
    negative_prompt: str,
) -> Tuple[bool, Optional[str]]:
    try:
        rsp = VideoSynthesis.async_call(
            model=model,
            prompt=build_final_prompt(enhanced_prompt),
            img_url=encode_file_base64(image_path),
            resolution=resolution,
            duration=duration,
            prompt_extend=prompt_extend,
            watermark=watermark,
            negative_prompt=negative_prompt,
        )
        if rsp.status_code != HTTPStatus.OK:
            return False, f"submit: {rsp.code} {rsp.message}"

        task_id = rsp.output.task_id
        rsp = VideoSynthesis.wait(rsp)
        if rsp.status_code != HTTPStatus.OK:
            return False, f"wait: {rsp.code} {rsp.message}"

        out_path = output_dir / f"{image_path.stem}_{task_id[:8]}.mp4"
        if not download_video(rsp.output.video_url, out_path):
            return False, "download failed"
        return True, str(out_path)
    except Exception as e:
        return False, str(e)


def run_batch(
    *,
    name: str,
    tasks: List[Dict[str, str]],
    image_dir: Path,
    output_dir: Path,
    skip_if_exists: bool,
    skip_first: int,
    limit: Optional[int],
    sleep_seconds: float,
    model: str,
    resolution: str,
    duration: int,
    prompt_extend: bool,
    watermark: bool,
    negative_prompt: str,
    manifest_path: Optional[Path],
) -> Dict[str, Any]:
    if skip_first:
        tasks = tasks[skip_first:]
    if limit is not None:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "name": name,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "total": len(tasks),
        "generated": 0,
        "skipped": 0,
        "failed": 0,
    }

    mf = open(manifest_path, "a", encoding="utf-8") if manifest_path else None
    print(f"\n=== {name}: {len(tasks)} tasks -> {output_dir} ===")

    try:
        for task in tqdm(tasks, desc=name):
            image_name = task["image_name"]
            image_path = image_dir / image_name

            if not image_path.exists():
                summary["failed"] += 1
                tqdm.write(f"  ✗ missing image: {image_name}")
                continue

            if skip_if_exists and has_existing_video(image_name, output_dir):
                summary["skipped"] += 1
                continue

            ok, msg = run_one_generation(
                image_path=image_path,
                enhanced_prompt=task["enhanced_prompt"],
                output_dir=output_dir,
                model=model,
                resolution=resolution,
                duration=duration,
                prompt_extend=prompt_extend,
                watermark=watermark,
                negative_prompt=negative_prompt,
            )
            if ok:
                summary["generated"] += 1
                if mf:
                    mf.write(
                        json.dumps(
                            {
                                "split": task.get("split"),
                                "image_name": image_name,
                                "instruction": task.get("instruction"),
                                "enhanced_prompt": task["enhanced_prompt"],
                                "video_path": msg,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    mf.flush()
            else:
                summary["failed"] += 1
                tqdm.write(f"  ✗ {image_name}: {msg}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if mf:
            mf.close()

    print(
        f"[{name}] generated={summary['generated']} "
        f"skipped={summary['skipped']} failed={summary['failed']}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wan2.2 I2V from enhanced_prompts JSON + source images."
    )
    p.add_argument(
        "--enhanced-prompts-json",
        type=Path,
        default=SCPE_ROOT / "data/enhanced_prompts_v7_official3.json",
        help="by_split 或扁平 enhanced_prompts JSON。",
    )
    p.add_argument(
        "--image-dir-l1l2",
        type=Path,
        default=DEFAULT_DATA_ROOT / "data_v7_L12",
    )
    p.add_argument(
        "--image-dir-l3",
        type=Path,
        default=DEFAULT_DATA_ROOT / "data_v7_L3",
    )
    p.add_argument(
        "--output-dir-l1l2",
        type=Path,
        default=SCPE_ROOT / "output/wan22_videos_official3_enhanced/L1L2",
    )
    p.add_argument(
        "--output-dir-l3",
        type=Path,
        default=SCPE_ROOT / "output/wan22_videos_official3_enhanced/L3",
    )
    p.add_argument(
        "--split",
        choices=["all", "L1L2", "L3"],
        default="all",
        help="只跑某一 split，或 all（L1L2+L3）。",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-first", type=int, default=0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--model", default="wan2.2-i2v-flash")
    p.add_argument("--resolution", default="720P")
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--disable-prompt-extend", action="store_true")
    p.add_argument("--enable-watermark", action="store_true")
    p.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
    )
    p.add_argument(
        "--api-base",
        default="https://dashscope.aliyuncs.com/api/v1",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=SCPE_ROOT / "output/wan22_videos_official3_enhanced/run_summary.json",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="jsonl 记录每条生成结果；默认写到各 output-dir/manifest.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("请设置 DASHSCOPE_API_KEY 或传入 --api-key")

    dashscope.base_http_api_url = args.api_base
    dashscope.api_key = args.api_key

    ep_path = args.enhanced_prompts_json.expanduser().resolve()
    all_tasks = load_enhanced_tasks(ep_path)
    skip_if_exists = not args.no_skip_existing
    prompt_extend = not args.disable_prompt_extend
    watermark = args.enable_watermark

    summaries: Dict[str, Any] = {"source_json": str(ep_path), "splits": {}}

    def tasks_for(split: str) -> List[Dict[str, str]]:
        return [t for t in all_tasks if t.get("split") == split]

    if args.split in ("all", "L1L2"):
        l1_tasks = tasks_for("L1L2")
        manifest = args.manifest or (args.output_dir_l1l2 / "manifest.jsonl")
        summaries["splits"]["L1L2"] = run_batch(
            name="L1L2",
            tasks=l1_tasks,
            image_dir=args.image_dir_l1l2.expanduser().resolve(),
            output_dir=args.output_dir_l1l2.expanduser().resolve(),
            skip_if_exists=skip_if_exists,
            skip_first=args.skip_first,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            model=args.model,
            resolution=args.resolution,
            duration=args.duration,
            prompt_extend=prompt_extend,
            watermark=watermark,
            negative_prompt=args.negative_prompt,
            manifest_path=manifest,
        )

    if args.split in ("all", "L3"):
        l3_tasks = tasks_for("L3")
        manifest = args.manifest or (args.output_dir_l3 / "manifest.jsonl")
        summaries["splits"]["L3"] = run_batch(
            name="L3",
            tasks=l3_tasks,
            image_dir=args.image_dir_l3.expanduser().resolve(),
            output_dir=args.output_dir_l3.expanduser().resolve(),
            skip_if_exists=skip_if_exists,
            skip_first=args.skip_first if args.split == "L3" else 0,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            model=args.model,
            resolution=args.resolution,
            duration=args.duration,
            prompt_extend=prompt_extend,
            watermark=watermark,
            negative_prompt=args.negative_prompt,
            manifest_path=manifest,
        )

    totals = {"generated": 0, "skipped": 0, "failed": 0}
    for s in summaries["splits"].values():
        for k in totals:
            totals[k] += s.get(k, 0)
    summaries["total"] = totals

    summary_path = args.summary_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== ALL DONE ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
