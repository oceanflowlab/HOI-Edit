"""
ace_v2f_qa2.py — 两次 VLM 选帧（无 Playbook 文件）

Call 1: 根据内嵌选帧规则生成 question + action_type
Call 2: 15 帧 + instruction + question + action_type 综合选帧

Prompt 文案见 data/qa2_prompts_cn.json / qa2_prompts_en.json（ACE_LANG 或 --ace-lang）。
"""
import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

from ace_i2v_locale import format_qa2_prompt, normalize_lang, qa2_quality_line

from google import genai
from google.genai import types
from google.genai.errors import APIError

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
ACE_LANG = normalize_lang(os.getenv("ACE_LANG"))

try:
    if not GEMINI_API_KEY:
        print("❗ 错误：GEMINI_API_KEY 环境变量未设置。")
        gemini_client = None
    else:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✅ Gemini 客户端已初始化 (模型: {MODEL_NAME}, ACE_LANG={ACE_LANG})")
except Exception as e:
    gemini_client = None
    print(f"❗ 无法初始化 Gemini API 客户端: {e}")


def _stems_for_video_match(stem: str) -> List[str]:
    out: List[str] = []
    for s in (stem, stem.replace(":", "_")):
        if s not in out:
            out.append(s)
    return out


def find_mp4_for_image(video_dir: Path, image_name: str) -> Optional[Path]:
    if not video_dir.is_dir():
        return None
    stem = Path(image_name).stem
    for st in _stems_for_video_match(stem):
        direct = video_dir / f"{st}.mp4"
        if direct.is_file():
            return direct
        top = sorted(video_dir.glob(f"{st}*.mp4"))
        if top:
            return top[0]
        sub = sorted(video_dir.rglob(f"{st}*.mp4"))
        if sub:
            return sub[0]
    return None


def save_frame_as_png(frame_bytes: bytes, output_path: Path):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        _, png_buffer = cv2.imencode(".png", img)
        with open(output_path, "wb") as f:
            f.write(png_buffer.tobytes())
    else:
        with open(output_path, "wb") as f:
            f.write(frame_bytes)


def build_media_part(frame_data: bytes) -> Optional[types.Part]:
    if not frame_data:
        return None
    return types.Part(inline_data=types.Blob(data=frame_data, mime_type="image/jpeg"))


def extract_video_frames(video_path: str, num_frames: int = 15) -> List[bytes]:
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []
    if total_frames < num_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames: List[bytes] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                frames.append(buffer.tobytes())
    cap.release()
    return frames


async def call_gemini_json(
    prompt_text: str,
    media_parts: Optional[List[types.Part]] = None,
    temperature: float = 0.2,
    schema_properties: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if gemini_client is None:
        return None
    parts = [types.Part(text=prompt_text)]
    if media_parts:
        parts.extend(media_parts)
    contents = [types.Content(role="user", parts=parts)]
    props = schema_properties or {"reasoning": {"type": "STRING"}}
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema={"type": "OBJECT", "properties": props},
    )
    try:
        def _generate():
            return gemini_client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config,
            )
        response = await asyncio.to_thread(_generate)
        text_content = (response.text or "").strip()
        if not text_content:
            return None
        return json.loads(text_content)
    except (APIError, json.JSONDecodeError, Exception) as e:
        print(f"   ❗ Gemini 调用失败: {e}")
        return None


async def generate_question_meta(instruction: str, lang: str = ACE_LANG) -> Optional[Dict[str, Any]]:
    """Call 1: 内嵌规则生成 question + action_type。"""
    prompt = format_qa2_prompt("call1", lang, instruction=instruction)
    return await call_gemini_json(
        prompt,
        media_parts=None,
        temperature=0.3,
        schema_properties={
            "action_type": {"type": "STRING", "enum": ["static", "dynamic"]},
            "question": {"type": "STRING"},
            "reasoning": {"type": "STRING"},
        },
    )


async def select_frame_qa2(
    instruction: str,
    question: str,
    action_type: str,
    frames_bytes_list: List[bytes],
    lang: str = ACE_LANG,
) -> Optional[Dict[str, Any]]:
    """Call 2: 综合选帧（Executor + Fallback 合并为一次）。"""
    if not frames_bytes_list:
        return None
    n = len(frames_bytes_list)
    quality = qa2_quality_line(action_type, lang)
    prompt = format_qa2_prompt(
        "call2",
        lang,
        n=str(n),
        instruction=instruction,
        question=question,
        action_type=action_type,
        quality=quality,
    )
    media_parts = [p for f in frames_bytes_list if (p := build_media_part(f))]
    if not media_parts:
        return None
    result = await call_gemini_json(
        prompt,
        media_parts=media_parts,
        temperature=0.1,
        schema_properties={
            "best_frame_index": {"type": "INTEGER"},
            "strict_pass": {"type": "BOOLEAN"},
            "selection_mode": {"type": "STRING"},
            "match_score": {"type": "NUMBER"},
            "reasoning": {"type": "STRING"},
        },
    )
    if not result or "best_frame_index" not in result:
        return None
    idx_1 = int(result["best_frame_index"])
    if idx_1 < 1 or idx_1 > n:
        print(f"   ⚠️ qa2 索引无效 ({idx_1})，使用第 1 帧")
        idx_1 = 1
        result["selection_mode"] = result.get("selection_mode", "invalid_index_fallback")
    result["best_frame_index_0_based"] = idx_1 - 1
    return result


async def select_frame_for_sample(
    instruction: str,
    frames: List[bytes],
    lang: str = ACE_LANG,
) -> Dict[str, Any]:
    """完整 qa2 流程，返回统一结果 dict。"""
    meta = await generate_question_meta(instruction, lang=lang)
    if not meta or "question" not in meta:
        return {"ok": False, "error": "call1_failed"}
    question = meta["question"]
    action_type = meta.get("action_type", "static")
    sel = await select_frame_qa2(instruction, question, action_type, frames, lang=lang)
    if not sel or "best_frame_index_0_based" not in sel:
        return {
            "ok": True,
            "index_0based": 0,
            "index_1based": 1,
            "route": "first_frame_call2_failed",
            "question": question,
            "action_type": action_type,
        }
    i = sel["best_frame_index_0_based"]
    return {
        "ok": True,
        "index_0based": i,
        "index_1based": i + 1,
        "route": sel.get("selection_mode", "qa2"),
        "strict_pass": sel.get("strict_pass"),
        "match_score": sel.get("match_score"),
        "reasoning": sel.get("reasoning"),
        "question": question,
        "action_type": action_type,
    }


def load_tasks_from_json(json_path: Path) -> List[Dict[str, str]]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        {"image_name": k, "instruction": m["instruction"]}
        for k, m in data.items()
        if m.get("instruction")
    ]


def shard_tasks(tasks: List[Dict[str, str]], shard_index: int, num_shards: int) -> List[Dict[str, str]]:
    if num_shards <= 1:
        return tasks
    return [t for i, t in enumerate(tasks) if i % num_shards == shard_index]


async def run_qa2_frame_selection(
    tasks: List[Dict[str, str]],
    source_video_dir: Path,
    fallback_video_dir: Path,
    output_frame_dir: Path,
    num_frames: int = 15,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    lang: str = ACE_LANG,
) -> None:
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    tasks = shard_tasks(tasks, shard_index, num_shards)
    desc = "QA2 frame select" if num_shards <= 1 else f"QA2 shard {shard_index+1}/{num_shards}"
    ok, skip, fail = 0, 0, 0
    for task in tqdm(tasks, desc=desc):
        image_name = task["image_name"]
        instruction = task["instruction"]
        out_path = output_frame_dir / Path(image_name).with_suffix(".png")
        if out_path.exists():
            skip += 1
            continue
        video_path = find_mp4_for_image(source_video_dir, image_name)
        if video_path is None:
            video_path = find_mp4_for_image(fallback_video_dir, image_name)
        if video_path is None:
            fail += 1
            continue
        frames = extract_video_frames(str(video_path), num_frames=num_frames)
        if not frames:
            fail += 1
            continue
        res = await select_frame_for_sample(instruction, frames, lang=lang)
        idx = res.get("index_0based", 0)
        save_frame_as_png(frames[idx], out_path)
        if res.get("ok") and res.get("route") != "first_frame_call2_failed":
            ok += 1
        else:
            fail += 1
    print(f"完成: ok={ok} skip={skip} fail={fail} -> {output_frame_dir}")


async def main():
    import argparse
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description="ace_v2f_qa2: 两次 VLM 选帧（无 Playbook 文件）")
    parser.add_argument("--json-path", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--fallback-video-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "v2f_qa2_frames"))
    parser.add_argument("--num-frames", type=int, default=15)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--ace-lang",
        choices=["cn", "en"],
        default=None,
        help="QA2 prompt 语言（默认 ACE_LANG 环境变量，否则 cn）",
    )
    args = parser.parse_args()

    global ACE_LANG
    ACE_LANG = normalize_lang(args.ace_lang or os.getenv("ACE_LANG"))
    print(f"✅ ACE_LANG={ACE_LANG}")

    if gemini_client is None:
        print("❗ GEMINI 未初始化")
        return
    json_path = Path(args.json_path or repo_root / "organized_hoi_dataset/filtered_dataset/collected_annotations_bboxes_v4_subset.json").expanduser().resolve()
    source = Path(args.video_dir or script_dir / "wan22_videos_official3_enhanced").expanduser().resolve()
    fallback = Path(args.fallback_video_dir or repo_root / "organized_hoi_dataset/filtered_dataset/videos").expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    tasks = load_tasks_from_json(json_path)
    if not tasks:
        return
    if args.num_shards > 1:
        print(f"分片 {args.shard_index + 1}/{args.num_shards}，本 worker {len(shard_tasks(tasks, args.shard_index, args.num_shards))} 条")
    await run_qa2_frame_selection(
        tasks, source, fallback, out, args.num_frames,
        shard_index=args.shard_index, num_shards=args.num_shards,
        lang=ACE_LANG,
    )


if __name__ == "__main__":
    asyncio.run(main())
