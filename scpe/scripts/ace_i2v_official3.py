"""
ACE I2V（Google 官方版）：多模态推理走 Google GenAI SDK（gemini-2.5-pro）；视频生成仍为 DashScope wan2.2-i2v-flash。
需 export GEMINI_API_KEY；可选 GEMINI_MODEL、ACE_I2V_RESPONSE_SCHEMA=0 关闭结构化输出。

角色 prompt 文案：data/ace_prompts_cn.json / data/ace_prompts_en.json（ACE_LANG 或 --ace-lang）。
Playbook 种子：data/playbook_seed_cn.json / data/playbook_seed_en.json。
Wan 视频 wrap：data/wan22_wrap_cn.json / data/wan22_wrap_en.json。
"""
import os
import json
import mimetypes
import time
import re
import shutil
import base64
import requests
import asyncio
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from http import HTTPStatus
from tqdm import tqdm

from ace_i2v_locale import format_ace_prompt, load_playbook_seed, normalize_lang

from google import genai
from google.genai import types
from google.genai.errors import APIError

import dashscope
from dashscope import VideoSynthesis

# --- Google Gemini（官方 SDK）---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"✅ Gemini 官方客户端已初始化 (模型: {MODEL_NAME})")
    except Exception as e:
        print(f"❗ 无法初始化 Gemini 客户端: {e}")
else:
    print("❗ GEMINI_API_KEY 未设置：ACE 推理将无法调用（请 export GEMINI_API_KEY）")

# --- 配置 DashScope API (真实的视频生成) ---
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if DASHSCOPE_API_KEY:
    print("✅ DashScope：已使用环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = DASHSCOPE_API_KEY
    print("✅ DashScope 客户端已配置 (I2V 模型: wan2.2-i2v-flash)")
else:
    print("⚠️ DashScope：未设置 DASHSCOPE_API_KEY；DashScope I2V 生成不可用")

# --- 路径配置 (在 main 中设置) ---
PLAYBOOK_FILE: str = "hoi_i2v_playbook.json" # 将在 main 中被设置为绝对路径
ACE_LANG: str = normalize_lang(os.getenv("ACE_LANG"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_meta(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "mime": mimetypes.guess_type(str(p))[0],
    }


class SampleTrace:
    """每样本各阶段 Gemini 输出与状态，写入 JSON + JSONL。"""

    def __init__(self, trace_dir: Optional[Path], image_name: str, index: int, total: int):
        self.enabled = trace_dir is not None
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.image_name = image_name
        self.index = index
        self.total = total
        self.record: Dict[str, Any] = {
            "ts": _utc_now(),
            "index": index,
            "total": total,
            "image_name": image_name,
            "status": "pending",
            "inputs": {},
            "stages": {},
        }

    def set_inputs(self, **kwargs: Any) -> None:
        self.record["inputs"].update(kwargs)

    def log_stage(self, stage: str, output: Any, **extra: Any) -> None:
        self.record["stages"][stage] = {"ts": _utc_now(), "output": output, **extra}

    def finish(self, status: str) -> None:
        self.record["status"] = status

    def summary_line(self) -> str:
        st = self.record["stages"]
        a = st.get("analysis", {}).get("output") if isinstance(st.get("analysis"), dict) else None
        success = a.get("success") if isinstance(a, dict) else None
        critique = (a.get("critique") or "")[:60] if isinstance(a, dict) else ""
        merged = st.get("merge", {}).get("updated") if isinstance(st.get("merge"), dict) else None
        parts = [
            f"[{self.index}/{self.total}] {self.image_name}",
            f"status={self.record['status']}",
        ]
        if success is not None:
            parts.append(f"analysis.success={success}")
        if critique:
            parts.append(f"critique={critique!r}")
        if merged is not None:
            parts.append(f"playbook_merged={merged}")
        return "   | " + " ".join(parts)

    def save(self) -> None:
        if not self.enabled or self.trace_dir is None:
            return
        safe = re.sub(r"[^\w.\-]+", "_", Path(self.image_name).stem)[:80]
        out = self.trace_dir / f"{self.index:04d}_{safe}.json"
        out.write_text(json.dumps(self.record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (self.trace_dir / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.record, ensure_ascii=False) + "\n")

# --- 实用工具函数 ---

def guess_mime_type(file_path):
    """根据文件扩展名推测 MIME 类型。"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if file_path.endswith(('.mp4', '.mov', '.avi', '.webm')):
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "video/mp4"
    return mime_type or "image/jpeg"

def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"```(json)?\n", "", t, count=1)
        t = re.sub(r"\n```$", "", t, count=1)
    return t.strip()


def build_media_part(media_path: str) -> Optional[types.Part]:
    """将图像或视频文件转换为 Gemini API 所需的 Part 对象。"""
    if not os.path.exists(media_path):
        print(f"   ❗ 媒体文件不存在: {media_path}")
        return None
    try:
        with open(media_path, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"   ❗ 无法读取媒体文件 {media_path}: {e}")
        return None
    mime_type = guess_mime_type(media_path)
    return types.Part(inline_data=types.Blob(data=data, mime_type=mime_type))


def build_ace_response_schema(prompt_text: str) -> Dict[str, Any]:
    """与 ace_i2v.py call_gemini_json 的 response_schema 分支一致。"""
    response_schema: Dict[str, Any] = {
        "type": "OBJECT",
        "properties": {
            "reasoning": {"type": "STRING"},
        },
    }
    if "ace_generator_prompt" in prompt_text:
        response_schema["properties"]["enhanced_prompt"] = {"type": "STRING"}
    elif "ace_analysis_prompt" in prompt_text:
        response_schema["properties"]["success"] = {"type": "BOOLEAN"}
        response_schema["properties"]["critique"] = {"type": "STRING"}
    elif "ace_reflector_prompt" in prompt_text:
        response_schema["properties"]["root_cause"] = {"type": "STRING"}
        response_schema["properties"]["key_insight"] = {"type": "STRING"}
    elif "ace_curator_prompt" in prompt_text:
        response_schema["properties"]["operations"] = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING"},
                    "section": {"type": "STRING"},
                    "content": {"type": "STRING"},
                },
            },
        }
    return response_schema


def encode_file_base64(file_path: str) -> str:
    """(DashScope) 格式为 data:{MIME_type};base64,{base64_data}"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

def download_video(video_url: str, save_path: Path) -> bool:
    """(DashScope) 从URL下载视频到本地"""
    try:
        print(f'  ... 正在下载视频到: {save_path}')
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = save_path.stat().st_size / (1024 * 1024)  # MB
        print(f'  ✓ 视频下载完成，大小: {file_size:.2f} MB')
        return True
    except Exception as e:
        print(f'  ✗ 下载视频失败: {str(e)}')
        return False

async def call_gemini_json(
    prompt_text: str,
    media_paths: Optional[List[str]] = None,
    temperature: float = 0.2,
    response_mime_type: str = "application/json",
    max_retries: int = 3,
    timeout: float = 120.0,
    retry_backoff: float = 1.7,
) -> Optional[Dict[str, Any]]:
    """通过 Google GenAI 官方 SDK 调用 Gemini，返回 JSON dict。"""
    if gemini_client is None:
        print("   ❗ Gemini 客户端未初始化（请设置 GEMINI_API_KEY）")
        return None

    parts: List[types.Part] = [types.Part(text=prompt_text)]
    if media_paths:
        for path in media_paths:
            part = build_media_part(path)
            if part is None:
                return None
            parts.append(part)

    contents = [types.Content(role="user", parts=parts)]

    use_schema = os.getenv("ACE_I2V_RESPONSE_SCHEMA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
    }
    if use_schema:
        config_kwargs["response_schema"] = build_ace_response_schema(prompt_text)
    config = types.GenerateContentConfig(**config_kwargs)

    sleep_interval = 0.8
    text_out: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:

            def _generate():
                return gemini_client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=config,
                )

            response = await asyncio.wait_for(
                asyncio.to_thread(_generate),
                timeout=timeout,
            )
            text_out = (response.text or "").strip()
            if not text_out:
                raise ValueError("empty text")
            return json.loads(strip_code_fence(text_out))

        except asyncio.TimeoutError:
            if attempt < max_retries:
                await asyncio.sleep(sleep_interval)
                sleep_interval *= retry_backoff
                continue
            print(f"   ❗ Gemini 请求超时（已重试 {max_retries} 次，timeout={timeout}s）")
            return None
        except APIError as e:
            if attempt < max_retries:
                await asyncio.sleep(sleep_interval)
                sleep_interval *= retry_backoff
                continue
            print(f"   ❗ Gemini API 错误: {e}")
            return None
        except json.JSONDecodeError as e:
            snippet = (text_out[:200] if text_out else "").replace("\n", " ")
            if attempt < max_retries:
                await asyncio.sleep(sleep_interval)
                sleep_interval *= retry_backoff
                continue
            print(f"   ❗ 解析 Gemini JSON 失败: {e} | 片段: {snippet!r}")
            return None
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(sleep_interval)
                sleep_interval *= retry_backoff
                continue
            snippet = (text_out[:200] if text_out else "").replace("\n", " ")
            print(f"   ❗ Gemini 调用失败: {e} | 片段: {snippet!r}")
            return None

    return None

# --- Playbook 管理 (ACE 论文中的非 LLM 逻辑) ---

def load_playbook(filepath: str = PLAYBOOK_FILE) -> Dict[str, List[str]]:
    """加载 Playbook，如果不存在则创建默认结构。"""
    try:
        default_playbook = load_playbook_seed(ACE_LANG)
    except FileNotFoundError:
        default_playbook = {
            "strategies": [
                "[strategy-001]: 视频 prompt 必须详细描述动作的开始、过程和结束。"
            ],
            "templates": [
                "[template-001]: 一个{视角}镜头，{人物}的{身体部位}从{方向}靠近{物体}..."
            ],
            "pitfalls": [
                "[pitfall-001]: 陷阱：使用模糊动词（如 '互动'）。反思：必须具体化为 '推', '拉', '转'。"
            ],
        }
    if not os.path.exists(filepath):
        print(f"   📘 Playbook 文件未找到，在 '{filepath}' 创建新的。")
        save_playbook(filepath, default_playbook)
        return default_playbook
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"   ❗ Playbook 文件 '{filepath}' 损坏，将使用默认 Playbook。")
        return default_playbook
    except Exception as e:
        print(f"   ❗ 加载 Playbook 失败: {e}")
        return default_playbook

def save_playbook(filepath: str, playbook: Dict[str, List[str]]):
    """将 Playbook 保存到文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(playbook, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   ❗ 保存 Playbook 失败: {e}")


# --- Playbook 整理 ---

PITFALL_MOVE_RE = re.compile(
    r"为防止|以防止|防止|以避免|避免模型|"
    r"to prevent|in order to prevent|prevent the model|avoid the model",
    re.IGNORECASE,
)
STRATEGY_PREFIX_RE = re.compile(r"^\[strategy-\d+\]:\s*", re.IGNORECASE)
PITFALL_PREFIX_RE = re.compile(r"^\[pitfall-\d+\]:\s*", re.IGNORECASE)


def _strip_section_prefix(entry: str, kind: str) -> str:
    if kind == "strategy":
        return STRATEGY_PREFIX_RE.sub("", entry, count=1).strip()
    return PITFALL_PREFIX_RE.sub("", entry, count=1).strip()


def _ensure_pitfall_style(body: str) -> str:
    body = body.strip()
    if body.startswith("陷阱：") or body.startswith("陷阱:"):
        return body
    if body.lower().startswith("pitfall:"):
        return body
    if re.search(r"[\u4e00-\u9fff]", body):
        return f"陷阱：{body}"
    return f"Pitfall: {body}"


def _assign_section_prefixes(bodies: List[str], kind: str) -> List[str]:
    tag = "strategy" if kind == "strategy" else "pitfall"
    return [f"[{tag}-{i:03d}]: {body}" for i, body in enumerate(bodies, start=1)]


def normalize_playbook_content(playbook: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """防/避类迁 pitfalls + 补全编号前缀。"""
    raw_strategies = playbook.get("strategies", [])
    raw_pitfalls = playbook.get("pitfalls", [])

    stay_strategy_bodies: List[str] = []
    moved_to_pitfall_bodies: List[str] = []

    for entry in raw_strategies:
        body = _strip_section_prefix(str(entry), "strategy")
        if PITFALL_MOVE_RE.search(body):
            moved_to_pitfall_bodies.append(_ensure_pitfall_style(body))
        else:
            stay_strategy_bodies.append(body)

    pitfall_bodies: List[str] = []
    for entry in raw_pitfalls:
        pitfall_bodies.append(_ensure_pitfall_style(_strip_section_prefix(str(entry), "pitfall")))
    pitfall_bodies.extend(moved_to_pitfall_bodies)

    new_playbook: Dict[str, List[str]] = {
        "strategies": _assign_section_prefixes(stay_strategy_bodies, "strategy"),
        "templates": list(playbook.get("templates", [])),
        "pitfalls": _assign_section_prefixes(pitfall_bodies, "pitfall"),
    }

    stats = {
        "strategies_in": len(raw_strategies),
        "strategies_out": len(new_playbook["strategies"]),
        "moved_to_pitfalls": len(moved_to_pitfall_bodies),
        "pitfalls_in": len(raw_pitfalls),
        "pitfalls_out": len(new_playbook["pitfalls"]),
    }
    return new_playbook, stats


def default_normalized_playbook_path(filepath: str) -> Path:
    path = Path(filepath).expanduser().resolve()
    if path.stem.endswith("_normalized"):
        return path
    return path.with_name(f"{path.stem}_normalized.json")


def finalize_playbook(
    filepath: str,
    playbook: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Optional[Path]:
    """Playbook 学习结束后整理并写出 normalized 副本（保留原始 playbook 文件）。"""
    src_path = Path(filepath).expanduser().resolve()
    if playbook is None:
        if not src_path.is_file():
            print(f"   ❗ finalize_playbook: 文件不存在 {src_path}")
            return None
        playbook = json.loads(src_path.read_text(encoding="utf-8"))

    new_playbook, stats = normalize_playbook_content(playbook)
    if output_path:
        out_path = Path(output_path).expanduser().resolve()
    else:
        out_path = default_normalized_playbook_path(str(src_path))

    if out_path == src_path:
        print("   ❗ normalized 输出路径与原始 Playbook 相同，已跳过写入")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(new_playbook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n--- 📘 Playbook normalize 完成 ---")
    print(f"   原始:       {src_path}")
    print(f"   normalized: {out_path}")
    print(
        f"   strategies: {stats['strategies_in']} -> {stats['strategies_out']} "
        f"(迁出 {stats['moved_to_pitfalls']} 条到 pitfalls)"
    )
    print(f"   pitfalls:   {stats['pitfalls_in']} -> {stats['pitfalls_out']}")
    return out_path


def merge_playbook(playbook: Dict[str, List[str]], delta_update: Dict[str, Any]) -> bool:
    """(角色 4: Merger) ACE 增量更新，防止“上下文崩塌”。"""
    updated = False
    if "operations" not in delta_update:
        return False
        
    for op in delta_update.get("operations", []):
        if op.get("type") == "ADD":
            section = op.get("section")
            content = op.get("content")
            
            if section in playbook and content:
                if content not in playbook[section]:
                    playbook[section].append(content)
                    print(f"   📘 Playbook 已更新 (板块: {section}): {content[:80]}...")
                    updated = True
                else:
                    print(f"   📘 (跳过冗余更新: {content[:50]}...)")
            elif section not in playbook:
                print(f"   ❗ (Curator 尝试添加到不存在的板块: {section})")

    if updated and PLAYBOOK_FILE:
        save_playbook(PLAYBOOK_FILE, playbook)
        n = sum(len(v) for v in playbook.values() if isinstance(v, list))
        print(f"   💾 已落盘 ({n} 条) -> {PLAYBOOK_FILE}", flush=True)

    return updated

# --- ACE 角色实现 ---

async def ace_generator(
    image_path: str, 
    instruction: str, 
    playbook: Dict[str, List[str]]
) -> Optional[Dict[str, Any]]:
    """(角色 1: Generator) 读取 Playbook 和用户输入，生成增强的 prompt。"""
    # print("--- 角色 1: Generator 启动 ---") # 减少日志噪音
    playbook_text = json.dumps(playbook, indent=2, ensure_ascii=False)
    
    prompt = format_ace_prompt(
        "generator",
        ACE_LANG,
        instruction=instruction,
        playbook_text=playbook_text,
    )
    
    return await call_gemini_json(
        prompt, 
        media_paths=[image_path], 
        temperature=0.5,
        timeout=180.0  # 处理图像需要更长时间，增加到3分钟
    )

async def ace_reflector(
    analysis_report: Dict[str, Any], 
    failed_prompt: str
) -> Optional[Dict[str, Any]]:
    """(角色 2b: Reflector) 将“特定的”失败报告“抽象”为“通用的”洞察。"""
    # print("--- 角色 2b: Reflector 启动 ---") # 减少日志噪音
    report_text = json.dumps(analysis_report, indent=2, ensure_ascii=False)
    
    prompt = format_ace_prompt(
        "reflector",
        ACE_LANG,
        failed_prompt=failed_prompt,
        report_text=report_text,
    )
    
    return await call_gemini_json(prompt, temperature=0.3)

async def ace_curator(
    insights: Dict[str, Any], 
    playbook: Dict[str, List[str]]
) -> Optional[Dict[str, Any]]:
    """(角色 3: Curator) 将“通用的”洞察转化为“增量的” Playbook 更新。"""
    # print("--- 角色 3: Curator 启动 ---") # 减少日志噪音
    insights_text = json.dumps(insights, indent=2, ensure_ascii=False)
    playbook_text = json.dumps(playbook, indent=2, ensure_ascii=False)

    prompt = format_ace_prompt(
        "curator",
        ACE_LANG,
        insights_text=insights_text,
        playbook_text=playbook_text,
    )
    
    return await call_gemini_json(prompt, temperature=0.1)

# --- 真实的 I2V 视频生成 (替换模拟器) ---

def run_video_generation(image_path: str, enhanced_prompt: str, output_dir: Path) -> Optional[str]:
    """
    (真实) 您的 I2V 模型。
    【核心更新】使用 DashScope API 真实生成视频。
    """
    print("   ...(真实) 视频生成 (DashScope) 启动...")
    try:
        # 1. Base64 编码图像 (DashScope 需要)
        img_url_base64 = encode_file_base64(image_path)
        
        from ace_i2v_locale import load_wan22_wrap

        wrap = load_wan22_wrap(ACE_LANG)
        prompt = (
            wrap["prefix"]
            + enhanced_prompt.strip()
            + wrap["suffix"]
        )
        print(f'   ... 提交任务 - 图像: {Path(image_path).name}')
        print(f'   ... 最终 Prompt: {prompt[:70]}...')

        # 3. 异步调用 API
        rsp = VideoSynthesis.async_call(
            model='wan2.2-i2v-flash',
            prompt=prompt,
            img_url=img_url_base64,
            resolution="720P",
            duration=5,
            prompt_extend=True,
            watermark=False,
            negative_prompt="",
        )
        
        if rsp.status_code != HTTPStatus.OK:
            print(f"   ✗ 提交任务失败: {rsp.code}, {rsp.message}")
            return None

        task_id = rsp.output.task_id
        print(f'   ... 任务已提交，task_id: {task_id}，等待完成...')
        
        # 4. 等待任务完成 (这是阻塞操作)
        rsp = VideoSynthesis.wait(rsp)
        
        if rsp.status_code == HTTPStatus.OK:
            video_url = rsp.output.video_url
            print(f"   ✓ 任务成功！video_url: {video_url}")
            
            # 5. 下载视频
            videos_dir = output_dir # <-- 【重要】使用传入的 output_dir
            image_name = Path(image_path).stem
            # 使用 task_id 确保文件名唯一，防止重复运行
            video_filename = f"{image_name}_{task_id[:8]}.mp4"
            video_local_path = videos_dir / video_filename
            
            if download_video(video_url, video_local_path):
                return str(video_local_path)
            else:
                print("   ✗ 视频下载失败。")
                return None
        else:
            print(f'   ✗ 等待任务完成失败: {rsp.code}, {rsp.message}')
            return None
            
    except Exception as e:
        print(f"   ✗ 视频生成过程中出错: {str(e)}")
        return None

async def run_automated_analysis(
    image_path: str,
    instruction: str, 
    enhanced_prompt: str, 
    video_path: str
) -> Optional[Dict[str, Any]]:
    """
    (真实角色 2a) 您的 LLM 分析模型 (Google Gemini 官方 API)。
    """
    # print("--- 角色 2a: 自动分析模型 (Gemini) 启动 ---") # 减少日志噪音
    
    prompt = format_ace_prompt(
        "analyzer",
        ACE_LANG,
        instruction=instruction,
        enhanced_prompt=enhanced_prompt,
    )
    
    return await call_gemini_json(
        prompt, 
        media_paths=[image_path, video_path], # 发送原始图像和"生成的视频"
        temperature=0.1, # 分析任务需要高确定性
        timeout=180.0  # 处理图像+视频需要更长时间，增加到3分钟
    )

# --- 【新】数据加载 ---

def load_tasks_from_json(json_path: Path) -> List[Dict[str, str]]:
    """从您提供的 JSON 文件中加载任务列表。"""
    if not json_path.exists():
        print(f"❗ 错误：JSON 任务文件未找到: {json_path}")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tasks = []
        for image_name, meta in data.items():
            instruction = meta.get('instruction')
            if instruction: # 只处理有指令的条目
                tasks.append({
                    'image_name': image_name,
                    'instruction': instruction
                })
        
        print(f"✅ 从 {json_path.name} 加载了 {len(tasks)} 个任务。")
        return tasks
    except Exception as e:
        print(f"❗ 加载 JSON 任务失败: {e}")
        return []

def resolve_video_path(image_name: str, video_dir: Path) -> Optional[Path]:
    """按图像 stem 解析视频：优先 {stem}.mp4，否则匹配 {stem}*.mp4（含 data_v7 hash 后缀）。"""
    if not video_dir.exists():
        return None
    stem = Path(image_name).stem
    exact = video_dir / f"{stem}.mp4"
    if exact.exists():
        return exact
    matches = sorted(video_dir.glob(f"{stem}*.mp4"))
    return matches[0] if matches else None


def has_existing_video(image_name: str, video_dir: Optional[Path]) -> bool:
    """检查指定目录中是否已经为该图像生成过视频。"""
    if not video_dir:
        return False
    return resolve_video_path(image_name, video_dir) is not None

async def generate_videos_from_playbook(
    tasks: List[Dict[str, str]],
    playbook: Dict,
    image_dir: Path,
    output_video_dir: Path,
    *,
    exclude_video_dir: Optional[Path] = None,
    skip_if_exists: bool = True,
) -> Dict[str, int]:
    """使用已有的 Playbook 直接生成视频，不对 Playbook 进行更新。"""
    print(f"\n--- 🚀 使用现有 Playbook 直接生成视频 (共 {len(tasks)} 个任务) 🚀 ---")
    print(f"   (输出目录: {output_video_dir})")
    if exclude_video_dir:
        print(f"   (排除已有视频目录: {exclude_video_dir})")

    output_video_dir.mkdir(parents=True, exist_ok=True)

    summary = {"generated": 0, "skipped": 0, "failed": 0}

    for task in tqdm(tasks, desc="Direct Generation"):
        image_name = task["image_name"]
        instruction = task["instruction"]
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            summary["failed"] += 1
            continue

        if exclude_video_dir and has_existing_video(image_name, exclude_video_dir):
            summary["skipped"] += 1
            continue

        if skip_if_exists and has_existing_video(image_name, output_video_dir):
            summary["skipped"] += 1
            continue

        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            summary["failed"] += 1
            continue

        enhanced_prompt = generator_output["enhanced_prompt"]
        video_path = run_video_generation(str(image_path), enhanced_prompt, output_video_dir)
        if not video_path:
            print(f"   (视频生成失败: {image_name})")
            summary["failed"] += 1
            continue

        summary["generated"] += 1

    print(f"--- ✅ 直接生成完成 ---")
    print(f"   已生成: {summary['generated']}")
    print(f"   已跳过: {summary['skipped']}")
    print(f"   失败: {summary['failed']}")
    return summary


async def run_enhance_prompts_only(
    tasks: List[Dict[str, str]],
    playbook: Dict,
    image_dir: Path,
    output_json: Path,
    *,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """使用 Playbook + Generator 为每条样本生成 enhanced_prompt（不生成视频）。"""
    print(f"\n--- 🚀 第二轮：优化 Prompt (共 {len(tasks)} 个样本) 🚀 ---")
    print(f"   输出: {output_json}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    if output_json.exists() and skip_existing:
        try:
            results = json.loads(output_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = {}

    summary = {"generated": 0, "skipped": 0, "failed": 0}

    for task in tqdm(tasks, desc="Enhance Prompts"):
        image_name = task["image_name"]
        instruction = task["instruction"]
        image_path = image_dir / image_name

        if skip_existing and image_name in results and results[image_name].get("enhanced_prompt"):
            summary["skipped"] += 1
            continue

        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            summary["failed"] += 1
            continue

        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            summary["failed"] += 1
            continue

        results[image_name] = {
            "instruction": instruction,
            "enhanced_prompt": generator_output["enhanced_prompt"],
            "reasoning": generator_output.get("reasoning"),
        }
        summary["generated"] += 1

        if summary["generated"] % 10 == 0:
            output_json.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"--- ✅ Prompt 优化完成 ---")
    print(f"   新生成: {summary['generated']}")
    print(f"   已跳过: {summary['skipped']}")
    print(f"   失败: {summary['failed']}")
    print(f"   合计条目: {len(results)}")
    return summary


# --- 【新】多轮 (Multi-Epoch) 编排 ---

async def run_epoch_0_warmup(
    tasks: List[Dict[str, str]],
    playbook: Dict,
    image_dir: Path,
    original_video_dir: Path,
    *,
    save_every: int = 10,
    trace_dir: Optional[Path] = None,
) -> Dict:
    """第 0 轮：从您已有的视频中学习，构建 Playbook V0。"""
    print(f"\n--- 🚀 开始 第 0 轮 (预热, {len(tasks)} 个样本) 🚀 ---")
    print(f"   (从已有视频中学习: {original_video_dir})")
    print(f"   💾 Playbook 每次新增条目后立即写入 -> {PLAYBOOK_FILE}")
    if trace_dir:
        print(f"   📝 每样本 trace: {trace_dir}/ (单条 JSON + trace.jsonl)")
    if save_every > 0:
        print(f"   💾 另每 {save_every} 条做一次全量 checkpoint")
    
    total = len(tasks)
    for i, task in enumerate(tqdm(tasks, desc="Epoch 0 (Warmup)"), start=1):
        image_name = task['image_name']
        instruction = task['instruction']
        trace = SampleTrace(trace_dir, image_name, i, total)
        trace.record["instruction"] = instruction
        
        image_path = image_dir / image_name
        video_path = resolve_video_path(image_name, original_video_dir)

        if not image_path.exists():
            trace.finish("skipped_no_image")
            tqdm.write(trace.summary_line())
            trace.save()
            continue
        if not video_path:
            trace.finish("skipped_no_video")
            tqdm.write(trace.summary_line())
            trace.save()
            continue

        trace.set_inputs(
            image=_media_meta(str(image_path)),
            video=_media_meta(str(video_path)),
            gemini_media=["image", "video"],
            instruction=instruction,
            enhanced_prompt="N/A (Original Video)",
        )

        # 1. (角色 2a) 分析现有视频
        analysis_report = await run_automated_analysis(
            str(image_path), 
            instruction, 
            "N/A (Original Video)", # 原始视频没有增强 prompt
            str(video_path),
        )
        trace.log_stage(
            "analysis",
            analysis_report,
            gemini_called=analysis_report is not None,
        )

        if not analysis_report:
            trace.finish("failed_analysis_api")
            tqdm.write(trace.summary_line())
            trace.save()
            continue
        
        # 2. 如果失败，则学习
        if not analysis_report.get("success", False):
            insights = await ace_reflector(analysis_report, "N/A (Original Video)")
            trace.log_stage("reflector", insights, gemini_called=insights is not None)
            if not insights or "key_insight" not in insights:
                trace.finish("failed_reflector")
                tqdm.write(trace.summary_line())
                trace.save()
                continue
            
            delta_update = await ace_curator(insights, playbook)
            trace.log_stage("curator", delta_update, gemini_called=delta_update is not None)
            if not delta_update or "operations" not in delta_update:
                trace.finish("failed_curator")
                tqdm.write(trace.summary_line())
                trace.save()
                continue

            merged = merge_playbook(playbook, delta_update)
            trace.log_stage("merge", {"updated": merged, "operations": delta_update.get("operations")})
            trace.finish("ok_learned" if merged else "ok_fail_no_new_rule")
        else:
            trace.finish("ok_pass")

        tqdm.write(trace.summary_line())
        trace.save()

        if save_every > 0 and i % save_every == 0:
            save_playbook(PLAYBOOK_FILE, playbook)
            n = sum(len(v) for v in playbook.values() if isinstance(v, list))
            print(f"   💾 checkpoint {i}/{len(tasks)} | Playbook 共 {n} 条 -> {PLAYBOOK_FILE}", flush=True)
    
    print(f"--- ✅ 第 0 轮 (预热) 完成 ---")
    save_playbook(PLAYBOOK_FILE, playbook)
    print(f"   ✅ Playbook V0 已保存到 {PLAYBOOK_FILE}")
    return playbook

async def run_epoch_1_learn(tasks: List[Dict[str, str]], playbook: Dict, image_dir: Path, output_video_dir: Path) -> Dict:
    """第 1 轮：使用 V0 Playbook 生成 V1 视频，并学习 V1 的失败，构建 Playbook V1。"""
    print(f"\n--- 🚀 开始 第 1 轮 (学习, {len(tasks)} 个样本) 🚀 ---")
    print(f"   (新视频将保存到: {output_video_dir})")
    output_video_dir.mkdir(parents=True, exist_ok=True)
    
    for task in tqdm(tasks, desc="Epoch 1 (Learn)"):
        image_name = task['image_name']
        instruction = task['instruction']
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            continue

        # 1. (角色 1) 生成增强 Prompt
        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            continue
        enhanced_prompt = generator_output["enhanced_prompt"]

        # 2. (真实) 生成 V1 视频
        video_path_v1 = run_video_generation(str(image_path), enhanced_prompt, output_video_dir)
        if not video_path_v1:
            print(f"   (视频生成失败: {image_name})")
            continue

        # 3. (角色 2a) 分析 V1 视频
        analysis_report = await run_automated_analysis(
            str(image_path), 
            instruction, 
            enhanced_prompt, 
            video_path_v1
        )
        if not analysis_report:
            print(f"   (分析失败: {image_name})")
            continue

        # 4. 如果 V1 失败，则学习
        if not analysis_report.get("success", False):
            # print(f"   ⚠️ 发现失败案例 (V1): {image_name}")
            insights = await ace_reflector(analysis_report, enhanced_prompt)
            if not insights or "key_insight" not in insights:
                continue
            
            delta_update = await ace_curator(insights, playbook)
            if not delta_update or "operations" not in delta_update:
                continue
            
            # 实时更新 Playbook (在内存中)
            merge_playbook(playbook, delta_update)

    print(f"--- ✅ 第 1 轮 (学习) 完成 ---")
    save_playbook(PLAYBOOK_FILE, playbook)
    print(f"   ✅ Playbook V1 已保存到 {PLAYBOOK_FILE}")
    return playbook

async def run_epoch_2_final(tasks: List[Dict[str, str]], playbook: Dict, image_dir: Path, output_video_dir: Path):
    """第 2 轮：使用 V1 Playbook 生成最终的 V2 视频。"""
    print(f"\n--- 🚀 开始 第 2 轮 (最终生成, {len(tasks)} 个样本) 🚀 ---")
    print(f"   (最终视频将保存到: {output_video_dir})")
    output_video_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for task in tqdm(tasks, desc="Epoch 2 (Final)"):
        image_name = task['image_name']
        instruction = task['instruction']
        image_path = image_dir / image_name
        
        if not image_path.exists():
            print(f"   (跳过 {image_name}: 原始图像未找到)")
            fail_count += 1
            continue
            
        # 1. (角色 1) 生成最终 Prompt
        generator_output = await ace_generator(str(image_path), instruction, playbook)
        if not generator_output or "enhanced_prompt" not in generator_output:
            print(f"   (Generator 失败: {image_name})")
            fail_count += 1
            continue
        enhanced_prompt = generator_output["enhanced_prompt"]

        # 2. (真实) 生成 V2 视频
        video_path_v2 = run_video_generation(str(image_path), enhanced_prompt, output_video_dir)
        if not video_path_v2:
            print(f"   (视频生成失败: {image_name})")
            fail_count += 1
            continue
        
        success_count += 1
        # print(f"   ✓ 最终视频已生成: {video_path_v2}") # 减少日志噪音

    print(f"--- ✅ 第 2 轮 (最终生成) 完成 ---")
    print(f"   成功: {success_count} / {len(tasks)}")
    print(f"   失败: {fail_count} / {len(tasks)}")

# --- 【新】主函数 (替换旧的 __main__) ---

async def main():
    """主编排函数"""
    parser = argparse.ArgumentParser(description="ACE I2V 多轮生成脚本（Google Gemini 官方 API + DashScope I2V）")
    default_base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--dataset",
        choices=["data_v7_12", "data_v7_3"],
        default=None,
        help="使用 data 预设：data_v7_12（子集 1+2）或 data_v7_3（子集 3），见 paths_data_v7.py。",
    )
    parser.add_argument(
        "--mode",
        choices=["full_pipeline", "epoch0_only", "enhance_prompts_only", "final_only"],
        default="full_pipeline",
        help="full_pipeline: 完整流程；epoch0_only: 从本地视频学 Playbook；enhance_prompts_only: 用 Playbook 生成 enhanced_prompt；final_only: 生成视频。",
    )
    parser.add_argument(
        "--json-path",
        default=str((default_base_dir / "filtered_dataset/collected_annotations_bboxes_v4_subset.json").resolve()),
        help="任务列表 JSON 文件路径。",
    )
    parser.add_argument(
        "--image-dir",
        default=str((default_base_dir / "filtered_dataset/images").resolve()),
        help="源图像目录。",
    )
    parser.add_argument(
        "--original-video-dir",
        default=str((default_base_dir / "filtered_dataset/videos").resolve()),
        help="已有视频目录（Epoch0 使用）。",
    )
    parser.add_argument(
        "--epoch1-video-dir",
        default=str((default_base_dir / "ace_generated_output/epoch_1_videos").resolve()),
        help="Epoch 1 输出目录。",
    )
    parser.add_argument(
        "--epoch2-video-dir",
        default=str((default_base_dir / "ace_generated_output/epoch_2_videos").resolve()),
        help="Epoch 2 输出目录。",
    )
    parser.add_argument(
        "--direct-output-dir",
        default=str((default_base_dir / "ace_generated_output/epoch_3_videos").resolve()),
        help="使用现有 Playbook 直接生成视频的输出目录。",
    )
    parser.add_argument(
        "--exclude-dir",
        default=None,
        help="如果指定，则跳过在该目录下已经生成过的视频任务。",
    )
    parser.add_argument(
        "--playbook-file",
        default=str((default_base_dir / "ace_generated_output/hoi_i2v_playbook.json").resolve()),
        help="Playbook 文件路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个任务，用于调试或分批执行。",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="跳过 JSON 中前 N 条（中断后续跑时用，例如已跑到 27/500 则传 27）。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Epoch0 每处理 N 条额外全量保存（0=关闭；新增条目已在 merge 时即时落盘）。",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="默认跳过输出目录中已存在的视频，启用该选项可强制重新生成。",
    )
    parser.add_argument(
        "--output-prompts-json",
        default=None,
        help="enhance_prompts_only 模式：写出 {image_name: {instruction, enhanced_prompt, ...}} 的 JSON。",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Epoch0：每样本保存 analysis/reflector/curator 等 JSON 到该目录。",
    )
    parser.add_argument(
        "--no-normalize-playbook",
        action="store_true",
        help="Playbook 学习结束后跳过 normalize（默认会写出 *_normalized.json）。",
    )
    parser.add_argument(
        "--normalized-playbook-file",
        default=None,
        help="normalize 输出路径；默认在原始 Playbook 同目录生成 <stem>_normalized.json。",
    )
    parser.add_argument(
        "--ace-lang",
        choices=["cn", "en"],
        default=None,
        help="Playbook 种子与 ACE 角色 prompt 语言（默认环境变量 ACE_LANG，否则 cn）。",
    )

    args = parser.parse_args()

    global ACE_LANG
    ACE_LANG = normalize_lang(args.ace_lang or os.getenv("ACE_LANG"))
    print(f"✅ ACE_LANG={ACE_LANG}")

    # 1. 检查 Gemini API Key
    if not GEMINI_API_KEY or gemini_client is None:
        print("❗ GEMINI_API_KEY 未设置或客户端初始化失败，退出。")
        return

    # 2. 解析路径（--dataset 使用 data/data_v7_L12 或 data/data_v7_L3 预设）
    if args.dataset:
        from paths_data_v7 import ensure_output_dirs, get_data_v7_paths

        bundle = get_data_v7_paths(args.dataset)
        ensure_output_dirs(bundle)
        print(f"✅ 使用预设 dataset={bundle.name}（子目录: {', '.join(bundle.subsets)}）")
        print(f"   任务 JSON: {bundle.json_path}")
        print(f"   图像根目录: {bundle.image_dir}")
        json_path = bundle.json_path
        image_dir = bundle.image_dir
        original_video_dir = bundle.original_video_dir
        epoch1_video_dir = bundle.epoch1_video_dir
        epoch2_video_dir = bundle.epoch2_video_dir
        direct_output_dir = bundle.direct_output_dir
        playbook_file = bundle.playbook_file
    else:
        json_path = Path(args.json_path).expanduser().resolve()
        image_dir = Path(args.image_dir).expanduser().resolve()
        original_video_dir = Path(args.original_video_dir).expanduser().resolve()
        epoch1_video_dir = Path(args.epoch1_video_dir).expanduser().resolve()
        epoch2_video_dir = Path(args.epoch2_video_dir).expanduser().resolve()
        direct_output_dir = Path(args.direct_output_dir).expanduser().resolve()
        playbook_file = Path(args.playbook_file).expanduser().resolve()

    exclude_dir = Path(args.exclude_dir).expanduser().resolve() if args.exclude_dir else None

    # Playbook 路径必须为全局变量，以兼容现有函数
    global PLAYBOOK_FILE
    PLAYBOOK_FILE = str(playbook_file)

    Path(PLAYBOOK_FILE).parent.mkdir(parents=True, exist_ok=True)

    # 3. 加载任务
    tasks = load_tasks_from_json(json_path)
    if args.skip_first:
        tasks = tasks[args.skip_first :]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("❗ 没有加载到任何任务，退出。")
        return

    # 4. 加载 Playbook
    playbook_v0 = load_playbook(PLAYBOOK_FILE)

    if args.mode == "final_only":
        await generate_videos_from_playbook(
            tasks,
            playbook_v0,
            image_dir,
            direct_output_dir,
            exclude_video_dir=exclude_dir,
            skip_if_exists=not args.no_skip_existing,
        )
        return

    trace_dir: Optional[Path] = None
    if args.trace_dir:
        trace_dir = Path(args.trace_dir).expanduser().resolve()
    elif os.getenv("ACE_TRACE_DIR", "").strip():
        trace_dir = Path(os.environ["ACE_TRACE_DIR"]).expanduser().resolve()

    if args.mode == "epoch0_only":
        await run_epoch_0_warmup(
            tasks,
            playbook_v0,
            image_dir,
            original_video_dir,
            save_every=args.save_every,
            trace_dir=trace_dir,
        )
        print("\n--- 🚀 Epoch 0 Playbook 学习完成 🚀 ---")
        if not args.no_normalize_playbook:
            finalize_playbook(PLAYBOOK_FILE, output_path=args.normalized_playbook_file)
        return

    if args.mode == "enhance_prompts_only":
        if not args.output_prompts_json:
            print("❗ enhance_prompts_only 需要 --output-prompts-json")
            return
        out_path = Path(args.output_prompts_json).expanduser().resolve()
        await run_enhance_prompts_only(
            tasks,
            playbook_v0,
            image_dir,
            out_path,
            skip_existing=not args.no_skip_existing,
        )
        print("\n--- 🚀 Prompt 优化完成 🚀 ---")
        return

    # 5. 执行多轮迭代
    try:
        epoch1_video_dir.mkdir(parents=True, exist_ok=True)
        epoch2_video_dir.mkdir(parents=True, exist_ok=True)

        playbook_v0_learned = await run_epoch_0_warmup(
            tasks, playbook_v0, image_dir, original_video_dir, save_every=args.save_every
        )
        playbook_v1_learned = await run_epoch_1_learn(tasks, playbook_v0_learned, image_dir, epoch1_video_dir)
        await run_epoch_2_final(tasks, playbook_v1_learned, image_dir, epoch2_video_dir)

        if not args.no_normalize_playbook:
            finalize_playbook(PLAYBOOK_FILE, output_path=args.normalized_playbook_file)

        print("\n--- 🚀 ACE 多轮迭代全部完成 🚀 ---")

    except Exception as e:
        print(f"\n--- ❗ 异步主循环中发生意外错误 ---")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行主异步函数
    asyncio.run(main())
