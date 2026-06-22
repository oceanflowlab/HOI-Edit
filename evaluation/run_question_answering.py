#!/usr/bin/env python3
"""
基于Gemini的图像问答脚本
针对L3和L1L2数据集，根据不同的question对图像进行问答
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Optional, Any
import re
from PIL import Image, ImageDraw

from tool_unmentioned_utils import (
    is_tool_unmentioned_sample,
    question_mentions_yellow_box,
    resolve_tool_bbox_for_qa,
)

# Gemini API相关导入
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import APIError
    HAS_GOOGLE_SDK = True
except ImportError:
    HAS_GOOGLE_SDK = False

# --- 配置（仅 Google 官方 Gemini API）---
MODEL_NAME = 'gemini-2.5-pro'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MAX_API_RETRIES = 3
API_RETRY_BACKOFF = 1.5

gemini_client = None


def init_gemini_client() -> None:
    """初始化 Google Gemini 客户端。"""
    global gemini_client
    gemini_client = None
    if not GEMINI_API_KEY:
        print("❗ 未设置 GEMINI_API_KEY，请 export GEMINI_API_KEY 或在 env/local.conf 中配置")
        raise SystemExit(1)
    if not HAS_GOOGLE_SDK:
        print("❗ 需要安装 google-genai SDK: pip install google-genai")
        raise SystemExit(1)
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API 客户端已初始化（Google 官方 SDK）")
    except Exception as e:
        print(f"❗ 无法初始化 Gemini API 客户端: {e}")
        raise SystemExit(1)


def guess_mime_type(file_path):
    """根据文件扩展名推测 MIME 类型，默认使用 image/jpeg。"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        return "image/jpeg"
    return mime_type


def build_image_part(image_path):
    """将图像文件转换为 Gemini API 所需的 Part 对象。"""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"   ❗ 图像文件不存在: {image_path}")
        return None
    except OSError as e:
        print(f"   ❗ 无法读取图像文件 {image_path}: {e}")
        return None

    mime_type = guess_mime_type(image_path)
    blob = types.Blob(data=data, mime_type=mime_type)
    return types.Part(inline_data=blob)


def call_gemini_vqa(prompt_text, image_paths=None, temperature=0.1):
    """调用 Gemini 模型进行视觉问答。"""
    image_paths = image_paths or []

    if gemini_client is None:
        print("   ❗ Gemini 客户端未初始化")
        return None

    parts = [types.Part(text=prompt_text)]
    for path in image_paths:
        part = build_image_part(path)
        if part is None:
            return None
        parts.append(part)

    contents = [types.Content(role="user", parts=parts)]
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
    except APIError as e:
        error_msg = f"Gemini API 错误: {e}"
        print(f"   ❗ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"调用 Gemini API 失败: {e}"
        print(f"   ❗ {error_msg}")
        return {"error": error_msg}

    text_content = (response.text or "").strip()
    if not text_content:
        error_msg = "Gemini API 返回为空"
        print(f"   ❗ {error_msg}")
        return {"error": error_msg}
    
    # 移除可能的markdown代码块标记
    if text_content.startswith("```"):
        lines = text_content.split("\n")
        if len(lines) >= 3:
            text_content = "\n".join(lines[1:-1]).strip()
    
    try:
        parsed_result = json.loads(text_content)
        # 检查返回类型：如果是 list，尝试提取第一个元素
        if isinstance(parsed_result, list):
            if len(parsed_result) > 0:
                print(f"   ⚠️  API 返回了数组，使用第一个元素")
                parsed_result = parsed_result[0]
            else:
                error_msg = "API 返回了空数组"
                print(f"   ❗ {error_msg}")
                return {"error": error_msg}
        # 确保返回的是字典
        if not isinstance(parsed_result, dict):
            error_msg = f"API 返回了意外的数据类型: {type(parsed_result).__name__}"
            print(f"   ❗ {error_msg}")
            return {"error": error_msg}
        return parsed_result
    except json.JSONDecodeError as e:
        snippet = text_content[:200].replace("\n", " ")
        error_msg = f"解析 API 响应失败: {e} | 片段: {snippet!r}"
        print(f"   ❗ {error_msg}")
        return {"error": error_msg}


def call_gemini_vqa_with_retry(prompt_text, image_paths=None, temperature=0.1):
    """带重试机制的 Gemini VQA 调用。"""
    sleep_interval = 0.3
    last_error = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        result = call_gemini_vqa(
            prompt_text,
            image_paths=image_paths,
            temperature=temperature
        )
        
        # 如果返回的是错误信息，记录最后一次错误
        if isinstance(result, dict) and "error" in result:
            last_error = result["error"]
            if attempt < MAX_API_RETRIES:
                time.sleep(sleep_interval)
                sleep_interval *= API_RETRY_BACKOFF
                continue
            else:
                # 最后一次重试也失败，返回错误信息
                return result
        elif result is not None:
            # 成功获取结果
            return result
        
        # result is None的情况
        if attempt < MAX_API_RETRIES:
            time.sleep(sleep_interval)
            sleep_interval *= API_RETRY_BACKOFF
        else:
            return {"error": f"API调用失败（重试{MAX_API_RETRIES}次后仍失败）: {last_error}"}
    
    return {"error": "未知错误"}


def extract_questions(sample: Dict[str, Any], question_field: str = "question") -> List[str]:
    """从样本中提取所有问题（仅标注字段，不用 generated_question）。

    question_field:
      - question（默认）: 优先 questions 列表，否则 question
      - question_v6: 使用 question_v6（列表或字符串）
      - 其他字段名: 直接从该字段读取（列表或字符串）

    无有效问题时返回 []，调用方跳过该样本。
    返回去重后的问题列表；多条时对同一图像逐条调用 API。
    """
    questions_list = []

    if question_field == "question":
        if "questions" in sample and isinstance(sample["questions"], list):
            questions_list = sample["questions"]
        elif "question" in sample:
            question_value = sample["question"]
            if question_value is None or question_value == "":
                pass
            elif isinstance(question_value, list):
                questions_list = question_value
            elif isinstance(question_value, str):
                q = question_value.strip()
                if q:
                    questions_list = [q]
    elif question_field in sample:
        question_value = sample[question_field]
        if question_value is None or question_value == "":
            pass
        elif isinstance(question_value, list):
            questions_list = question_value
        elif isinstance(question_value, str):
            q = question_value.strip()
            if q:
                questions_list = [q]

    # 提取有效的问题（过滤掉None、空字符串等）
    questions = [q.strip() for q in questions_list if q and isinstance(q, str) and q.strip()]
    
    # 去重并保持顺序
    seen = set()
    unique_questions = []
    for q in questions:
        if q and q not in seen:
            seen.add(q)
            unique_questions.append(q)
    
    return unique_questions


def parse_tool_bboxes(tool_bboxes_str: str) -> Optional[List[int]]:
    """解析tool_bboxes字符串，提取坐标。
    
    Args:
        tool_bboxes_str: 格式如 "[   2,   1146,   203,   1482 ]" 或 "[2, 1146, 203, 1482]"
                        可能是JSON字符串格式，需要先去除引号
    
    Returns:
        坐标列表 [x1, y1, x2, y2]，如果解析失败返回None
    """
    if not tool_bboxes_str:
        return None
    
    # 如果是非字符串类型，尝试转换
    if not isinstance(tool_bboxes_str, str):
        # 如果是列表，直接返回
        if isinstance(tool_bboxes_str, list) and len(tool_bboxes_str) >= 4:
            try:
                return [int(x) for x in tool_bboxes_str[:4]]
            except (ValueError, TypeError):
                return None
        return None
    
    # 去除首尾空白字符和可能的JSON转义引号
    tool_bboxes_str = tool_bboxes_str.strip().strip('"').strip("'")
    
    # 如果为空字符串，返回None
    if not tool_bboxes_str:
        return None
    
    # 尝试使用json.loads解析（如果是有效的JSON数组字符串）
    try:
        parsed = json.loads(tool_bboxes_str)
        if isinstance(parsed, list) and len(parsed) >= 4:
            return [int(x) for x in parsed[:4]]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # 如果JSON解析失败，使用正则表达式提取数字
    numbers = re.findall(r'\d+', tool_bboxes_str)
    if len(numbers) >= 4:
        try:
            coords = [int(n) for n in numbers[:4]]
            return coords
        except ValueError:
            return None
    
    return None


def draw_yellow_box_on_image(image_path: str, tool_bbox: List[int], output_path: Optional[str] = None) -> str:
    """在图像上绘制黄色框。
    
    Args:
        image_path: 输入图像路径
        tool_bbox: 边界框坐标 [x1, y1, x2, y2]
        output_path: 输出图像路径，如果为None则创建临时文件
    
    Returns:
        输出图像路径
    """
    if len(tool_bbox) != 4:
        return image_path
    
    try:
        # 打开图像
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        x1, y1, x2, y2 = tool_bbox
        
        # 绘制黄色框（RGB: 255, 255, 0）
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
        
        # 保存图像
        if output_path is None:
            # 创建临时文件
            base_name = os.path.splitext(image_path)[0]
            ext = os.path.splitext(image_path)[1]
            output_path = f"{base_name}_with_yellow_box{ext}"
        
        img.save(output_path)
        return output_path
    except Exception as e:
        print(f"   ⚠️  绘制黄色框失败: {e}")
        return image_path


def should_draw_tool_yellow_box(
    question: str,
    tool_bboxes: Any,
    *,
    draw_tool_yellow_box: bool,
    is_tool_unmentioned: bool,
    is_edited: bool,
) -> bool:
    """Whether to overlay yellow tool box on the image sent to VQA."""
    if not tool_bboxes or not parse_tool_bboxes(tool_bboxes):
        return False
    if draw_tool_yellow_box:
        if not is_edited:
            return False
        if not is_tool_unmentioned:
            return False
        return question_mentions_yellow_box(question, legacy_only=False)
    # Legacy: only exact "yellow box" substring (any image type)
    return question_mentions_yellow_box(question, legacy_only=True)


def answer_question(image_path: str, question: str, 
                    subject_bbox: Optional[List[int]] = None,
                    object_bbox: Optional[List[int]] = None,
                    instruction: Optional[str] = None,
                    tool_bboxes: Optional[str] = None,
                    *,
                    is_edited: bool = False,
                    draw_tool_yellow_box: bool = False,
                    is_tool_unmentioned: bool = False,
                    tool_track_dir: Optional[str] = None,
                    image_name: Optional[str] = None) -> Dict[str, Any]:
    """对单个图像和问题进行问答。"""
    
    actual_image_path = image_path
    temp_image_path = None

    draw_yellow = should_draw_tool_yellow_box(
        question,
        tool_bboxes,
        draw_tool_yellow_box=draw_tool_yellow_box,
        is_tool_unmentioned=is_tool_unmentioned,
        is_edited=is_edited,
    )
    tool_bbox = None
    if draw_yellow:
        tool_bbox = resolve_tool_bbox_for_qa(
            tool_bboxes,
            parse_tool_bboxes_fn=parse_tool_bboxes,
            is_edited=is_edited,
            tool_track_dir=tool_track_dir,
            image_name=image_name or os.path.basename(image_path),
        )
        if tool_bbox:
            tqdm.write(f"      🟡 绘制工具黄框: {tool_bbox}")
            base_name = os.path.splitext(image_path)[0]
            ext = os.path.splitext(image_path)[1]
            temp_image_path = f"{base_name}_with_yellow_box{ext}"
            actual_image_path = draw_yellow_box_on_image(image_path, tool_bbox, temp_image_path)
            if actual_image_path != image_path:
                tqdm.write(f"      ✅ 已绘制黄色框，使用图像: {os.path.basename(actual_image_path)}")
    
    # 构建prompt
    bbox_info = ""
    if subject_bbox and len(subject_bbox) == 4:
        bbox_info += f"\n- Subject (person) bounding box: {subject_bbox} (shown in BLUE box)\n"
    if object_bbox and len(object_bbox) == 4:
        bbox_info += f"- Object bounding box: {object_bbox} (shown in RED box)\n"
    
    yellow_box_info = ""
    if draw_yellow and tool_bbox:
        yellow_box_info = f"\n- Tool bounding box: {tool_bbox} (shown in YELLOW box)\n"
    
    instruction_info = ""
    if instruction:
        instruction_info = f"\nNote: The editing instruction was: \"{instruction}\"\n"
    
    prompt = f"""You are analyzing an image to answer a question about human-object interaction.

{bbox_info}{yellow_box_info}{instruction_info}
Please answer the following question based on what you see in the image:

Question: {question}

Provide your answer in a strict JSON format with the following keys:
- "answer": Your answer to the question (can be "yes", "no", or a detailed explanation)
- "confidence": Your confidence in the answer, from 0.0 to 1.0
- "reasoning": A brief explanation of why you chose this answer

Example output:
{{"answer": "yes", "confidence": 0.95, "reasoning": "The person is clearly holding the object in their hand."}}
"""
    
    result = call_gemini_vqa_with_retry(
        prompt,
        image_paths=[actual_image_path],
        temperature=0.1
    )
    
    # 清理临时文件（如果创建了）
    if temp_image_path and os.path.exists(temp_image_path) and temp_image_path != image_path:
        try:
            os.remove(temp_image_path)
        except Exception as e:
            tqdm.write(f"      ⚠️  清理临时文件失败: {e}")
    
    if result is None:
        return {"error": "无法从 Gemini API 获取结果"}
    
    if isinstance(result, dict) and "error" in result:
        return result
    
    # 确保返回结果包含必要的字段
    if "answer" not in result:
        result["answer"] = "unknown"
    if "confidence" not in result:
        result["confidence"] = 0.0
    if "reasoning" not in result:
        result["reasoning"] = ""
    
    return result


def _find_edited_under_root(edited_root: str, image_name: str) -> Optional[str]:
    """在指定根目录（含可选 1/2/3 子目录）查找编辑图。"""
    if not edited_root or not os.path.isdir(edited_root):
        return None
    image_stem = os.path.splitext(image_name)[0]
    image_ext = os.path.splitext(image_name)[1]
    candidate_names = [
        image_name,
        f"{image_stem}_edited.png",
        f"{image_stem}_edited.jpg",
        f"{image_stem}.png",
        f"{image_stem}{image_ext}",
    ]
    search_roots = [edited_root]
    for sub in ("1", "2", "3"):
        subdir = os.path.join(edited_root, sub)
        if os.path.isdir(subdir):
            search_roots.append(subdir)
    for root in search_roots:
        for name in candidate_names:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
    return None


def find_image_path(
    image_name: str,
    image_dir: str,
    model_name: Optional[str] = None,
    is_edited: bool = False,
    dataset_type: Optional[str] = None,
    model_edited_dirs: Optional[Dict[str, str]] = None,
    use_v7_orig: bool = False,
) -> Optional[str]:
    """查找图像文件路径。
    
    Args:
        image_name: 图像文件名（如 "xxx.png"）
        image_dir: 基础图像目录
        model_name: 模型名称（如 "nanobanana", "qwen_plus"），如果提供则查找编辑后的图像
        is_edited: 是否查找编辑后的图像（添加_edited后缀）
        dataset_type: "L1L2" / "L3" 时优先匹配对应后缀的编辑目录，避免 L1L2 任务误用 L3 图
    
    Returns:
        找到的图像路径，如果未找到则返回None
    """
    # 如果指定了模型名称，查找编辑后的图像
    if model_name:
        if model_edited_dirs and model_name in model_edited_dirs:
            found = _find_edited_under_root(model_edited_dirs[model_name], image_name)
            if found:
                return found

        workspace = os.environ.get("EVAL_WORKSPACE", "")
        if not workspace:
            workspace = str(Path(__file__).resolve().parent.parent)
        data_cr = os.path.join(workspace, "data_v7", "CR")
        split = dataset_type if dataset_type in ("L1L2", "L3") else "L3"
        default_dir = os.path.join(data_cr, f"{model_name}_frames", split)
        found = _find_edited_under_root(default_dir, image_name)
        if found:
            return found

    # 查找原始图像
    if use_v7_orig:
        try:
            from v7_path_utils import resolve_original_image_path
            orig = resolve_original_image_path(image_dir, image_name)
            if os.path.isfile(orig):
                return orig
        except ImportError:
            pass

    direct_path = os.path.join(image_dir, image_name)
    if os.path.exists(direct_path):
        return direct_path
    
    # 尝试不同的目录结构
    possible_dirs = [
        image_dir,
        os.path.join(image_dir, "v6_images_L3"),
        os.path.join(image_dir, "v6_images_L1L2"),
        os.path.join(image_dir, "organized_hoi_dataset", "data_v6", "v6_images_L3"),
        os.path.join(image_dir, "organized_hoi_dataset", "data_v6", "v6_images_L1L2"),
    ]
    
    for base_dir in possible_dirs:
        full_path = os.path.join(base_dir, image_name)
        if os.path.exists(full_path):
            return full_path
    
    return None


def _qa_sample_complete(sample_results: dict, models: Optional[List[str]]) -> bool:
    """resume 模式下判断该样本是否已有全部目标模型的有效答案。"""
    questions = sample_results.get("questions") or []
    if not questions:
        return False
    if not models:
        return bool(questions)
    for q in questions:
        if not isinstance(q, dict):
            return False
        answers = q.get("answers") or {}
        for model_name in models:
            ans = answers.get(model_name)
            if not isinstance(ans, dict) or ans.get("error") or "answer" not in ans:
                return False
    return True


def process_dataset(json_path: str, image_dir: str, output_path: str, 
                   dataset_type: str = "L3", models: Optional[List[str]] = None,
                   include_original: bool = False,
                   model_edited_dirs: Optional[Dict[str, str]] = None,
                   use_v7_orig: bool = False,
                   question_field: str = "question",
                   draw_tool_yellow_box: bool = False,
                   tool_track_dir: Optional[str] = None,
                   resume: bool = False):
    """处理整个数据集。"""
    print(f"\n{'='*80}")
    print(f"📊 处理数据集: {dataset_type}")
    print(f"📁 JSON文件: {json_path}")
    print(f"🖼️  图像目录: {image_dir}")
    print(f"💾 输出文件: {output_path}")
    print(f"❓ 问题字段: {question_field}")
    if draw_tool_yellow_box:
        print(f"🟡 tool_unmentioned 编辑图黄框: 开启")
        print(f"   tool 追踪目录: {tool_track_dir or '(未指定，使用 JSON tool_bboxes)'}")
    print(f"{'='*80}\n")
    
    # 读取JSON文件
    print("📖 读取JSON文件...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 找到 {len(data)} 个样本\n")
    
    prior_results: Dict[str, dict] = {}
    if resume and os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict) and isinstance(old.get("results"), dict):
                prior_results = old["results"]
                print(f"♻️  resume: 载入已有 QA 结果 {len(prior_results)} 条")
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  resume: 无法读取已有输出，将全量重跑: {e}")
    
    # 准备输出结果
    results = {}
    total_questions = 0
    processed_samples = 0
    error_samples = 0
    skipped_resume = 0
    
    # 如果没有指定模型，默认只处理原始图像
    if models is None:
        models = []
    
    # 处理每个样本
    for image_name, sample in tqdm(data.items(), desc="处理样本"):
        # 提取问题
        questions = extract_questions(sample, question_field=question_field)
        if not questions:
            print(f"⚠️  样本 {image_name} 没有找到问题（字段: {question_field}）")
            results[image_name] = prior_results.get(image_name) or {
                "warning": f"没有找到问题（字段: {question_field}）",
                "questions": []
            }
            continue
        
        if resume and image_name in prior_results and _qa_sample_complete(prior_results[image_name], models):
            results[image_name] = prior_results[image_name]
            processed_samples += 1
            skipped_resume += 1
            total_questions += len(questions)
            continue
        
        total_questions += len(questions)
        
        # 获取边界框信息
        subject_bbox = sample.get("subject_bounding_box")
        object_bbox = sample.get("object_bounding_box")
        instruction = sample.get("instruction", "")
        tool_bboxes = sample.get("tool_bboxes", "")
        sample_is_tool_unmentioned = is_tool_unmentioned_sample(sample)
        
        # 准备结果结构
        sample_results = {
            "instruction": instruction,
            "subject_bbox": subject_bbox,
            "object_bbox": object_bbox,
            "tool_bboxes": tool_bboxes,
            "questions": []
        }
        
        # 对每个问题进行问答
        for idx, question in enumerate(questions, 1):
            tqdm.write(f"\n📝 [{image_name}] 问题 {idx}/{len(questions)}: {question[:60]}...")
            
            question_result = {
                "question": question,
                "answers": {}
            }
            
            # 1. 对原始图像进行问答（可选，默认跳过）
            if include_original:
                original_image_path = find_image_path(
                    image_name, image_dir, dataset_type=dataset_type, use_v7_orig=use_v7_orig
                )
                if original_image_path:
                    tqdm.write(f"   🖼️  原始图像: {os.path.basename(original_image_path)}")
                    original_result = answer_question(
                        image_path=original_image_path,
                        question=question,
                        subject_bbox=subject_bbox,
                        object_bbox=object_bbox,
                        instruction=instruction,
                        tool_bboxes=tool_bboxes,
                        is_edited=False,
                        draw_tool_yellow_box=draw_tool_yellow_box,
                        is_tool_unmentioned=sample_is_tool_unmentioned,
                        tool_track_dir=tool_track_dir,
                        image_name=image_name,
                    )
                    
                    # 只有成功获取结果才保存
                    if original_result and "error" not in original_result:
                        question_result["answers"]["original"] = original_result
                        answer = original_result.get("answer", "unknown")
                        confidence = original_result.get("confidence", 0.0)
                        tqdm.write(f"      ✅ 答案: {answer} (置信度: {confidence:.2f})")
                    else:
                        tqdm.write(f"      ⏭️  跳过（API错误）")
                    
                    time.sleep(0.3)
                else:
                    tqdm.write(f"      ⏭️  跳过原始图像（未找到）")
            
            # 2. 对不同模型的编辑后图像进行问答（如果找到）
            for model_name in models:
                edited_image_path = find_image_path(
                    image_name,
                    image_dir,
                    model_name=model_name,
                    is_edited=True,
                    dataset_type=dataset_type,
                    model_edited_dirs=model_edited_dirs,
                    use_v7_orig=use_v7_orig,
                )
                if edited_image_path:
                    tqdm.write(f"   🎨 {model_name}: {os.path.basename(edited_image_path)}")
                    edited_result = answer_question(
                        image_path=edited_image_path,
                        question=question,
                        subject_bbox=subject_bbox,
                        object_bbox=object_bbox,
                        instruction=instruction,
                        tool_bboxes=tool_bboxes,
                        is_edited=True,
                        draw_tool_yellow_box=draw_tool_yellow_box,
                        is_tool_unmentioned=sample_is_tool_unmentioned,
                        tool_track_dir=tool_track_dir,
                        image_name=image_name,
                    )
                    
                    # 只有成功获取结果才保存
                    if edited_result and "error" not in edited_result:
                        question_result["answers"][model_name] = edited_result
                        answer = edited_result.get("answer", "unknown")
                        confidence = edited_result.get("confidence", 0.0)
                        tqdm.write(f"      ✅ 答案: {answer} (置信度: {confidence:.2f})")
                    else:
                        tqdm.write(f"      ⏭️  跳过（API错误）")
                    
                    time.sleep(0.3)
                else:
                    tqdm.write(f"      ⏭️  跳过{model_name}（图像未找到）")
            
            # 只有当至少有一个答案时才保存问题结果
            if question_result["answers"]:
                sample_results["questions"].append(question_result)
            else:
                tqdm.write(f"      ⚠️  该问题没有可用的答案，跳过")
        
        # 只有当至少有一个问题的答案时才保存样本结果
        if sample_results["questions"]:
            # 保存图像路径信息（只保存找到的）
            if include_original:
                original_image_path = find_image_path(
                    image_name, image_dir, dataset_type=dataset_type, use_v7_orig=use_v7_orig
                )
                if original_image_path:
                    sample_results["original_image_path"] = original_image_path
            
            if models:
                sample_results["edited_image_paths"] = {}
                for model_name in models:
                    edited_path = find_image_path(
                        image_name,
                        image_dir,
                        model_name=model_name,
                        is_edited=True,
                        dataset_type=dataset_type,
                        model_edited_dirs=model_edited_dirs,
                        use_v7_orig=use_v7_orig,
                    )
                    if edited_path:
                        sample_results["edited_image_paths"][model_name] = edited_path
            
            results[image_name] = sample_results
            processed_samples += 1
        else:
            tqdm.write(f"   ⚠️  样本 {image_name} 没有可用的答案，跳过")
            error_samples += 1
    
    # 保存结果
    print(f"\n💾 保存结果到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    output_data = {
        "dataset_type": dataset_type,
        "total_samples": len(data),
        "processed_samples": processed_samples,
        "error_samples": error_samples,
        "total_questions": total_questions,
        "include_original": include_original,
        "models": models if models else [],
        "draw_tool_yellow_box": draw_tool_yellow_box,
        "tool_track_dir": tool_track_dir,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 处理完成！")
    print(f"   - 总样本数: {len(data)}")
    print(f"   - 成功处理: {processed_samples}")
    if resume and skipped_resume:
        print(f"   - resume 跳过: {skipped_resume}")
    print(f"   - 错误样本: {error_samples}")
    print(f"   - 总问题数: {total_questions}")
    print(f"   - 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="基于Gemini的图像问答脚本")
    parser.add_argument("--json", type=str, required=True,
                       help="输入的JSON文件路径（包含问题和样本信息）")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="原始图像目录路径")
    parser.add_argument("--output", type=str, required=True,
                       help="输出JSON文件路径")
    parser.add_argument("--dataset-type", type=str, default="L3",
                       choices=["L3", "L1L2"],
                       help="数据集类型（L3或L1L2）")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                       help="要处理的模型列表（如：nanobanana qwen_plus wan_camera），必须至少指定一个模型")
    parser.add_argument("--include-original", action="store_true",
                       help="是否同时处理原始图像（默认不处理）")
    parser.add_argument("--v7", action="store_true",
                       help="V7 数据集：扁平原图目录 + 支持 final_eval_data_edited_*_v6_V7_resized")
    parser.add_argument("--model-edited-dir", action="append", default=[],
                       help="显式指定模型编辑图目录，格式 MODEL=DIR，可重复")
    parser.add_argument("--question-field", type=str, default="question",
                       help="从 JSON 读取问题的字段名（默认 question；CR 新版用 question_v6）")
    parser.add_argument(
        "--draw-tool-yellow-box",
        action="store_true",
        help="对 tool_unmentioned 样本在编辑图上按 tool 框（优先 SAM2 追踪）绘制黄框；默认关闭",
    )
    parser.add_argument(
        "--tool-track-dir",
        type=str,
        default=None,
        help="SAM2 tool 追踪输出根目录（含 tool_bboxes/ 子目录）；与 --draw-tool-yellow-box 联用",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="增量续跑：跳过输出 JSON 中已有有效答案的样本",
    )
    
    args = parser.parse_args()

    init_gemini_client()

    model_edited_dirs: Dict[str, str] = {}
    for item in args.model_edited_dir:
        if "=" not in item:
            print(f"❌ --model-edited-dir 格式错误（需 MODEL=DIR）: {item}")
            exit(1)
        model_key, edited_dir = item.split("=", 1)
        model_key = model_key.strip()
        edited_dir = edited_dir.strip()
        if not model_key or not edited_dir:
            print(f"❌ --model-edited-dir 格式错误: {item}")
            exit(1)
        model_edited_dirs[model_key] = edited_dir
    
    # 检查输入文件是否存在
    if not os.path.exists(args.json):
        print(f"❌ JSON文件不存在: {args.json}")
        exit(1)
    
    if not os.path.exists(args.image_dir):
        print(f"❌ 图像目录不存在: {args.image_dir}")
        exit(1)
    
    # 检查是否至少指定了模型或原始图像
    if not args.models and not args.include_original:
        print("❌ 错误: 必须至少指定一个模型（--models）或启用原始图像处理（--include-original）")
        exit(1)
    
    # 处理数据集
    process_dataset(
        json_path=args.json,
        image_dir=args.image_dir,
        output_path=args.output,
        dataset_type=args.dataset_type,
        models=args.models,
        include_original=args.include_original,
        model_edited_dirs=model_edited_dirs or None,
        use_v7_orig=args.v7,
        question_field=args.question_field,
        draw_tool_yellow_box=args.draw_tool_yellow_box,
        tool_track_dir=args.tool_track_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
