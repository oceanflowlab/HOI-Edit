import os
import json
import mimetypes
import signal

# 代理：优先使用环境变量；否则 workspace 的 CR_DEFAULT_PROXY_URL
if not os.getenv('http_proxy') and not os.getenv('HTTP_PROXY'):
    _proxy = os.getenv('CR_DEFAULT_PROXY_URL', '')
    if _proxy:
        os.environ['http_proxy'] = _proxy
        os.environ['https_proxy'] = _proxy
        os.environ['HTTP_PROXY'] = _proxy
        os.environ['HTTPS_PROXY'] = _proxy

from PIL.Image import fromqimage
import cv2
import re
import numpy as np
from tqdm import tqdm
import argparse
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- 1. 配置 ---
# Gemini API 配置
MODEL_NAME = 'gemini-2.5-pro'  # 使用 Gemini 2.5 Pro
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# API 调用重试配置
MAX_API_RETRIES = 3
API_RETRY_BACKOFF = 1.5  # 每次重试的退避倍数
API_TIMEOUT_SECONDS = 120  # API调用超时时间（秒）

try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY 未设置")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API 客户端已初始化")
except Exception as e:
    gemini_client = None
    print(f"❗ 无法初始化 Gemini API 客户端: {e}")


# --- 超时处理类 ---
class TimeoutError(Exception):
    """API调用超时异常"""
    pass


def timeout_handler(signum, frame):
    """信号处理函数，超时时抛出异常"""
    raise TimeoutError("API调用超时")


def guess_mime_type(file_path):
    """根据文件扩展名推测 MIME 类型，默认使用 image/jpeg。"""
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
        tqdm.write(f"   ❗ 图像文件不存在: {image_path}")
        return None
    except OSError as e:
        tqdm.write(f"   ❗ 无法读取图像文件 {image_path}: {e}")
        return None

    mime_type = guess_mime_type(image_path)
    blob = types.Blob(data=data, mime_type=mime_type)
    return types.Part(inline_data=blob)


def call_gemini_json(
        prompt_text,
        image_paths=None,
        temperature=0.1,
        response_mime_type="application/json"):
    """调用 Gemini 模型并返回 JSON 结果。"""
    if gemini_client is None:
        tqdm.write("   ❗ Gemini 客户端未初始化")
        return None

    image_paths = image_paths or []
    parts = [types.Part(text=prompt_text)]

    for path in image_paths:
        part = build_image_part(path)
        if part is None:
            return None
        parts.append(part)

    contents = [
        types.Content(
            role="user",
            parts=parts
        )
    ]

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type=response_mime_type
    )

    try:
        # 设置超时信号处理
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(API_TIMEOUT_SECONDS)
        
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
        finally:
            # 取消超时，恢复原来的信号处理
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    except TimeoutError:
        error_msg = f"Gemini API 调用超时（{API_TIMEOUT_SECONDS}秒）"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}
    except APIError as e:
        # 确保取消超时
        signal.alarm(0)
        error_msg = f"Gemini API 错误: {e}"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        # 确保取消超时
        signal.alarm(0)
        error_msg = f"调用 Gemini API 失败: {e}"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}

    text_content = (response.text or "").strip()
    if not text_content:
        error_msg = "Gemini API 返回为空"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}

    if text_content.startswith("```"):
        lines = text_content.split("\n")
        if len(lines) >= 3:
            text_content = "\n".join(lines[1:-1]).strip()

    try:
        parsed_result = json.loads(text_content)
        # 检查返回类型：如果是 list，尝试提取第一个元素（某些情况下 API 可能返回数组）
        if isinstance(parsed_result, list):
            if len(parsed_result) > 0:
                tqdm.write(f"   ⚠️  Gemini API 返回了数组，使用第一个元素")
                parsed_result = parsed_result[0]
            else:
                error_msg = "Gemini API 返回了空数组"
                tqdm.write(f"   ❗ {error_msg}")
                return {"error": error_msg}
        # 确保返回的是字典
        if not isinstance(parsed_result, dict):
            error_msg = f"Gemini API 返回了意外的数据类型: {type(parsed_result).__name__}，期望 dict。返回内容: {str(parsed_result)[:200]}"
            tqdm.write(f"   ❗ {error_msg}")
            return {"error": error_msg}
        return parsed_result
    except json.JSONDecodeError as e:
        snippet = text_content[:200].replace("\n", " ")
        error_msg = f"解析 Gemini 响应失败: {e} | 片段: {snippet!r}"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}


def call_gemini_json_with_retry(prompt_text, image_paths=None, temperature=0.1,
                                response_mime_type="application/json"):
    """带重试机制的 Gemini JSON 调用。"""
    sleep_interval = 0.3
    last_error = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        result = call_gemini_json(
            prompt_text,
            image_paths=image_paths,
            temperature=temperature,
            response_mime_type=response_mime_type
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

        # result is None的情况（不应该发生，因为call_gemini_json现在总是返回dict）
        if attempt < MAX_API_RETRIES:
            time.sleep(sleep_interval)
            sleep_interval *= API_RETRY_BACKOFF

    # 所有重试都失败，返回错误信息
    if last_error:
        return {"error": f"API调用失败（重试{MAX_API_RETRIES}次）: {last_error}"}
    return {"error": f"API调用失败（重试{MAX_API_RETRIES}次）: 无法获取响应"}


# 绘图常量
BLUE = (255, 0, 0)
RED = (0, 0, 255)

IMAGES_REQUIRE_COMPARISON = {
    "P6enFEL2uzs-0:09:37.540-0:09:42.200_frame_0003_112.png",
    "k7BygHbdpTA-0:00:35.000-0:00:42.040_frame_0002_84.png",
    "S3ltKKQ3UP8-0:00:53.920-0:00:56.480_frame_0001_3.png",
    "JBmIDysRvn4-0:13:52.560-0:14:07.080_frame_0001_0.png",
    "D_udKnvx6tQ-0:01:00.794-0:01:05.098_frame_0002_59.png",
    "DCLa54pLgTE-0:14:39.166-0:14:47.233_frame_0001_5.png",
    "9r5QHiNP3Vg-0:03:26.960-0:03:31.800_frame_0003_120.png",
    "8411368_resize1080p_scene-6_frame_0002_28.png",
    "8358813_resize1080p_scene-7_frame_0001_0.png",
    "7UWSSX7gLYU-0:01:21.360-0:01:26.520_frame_0003_129.png",
    "2M2qmdN5KRk-0:04:31.800-0:04:38.520_frame_0002_85.png",
    "18877220_resize1080p_scene-4_frame_0003_59.png",
    "18599775_resize1080p_scene-3_frame_0003_59.png",
}
WHITE = (255, 255, 255)

# --- 2. 命令行参数解析 ---


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini VQA - 寻找最佳主客体检测框配对")
    parser.add_argument(
        "--input_json_path",
        type=str,
        required=True,
        help="包含指令和图像名称的输入JSON文件路径。")
    parser.add_argument(
        "--image_dir_path",
        type=str,
        required=True,
        help="存放编辑后图像的目录路径。")
    parser.add_argument(
        "--original_image_dir_path",
        type=str,
        required=False,
        help="存放原始图像（编辑前）的目录路径。如果不提供，将从image_dir_path推导。")
    parser.add_argument(
        "--person_dir_path",
        type=str,
        required=True,
        help="存放人物检测结果 (.txt) 的目录路径。")
    parser.add_argument(
        "--object_dir_path",
        type=str,
        required=False,
        help="存放物体检测结果 (.txt) 的目录路径。")
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="保存最终结果的输出JSON文件路径。")
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp_vqa_images",
        help="存放VQA临时图像的目录。")
    parser.add_argument(
        "--object_track_dir",
        type=str,
        required=False,
        help="存放跟踪生成的物体bbox JSON的目录（需包含bboxes子目录）。")
    parser.add_argument(
        "--track_frame_key",
        type=str,
        default="frame_00001",
        help="从跟踪JSON中读取的帧键 (默认: frame_00001)")
    parser.add_argument(
        "--image-list",
        type=str,
        default=None,
        dest="image_list",
        help="要处理的图像列表文件路径（每行一个图像名称，支持部分匹配）。如果不提供，处理所有图像。")
    parser.add_argument(
        "--image-names",
        type=str,
        default=None,
        dest="image_names",
        help="要处理的图像名称列表（逗号分隔，支持部分匹配）。例如: 'img1.png,img2.png'")
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理所有图像，即使已有结果也会跳过检查并重新处理。")
    return parser.parse_args()

# --- 4. 核心功能函数 ---


def parse_detection_file(file_path, return_scores=False):
    """从给定的txt文件中解析出所有的边界框坐标和置信度。

    Args:
        file_path: 检测结果文件路径
        return_scores: 是否返回置信度分数

    Returns:
        如果return_scores=True，返回 [(box, score), ...]
        否则返回 [box, ...]
    """
    if not os.path.exists(file_path):
        return []

    boxes = []
    with open(file_path, 'r') as f:
        content = f.read()

        if return_scores:
            # 解析置信度和坐标
            # Detection 0: human(0.54)
            # Box coordinates: [296, 431, 1048, 765]
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                # 查找Detection行
                if line.startswith('Detection'):
                    # 提取置信度
                    score_match = re.search(r'\(([0-9.]+)\)', line)
                    score = float(score_match.group(1)) if score_match else 0.0

                    # 下一行应该是Box coordinates
                    i += 1
                    if i < len(lines):
                        box_line = lines[i].strip()
                        box_match = re.search(
                            r"Box coordinates: \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", box_line)
                        if box_match:
                            box = [int(coord) for coord in box_match.groups()]
                            boxes.append((box, score))
                i += 1
        else:
            # 原有逻辑，只返回坐标
            matches = re.findall(
                r"Box coordinates: \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", content)
            for match in matches:
                boxes.append([int(coord) for coord in match])

    return boxes


def load_tracked_bbox(track_dir, image_base_name, frame_key="frame_00001"):
    """从跟踪结果JSON加载指定帧的边界框

    Args:
        track_dir: 跟踪结果目录
        image_base_name: 图像基础名称
        frame_key: 帧键（如 "frame_00001"）

    Returns:
        边界框 [x1, y1, x2, y2]，基于原始图像尺寸
        注意：如果原始图像和编辑后图像尺寸不同，需要在调用后手动缩放
    """
    if not track_dir:
        return None

    json_path = os.path.join(track_dir, "bboxes", f"{image_base_name}.json")

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracked = data.get("tracked_bboxes", {}) or {}
        box = tracked.get(frame_key)

        if not box:
            return None

        return box
    except (json.JSONDecodeError, OSError, ValueError):
        return None

    return None


def draw_box_and_label(
        image,
        box,
        label,
        color,
        line_type='solid',
        thickness=3):
    """在图像上绘制一个边界框和标签。

    Args:
        image: 输入图像
        box: 边界框坐标 [x1, y1, x2, y2]
        label: 标签文本
        color: 颜色 (B, G, R)
        line_type: 线条类型 'solid'(实线) 或 'dashed'(虚线)
        thickness: 线条粗细
    """
    if box is None:
        return image
    x1, y1, x2, y2 = map(int, box)
    thickness = 2

    if line_type == 'dashed':
        # 绘制虚线框
        dash_length = 10
        gap_length = 5

        # 上边
        for i in range(x1, x2, dash_length + gap_length):
            cv2.line(
                image, (i, y1), (min(
                    i + dash_length, x2), y1), color, thickness)
        # 下边
        for i in range(x1, x2, dash_length + gap_length):
            cv2.line(
                image, (i, y2), (min(
                    i + dash_length, x2), y2), color, thickness)
        # 左边
        for i in range(y1, y2, dash_length + gap_length):
            cv2.line(
                image, (x1, i), (x1, min(
                    i + dash_length, y2)), color, thickness)
        # 右边
        for i in range(y1, y2, dash_length + gap_length):
            cv2.line(
                image, (x2, i), (x2, min(
                    i + dash_length, y2)), color, thickness)
    else:
        # 绘制实线框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # if label:
    #     (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    #     cv2.rectangle(image, (x1, y1 - 25), (x1 + tw, y1), color, -1)
    #     cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    return image


def concatenate_images_horizontally(img1, img2, gap=20):
    """将两张图片横向拼接，中间留有间隙

    Args:
        img1: 第一张图片（编辑前）
        img2: 第二张图片（编辑后）
        gap: 中间间隙的宽度（像素）

    Returns:
        拼接后的图片
    """
    # 获取两张图片的高度和宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 使用较大的高度
    max_height = max(h1, h2)

    # 如果高度不同，调整图片大小使其高度一致
    if h1 != max_height:
        scale = max_height / h1
        img1 = cv2.resize(img1, (int(w1 * scale), max_height))
        w1 = int(w1 * scale)

    if h2 != max_height:
        scale = max_height / h2
        img2 = cv2.resize(img2, (int(w2 * scale), max_height))
        w2 = int(w2 * scale)

    # 创建拼接画布（白色背景）
    total_width = w1 + gap + w2
    concatenated = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

    # 放置两张图片
    concatenated[:, :w1] = img1
    concatenated[:, w1 + gap:] = img2

    return concatenated


def draw_numbered_boxes(image, boxes, entity_type, scores=None):
    """
    在图像上绘制所有检测框并标号

    Args:
        image: 原始图像
        boxes: 检测框列表
        entity_type: "subject" 或 "object"
        scores: 对应的置信度分数（可选）

    Returns:
        绘制了所有框的图像
    """
    img_copy = image.copy()

    # 根据实体类型选择颜色
    if entity_type == "subject":
        color = (0, 0, 255)  # 红色用于主体
        prefix = "S"
    else:
        color = (255, 0, 0)  # 蓝色用于客体
        prefix = "O"

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # 绘制边界框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)

        # 添加标号
        if scores and idx < len(scores):
            label_text = f"{prefix}{idx} ({scores[idx]:.2f})"
        else:
            label_text = f"{prefix}{idx}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(
            label_text, font, font_scale, font_thickness)

        # 标签背景
        label_y = y1 - 10
        if label_y < 20:
            label_y = y1 + text_h + 15

        cv2.rectangle(img_copy, (x1, label_y - text_h - 5),
                      (x1 + text_w + 10, label_y + 5), color, -1)
        cv2.putText(img_copy, label_text, (x1 + 5, label_y),
                    font, font_scale, (255, 255, 255), font_thickness)

    return img_copy


def add_title_to_image(image, title, title_height=80):
    """在图像上方添加标题栏"""
    h, w = image.shape[:2]

    # 创建新的画布（原图高度 + 标题栏高度）
    new_img = np.ones((h + title_height, w, 3), dtype=np.uint8) * 255

    # 将原图放在下方
    new_img[title_height:, :] = image

    # 添加标题文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(
        title, font, font_scale, font_thickness)

    # 文字居中
    text_x = (w - text_w) // 2
    text_y = (title_height + text_h) // 2

    # 绘制文字（黑色）
    cv2.putText(new_img, title, (text_x, text_y),
                font, font_scale, (0, 0, 0), font_thickness)

    return new_img


def evaluate_similarity_with_gemini(
        before_img_path,
        after_img_path,
        entity_type,
        detections,
        no_detection_mode=False,
        image_dimensions=None,
        edit_instruction=None,
        object_name=None):
    """
    使用 Gemini 评估编辑前后主客体的相似度 (ACE-Metric Final Version)

    集成特性:
    - Context-Aware: 理解编辑指令
    - Generative Infilling Compatible: 允许合理的补全
    - Uncanny Valley Penalty: 严厉惩罚低质量人脸
    - Geometric Integrity Check: 严厉惩罚物体形状坍塌

    Args:
        before_img_path: 编辑前图像路径（带红框标注原主客体）
        after_img_path: 编辑后图像路径（带多个标号框或无框）
        entity_type: "subject" 或 "object"
        detections: 检测结果列表（boxes）
        no_detection_mode: 是否为无检测框模式
        image_dimensions: 图像尺寸 (height, width)，用于无检测模式
        edit_instruction: 编辑指令（用于上下文理解）
        object_name: 客体类别名称（如 "dog", "phone", "ball" 等），仅在 entity_type=="object" 时使用

    Returns:
        包含最佳匹配ID和相似度得分的字典
    """

    if not no_detection_mode and len(detections) == 0:
        return None

    # 1. 构建上下文描述 (Context Injection)
    # 这是解题的关键：告诉 Gemini "原本打算做什么"
    context_str = ""
    if edit_instruction:
        context_str = f"\nℹ️ CONTEXT - EDIT INSTRUCTION: \"{edit_instruction}\"\nUse this instruction to distinguish between 'errors' and 'intended changes' (e.g., 'turn around' implies face disappears/appears, 'drop' implies position change)."

    prefix = "S" if entity_type == "subject" else "O"
    det_list_str = ", ".join(
        [f"{prefix}{i}" for i in range(len(detections))]) if detections else ""

    # 构建客体描述（如果有具体类别）
    object_desc = object_name if (
        entity_type == "object" and object_name and object_name != "object") else "object"

    # 2. 核心评分标准 (Rubric) - 针对 HOI 任务的最终优化版

    # --- SUBJECT RUBRIC (主体) ---
    subject_rubric = """
   a) Identity Score (0.0-1.0): Is it the same person?
      - 1.0 = Same person (or compatible infilling).
        * **Generative Infilling:** If the original view was partial (e.g., no head), accept generated parts if they are COMPATIBLE (skin tone, build, clothing style match).
        * **Valid Occlusion:** If parts disappear (e.g., turning back), rate 1.0 if remaining features match.
      - 0.0-0.3 = Direct contradiction (e.g., different race, different gender, completely different outfit).

   b) Quality Score (0.0-1.0): Anatomic & Visual Realism [CRITICAL]
      - 1.0 = Photorealistic. Indistinguishable from a real photo.
      - **CRITICAL PENALTY:** Rate < 0.4 IMMEDIATELY if you detect:
        * "Uncanny Valley" faces (waxy skin, dead eyes, asymmetry).
        * Distorted hands (sausage fingers, extra joints).
        * "Pasted" look (lighting on person doesn't match background).
      - **Logic:** Even if Identity is 1.0, if the person looks like a bad 3D render, Quality must be low.

   c) Similarity Score (0.0-1.0):
      - Calculation: Identity Score × Quality Score.
    """

    # --- OBJECT RUBRIC (客体) ---
    object_rubric = f"""
   a) Identity Score (0.0-1.0): Is it the same {object_desc} instance?
      - 1.0 = Same {object_desc} instance.
        * **Check Instruction:** Does the prompt imply movement/usage?
        * **Allow Interaction Changes:** 1. State (open/close) -> 1.0
          2. Location (in hand/on table/falling) -> 1.0
          3. Occlusion (grasped/released) -> 1.0
      - 0.0-0.5 = Core Identity Loss (e.g., material change, semantic change).

   b) Quality Score (0.0-1.0): Geometric Integrity & Physics [CRITICAL]
      - 1.0 = Solid, rigid structure preserved.
      - **CRITICAL PENALTY:** Rate < 0.4 if specific 3D structure is lost, even if color is correct.
        * Example: A tall cylinder becoming a flat box = 0.2.
        * Example: A metal pot becoming "melty" or soft = 0.2.

   c) Similarity Score (0.0-1.0):
      - Calculation: Identity Score × Quality Score.
    """

    rubric = subject_rubric if entity_type == "subject" else object_rubric

    # 3. Prompt 组装

    if no_detection_mode:
        # --- 模式 A: 无检测框 (Gemini 作为 Detector) ---
        dim_str = f"Dimensions: {image_dimensions[0]}x{image_dimensions[1]}." if image_dimensions else ""

        entity_desc = entity_type if entity_type == "subject" else object_desc
        prompt_text = f"""
Image 1: "BEFORE EDITING" ({entity_desc} in RED BOX).

Image 2: "AFTER EDITING" (No boxes detected).

{dim_str}

{context_str}

⚠️ SYSTEM FAILURE: Automatic detection failed. You are the backup detector.

Task 1: LOCATE

- Find the {entity_desc} in Image 2.

- NOTE: Position may change drastically based on the instruction (e.g., object falling, person sitting).

Task 2: EVALUATE

Rate based on these strict rules:

{rubric}

Return JSON ONLY:

{{"best_match_id": "manual", "identity_score": 0.0, "quality_score": 0.0, "similarity_score": 0.0, "box_coords": [x1, y1, x2, y2], "reasoning": "Explain based on instruction compliance, visual compatibility, and realism/geometry check."}}

"""
    else:
        # --- 模式 B: 有检测框 (Gemini 作为 Selector & Judge) ---
        entity_desc = entity_type if entity_type == "subject" else object_desc
        prompt_text = f"""
Image 1: "BEFORE EDITING" ({entity_desc} in RED BOX).

Image 2: "AFTER EDITING" (Candidates: {det_list_str}).

{context_str}

Task 1: MATCH

- Which candidate in Image 2 is the {entity_desc} from Image 1?

- **Context is Key:** If instruction says "release {object_desc}" or "drop {object_desc}", look for it falling/on table, NOT in hand.

- Ignore spatial distance if the action requires movement.

Task 2: EVALUATE

For the best match, rate based on these strict rules:

{rubric}

Return JSON ONLY:

{{"best_match_id": "{prefix}X", "identity_score": 0.0, "quality_score": 0.0, "similarity_score": 0.0, "reasoning": "Explain logic. Did it handle the interaction correctly? Is the quality/geometry preserved?"}}

"""

    result = call_gemini_json_with_retry(
        prompt_text,
        image_paths=[before_img_path, after_img_path],
        temperature=0.1
    )

    if result is None:
        error_msg = "无法从 Gemini API 获取相似度结果（API调用失败或返回为空）"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}

    # 检查返回结果中是否包含错误信息
    if isinstance(result, dict) and "error" in result:
        return result

    # 检查返回结果类型：必须是字典
    if not isinstance(result, dict):
        error_msg = f"Gemini API 返回了意外的数据类型: {type(result).__name__}，期望 dict。返回内容: {str(result)[:200]}"
        tqdm.write(f"   ❗ {error_msg}")
        return {"error": error_msg}

    match_id_str = result.get('best_match_id', '')
    match_id = None
    if match_id_str and match_id_str != 'manual':
        num_match = re.search(r'\d+', str(match_id_str))
        if num_match:
            try:
                match_id = int(num_match.group())
            except ValueError:
                match_id = None
    elif match_id_str == 'manual':
        match_id = 'manual'

    result['best_match_id_parsed'] = match_id
    return result


def generate_question_from_instruction(
        instruction,
        subject="person",
        object_name="the object"):
    """使用Gemini将陈述性指令转换为Yes/No问句。"""
    prompt = f"""
    Analyze the following instruction and convert it into a simple yes/no question to verify if the action has been completed.
    The subject is in a blue bounding box, and the object is in a red bounding box.
    The question should follow the format: "Has the {subject} in the blue bounding box [past participle] the {object_name} in the red bounding box?"

    Example 1:
    Instruction: "Make the child chase the dog."
    Subject: "child"
    Object Name: "dog"
    Output JSON: {{"question": "Has the child in the blue bounding box chased the dog in the red bounding box?"}}

    Example 2:
    Instruction: "Turn the phone over."
    Subject: "person"
    Object Name: "phone"
    Output JSON: {{"question": "Has the person in the blue bounding box turned over the phone in the red bounding box?"}}

    Now, process this input:
    Instruction: "{instruction}"
    Subject: "{subject}"
    Object Name: "{object_name}"

    Output JSON:
    """

    result = call_gemini_json_with_retry(
        prompt,
        image_paths=None,
        temperature=0.1
    )

    if result is None:
        tqdm.write(" - ❗ 无法从 Gemini API 获取问题生成结果")
        return None

    question = result.get("question")
    if not question:
        tqdm.write(" - ❗ Gemini API 响应中缺少 question 字段")
        return None

    return question


def perform_vqa_on_pair_single(before_image_path, after_image_path, question):
    """对编辑前后两张带标注框的图像进行对比VQA (使用Gemini API)。"""
    prompt = f"""
    You will see ONE image (AFTER EDITING) with blue (subject/person) and red (object) bounding boxes showing the DETECTED positions of the subject and object for interaction after editing.

    Answer the following question based solely on this edited image:
    Does the interaction shown in the question occur in the edited image?

    Consider:
    - Focus on the subject and object in the bounding boxes in the edited image
    - Check if the subject and object are in positions/poses that suggest the interaction
    - The interaction should be visually evident in the edited image

    Provide your answer in a strict JSON format with two keys: "answer" and "confidence".
    The "answer" must be "yes" or "no".
    The "confidence" score should reflect your certainty in the "yes" answer, from 0.0 to 1.0. If the answer is "no", confidence must be 0.0.
    Answer nothing else.

    Question: "{question}"

    Example output for a "yes" answer: {{"answer": "yes", "confidence": 0.95}}
    Example output for a "no" answer: {{"answer": "no", "confidence": 0.0}}
    """

    result = call_gemini_json_with_retry(
        prompt,
        image_paths=[after_image_path],
        temperature=0.1
    )

    if result is None:
        error_msg = "无法从 Gemini API 获取 VQA 结果（API调用失败或返回为空）"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    # 检查返回结果中是否包含错误信息
    if isinstance(result, dict) and "error" in result:
        return result

    answer = result.get("answer")
    confidence = result.get("confidence")

    if isinstance(answer, str):
        result["answer"] = answer.strip().lower()
    elif answer is None:
        error_msg = "VQA 结果缺少 answer 字段"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            error_msg = f"VQA 置信度无法解析为数字: {confidence}"
            tqdm.write(f"   - ❗ {error_msg}")
            return {"error": error_msg}
        result["confidence"] = confidence
    elif confidence is None:
        error_msg = "VQA 结果缺少 confidence 字段"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    return result


def perform_vqa_on_pair_with_comparison(
        before_image_path,
        after_image_path,
        question):
    """对编辑前后两张图像进行对比 VQA（使用 Gemini API）。"""
    prompt = f"""
    You will receive TWO images in chronological order:
    1. BEFORE EDITING (original state)
    2. AFTER EDITING (edited result)

    Both images contain blue (subject/person) and red (object) bounding boxes highlighting the relevant regions.

    Your task is to answer the given yes/no question for BOTH images, and to judge whether the edit successfully introduced the interaction.

    Consider:
    - Focus on the subject and object inside the bounding boxes
    - Determine if the interaction described in the question occurs in each image
    - Compare the two images and conclude whether the edit improved the interaction outcome

    Provide ONLY a JSON object with the following keys:
    {{
      "before_answer": "yes" or "no",
      "before_confidence": number between 0.0 and 1.0,
      "after_answer": "yes" or "no",
      "after_confidence": number between 0.0 and 1.0,
      "verdict": one of ["improved", "regressed", "unchanged"],
      "verdict_confidence": number between 0.0 and 1.0,
      "notes": short explanation (string, <= 30 words)
    }}

    Question: "{question}"

    Make sure numerical fields are valid JSON numbers, not strings.
    """
    result = call_gemini_json_with_retry(
        prompt,
        image_paths=[before_image_path, after_image_path],
        temperature=0.1
    )

    if result is None:
        error_msg = "无法从 Gemini API 获取 VQA 对比结果（API调用失败或返回为空）"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    # 检查返回结果中是否包含错误信息
    if isinstance(result, dict) and "error" in result:
        return result

    for key in ("before_answer", "after_answer", "verdict"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = value.strip().lower()
        elif value is None:
            error_msg = f"VQA 结果缺少 {key} 字段"
            tqdm.write(f"   - ❗ {error_msg}")
            return {"error": error_msg}

    for key in ("before_confidence", "after_confidence", "verdict_confidence"):
        value = result.get(key)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                error_msg = f"VQA 置信度字段 {key} 无法解析为数字: {value}"
                tqdm.write(f"   - ❗ {error_msg}")
                return {"error": error_msg}
            result[key] = value
        elif value is None:
            error_msg = f"VQA 结果缺少 {key} 字段"
            tqdm.write(f"   - ❗ {error_msg}")
            return {"error": error_msg}

    notes = result.get("notes")
    if isinstance(notes, str):
        result["notes"] = notes.strip()

    return result


def perform_vqa_on_pair(before_image_path, after_image_path, question):
    """对编辑前后两张带标注框的图像进行对比VQA (使用 Gemini API)。

    Args:
        before_image_path: 编辑前的图片路径（带原始主客体框标注）
        after_image_path: 编辑后的图片路径（带检测到的主客体框标注）
        question: 要回答的问题
    """
    prompt = f"""
    You will see TWO images for comparison:
    1. FIRST IMAGE (BEFORE EDITING): Original image with blue (subject/person) and red (object) bounding boxes showing the ORIGINAL positions of the subject and object for interaction.
    2. SECOND IMAGE (AFTER EDITING): Edited image with blue (subject/person) and red (object) bounding boxes showing the DETECTED positions of the subject and object for interaction after editing.

    Compare these two images and answer the following question based on the SECOND IMAGE (after editing):
    Does the interaction shown in the question occur in the edited image?

    Consider:
    - Focus on the subject and object in the bounding boxes in the AFTER EDITING image
    - Check if the subject and object are in positions/poses that suggest the interaction
    - The interaction should be visually evident in the edited image

    ⚠️ CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no explanations, no additional text.

    Required JSON format:
    {{
        "answer": "yes" or "no",
        "confidence": 0.0 to 1.0,
        "reasoning": "brief explanation of why you chose this answer",
    }}

    Rules:
    - The "answer" must be exactly "yes" or "no" (lowercase, in quotes)
    - The "confidence" must be a number between 0.0 and 1.0
    - If answer is "no", confidence must be 0.0
    - "reasoning" is optional but recommended for explainability
    - Do NOT include any markdown formatting (no ```json, no **bold**, etc.)
    - Do NOT include any explanatory text before or after the JSON
    - Answer ONLY the JSON object, nothing else!

    Question: "{question}"

    Example for "yes":
    {{
        "answer": "yes",
        "confidence": 0.95,
        "reasoning": "The person is clearly placing the object on the ground, with their body positioned in a crouching pose indicating the placement action."
    }}

    Example for "no":
    {{
        "answer": "no",
        "confidence": 0.0,
        "reasoning": "The person is still holding the object, and there is no visual evidence of the placement interaction occurring."
    }}
    """
    result = call_gemini_json_with_retry(
        prompt,
        image_paths=[before_image_path, after_image_path],  # 传入两张图进行对比
        temperature=0.1
    )

    if result is None:
        error_msg = "无法从 Gemini API 获取 VQA 结果（API调用失败或返回为空）"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    # 检查返回结果中是否包含错误信息
    if isinstance(result, dict) and "error" in result:
        return result

    answer = result.get("answer")
    confidence = result.get("confidence")

    if isinstance(answer, str):
        result["answer"] = answer.strip().lower()
    elif answer is None:
        error_msg = "VQA 结果缺少 answer 字段"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            error_msg = f"VQA 置信度无法解析为数字: {confidence}"
            tqdm.write(f"   - ❗ {error_msg}")
            return {"error": error_msg}
        result["confidence"] = confidence
    elif confidence is None:
        error_msg = "VQA 结果缺少 confidence 字段"
        tqdm.write(f"   - ❗ {error_msg}")
        return {"error": error_msg}

    return result

# --- 5. 主批处理流程 ---


def mark_error_in_result(final_results, image_name, error_type="api"):
    """统一标记结果中的错误状态

    Args:
        final_results: 结果字典
        image_name: 图像名称
        error_type: 错误类型，"processing" 或 "api"
    """
    if image_name not in final_results:
        final_results[image_name] = {}
    final_results[image_name]["has_error"] = True
    final_results[image_name]["error_type"] = error_type


def load_skip_object_bbox_list(dataset_type):
    """加载跳过客体框的图像列表
    
    Args:
        dataset_type: "L1L2" 或 "L3"
    
    Returns:
        set: 图像名称集合
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    list_file = os.path.join(script_dir, f"skip_object_bbox_{dataset_type}.txt")
    
    if not os.path.exists(list_file):
        return set()
    
    skip_images = set()
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line and not line.startswith('#'):
                skip_images.add(line)
    
    return skip_images


def remove_red_box_from_question(question):
    """从问题中移除"in the red bounding box"相关文本
    
    Args:
        question: 原始问题字符串
    
    Returns:
        str: 修改后的问题
    """
    if not question:
        return question
    
    # 移除各种可能的"red bounding box"表述
    patterns = [
        r'\s*in\s+the\s+red\s+bounding\s+box\s*',
        r'\s*in\s+the\s+red\s+box\s*',
        r'\s*in\s+red\s+bounding\s+box\s*',
        r'\s*in\s+red\s+box\s*',
    ]
    
    result = question
    for pattern in patterns:
        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
    
    # 清理多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def main():
    args = parse_args()
    os.makedirs(args.temp_dir, exist_ok=True)

    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    # 检测数据集类型（L1L2 / L3 / 2jiehe）；优先环境变量，避免路径里误匹配 L1L2 子串
    dataset_type = os.environ.get("EVAL_V7_SPLIT_TAG", "").strip() or None
    if not dataset_type:
        if "2jiehe" in args.input_json_path or "2jiehe" in args.image_dir_path:
            dataset_type = "2jiehe"
        elif "L1L2" in args.input_json_path or "/L1L2" in args.image_dir_path or "_L1L2_" in args.image_dir_path:
            dataset_type = "L1L2"
        elif "L3" in args.input_json_path or "/L3" in args.image_dir_path or "_L3_" in args.image_dir_path:
            dataset_type = "L3"
    
    # 加载跳过客体框的图像列表
    skip_object_bbox_images = set()
    if dataset_type:
        skip_object_bbox_images = load_skip_object_bbox_list(dataset_type)
        if skip_object_bbox_images:
            print(f"📋 加载了 {len(skip_object_bbox_images)} 个跳过客体框的图像（{dataset_type}）")

    # 加载要处理的图像列表（如果指定）
    target_images = None
    if args.image_list:
        # 从文件读取图像列表
        if os.path.exists(args.image_list):
            with open(args.image_list, 'r', encoding='utf-8') as f:
                target_images = [line.strip() for line in f if line.strip()]
            print(f"📋 从文件加载了 {len(target_images)} 个目标图像: {args.image_list}")
        else:
            print(f"⚠️  警告: 图像列表文件不存在: {args.image_list}")
    elif args.image_names:
        # 从命令行参数读取图像列表
        target_images = [name.strip()
                         for name in args.image_names.split(',') if name.strip()]
        print(f"📋 从命令行参数加载了 {len(target_images)} 个目标图像")

    # 如果指定了目标图像列表，过滤任务
    if target_images:
        # 支持部分匹配（如果图像名称包含列表中的任何字符串）
        filtered_tasks = {}
        for image_name, data in tasks.items():
            # 检查图像名称是否匹配列表中的任何项（支持部分匹配）
            image_base = os.path.splitext(image_name)[0]  # 去除扩展名
            for target in target_images:
                target_base = os.path.splitext(target)[0]  # 去除扩展名
                if target in image_name or image_name in target or target_base in image_base or image_base in target_base:
                    filtered_tasks[image_name] = data
                    break
        tasks = filtered_tasks
        print(f"✅ 过滤后剩余 {len(tasks)} 个图像需要处理")
        if len(tasks) == 0:
            print("⚠️  警告: 没有匹配的图像，退出")
            return

    final_results = {}
    # 如果使用 --force 参数，不加载已有结果，强制重新处理所有图像
    if args.force:
        print("🔄 强制重新运行模式：忽略已有结果，重新处理所有图像")
        final_results = {}
    elif os.path.exists(args.output_json_path):
        with open(args.output_json_path, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
    else:
        final_results = {}
    processed_count = 0
    skipped_count = 0
    error_count = 0
    for image_name, data in tqdm(tasks.items(), desc="Processing Images"):
        try:
            instruction = data.get("instruction", "")
            object_name = data.get("object", "object")  # 默认为 "object"
            tags_v4 = data.get("tags_v4", "")
            orig_subject_box = data.get("subject_bounding_box")
            orig_object_box = data.get("object_bounding_box")

            image_base_name = os.path.splitext(image_name)[0]
            if tags_v4 == "":
                skipped_count += 1
                tqdm.write(
                    f"⏭️  跳过图像 {image_name}: tags_v4 为空 (已跳过 {skipped_count} 张)")
                continue

            # 检查是否已经处理过（如果结果存在且有有效数据，则跳过）
            # 如果使用 --force 参数，跳过此检查
            if not args.force and image_name in final_results:
                existing_result = final_results[image_name]
                # 检查是否有有效的结果：
                # 1. processing_mode 不为空（说明已经完整处理过）
                # 2. generated_question 不为空（说明已经生成过问题）
                # 注意：max_yes_confidence 初始化为 0.0，不能用来判断是否处理过
                has_processing_mode = existing_result.get(
                    "processing_mode", "").strip()
                has_question = existing_result.get(
                    "generated_question", "").strip()

                # 检查是否有任何错误，如果有错误需要重新处理
                subject_error = existing_result.get("subject_similarity_error")
                object_error = existing_result.get("object_similarity_error")
                vqa_error = existing_result.get("vqa_error")
                processing_error = existing_result.get("processing_error")
                has_error_flag = existing_result.get("has_error", False)
                
                # 如果有任何错误，需要重新处理
                has_any_error = bool(
                    subject_error or object_error or vqa_error or 
                    processing_error or has_error_flag
                )

                # 特殊处理：如果相似度为0且没有错误信息，可能是API调用失败，需要重新计算
                subject_sim = existing_result.get("subject_similarity", None)
                object_sim = existing_result.get("object_similarity", None)

                # 如果相似度都是0且没有错误信息，说明可能是API失败，需要重新计算
                need_rerun_similarity = (
                    subject_sim == 0.0 and object_sim == 0.0 and
                    not subject_error and not object_error and
                    has_question  # 确保已经生成过问题，只是相似度计算失败
                )

                # 如果有错误，需要重新处理（不跳过）
                if has_any_error:
                    error_types = []
                    if subject_error:
                        error_types.append("subject_similarity_error")
                    if object_error:
                        error_types.append("object_similarity_error")
                    if vqa_error:
                        error_types.append("vqa_error")
                    if processing_error:
                        error_types.append("processing_error")
                    if has_error_flag:
                        error_types.append("has_error")
                    tqdm.write(
                        f"🔄 重新处理有错误的图像: {image_name} (错误类型: {', '.join(error_types)})")
                    # 不清除结果，让脚本重新处理
                    # continue 语句被移除，继续执行后续处理逻辑
                elif has_processing_mode or has_question:
                    # 没有错误且已处理过，可以跳过
                    if need_rerun_similarity:
                        tqdm.write(f"🔄 重新计算相似度: {image_name} (相似度为0且无错误信息)")
                        # 清除相似度相关字段，强制重新计算
                        existing_result.pop("subject_similarity", None)
                        existing_result.pop("object_similarity", None)
                        existing_result.pop("subject_identity_score", None)
                        existing_result.pop("object_identity_score", None)
                        existing_result.pop("subject_quality_score", None)
                        existing_result.pop("object_quality_score", None)
                        existing_result.pop("subject_similarity_error", None)
                        existing_result.pop("object_similarity_error", None)
                    else:
                        tqdm.write(
                            f"⏭️  跳过已处理的图像: {image_name} (processing_mode={has_processing_mode or 'N/A'})")
                        continue

            # 检查是否存在 L1-interaction-relation_not_occur 标签
            is_relation_not_occur = "L1-interaction-relation_not_occur" in tags_v4

            # 构造文件路径
            # 编辑后的图片路径（支持 *_edited.png、*_edited.jpg、带哈希后缀的文件名，以及 .png/.jpg
            # 格式互换）
            edited_image_candidates = []

            # 获取基础文件名（不含扩展名）
            base_name = os.path.splitext(image_name)[0]

            # 尝试多种可能的文件名组合

            # 1. _edited.png 格式

            edited_image_candidates.append(
                os.path.join(
                    args.image_dir_path,
                    f"{base_name}_edited.png"))
            # 2. _edited.jpg 格式

            edited_image_candidates.append(
                os.path.join(
                    args.image_dir_path,
                    f"{base_name}_edited.jpg"))
            # 3. 原文件名（保持原扩展名）
            edited_image_candidates.append(
                os.path.join(args.image_dir_path, image_name))
            # 4. 原文件名但扩展名改为 .png
            if not image_name.endswith(".png"):
                edited_image_candidates.append(os.path.join(
                    args.image_dir_path, f"{base_name}.png"))
            # 5. 原文件名但扩展名改为 .jpg
            if not image_name.endswith(".jpg"):
                edited_image_candidates.append(os.path.join(
                    args.image_dir_path, f"{base_name}.jpg"))

            # 6. 尝试匹配带哈希后缀的文件（如 base_name_hash.jpg）
            # 先检查目录中是否有以 base_name 开头的文件
            if os.path.isdir(args.image_dir_path):
                try:
                    import glob
                    # 匹配 base_name_*.png 和 base_name_*.jpg（排除 _edited 格式）
                    pattern_png = os.path.join(
                        args.image_dir_path, f"{base_name}_*.png")
                    pattern_jpg = os.path.join(
                        args.image_dir_path, f"{base_name}_*.jpg")
                    matching_files = glob.glob(
                        pattern_png) + glob.glob(pattern_jpg)
                    # 排除 _edited 格式（已经在前面尝试过了）
                    matching_files = [f for f in matching_files
                                      if not os.path.basename(f).endswith("_edited.png")
                                      and not os.path.basename(f).endswith("_edited.jpg")]
                    edited_image_candidates.extend(matching_files)
                except Exception:
                    pass  # 如果 glob 失败，继续使用其他候选路径

            edited_image_path = next(
                (p for p in edited_image_candidates if os.path.exists(p)), None)

            if not edited_image_path:
                skipped_count += 1
                tqdm.write(
                    f"警告: 找不到编辑后图像 {image_name}，跳过。 (已跳过 {skipped_count} 张)")
                # 只显示前3个避免日志过长
                tqdm.write(f"  尝试过的路径: {edited_image_candidates[:3]}...")
                final_results.pop(image_name, None)
                continue

            # 原图路径（编辑前，支持 .png/.jpg 格式互换）
            original_images_root = args.original_image_dir_path or \
                os.environ.get("EVAL_V7_ORIG_FALLBACK_DIRS", "").split(":")[0] or ""

            # 尝试多种可能的原图文件名组合
            original_image_candidates = [
                os.path.join(original_images_root, image_name),  # 原文件名
            ]

            # 尝试文件名格式转换（连字符 <-> 冒号）
            # JSON中可能使用连字符，但实际文件名可能使用冒号
            # 文件名格式: VIDEO_ID-时-分-秒.毫秒-时-分-秒.毫秒_其他.png
            # 需要转换为: VIDEO_ID-时:分:秒.毫秒-时:分:秒.毫秒_其他.png
            colon_name = None
            hyphen_name = None

            import re

            # 检查是否包含连字符格式的时间戳（-数字-数字-数字.）
            if re.search(r'-\d+-\d+-\d+\.', image_name):
                # 尝试将连字符替换为冒号
                # 匹配模式: -数字-数字-数字. 替换为 -数字:数字:数字.
                colon_name = re.sub(
                    r'-(\d+)-(\d+)-(\d+)\.',
                    r'-\1:\2:\3.',
                    image_name)
                original_image_candidates.append(
                    os.path.join(original_images_root, colon_name))
            # 检查是否包含冒号格式的时间戳（-数字:数字:数字.）
            elif re.search(r'-\d+:\d+:\d+\.', image_name):
                # 尝试将冒号替换为连字符
                # 匹配模式: -数字:数字:数字. 替换为 -数字-数字-数字.
                hyphen_name = re.sub(
                    r'-(\d+):(\d+):(\d+)\.', r'-\1-\2-\3.', image_name)
                original_image_candidates.append(
                    os.path.join(original_images_root, hyphen_name))

            # 如果原文件名不是 .png，尝试 .png 格式
            if not image_name.endswith(".png"):
                original_image_candidates.append(os.path.join(
                    original_images_root, f"{base_name}.png"))
                # 也尝试格式转换后的 .png
                if colon_name:
                    colon_base = os.path.splitext(colon_name)[0]
                    original_image_candidates.append(os.path.join(
                        original_images_root, f"{colon_base}.png"))
                elif hyphen_name:
                    hyphen_base = os.path.splitext(hyphen_name)[0]
                    original_image_candidates.append(os.path.join(
                        original_images_root, f"{hyphen_base}.png"))
            # 如果原文件名不是 .jpg，尝试 .jpg 格式
            if not image_name.endswith(".jpg"):
                original_image_candidates.append(os.path.join(
                    original_images_root, f"{base_name}.jpg"))
                # 也尝试格式转换后的 .jpg
                if colon_name:
                    colon_base = os.path.splitext(colon_name)[0]
                    original_image_candidates.append(os.path.join(
                        original_images_root, f"{colon_base}.jpg"))
                elif hyphen_name:
                    hyphen_base = os.path.splitext(hyphen_name)[0]
                    original_image_candidates.append(os.path.join(
                        original_images_root, f"{hyphen_base}.jpg"))

            original_image_path = next(
                (p for p in original_image_candidates if os.path.exists(p)), None)

            # 如果都找不到，使用第一个候选路径（保持向后兼容）
            if original_image_path is None:
                original_image_path = original_image_candidates[0]

            # DINO检测脚本的目录结构：
            # - 目录名：从JSON key提取的base_name（不包含_edited），如 "xxx"
            # - 文件名：从实际图片文件名提取的base_name（包含_edited），如 "detection_results_xxx_edited.txt"
            # 需要确定实际图片文件名中的base_name
            actual_image_base_name = image_base_name

            if edited_image_path:
                actual_image_filename = os.path.basename(edited_image_path)
                actual_image_base_name = os.path.splitext(
                    actual_image_filename)[0]

            # 尝试查找检测结果文件
            # DINO脚本创建的目录名是 image_base_name（从JSON key提取，不含_edited）
            # 但文件名中的base_name是 actual_image_base_name（从实际图片文件名提取，可能含_edited）
            person_txt_candidates = [

                # 最可能的路径：目录名=image_base_name，文件名=actual_image_base_name
                os.path.join(
                    args.person_dir_path,
                    image_base_name,
                    f"detection_results_{actual_image_base_name}.txt"),
                # 备用路径：目录名和文件名都是image_base_name
                os.path.join(
                    args.person_dir_path,
                    image_base_name,
                    f"detection_results_{image_base_name}.txt"),
                # 备用路径：目录名和文件名都是actual_image_base_name
                os.path.join(
                    args.person_dir_path,
                    actual_image_base_name,
                    f"detection_results_{actual_image_base_name}.txt"),
            ]

            person_txt_path = next(
                (p for p in person_txt_candidates if os.path.exists(p)), None)

            object_txt_candidates = []

            if args.object_dir_path:
                object_txt_candidates = [
                    os.path.join(
                        args.object_dir_path,
                        image_base_name,
                        f"detection_results_{actual_image_base_name}.txt"),
                    os.path.join(
                        args.object_dir_path,
                        image_base_name,
                        f"detection_results_{image_base_name}.txt"),
                    os.path.join(
                        args.object_dir_path,
                        actual_image_base_name,
                        f"detection_results_{actual_image_base_name}.txt"),
                ]
            object_txt_path = next(
                (p for p in object_txt_candidates if os.path.exists(p)),
                None) if args.object_dir_path else None

            object_track_json_path = os.path.join(
                args.object_track_dir,
                "bboxes",
                f"{image_base_name}.json") if args.object_track_dir else None

            # 加载跟踪框（稍后会在读取图像后根据尺寸差异进行缩放）

            # 注意：这里先加载原始跟踪框，实际的缩放会在读取图像后完成

            tracked_object_box = load_tracked_bbox(

                args.object_track_dir,
                image_base_name,
                args.track_frame_key
            )

            # 初始化结果

            final_results[image_name] = {
                "max_yes_confidence": 0.0,
                "best_person_box": None,
                "best_object_box": None,
                "generated_question": "",
                "subject_similarity": 0.0,
                "subject_identity_score": 0.0,
                "subject_quality_score": 0.0,
                "object_similarity": 0.0,
                "object_identity_score": 0.0,
                "object_quality_score": 0.0,
                "processing_mode": "",
                "object_bbox_source": "",
                "object_track_json": object_track_json_path if tracked_object_box is not None else ""}

            if not person_txt_path:
                skipped_count += 1
                tqdm.write(
                    f"警告: {image_name} 缺少人物检测结果文件，跳过。 (已跳过 {skipped_count} 张)")
                tqdm.write(f"  尝试过的路径:")
                for candidate in person_txt_candidates:
                    tqdm.write(f"  - {candidate}")
                final_results.pop(image_name, None)
                continue

            object_detection_available = os.path.exists(
                object_txt_path) if object_txt_path else False

            if tracked_object_box is None and not object_detection_available:
                skipped_count += 1
                tqdm.write(
                    f"警告: {image_name} 缺少物体跟踪/检测结果，跳过。 (已跳过 {skipped_count} 张)")
                final_results.pop(image_name, None)
                continue

            # 读取编辑后的图像（用于检测框绘制和VQA）
            edited_image = cv2.imread(edited_image_path)

            if edited_image is None:
                skipped_count += 1
                tqdm.write(
                    f"警告: 无法读取编辑后图像 {edited_image_path}，跳过。 (已跳过 {skipped_count} 张)")
                final_results.pop(image_name, None)
                continue

            # 读取原图（编辑前，用于相似度对比和最终可视化）
            before_edit_image = cv2.imread(original_image_path)

            if before_edit_image is None:
                tqdm.write(f"警告: 无法读取原图 {original_image_path}")
                before_edit_image = None

            # ⚠️ 关键检查：验证DINO检测框坐标是否与当前编辑后图像尺寸匹配
            # DINO检测输出的坐标基于检测时的图像尺寸，需要验证是否与当前使用的图像尺寸一致
            edited_image_h, edited_image_w = edited_image.shape[:2]

            # ⚠️ 关键修复：如果跟踪框存在，需要先根据原始图像和编辑后图像的尺寸差异进行缩放
            # 必须在赋值给 object_boxes 之前完成缩放！
            if tracked_object_box is not None and before_edit_image is not None:
                orig_h, orig_w = before_edit_image.shape[:2]
                edit_h, edit_w = edited_image.shape[:2]

                # 初始化缩放比例（默认1.0，表示不需要缩放）
                scale_x = 1.0
                scale_y = 1.0

                # 如果尺寸不同，需要缩放跟踪框
                if orig_h != edit_h or orig_w != edit_w:
                    scale_x = edit_w / orig_w
                    scale_y = edit_h / orig_h

                    # 保存原始坐标用于日志
                    original_box = tracked_object_box.copy()

                    # 缩放边界框坐标 [x1, y1, x2, y2]
                    scaled_x1 = tracked_object_box[0] * scale_x
                    scaled_y1 = tracked_object_box[1] * scale_y
                    scaled_x2 = tracked_object_box[2] * scale_x
                    scaled_y2 = tracked_object_box[3] * scale_y

                    # 裁剪到图像边界内，并确保有效的边界框
                    scaled_x1 = max(0, min(int(scaled_x1), edit_w - 1))
                    scaled_y1 = max(0, min(int(scaled_y1), edit_h - 1))
                    scaled_x2 = max(0, min(int(scaled_x2), edit_w - 1))
                    scaled_y2 = max(0, min(int(scaled_y2), edit_h - 1))

                    # 确保 x1 < x2, y1 < y2
                    if scaled_x1 >= scaled_x2:
                        scaled_x2 = min(scaled_x1 + 1, edit_w - 1)
                    if scaled_y1 >= scaled_y2:
                        scaled_y2 = min(scaled_y1 + 1, edit_h - 1)

                    tracked_object_box = [
                        scaled_x1, scaled_y1, scaled_x2, scaled_y2]
                    tqdm.write(
                        f"    ⚠️  检测到图像尺寸差异，已缩放跟踪框: 原始({orig_w}x{orig_h}) -> 编辑后({edit_w}x{edit_h}), 缩放比例({scale_x:.3f}, {scale_y:.3f})")
                    tqdm.write(
                        f"        缩放前: {original_box} -> 缩放后: {tracked_object_box}")

            # 解析人框（带置信度）和物体框

            person_boxes_with_scores = parse_detection_file(
                person_txt_path, return_scores=True)

            # 验证并调整人框坐标（如果需要）

            validated_person_boxes = []

            for box, score in person_boxes_with_scores:
                x1, y1, x2, y2 = box
                # 检查坐标是否在图像边界内
                if x1 < 0 or y1 < 0 or x2 > edited_image_w or y2 > edited_image_h:
                    tqdm.write(
                        f"    ⚠️  警告: 人框坐标超出图像边界: {box}, 图像尺寸: {edited_image_w}x{edited_image_h}")
                    # 裁剪到边界内
                    x1 = max(0, min(x1, edited_image_w - 1))
                    y1 = max(0, min(y1, edited_image_h - 1))
                    x2 = max(0, min(x2, edited_image_w - 1))
                    y2 = max(0, min(y2, edited_image_h - 1))
                    if x1 >= x2:
                        x2 = min(x1 + 1, edited_image_w - 1)
                    if y1 >= y2:
                        y2 = min(y1 + 1, edited_image_h - 1)
                    box = [x1, y1, x2, y2]
                validated_person_boxes.append((box, score))
            person_boxes_with_scores = validated_person_boxes

            object_boxes = []
            
            if tracked_object_box is not None:
                # 此时 tracked_object_box 已经完成缩放（如果尺寸不同）
                object_boxes = [tracked_object_box]
                final_results[image_name]["object_bbox_source"] = f"track:{args.track_frame_key}"
                final_results[image_name]["object_track_json"] = object_track_json_path
                tqdm.write(
                    f"    客体: 使用跟踪框 {args.track_frame_key} -> {tracked_object_box}")
            elif object_detection_available:
                object_boxes_raw = parse_detection_file(object_txt_path)
                # 验证并调整物体框坐标（如果需要）
                for box in object_boxes_raw:
                    x1, y1, x2, y2 = box
                    # 检查坐标是否在图像边界内
                    if x1 < 0 or y1 < 0 or x2 > edited_image_w or y2 > edited_image_h:
                        tqdm.write(
                            f"    ⚠️  警告: 物体框坐标超出图像边界: {box}, 图像尺寸: {edited_image_w}x{edited_image_h}")
                        # 裁剪到边界内
                        x1 = max(0, min(x1, edited_image_w - 1))
                        y1 = max(0, min(y1, edited_image_h - 1))
                        x2 = max(0, min(x2, edited_image_w - 1))
                        y2 = max(0, min(y2, edited_image_h - 1))
                        if x1 >= x2:
                            x2 = min(x1 + 1, edited_image_w - 1)
                        if y1 >= y2:
                            y2 = min(y1 + 1, edited_image_h - 1)
                        box = [x1, y1, x2, y2]
                    object_boxes.append(box)
                if "1ouT6zrcpD0-0:08:01.914-0:08:36.949_frame_0002_53" not in image_name:
                    object_boxes = object_boxes[:5]
                final_results[image_name]["object_bbox_source"] = "detection"
                final_results[image_name]["object_track_json"] = object_track_json_path if object_track_json_path and os.path.exists(
                    object_track_json_path) else ""

            # 从人框中选择置信度最高的那个
            if person_boxes_with_scores:
                # 按置信度排序，取最高的
                person_boxes_with_scores.sort(key=lambda x: x[1], reverse=True)
                best_person_box_initial = person_boxes_with_scores[0][0]  # 取坐标
                best_person_score = person_boxes_with_scores[0][1]  # 取置信度
                person_boxes = [best_person_box_initial]  # 只保留最佳的人框
                tqdm.write(
                    f"    选择最佳人框（置信度: {best_person_score:.3f}），共有 {len(person_boxes_with_scores)} 个候选")
            else:
                person_boxes = []

            # 注意：edited_image 和 before_edit_image 已经在前面读取过了

            img_height, img_width = edited_image_h, edited_image_w

            subject_name = "child" if "child" in instruction.lower() else "person"

            # 检查是否需要跳过客体框
            skip_object_bbox = image_name in skip_object_bbox_images
            
            # 生成问题
            if "generated_question" in data and data["generated_question"] and data["generated_question"] != "":
                question = data["generated_question"]
            else:
                question = generate_question_from_instruction(
                    instruction, subject_name, object_name)
            if not question:
                skipped_count += 1
                tqdm.write(f"警告: {image_name} 未能生成问题，跳过。 (已跳过 {skipped_count} 张)")
                continue
            
            # 如果需要跳过客体框，从问题中移除"in the red bounding box"
            if skip_object_bbox:
                original_question = question
                question = remove_red_box_from_question(question)
                if question != original_question:
                    tqdm.write(f"    📝 已从问题中移除'red bounding box': {original_question} -> {question}")
            
            final_results[image_name]["generated_question"] = question

            # ==================== 根据标签选择不同的处理模式 ====================
            if is_relation_not_occur:
                # 模式1: 先评估相似度，再判断关系（用于 L1-interaction-relation_not_occur）
                final_results[image_name]["processing_mode"] = "similarity_first"
                tqdm.write(
                    f"  -> {image_name}: 使用【相似度优先】模式（L1-interaction-relation_not_occur）")

            # 创建临时目录
            temp_subdir = os.path.join(args.temp_dir, image_base_name)
            os.makedirs(temp_subdir, exist_ok=True)

            best_person_box = None
            best_object_box = None
            subject_similarity = 0.0
            object_similarity = 0.0
            max_confidence = 0.0  # 初始化max_confidence

            # --- 步骤1: 评估主体相似度 ---
            if orig_subject_box:
                if len(person_boxes) > 0:
                    tqdm.write(f"    主体: 找到 {len(person_boxes)} 个候选框，评估相似度...")

                    # 绘制原始主体框（在原图上）
                    before_subj = draw_box_and_label(
                        (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                        orig_subject_box,
                        "ORIGINAL",
                        RED)
                    before_subj_titled = add_title_to_image(
                        before_subj, "BEFORE EDITING")
                    before_subj_path = os.path.join(
                        temp_subdir, 'before_subject.jpg')
                    cv2.imwrite(before_subj_path, before_subj_titled)

                    # 绘制所有候选框（在编辑后的图上）
                    after_subj_all = draw_numbered_boxes(
                        edited_image.copy(), person_boxes, "subject")
                    after_subj_titled = add_title_to_image(
                        after_subj_all, "AFTER EDITING")
                    after_subj_path = os.path.join(
                        temp_subdir, 'after_subject_all.jpg')
                    cv2.imwrite(after_subj_path, after_subj_titled)

                    # 使用Gemini评估相似度
                    subj_result = evaluate_similarity_with_gemini(
                        before_subj_path,
                        after_subj_path,
                        "subject",
                        person_boxes,
                        no_detection_mode=False,
                        edit_instruction=instruction,
                        object_name=None)

                    if subj_result and "error" not in subj_result:
                        match_id = subj_result.get('best_match_id_parsed')
                        subject_similarity = subj_result.get(
                            'similarity_score', 0.0)

                        if match_id is not None and match_id != 'manual' and match_id < len(
                                person_boxes):
                            best_person_box = [person_boxes[match_id]]

                        final_results[image_name]["subject_similarity"] = subject_similarity
                        final_results[image_name]["subject_identity_score"] = subj_result.get(
                            'identity_score', 0.0)
                        final_results[image_name]["subject_quality_score"] = subj_result.get(
                            'quality_score', 0.0)

                        tqdm.write(
                            f"      ✓ 最佳匹配: {match_id}, 相似度: {subject_similarity:.3f}")
                    elif subj_result and "error" in subj_result:
                        # API调用失败，记录错误信息
                        error_msg = subj_result["error"]
                        final_results[image_name]["subject_similarity_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ 主体相似度计算失败: {error_msg}")
                else:
                    # 没有检测框，让Gemini自己定位
                    tqdm.write(f"    ⚠️  主体: 无检测框，使用Gemini手动定位...")

                    before_subj = draw_box_and_label(
                        (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                        orig_subject_box,
                        "ORIGINAL",
                        RED)
                    before_subj_titled = add_title_to_image(
                        before_subj, "BEFORE EDITING")
                    before_subj_path = os.path.join(
                        temp_subdir, 'before_subject.jpg')
                    # cv2.imwrite(before_subj_path, before_subj_titled)

                    after_subj_titled = add_title_to_image(
                        edited_image.copy(), "AFTER EDITING")
                    after_subj_path = os.path.join(
                        temp_subdir, 'after_subject_no_detection.jpg')
                    # cv2.imwrite(after_subj_path, after_subj_titled)

                    subj_result = evaluate_similarity_with_gemini(
                        before_subj_path,
                        after_subj_path,
                        "subject",
                        [],
                        no_detection_mode=True,
                        image_dimensions=(
                            img_height,
                            img_width),
                        edit_instruction=instruction,
                        object_name=None)

                    if subj_result and "error" not in subj_result and 'box_coords' in subj_result:
                        best_person_box = [subj_result['box_coords']]
                        subject_similarity = subj_result.get(
                            'similarity_score', 0.0)
                        final_results[image_name]["subject_similarity"] = subject_similarity
                        final_results[image_name]["subject_identity_score"] = subj_result.get(
                            'identity_score', 0.0)
                        final_results[image_name]["subject_quality_score"] = subj_result.get(
                            'quality_score', 0.0)
                        tqdm.write(
                            f"      ✓ 手动定位完成, 相似度: {subject_similarity:.3f}")
                    elif subj_result and "error" in subj_result:
                        # API调用失败，记录错误信息
                        error_msg = subj_result["error"]
                        final_results[image_name]["subject_similarity_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ 主体相似度计算失败: {error_msg}")

            # --- 步骤2: 评估客体相似度 ---
            if skip_object_bbox:
                # 跳过客体框绘制和相似度评估，直接设为1.0
                tqdm.write(f"    ⏭️  跳过客体相似度评估（图像在跳过列表中），直接设为1.0")
                object_similarity = 1.0
                final_results[image_name]["object_similarity"] = object_similarity
                final_results[image_name]["object_identity_score"] = 1.0
                final_results[image_name]["object_quality_score"] = 1.0
                # 使用原始物体框作为best_object_box（如果存在）
                if orig_object_box:
                    best_object_box = [orig_object_box]
                elif object_boxes:
                    best_object_box = [object_boxes[0]]
                else:
                    best_object_box = None
            elif orig_object_box:
                if len(object_boxes) > 0:
                    tqdm.write(f"    客体: 找到 {len(object_boxes)} 个候选框，评估相似度...")

                    before_obj = draw_box_and_label(
                        (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                        orig_object_box,
                        "ORIGINAL",
                        BLUE)
                    before_obj_titled = add_title_to_image(
                        before_obj, "BEFORE EDITING")
                    before_obj_path = os.path.join(
                        temp_subdir, 'before_object.jpg')
                    cv2.imwrite(before_obj_path, before_obj_titled)

                    after_obj_all = draw_numbered_boxes(
                        edited_image.copy(), object_boxes, "object")
                    after_obj_titled = add_title_to_image(
                        after_obj_all, "AFTER EDITING")
                    after_obj_path = os.path.join(
                        temp_subdir, 'after_object_all.jpg')
                    cv2.imwrite(after_obj_path, after_obj_titled)

                    obj_result = evaluate_similarity_with_gemini(
                        before_obj_path,
                        after_obj_path,
                        "object",
                        object_boxes,
                        no_detection_mode=False,
                        edit_instruction=instruction,
                        object_name=object_name)

                    if obj_result and "error" not in obj_result:
                        match_id = obj_result.get('best_match_id_parsed')
                        object_similarity = obj_result.get(
                            'similarity_score', 0.0)

                        if match_id is not None and match_id != 'manual' and match_id < len(
                                object_boxes):
                            best_object_box = [object_boxes[match_id]]

                        final_results[image_name]["object_similarity"] = object_similarity
                        final_results[image_name]["object_identity_score"] = obj_result.get(
                            'identity_score', 0.0)
                        final_results[image_name]["object_quality_score"] = obj_result.get(
                            'quality_score', 0.0)

                        tqdm.write(
                            f"      ✓ 最佳匹配: {match_id}, 相似度: {object_similarity:.3f}")
                    elif obj_result and "error" in obj_result:
                        # API调用失败，记录错误信息
                        error_msg = obj_result["error"]
                        final_results[image_name]["object_similarity_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ 客体相似度计算失败: {error_msg}")
                else:
                    # 没有检测框，让Gemini自己定位
                    tqdm.write(f"    ⚠️  客体: 无检测框，使用Gemini手动定位...")

                    before_obj = draw_box_and_label(
                        (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                        orig_object_box,
                        "ORIGINAL",
                        BLUE)
                    before_obj_titled = add_title_to_image(
                        before_obj, "BEFORE EDITING")
                    before_obj_path = os.path.join(
                        temp_subdir, 'before_object.jpg')
                    cv2.imwrite(before_obj_path, before_obj_titled)

                    after_obj_titled = add_title_to_image(
                        edited_image.copy(), "AFTER EDITING")
                    after_obj_path = os.path.join(
                        temp_subdir, 'after_object_no_detection.jpg')
                    cv2.imwrite(after_obj_path, after_obj_titled)

                    obj_result = evaluate_similarity_with_gemini(
                        before_obj_path,
                        after_obj_path,
                        "object",
                        [],
                        no_detection_mode=True,
                        image_dimensions=(
                            img_height,
                            img_width),
                        edit_instruction=instruction,
                        object_name=object_name)

                    if obj_result and "error" not in obj_result and 'box_coords' in obj_result:
                        best_object_box = [obj_result['box_coords']]
                        object_similarity = obj_result.get(
                            'similarity_score', 0.0)
                        final_results[image_name]["object_similarity"] = object_similarity
                        final_results[image_name]["object_identity_score"] = obj_result.get(
                            'identity_score', 0.0)
                        final_results[image_name]["object_quality_score"] = obj_result.get(
                            'quality_score', 0.0)
                        tqdm.write(
                            f"      ✓ 手动定位完成, 相似度: {object_similarity:.3f}")
                    elif obj_result and "error" in obj_result:
                        # API调用失败，记录错误信息
                        error_msg = obj_result["error"]
                        final_results[image_name]["object_similarity_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ 客体相似度计算失败: {error_msg}")

            # --- 步骤3: 判断最相似的主客体之间的关系 ---
            if best_person_box and (best_object_box or skip_object_bbox):
                tqdm.write(f"    判断最相似主客体之间的关系...")

                # 绘制编辑前的图（带原始主客体框）
                before_image_with_boxes = before_edit_image.copy(
                ) if before_edit_image is not None else edited_image.copy()
                if orig_subject_box:
                    draw_box_and_label(
                        before_image_with_boxes, orig_subject_box, False, BLUE)
                # 如果不在跳过列表中，才绘制客体框
                if orig_object_box and not skip_object_bbox:
                    draw_box_and_label(
                        before_image_with_boxes, orig_object_box, False, RED)

                # 添加标题说明这是编辑前的图
                before_image_titled = add_title_to_image(
                    before_image_with_boxes, "BEFORE EDITING")
                before_temp_path = os.path.join(
                    temp_subdir, f"{image_base_name}_before_pair.png")
                cv2.imwrite(before_temp_path, before_image_titled)

                # 绘制编辑后的图（带检测到的主客体框）
                after_image_with_boxes = edited_image.copy()
                if best_person_box and isinstance(best_person_box, list) and len(best_person_box) > 0:
                    draw_box_and_label(
                        after_image_with_boxes,
                        best_person_box[0],
                        False,
                        BLUE)
                # 如果不在跳过列表中，才绘制客体框
                if best_object_box and isinstance(best_object_box, list) and len(best_object_box) > 0 and not skip_object_bbox:
                    draw_box_and_label(
                        after_image_with_boxes,
                        best_object_box[0],
                        False,
                        RED)

                # 添加标题说明这是编辑后的图
                after_image_titled = add_title_to_image(
                    after_image_with_boxes, "AFTER EDITING")
                after_temp_path = os.path.join(
                    temp_subdir, f"{image_base_name}_after_pair.png")
                cv2.imwrite(after_temp_path, after_image_titled)

                # 使用VQA判断关系（传入编辑前后两张图）
                requires_comparison = image_name in IMAGES_REQUIRE_COMPARISON
                vqa_result = None

                if requires_comparison:
                    comparison_result = perform_vqa_on_pair_with_comparison(
                        before_temp_path,
                        after_temp_path,
                        question
                    )
                    if comparison_result and "error" not in comparison_result:
                        final_results[image_name]["comparison_vqa"] = comparison_result
                        final_results[image_name]["comparison_verdict"] = comparison_result.get(
                            "verdict")
                        final_results[image_name]["comparison_verdict_confidence"] = comparison_result.get(
                            "verdict_confidence")
                        final_results[image_name]["comparison_notes"] = comparison_result.get(
                            "notes")
                        final_results[image_name]["comparison_before_answer"] = comparison_result.get(
                            "before_answer")
                        final_results[image_name]["comparison_before_confidence"] = comparison_result.get(
                            "before_confidence")

                        answer = comparison_result.get("after_answer")
                        confidence = comparison_result.get("after_confidence")

                        if isinstance(answer, str):
                            answer = answer.strip().lower()
                        else:
                            tqdm.write("      ✗ 对比 VQA 缺少 after_answer 字段")
                            answer = None

                        if isinstance(confidence, (int, float)):
                            confidence = float(confidence)
                        else:
                            tqdm.write("      ✗ 对比 VQA 缺少 after_confidence 字段")
                            confidence = None

                        if answer is not None and confidence is not None:
                            vqa_result = {
                                "answer": answer, "confidence": confidence}
                    elif comparison_result and "error" in comparison_result:
                        # API调用失败，记录错误信息
                        error_msg = comparison_result["error"]
                        final_results[image_name]["vqa_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ VQA对比失败: {error_msg}")
                        vqa_result = None
                else:
                    vqa_result = perform_vqa_on_pair(
                        before_temp_path, after_temp_path, question)

                # 如果成功，保存思考过程（reasoning）到结果中
                if vqa_result and isinstance(
                        vqa_result, dict) and "error" not in vqa_result:
                    if "reasoning" in vqa_result:
                        final_results[image_name]["vqa_reasoning"] = vqa_result.get(
                            "reasoning")

                # 确保 vqa_result 是 dict 类型
                if vqa_result and isinstance(
                        vqa_result,
                        dict) and "error" not in vqa_result and vqa_result.get(
                        "answer",
                        "").lower() == "yes":
                    max_confidence = vqa_result.get("confidence", 0.0)
                    final_results[image_name]["max_yes_confidence"] = max_confidence
                    tqdm.write(
                        f"      ✓ 关系判断: Yes (置信度: {max_confidence:.3f})")
                elif vqa_result and isinstance(vqa_result, dict) and "error" in vqa_result:
                    # API调用失败，记录错误信息
                    error_msg = vqa_result["error"]
                    final_results[image_name]["vqa_error"] = error_msg
                    final_results[image_name]["max_yes_confidence"] = 0.0
                    mark_error_in_result(
                        final_results, image_name, error_type="api")
                    tqdm.write(f"      ❗ VQA失败: {error_msg}")
                else:
                    final_results[image_name]["max_yes_confidence"] = 0.0
                    tqdm.write(f"      ✗ 关系判断: No")

                # 清理临时文件
                if os.path.exists(before_temp_path):
                    os.remove(before_temp_path)
                if os.path.exists(after_temp_path):
                    os.remove(after_temp_path)
            else:
                tqdm.write(f"    ⚠️  未找到有效的主客体配对")
                best_person_box = person_boxes if person_boxes else None
                best_object_box = object_boxes if object_boxes else None

            # 清理临时目录
            import shutil
            if os.path.exists(temp_subdir):
                shutil.rmtree(temp_subdir)

            final_results[image_name]["best_person_box"] = best_person_box
            final_results[image_name]["best_object_box"] = best_object_box
            
            # 模式2: 原有逻辑 - 遍历所有配对找最佳的
            final_results[image_name]["processing_mode"] = "confidence_first"
            tqdm.write(f"  -> {image_name}: 使用【置信度优先】模式")
            
            # 处理无检测框的情况
            if not person_boxes or not object_boxes:
                tqdm.write(f"    ⚠️  缺少检测框，尝试手动定位...")

                temp_subdir = os.path.join(args.temp_dir, image_base_name)
                os.makedirs(temp_subdir, exist_ok=True)

                best_person_box = None
                best_object_box = None

                # 如果需要跳过客体框，直接设置相似度为1.0
                if skip_object_bbox:
                    tqdm.write(f"    ⏭️  跳过客体相似度评估（图像在跳过列表中），直接设为1.0")
                    object_similarity = 1.0
                    final_results[image_name]["object_similarity"] = object_similarity
                    final_results[image_name]["object_identity_score"] = 1.0
                    final_results[image_name]["object_quality_score"] = 1.0
                    if orig_object_box:
                        best_object_box = [orig_object_box]

                # 如果没有人物框，让Gemini定位
                if not person_boxes and orig_subject_box:
                    before_subj = draw_box_and_label(
                        (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                        orig_subject_box,
                        "ORIGINAL",
                        RED)
                    before_subj_titled = add_title_to_image(
                        before_subj, "BEFORE EDITING")
                    before_subj_path = os.path.join(
                        temp_subdir, 'before_subject.jpg')
                    cv2.imwrite(before_subj_path, before_subj_titled)

                    after_subj_titled = add_title_to_image(
                        edited_image.copy(), "AFTER EDITING")
                    after_subj_path = os.path.join(
                        temp_subdir, 'after_subject_no_detection.jpg')
                    cv2.imwrite(after_subj_path, after_subj_titled)

                    subj_result = evaluate_similarity_with_gemini(
                        before_subj_path,
                        after_subj_path,
                        "subject",
                        [],
                        no_detection_mode=True,
                        image_dimensions=(
                            img_height,
                            img_width),
                        edit_instruction=instruction,
                        object_name=None)

                    if subj_result and "error" not in subj_result and 'box_coords' in subj_result and subj_result[
                            'box_coords'] is not None:
                        best_person_box = [subj_result['box_coords']]
                        final_results[image_name]["subject_similarity"] = subj_result.get(
                            'similarity_score', 0.0)
                        final_results[image_name]["subject_identity_score"] = subj_result.get(
                            'identity_score', 0.0)
                        final_results[image_name]["subject_quality_score"] = subj_result.get(
                            'quality_score', 0.0)
                        person_boxes = [
                            subj_result['box_coords']]  # 添加到列表中以便后续处理
                    elif subj_result and "error" in subj_result:
                        # API调用失败，记录错误信息
                        error_msg = subj_result["error"]
                        final_results[image_name]["subject_similarity_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ 主体相似度计算失败: {error_msg}")

                # 如果没有物体框，让Gemini定位（但跳过列表中的图像除外）
                if not object_boxes and orig_object_box and not skip_object_bbox:
                    before_obj = draw_box_and_label(
                        (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                        orig_object_box,
                        "ORIGINAL",
                        BLUE)
                    before_obj_titled = add_title_to_image(
                        before_obj, "BEFORE EDITING")
                    before_obj_path = os.path.join(
                        temp_subdir, 'before_object.jpg')
                    cv2.imwrite(before_obj_path, before_obj_titled)

                    after_obj_titled = add_title_to_image(
                        edited_image.copy(), "AFTER EDITING")
                    after_obj_path = os.path.join(
                        temp_subdir, 'after_object_no_detection.jpg')
                    cv2.imwrite(after_obj_path, after_obj_titled)

                    obj_result = evaluate_similarity_with_gemini(
                        before_obj_path,
                        after_obj_path,
                        "object",
                        [],
                        no_detection_mode=True,
                        image_dimensions=(
                            img_height,
                            img_width),
                        edit_instruction=instruction,
                        object_name=object_name)

                    if obj_result and "error" not in obj_result and 'box_coords' in obj_result and obj_result[
                            'box_coords'] is not None:
                        best_object_box = [obj_result['box_coords']]
                        final_results[image_name]["object_similarity"] = obj_result.get(
                            'similarity_score', 0.0)
                        final_results[image_name]["object_identity_score"] = obj_result.get(
                            'identity_score', 0.0)
                        final_results[image_name]["object_quality_score"] = obj_result.get(
                            'quality_score', 0.0)
                        object_boxes = [
                            obj_result['box_coords']]  # 添加到列表中以便后续处理
                    elif obj_result and "error" in obj_result:
                        # API调用失败，记录错误信息
                        error_msg = obj_result["error"]
                        final_results[image_name]["object_similarity_error"] = error_msg
                        mark_error_in_result(
                            final_results, image_name, error_type="api")
                        tqdm.write(f"      ❗ 客体相似度计算失败: {error_msg}")

                import shutil
                if os.path.exists(temp_subdir):
                    shutil.rmtree(temp_subdir)

                # 如果仍然没有检测框，跳过（但如果skip_object_bbox为True，允许继续）
                if not person_boxes or (not object_boxes and not skip_object_bbox):
                    tqdm.write(f"    ⚠️  仍然缺少检测框，跳过")
                    # 如果skip_object_bbox为True，需要确保object_boxes至少有一个元素用于后续配对
                    if skip_object_bbox and orig_object_box and not object_boxes:
                        object_boxes = [orig_object_box]
                        tqdm.write(f"    📝 跳过列表图像：使用原始客体框用于VQA配对")
            # continue

            # 使用最佳人框与所有物体框配对
            max_confidence = 0.0

            best_person_box = None

            best_object_box = None

            # 现在person_boxes只有1个元素（最高置信度的人框）

            total_pairs = len(person_boxes) * len(object_boxes)

            tqdm.write(
                f"    开始处理 {total_pairs} 个边界框配对（1个最佳人框 × {len(object_boxes)}个物体框）...")

            temp_image_paths = []

            best_index = 0

            requires_comparison = image_name in IMAGES_REQUIRE_COMPARISON

            comparison_result_best = None

            best_vqa_result = None  # 保存最佳结果的完整信息（包括思考过程）

            real_best_index = 0
            
            for p_idx, p_box in enumerate(person_boxes):
                for o_idx, o_box in enumerate(object_boxes):
                    # 绘制编辑前的图（带原始主客体框）
                    before_image_copy = before_edit_image.copy(
                    ) if before_edit_image is not None else edited_image.copy()
                    if orig_subject_box:
                        draw_box_and_label(
                            before_image_copy, orig_subject_box, False, BLUE)
                    # 如果不在跳过列表中，才绘制客体框
                    if orig_object_box and not skip_object_bbox:
                        draw_box_and_label(
                            before_image_copy, orig_object_box, False, RED)
                    
                    # 添加标题说明这是编辑前的图
                    before_image_titled = add_title_to_image(
                        before_image_copy, "BEFORE EDITING")
                    before_temp_path = os.path.join(
                        args.temp_dir, f"{image_base_name}_before_p{p_idx}_o{o_idx}.png")
                    cv2.imwrite(before_temp_path, before_image_titled)
                    temp_image_paths.append(before_temp_path)

                    # 绘制编辑后的图（带检测到的主客体框）
                    after_image_copy = edited_image.copy()

                    if p_box is not None:
                        draw_box_and_label(after_image_copy, p_box, False, BLUE)
                    # 如果不在跳过列表中，才绘制客体框
                    if o_box is not None and not skip_object_bbox:
                        draw_box_and_label(after_image_copy, o_box, False, RED)

                    # 添加标题说明这是编辑后的图
                    after_image_titled = add_title_to_image(
                        after_image_copy, "AFTER EDITING")
                    after_temp_path = os.path.join(
                        args.temp_dir, f"{image_base_name}_after_p{p_idx}_o{o_idx}.png")
                    cv2.imwrite(after_temp_path, after_image_titled)
                    temp_image_paths.append(after_temp_path)
                    # 使用VQA判断关系（传入编辑前后两张图）
                    vqa_result = None
                    comparison_result = None
                    if requires_comparison:
                        comparison_result = perform_vqa_on_pair_with_comparison(
                            before_temp_path,
                            after_temp_path,
                            question
                        )
                    if comparison_result and "error" not in comparison_result:
                        answer = comparison_result.get("after_answer")
                        confidence = comparison_result.get("after_confidence")

                        if isinstance(answer, str):
                            answer = answer.strip().lower()
                        else:
                            answer = None

                        if isinstance(confidence, (int, float)):
                            confidence = float(confidence)
                        else:
                            confidence = None

                        if answer is not None and confidence is not None:
                            vqa_result = {
                                "answer": answer, "confidence": confidence}
                            if (comparison_result_best is None or confidence >
                                    comparison_result_best.get("after_confidence", -1.0)):
                                comparison_result_best = comparison_result
                    elif comparison_result and "error" in comparison_result:
                        # API调用失败，记录错误信息（但继续处理其他配对）
                        error_msg = comparison_result["error"]
                        if "vqa_error" not in final_results.get(
                                image_name, {}):
                            final_results[image_name]["vqa_error"] = error_msg
                        tqdm.write(
                            f"      ❗ VQA对比失败 (配对 {p_idx}-{o_idx}): {error_msg}")
                        vqa_result = None
                    else:
                        vqa_result = perform_vqa_on_pair(
                            before_temp_path, after_temp_path, question)

                    # 确保 vqa_result 是 dict 类型
                    if vqa_result and isinstance(
                        vqa_result,
                        dict) and "error" not in vqa_result and vqa_result.get(
                        "answer",
                        "").lower() == "yes":
                            confidence = vqa_result.get("confidence", 0.0)
                            if confidence > max_confidence:
                                max_confidence = confidence
                                best_person_box = [p_box]
                                best_object_box = [o_box]
                                real_best_index = best_index
                                # 保存最佳结果的完整信息（包括思考过程）
                                best_vqa_result = vqa_result.copy()
                    elif vqa_result and isinstance(vqa_result, dict) and "error" in vqa_result:
                        # API调用失败，记录错误信息（但继续处理其他配对）
                        error_msg = vqa_result["error"]
                        if "vqa_error" not in final_results.get(image_name, {}):
                            final_results[image_name]["vqa_error"] = error_msg
                        tqdm.write(
                            f"      ❗ VQA失败 (配对 {p_idx}-{o_idx}): {error_msg}")
                        best_index += 1
            
            if best_person_box is None or best_object_box is None:
                best_person_box = person_boxes
                best_object_box = object_boxes
            
            # 保存最佳结果的思考过程（如果存在）
            if best_vqa_result and "reasoning" in best_vqa_result:
                final_results[image_name]["vqa_reasoning"] = best_vqa_result.get(
                    "reasoning")

            # 清理临时图像文件
            if comparison_result_best and "error" not in comparison_result_best:
                final_results[image_name]["comparison_vqa"] = comparison_result_best
                final_results[image_name]["comparison_verdict"] = comparison_result_best.get(
                    "verdict")
                final_results[image_name]["comparison_verdict_confidence"] = comparison_result_best.get(
                    "verdict_confidence")
                final_results[image_name]["comparison_notes"] = comparison_result_best.get(
                    "notes")
                final_results[image_name]["comparison_before_answer"] = comparison_result_best.get(
                    "before_answer")
                final_results[image_name]["comparison_before_confidence"] = comparison_result_best.get(
                    "before_confidence")

            print(temp_image_paths)
            
            for path in temp_image_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            # --- 计算最佳配对与原主客体的相似度 ---
            if best_person_box and orig_subject_box:
                tqdm.write(f"    计算最佳主体框与原主体的相似度...")
                
                # 创建临时目录用于相似度评估
                temp_subdir = os.path.join(
                    args.temp_dir, f"{image_base_name}_similarity")
                os.makedirs(temp_subdir, exist_ok=True)
                
                # 绘制原始主体框（在原图上）
                before_subj = draw_box_and_label(
                    (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                    orig_subject_box,
                    "ORIGINAL",
                    RED)
                before_subj_titled = add_title_to_image(
                    before_subj, "BEFORE EDITING")
                before_subj_path = os.path.join(temp_subdir, 'before_subject.jpg')
                cv2.imwrite(before_subj_path, before_subj_titled)
                
                # 绘制最佳主体框（在编辑后的图上，只用第一个，即最佳的那个）
                if best_person_box and isinstance(best_person_box, list) and len(best_person_box) > 0:
                    person_box_to_use = best_person_box[0]
                else:
                    tqdm.write(f"      ⚠️  警告: best_person_box 无效，跳过主体相似度计算")
                    person_box_to_use = None
                
                if person_box_to_use is not None:
                    after_subj = draw_box_and_label(
                        edited_image.copy(), person_box_to_use, "BEST", BLUE)
                    after_subj_titled = add_title_to_image(after_subj, "AFTER EDITING")
                    after_subj_path = os.path.join(
                        temp_subdir, 'after_subject_best.jpg')
                    cv2.imwrite(after_subj_path, after_subj_titled)
                    
                    # 评估相似度（只传入最佳的那个框）
                    subj_result = evaluate_similarity_with_gemini(
                        before_subj_path, after_subj_path, "subject", [
                            person_box_to_use], no_detection_mode=False, edit_instruction=instruction, object_name=None)
                else:
                    subj_result = None
                
                if subj_result and "error" not in subj_result:
                    final_results[image_name]["subject_similarity"] = subj_result.get(
                        'similarity_score', 0.0)
                    final_results[image_name]["subject_identity_score"] = subj_result.get(
                        'identity_score', 0.0)
                    final_results[image_name]["subject_quality_score"] = subj_result.get(
                        'quality_score', 0.0)
                    tqdm.write(
                        f"      ✓ 主体相似度: {subj_result.get('similarity_score', 0.0):.3f}")
                elif subj_result and "error" in subj_result:
                    # API调用失败，记录错误信息
                    error_msg = subj_result["error"]
                    final_results[image_name]["subject_similarity_error"] = error_msg
                    mark_error_in_result(final_results, image_name, error_type="api")
                    tqdm.write(f"      ❗ 主体相似度计算失败: {error_msg}")
                
                # 清理临时文件
                import shutil
                if os.path.exists(temp_subdir):
                    shutil.rmtree(temp_subdir)

            # --- 计算最佳客体框与原客体的相似度 ---
            if skip_object_bbox:
                # 跳过客体相似度评估，直接设为1.0
                tqdm.write(f"    ⏭️  跳过客体相似度评估（图像在跳过列表中），直接设为1.0")
                if not final_results[image_name].get("object_similarity"):
                    final_results[image_name]["object_similarity"] = 1.0
                    final_results[image_name]["object_identity_score"] = 1.0
                    final_results[image_name]["object_quality_score"] = 1.0
            elif best_object_box and orig_object_box:
                tqdm.write(f"    计算最佳客体框与原客体的相似度...")
                
                # 创建临时目录用于相似度评估
                temp_subdir = os.path.join(
                    args.temp_dir, f"{image_base_name}_similarity")
                os.makedirs(temp_subdir, exist_ok=True)
                
                # 绘制原始客体框（在原图上）
                before_obj = draw_box_and_label(
                    (before_edit_image.copy() if before_edit_image is not None else edited_image.copy()),
                    orig_object_box,
                    "ORIGINAL",
                    BLUE)
                before_obj_titled = add_title_to_image(
                    before_obj, "BEFORE EDITING")
                before_obj_path = os.path.join(temp_subdir, 'before_object.jpg')
                cv2.imwrite(before_obj_path, before_obj_titled)
                
                # 绘制最佳客体框（在编辑后的图上，只用第一个，即最佳的那个）
                after_obj = draw_box_and_label(
                    edited_image.copy(), best_object_box[0], "BEST", RED)
                after_obj_titled = add_title_to_image(after_obj, "AFTER EDITING")
                after_obj_path = os.path.join(temp_subdir, 'after_object_best.jpg')
                cv2.imwrite(after_obj_path, after_obj_titled)
                
                # 评估相似度（只传入最佳的那个框）
                obj_result = evaluate_similarity_with_gemini(
                    before_obj_path,
                    after_obj_path,
                    "object",
                    [
                        best_object_box[0]],
                    no_detection_mode=False,
                    edit_instruction=instruction,
                    object_name=object_name)
                
                if obj_result and "error" not in obj_result:
                    final_results[image_name]["object_similarity"] = obj_result.get(
                        'similarity_score', 0.0)
                    final_results[image_name]["object_identity_score"] = obj_result.get(
                        'identity_score', 0.0)
                    final_results[image_name]["object_quality_score"] = obj_result.get(
                        'quality_score', 0.0)
                    tqdm.write(
                        f"      ✓ 客体相似度: {obj_result.get('similarity_score', 0.0):.3f}")
                elif obj_result and "error" in obj_result:
                    # API调用失败，记录错误信息
                    error_msg = obj_result["error"]
                    final_results[image_name]["object_similarity_error"] = error_msg
                    mark_error_in_result(final_results, image_name, error_type="api")
                    tqdm.write(f"      ❗ 客体相似度计算失败: {error_msg}")
                
                # 清理临时文件
                import shutil
                if os.path.exists(temp_subdir):
                    shutil.rmtree(temp_subdir)

            final_results[image_name]["max_yes_confidence"] = max_confidence

            final_results[image_name]["best_person_box"] = best_person_box

            final_results[image_name]["best_object_box"] = best_object_box

            # --- 保存最终选择的主客体框的可视化结果 ---
            # 如果跳过客体框，只要有best_person_box就可以生成可视化
            if best_person_box and (best_object_box or skip_object_bbox):
                try:
                    # 创建可视化目录
                    visualization_dir = os.path.join(
                        args.temp_dir, "final_visualizations")
                    os.makedirs(visualization_dir, exist_ok=True)

                    # 使用之前读取的原图
                    if before_edit_image is not None:
                        before_edit_vis = before_edit_image.copy()

                        # 在编辑前图片上绘制原始主客体框
                        if orig_subject_box:
                            draw_box_and_label(
                                before_edit_vis,
                                orig_subject_box,
                                "Subject",
                                BLUE,
                                line_type='solid',
                                thickness=3)

                        # 如果不在跳过列表中，才绘制客体框
                        if orig_object_box and not skip_object_bbox:
                            draw_box_and_label(
                                before_edit_vis,
                                orig_object_box,
                                "Object",
                                RED,
                                line_type='solid',
                                thickness=3)

                        # 添加标题
                        before_edit_titled = add_title_to_image(
                            before_edit_vis, "BEFORE EDITING")
                    else:
                        tqdm.write(f"    ⚠️  警告：原图不可用")
                        before_edit_titled = None

                    # 在编辑后图片上绘制检测到的框
                    after_edit_vis = edited_image.copy()

                    # 统一处理框格式：确保是 [x1, y1, x2, y2] 格式
                    def normalize_box(box):
                        """统一框格式为 [x1, y1, x2, y2]"""
                        if box is None:
                            return None
                        # 如果是嵌套列表 [[x1,y1,x2,y2]]，取第一个
                        if isinstance(box, list) and len(box) > 0:
                            if isinstance(box[0], list):
                                return box[0]
                            # 如果已经是 [x1,y1,x2,y2] 格式
                            if len(box) == 4:
                                return box
                        return None

                    person_box_normalized = normalize_box(best_person_box)
                    # 如果跳过客体框，object_box_normalized设为None
                    object_box_normalized = None if skip_object_bbox else normalize_box(best_object_box)

                    if person_box_normalized:
                        # 验证坐标是否在图像范围内
                        h, w = edited_image.shape[:2]
                        x1, y1, x2, y2 = person_box_normalized
                        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                            draw_box_and_label(
                                after_edit_vis,
                                person_box_normalized,
                                "Subject",
                                BLUE,
                                line_type='solid',
                                thickness=3)
                        else:
                            tqdm.write(
                                f"    ⚠️  警告: 人框坐标超出图像范围: {person_box_normalized}, 图像尺寸: {w}x{h}")

                    # 如果不在跳过列表中，才绘制客体框
                    if object_box_normalized and not skip_object_bbox:
                        # 验证坐标是否在图像范围内
                        h, w = edited_image.shape[:2]
                        x1, y1, x2, y2 = object_box_normalized
                        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                            draw_box_and_label(
                                after_edit_vis,
                                object_box_normalized,
                                "Object",
                                RED,
                                line_type='solid',
                                thickness=3)
                        else:
                            tqdm.write(
                                f"    ⚠️  警告: 物体框坐标超出图像范围: {object_box_normalized}, 图像尺寸: {w}x{h}")

                    # 添加标题（包含处理模式和置信度）

                    processing_mode = final_results[image_name]["processing_mode"]
                    title = f"AFTER EDITING | {processing_mode.upper()} | Conf: {max_confidence:.3f}"
                    after_edit_titled = add_title_to_image(after_edit_vis, title)

                    # 拼接两张图片
                    if before_edit_titled is not None:
                        final_vis_image = concatenate_images_horizontally(
                            before_edit_titled, after_edit_titled, gap=30)
                    else:
                        final_vis_image = after_edit_titled

                    # 保存可视化结果
                    vis_save_path = os.path.join(
                        visualization_dir, f"{image_base_name}_final.jpg")
                    cv2.imwrite(vis_save_path, final_vis_image)

                    tqdm.write(f"    💾 最终框可视化已保存: {vis_save_path}")

                except Exception as e:
                    tqdm.write(f"    ⚠️  保存可视化失败: {e}")

            # 标记成功处理（如果没有错误标记）
            if image_name in final_results and "has_error" not in final_results[image_name]:
                final_results[image_name]["has_error"] = False
                final_results[image_name]["error_type"] = "none"

            # 保存结果（每次循环后保存，以便实时保存进度）
            with open(args.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4, ensure_ascii=False)

            processed_count += 1

        except Exception as e:
            error_count += 1
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            tqdm.write(f"\n❌ 处理失败: {image_name}")
            tqdm.write(f"   错误信息: {error_msg}")
            tqdm.write(
                f"   错误位置: {error_traceback.split(chr(10))[-2] if len(error_traceback.split(chr(10))) > 1 else 'unknown'}")

            # 记录错误信息到结果中
            if image_name not in final_results:
                final_results[image_name] = {}
            final_results[image_name]["processing_error"] = error_msg
            final_results[image_name]["processing_error_traceback"] = error_traceback
            mark_error_in_result(
                final_results,
                image_name,
                error_type="processing")

            # 保存错误结果
            try:
                with open(args.output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)
            except Exception as save_error:
                tqdm.write(f"    ⚠️  保存错误结果失败: {save_error}")

            tqdm.write(f"   ⏭️  跳过当前样本，继续处理下一个...\n")
            continue

    print(f"\n处理完成！结果已保存到 '{args.output_json_path}'。")
    print(
        f"处理统计: 成功处理 {processed_count} 张，跳过 {skipped_count} 张，错误 {error_count} 张，总计 {len(tasks)} 张")


# --- 6. 脚本执行入口 ---
if __name__ == '__main__':
    main()
