#!/usr/bin/env python3
"""
SAM2 Object Tracking Script for Evaluation
基于run_sam2_on_pair.py，支持命令行参数，用于评测流程中的物体追踪
"""

# --- 1. 关键设置：使用非交互式后端，确保在无GUI环境下也能保存图片 ---
import matplotlib
matplotlib.use('Agg')
import json
import argparse

# --- 2. 导入必要的库 ---
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2
from tqdm import tqdm   
import shutil
from typing import Optional

# V7 flat original dir: JSON keys may be "3/foo.jpg" while files live at "foo.jpg"
_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from v7_path_utils import resolve_original_image_path
from tool_unmentioned_utils import (
    has_valid_tool_bboxes,
    is_tool_unmentioned_sample,
)

ANN_OBJ_ID = 1


def _parse_tool_bbox_list(tool_bboxes) -> Optional[list]:
    if not has_valid_tool_bboxes(tool_bboxes):
        return None
    if isinstance(tool_bboxes, list) and len(tool_bboxes) >= 4:
        try:
            return [float(x) for x in tool_bboxes[:4]]
        except (TypeError, ValueError):
            return None
    return None


def _save_tracked_bbox_json(
    output_path: str,
    image_name: str,
    original_bbox: list,
    predicted_bboxes: dict,
    bbox_type: str = "object",
) -> None:
    bbox_payload = {
        "image_name": image_name,
        "bbox_type": bbox_type,
        "original_bbox": [float(x) for x in original_bbox],
        "tracked_bboxes": {
            f"frame_{frame_idx:05d}": bbox if bbox is None else [float(val) for val in bbox]
            for frame_idx, bbox in sorted(predicted_bboxes.items())
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as bbox_file:
        json.dump(bbox_payload, bbox_file, ensure_ascii=False, indent=2)

# --- 辅助函数 (内部使用，无需修改) ---
def _show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def _show_box(box, ax, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2, linestyle='-', label=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    patch = plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=facecolor, lw=lw, linestyle=linestyle)
    if label is not None:
        patch.set_label(label)
    ax.add_patch(patch)
    return patch

def _mask_to_bbox(mask):
    if mask.dtype != np.bool_:
        mask = mask > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

def _extract_bboxes_from_segments(video_segments, target_hw, obj_id=None):
    target_h, target_w = target_hw
    extracted = {}
    for frame_idx, masks in video_segments.items():
        mask = None
        if obj_id is not None and obj_id in masks:
            mask = masks[obj_id]
        elif masks:
            first_key = next(iter(masks.keys()))
            mask = masks[first_key]

        if mask is None:
            extracted[frame_idx] = None
            continue

        mask_np = np.array(mask)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]

        if mask_np.shape != (target_h, target_w):
            mask_np = cv2.resize(mask_np.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mask_np = mask_np.astype(bool)
        else:
            mask_np = mask_np.astype(bool)

        extracted[frame_idx] = _mask_to_bbox(mask_np)

    return extracted

# --- 3. 功能函数 1: 创建临时视频 ---
def create_temp_video_from_images(image_path_1, image_path_2, output_video_path, fps=2):
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)

    if img1 is None or img2 is None:
        raise ValueError("无法读取图片，请检查文件路径。")

    if img1.shape != img2.shape:
        print(f"警告: 两张图片尺寸不同，将自动 resize 第二张图片以匹配第一张。")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    height, width, _ = img1.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    video.write(img1)
    video.write(img2)
    video.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 临时视频已成功创建: {output_video_path}")

# --- 4. 功能函数 2: 使用 SAM2 处理视频（修改后：接收已加载的predictor） ---
def process_video_with_sam2(video_path, predictor, box_coords):
    # 直接使用传入的predictor，无需重新构建
    inference_state = predictor.init_state(video_path=video_path)

    ann_frame_idx = 0
    ann_obj_id = ANN_OBJ_ID

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box_coords,
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    total_frames = 2
    for frame_idx in range(total_frames):
        if frame_idx not in video_segments:
            first_mask_shape = next(iter(video_segments.values())).get(ann_obj_id, np.zeros((1, 1, 1))).shape
            video_segments[frame_idx] = {ann_obj_id: np.zeros(first_mask_shape, dtype=bool)}

    print(f"[INFO] SAM2 处理完成，共处理 {len(video_segments)} 帧。")
    return video_segments

# --- 5. 功能函数 3: 可视化并保存结果 ---
def visualize_and_save_results(image_path_1, image_path_2, video_segments, box_coords, output_dir, predicted_bboxes=None):
    os.makedirs(output_dir, exist_ok=True)
    
    img1_pil = Image.open(image_path_1).convert("RGB")
    img2_pil = Image.open(image_path_2).convert("RGB")
    
    if img2_pil.size != img1_pil.size:
        img2_pil = img2_pil.resize(img1_pil.size, Image.Resampling.LANCZOS)
    
    try:
        first_frame_masks = next(iter(video_segments.values()))
        obj_id = next(iter(first_frame_masks.keys()))
    except StopIteration:
        raise ValueError("[ERROR] 分割结果字典 video_segments 为空，无法可视化！")

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    plt.title("Frame 0 with Mask")
    plt.imshow(img1_pil)
    handles_frame0 = []
    handles_frame0.append(_show_box(box_coords, plt.gca(), edgecolor='green', label='Original BBox'))
    _show_mask(video_segments[0][obj_id], plt.gca(), obj_id=obj_id)
    if predicted_bboxes and 0 in predicted_bboxes and predicted_bboxes[0] is not None:
        handles_frame0.append(_show_box(predicted_bboxes[0], plt.gca(), edgecolor='red', linestyle='--', label='Tracked BBox'))
    if handles_frame0:
        unique_handles = {h.get_label(): h for h in handles_frame0 if h.get_label() is not None}
        if unique_handles:
            ax1.legend(unique_handles.values(), unique_handles.keys(), loc='lower right')
    plt.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    plt.title("Frame 1 with Mask")
    plt.imshow(img2_pil)
    _show_mask(video_segments[1][obj_id], plt.gca(), obj_id=obj_id)
    handles_frame1 = []
    if predicted_bboxes and 1 in predicted_bboxes and predicted_bboxes[1] is not None:
        handles_frame1.append(_show_box(predicted_bboxes[1], plt.gca(), edgecolor='red', linestyle='--', label='Tracked BBox'))
    if handles_frame1:
        unique_handles = {h.get_label(): h for h in handles_frame1 if h.get_label() is not None}
        if unique_handles:
            ax2.legend(unique_handles.values(), unique_handles.keys(), loc='lower right')
    plt.axis('off')

    result_image_path = os.path.join(output_dir, "segmentation_result.png")
    plt.savefig(result_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 分割结果对比图已保存: {result_image_path}")

    mask_output_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_output_dir, exist_ok=True)
    for frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            mask_image = mask.squeeze(0)
            if mask_image.shape != (img1_pil.height, img1_pil.width):
                mask_image = cv2.resize(mask_image.astype(np.uint8), 
                                       (img1_pil.width, img1_pil.height),
                                       interpolation=cv2.INTER_NEAREST)
            mask_image = (mask_image * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_image)
            mask_pil.save(os.path.join(mask_output_dir, f"frame_{frame_idx:05d}.png"))
    print(f"[INFO] 单独的mask文件已保存到: {mask_output_dir}")

# --- 6. 主程序：串联所有功能 ---
def main():
    parser = argparse.ArgumentParser(description="SAM2 Object Tracking for Evaluation")
    
    # 必需参数
    parser.add_argument("--input_json", type=str, required=True,
                        help="输入JSON文件路径（包含图像名称和object_bounding_box）")
    parser.add_argument("--original_image_dir", type=str, required=True,
                        help="原始图像目录路径")
    parser.add_argument("--edited_image_dir", type=str, required=True,
                        help="编辑后图像目录路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录路径（将创建bboxes子目录）")
    
    # 可选参数
    parser.add_argument("--sam_checkpoint", type=str, 
                        default="../sam2.1_hiera_large.pt",
                        help="SAM2模型checkpoint路径（相对于sam2目录）")
    parser.add_argument("--model_config", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM2模型配置文件路径（相对于sam2目录）")
    parser.add_argument("--temp_video_path", type=str,
                        default="/tmp/temp_two_frames.mp4",
                        help="临时视频文件路径")
    parser.add_argument("--temp_output_dir", type=str,
                        default="/tmp/sam2_tracking_temp",
                        help="临时输出目录")
    parser.add_argument(
        "--track-tool-bboxes",
        action="store_true",
        help="额外对 tool_bboxes 做 SAM2 追踪（默认关闭，不影响旧流程）",
    )
    parser.add_argument(
        "--tool-unmentioned-all",
        action="store_true",
        help="与 --track-tool-bboxes 联用：追踪所有含 tool_bboxes 的样本（默认仅 tool_unmentioned）",
    )

    args = parser.parse_args()
    
    # 加载JSON文件
    print(f"[INFO] 加载JSON文件: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        annos = json.load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    bbox_output_dir = os.path.join(args.output_dir, "bboxes")
    tool_bbox_output_dir = os.path.join(args.output_dir, "tool_bboxes")
    os.makedirs(bbox_output_dir, exist_ok=True)
    if args.track_tool_bboxes:
        os.makedirs(tool_bbox_output_dir, exist_ok=True)
    os.makedirs(args.temp_output_dir, exist_ok=True)
    tool_processed_count = 0
    tool_skipped_count = 0
    
    # 加载SAM2模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 正在加载 SAM2 模型（设备：{device}）...")
    print(f"      Checkpoint: {args.sam_checkpoint}")
    print(f"      Config: {args.model_config}")
    
    # 切换到sam2目录以正确加载配置文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    
    try:
        predictor = build_sam2_video_predictor(args.model_config, args.sam_checkpoint, device=device)
        print(f"[INFO] SAM2 模型加载完成！开始处理数据...")
    finally:
        os.chdir(original_cwd)
    
    # 处理每个图像
    processed_count = 0
    skipped_count = 0
    
    for image_name in tqdm(annos.keys(), desc="处理图像"):
        if "object_bounding_box" not in annos[image_name]:
            print(f"[WARN] 跳过 {image_name}: 缺少 object_bounding_box")
            skipped_count += 1
            continue
        
        object_bboxes = annos[image_name]["object_bounding_box"]
        if not object_bboxes or len(object_bboxes) != 4:
            print(f"[WARN] 跳过 {image_name}: object_bounding_box 格式不正确")
            skipped_count += 1
            continue
        
        box_coords = np.array(object_bboxes, dtype=np.float32).reshape(-1)
        
        # 构建图像路径
        origin_image_path = resolve_original_image_path(args.original_image_dir, image_name)
        
        # 编辑后图像：统一为 {stem}_edited.png（兼容 JSON key 为 .jpg / .png）
        stem_no_ext, _ext = os.path.splitext(image_name)
        edited_image_name = f"{stem_no_ext}_edited.png"
        edited_image_path = os.path.join(args.edited_image_dir, edited_image_name)
        
        # 检查文件是否存在
        if not os.path.exists(origin_image_path):
            print(f"[WARN] 跳过 {image_name}: 原始图像不存在 - {origin_image_path}")
            skipped_count += 1
            continue
        
        if not os.path.exists(edited_image_path):
            print(f"[WARN] 跳过 {image_name}: 编辑图像不存在 - {edited_image_path}")
            skipped_count += 1
            continue
        
        try:
            # 创建临时视频
            create_temp_video_from_images(origin_image_path, edited_image_path, args.temp_video_path)
            
            # 处理视频
            video_segments = process_video_with_sam2(
                video_path=args.temp_video_path,
                predictor=predictor,
                box_coords=box_coords
            )
            
            # 提取边界框
            with Image.open(origin_image_path) as _img_size_ref:
                target_hw = (_img_size_ref.height, _img_size_ref.width)
            predicted_bboxes = _extract_bboxes_from_segments(video_segments, target_hw, obj_id=ANN_OBJ_ID)
            
            # 可视化保存
            visualize_and_save_results(
                origin_image_path,
                edited_image_path,
                video_segments,
                box_coords,
                args.temp_output_dir,
                predicted_bboxes=predicted_bboxes
            )
            
            # 保存 object 边界框结果
            bbox_stem, _bbox_ext = os.path.splitext(image_name)
            bbox_file_path = os.path.join(bbox_output_dir, f"{bbox_stem}.json")
            _save_tracked_bbox_json(
                bbox_file_path,
                image_name,
                [float(x) for x in box_coords.tolist()],
                predicted_bboxes,
                bbox_type="object",
            )

            # 可选：tool_bboxes 追踪（tool_unmentioned）
            if args.track_tool_bboxes:
                sample = annos[image_name]
                tool_coords = _parse_tool_bbox_list(sample.get("tool_bboxes"))
                should_track_tool = tool_coords is not None
                if should_track_tool and not args.tool_unmentioned_all:
                    if not is_tool_unmentioned_sample(sample):
                        should_track_tool = False
                if should_track_tool:
                    try:
                        tool_box = np.array(tool_coords, dtype=np.float32).reshape(-1)
                        tool_segments = process_video_with_sam2(
                            video_path=args.temp_video_path,
                            predictor=predictor,
                            box_coords=tool_box,
                        )
                        tool_predicted = _extract_bboxes_from_segments(
                            tool_segments, target_hw, obj_id=ANN_OBJ_ID
                        )
                        tool_bbox_path = os.path.join(tool_bbox_output_dir, f"{bbox_stem}.json")
                        _save_tracked_bbox_json(
                            tool_bbox_path,
                            image_name,
                            tool_coords,
                            tool_predicted,
                            bbox_type="tool",
                        )
                        tool_processed_count += 1
                    except Exception as tool_err:
                        print(f"[WARN] tool_bboxes 追踪失败 {image_name}: {tool_err}")
                        tool_skipped_count += 1
                else:
                    tool_skipped_count += 1

            # 移动结果图
            result_image_src = os.path.join(args.temp_output_dir, "segmentation_result.png")
            result_image_dst = os.path.join(args.output_dir, image_name)
            if os.path.exists(result_image_src):
                shutil.copy2(result_image_src, result_image_dst)
            
            processed_count += 1
            
        except Exception as e:
            print(f"[ERROR] 处理 {image_name} 时出错: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"\n[SUCCESS] 所有任务已完成！")
    print(f"  成功处理: {processed_count} 张图像")
    print(f"  跳过: {skipped_count} 张图像")
    if args.track_tool_bboxes:
        print(f"  tool_bboxes 追踪成功: {tool_processed_count}")
        print(f"  tool_bboxes 跳过: {tool_skipped_count}")
        print(f"  tool 输出: {tool_bbox_output_dir}")
    print(f"  输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()

