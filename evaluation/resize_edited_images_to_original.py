#!/usr/bin/env python3
"""
将编辑后的图像缩放到原始图像的尺寸
用于解决尺寸不匹配导致的框错位问题
"""

import os
import sys
import json
import time
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from v7_path_utils import resolve_original_image_path


def _is_same_existing_file(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return False
    try:
        return os.path.samefile(src, dst)
    except (OSError, FileNotFoundError):
        return False


def _write_image_with_retry(path: str, img, max_retries: int = 3) -> bool:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    for attempt in range(max_retries):
        if cv2.imwrite(path, img):
            return True
        if attempt < max_retries - 1:
            time.sleep(0.5 * (attempt + 1))
    return False


def resize_image_to_match_original(edited_image_path, original_image_path, output_path):
    """
    将编辑后的图像缩放到原始图像的尺寸
    
    Args:
        edited_image_path: 编辑后图像的路径
        original_image_path: 原始图像的路径
        output_path: 输出图像的路径
    
    Returns:
        bool: 是否成功缩放
    """
    # 读取原始图像获取尺寸
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"⚠️  无法读取原始图像: {original_image_path}")
        return False
    
    orig_h, orig_w = original_img.shape[:2]
    
    # 读取编辑后图像
    edited_img = cv2.imread(edited_image_path)
    if edited_img is None:
        print(f"⚠️  无法读取编辑后图像: {edited_image_path}")
        return False
    
    edit_h, edit_w = edited_img.shape[:2]
    
    # 如果尺寸相同，用内存中的图像写入（避免 samefile/shutil 在 NFS 上的问题）
    if orig_w == edit_w and orig_h == edit_h:
        if _is_same_existing_file(edited_image_path, output_path):
            return True
        return _write_image_with_retry(output_path, edited_img)

    resized_img = cv2.resize(edited_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return _write_image_with_retry(output_path, resized_img)


def main():
    parser = argparse.ArgumentParser(description="将编辑后的图像缩放到原始图像的尺寸")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="输入JSON文件路径（包含图像列表）"
    )
    parser.add_argument(
        "--original_image_dir",
        type=str,
        required=True,
        help="原始图像目录"
    )
    parser.add_argument(
        "--edited_image_dir",
        type=str,
        required=True,
        help="编辑后图像目录"
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        default=None,
        help="输出图像目录（缩放后的编辑图像）。如果未指定，则直接覆盖原编辑图像"
    )
    parser.add_argument(
        "--backup_original",
        action="store_true",
        help="是否备份原始编辑图像（重命名为_edited_original.png）"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="是否在原图像路径下直接resize（覆盖原图像）。如果指定，则忽略--output_image_dir"
    )
    
    args = parser.parse_args()
    
    # 读取JSON文件
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确定输出目录
    if args.inplace:
        # 在原图像路径下直接resize
        output_image_dir = args.edited_image_dir
        print("⚠️  模式: 在原图像路径下直接resize（将覆盖原图像）")
    else:
        # 使用指定的输出目录
        if args.output_image_dir is None:
            print("❌ 错误: 必须指定 --output_image_dir 或使用 --inplace")
            return
        output_image_dir = args.output_image_dir
        print("📁 模式: 保存到新目录（不覆盖原图像）")
    
    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    
    # 统计信息
    total_count = 0
    resized_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"开始处理 {len(data)} 个样本...")
    print(f"原始图像目录: {args.original_image_dir}")
    print(f"编辑图像目录: {args.edited_image_dir}")
    print(f"输出图像目录: {output_image_dir}")
    print()
    
    # 处理每个样本
    for image_name in tqdm(data.keys(), desc="缩放图像"):
        total_count += 1
        
        # 构建路径（扁平原图目录无 1/2/3 子目录时自动回退到 basename）
        original_image_path = resolve_original_image_path(args.original_image_dir, image_name)
        
        # 尝试多种可能的编辑图像文件名
        base_name = os.path.splitext(image_name)[0]
        edited_image_candidates = [
            os.path.join(args.edited_image_dir, f"{base_name}_edited.png"),
            os.path.join(args.edited_image_dir, f"{base_name}_edited.jpg"),
            os.path.join(args.edited_image_dir, image_name),
        ]
        
        edited_image_path = None
        edited_image_basename = None
        for candidate in edited_image_candidates:
            if os.path.exists(candidate):
                edited_image_path = candidate
                edited_image_basename = os.path.basename(candidate)
                break
        
        if edited_image_path is None:
            tqdm.write(f"⚠️  未找到编辑图像: {image_name}")
            error_count += 1
            continue
        
        if not os.path.exists(original_image_path):
            tqdm.write(f"⚠️  未找到原始图像: {original_image_path}")
            error_count += 1
            continue
        
        # 输出路径
        # 如果inplace模式，输出路径就是编辑图像路径本身
        if args.inplace:
            output_image_path = edited_image_path
        else:
            # 保持编辑图像的相对路径（如 data_v7 下 1/xxx_edited.png），避免展平后 DINO 找不到 1/ 前缀
            if edited_image_basename:
                try:
                    edited_dir_abs = os.path.abspath(args.edited_image_dir)
                    edited_path_abs = os.path.abspath(edited_image_path)
                    rel_out = os.path.relpath(edited_path_abs, edited_dir_abs)
                    if rel_out.startswith(".."):
                        output_image_path = os.path.join(output_image_dir, edited_image_basename)
                    else:
                        output_image_path = os.path.join(output_image_dir, rel_out)
                except ValueError:
                    output_image_path = os.path.join(output_image_dir, edited_image_basename)
            else:
                output_image_path = os.path.join(output_image_dir, image_name)
        
        # 如果需要备份原始编辑图像
        if args.backup_original:
            backup_path = edited_image_path.replace('.png', '_original.png').replace('.jpg', '_original.jpg')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(edited_image_path, backup_path)
        
        # 检查是否需要缩放
        original_img = cv2.imread(original_image_path)
        edited_img = cv2.imread(edited_image_path)
        
        if original_img is None or edited_img is None:
            tqdm.write(f"⚠️  无法读取图像: {image_name}")
            error_count += 1
            continue
        
        orig_h, orig_w = original_img.shape[:2]
        edit_h, edit_w = edited_img.shape[:2]
        
        if orig_w == edit_w and orig_h == edit_h:
            if _is_same_existing_file(edited_image_path, output_image_path):
                skipped_count += 1
            elif _write_image_with_retry(output_image_path, edited_img):
                skipped_count += 1
            else:
                tqdm.write(f"⚠️  写入失败: {output_image_path}")
                error_count += 1
        else:
            # 需要缩放
            if resize_image_to_match_original(edited_image_path, original_image_path, output_image_path):
                resized_count += 1
            else:
                error_count += 1
    
    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总样本数: {total_count}")
    print(f"已缩放: {resized_count}")
    print(f"已跳过（尺寸相同）: {skipped_count}")
    print(f"错误: {error_count}")
    print(f"输出目录: {output_image_dir}")


if __name__ == "__main__":
    main()


