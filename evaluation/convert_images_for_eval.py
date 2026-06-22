#!/usr/bin/env python3
"""
图片格式转换脚本
- 将.jpg文件转换为.png
- 根据JSON中的key匹配文件名，确保文件名为 *_edited.png 格式
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def load_json(json_path):
    """加载JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_base_name(filename):
    """获取文件名的基础部分（去除扩展名和_edited后缀）"""
    # 移除扩展名
    name = Path(filename).stem
    # 移除_edited后缀（如果存在）
    if name.endswith('_edited'):
        name = name[:-7]  # 移除 '_edited'
    return name


def find_matching_files(image_dir, json_keys):
    """查找匹配的文件"""
    image_dir = Path(image_dir)
    
    # 创建JSON key到基础名的映射
    json_base_names = {}
    for key in json_keys:
        base_name = get_base_name(key)
        json_base_names[base_name] = key
    
    # 查找所有图片文件（含 1/2/3 子目录）
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(image_dir.rglob(ext))
    
    # 匹配文件
    matched_files = []
    for img_file in image_files:
        base_name = get_base_name(img_file.name)
        if base_name in json_base_names:
            matched_files.append({
                'file_path': img_file,
                'base_name': base_name,
                'json_key': json_base_names[base_name],
                'current_name': img_file.name
            })
    
    return matched_files


def convert_image_to_png(input_path, output_path):
    """将图片转换为PNG格式"""
    try:
        img = Image.open(input_path)
        # 如果是RGBA模式，保持；否则转换为RGB
        if img.mode in ('RGBA', 'LA'):
            img.save(output_path, 'PNG')
        else:
            img.convert('RGB').save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"❌ 转换失败 {input_path}: {e}")
        return False


def normalize_image_files(image_dir, json_path, dry_run=False):
    """
    规范化图片文件：
    1. 将.jpg转换为.png
    2. 确保文件名为 *_edited.png 格式
    """
    print(f"📁 图像目录: {image_dir}")
    print(f"📄 JSON文件: {json_path}")
    print(f"🔍 模式: {'预览模式（不实际转换）' if dry_run else '实际转换模式'}")
    print("")
    
    # 加载JSON
    json_data = load_json(json_path)
    json_keys = list(json_data.keys())
    print(f"📊 JSON中共有 {len(json_keys)} 条记录")
    
    # 查找匹配的文件
    matched_files = find_matching_files(image_dir, json_keys)
    print(f"✅ 找到 {len(matched_files)} 个匹配的文件")
    print("")
    
    if len(matched_files) == 0:
        print("⚠️  没有找到匹配的文件，请检查文件名格式")
        return
    
    # 统计需要转换的文件
    to_convert = []
    to_rename = []
    
    for item in matched_files:
        file_path = item['file_path']
        base_name = item['base_name']
        json_key = item['json_key']
        
        # 目标文件名：base_name + _edited.png
        target_name = f"{base_name}_edited.png"
        target_path = file_path.parent / target_name
        
        # 检查是否需要转换格式
        needs_format_convert = file_path.suffix.lower() in ['.jpg', '.jpeg']
        
        # 检查是否需要重命名
        needs_rename = file_path.name != target_name
        
        if needs_format_convert or needs_rename:
            if needs_format_convert:
                to_convert.append({
                    'source': file_path,
                    'target': target_path,
                    'reason': '格式转换（JPG -> PNG）'
                })
            elif needs_rename:
                to_rename.append({
                    'source': file_path,
                    'target': target_path,
                    'reason': '重命名'
                })
    
    # 显示需要转换的文件
    if to_convert:
        print(f"🔄 需要转换格式的文件 ({len(to_convert)} 个):")
        for item in to_convert[:10]:  # 只显示前10个
            print(f"   {item['source'].name} -> {item['target'].name} ({item['reason']})")
        if len(to_convert) > 10:
            print(f"   ... 还有 {len(to_convert) - 10} 个文件")
        print("")
    
    if to_rename:
        print(f"📝 需要重命名的文件 ({len(to_rename)} 个):")
        for item in to_rename[:10]:  # 只显示前10个
            print(f"   {item['source'].name} -> {item['target'].name} ({item['reason']})")
        if len(to_rename) > 10:
            print(f"   ... 还有 {len(to_rename) - 10} 个文件")
        print("")
    
    if not to_convert and not to_rename:
        print("✅ 所有文件格式正确，无需转换")
        return
    
    if dry_run:
        print("ℹ️  预览模式：未实际执行转换")
        return
    
    # 执行转换
    print("🚀 开始转换...")
    print("")
    
    converted_count = 0
    renamed_count = 0
    failed_count = 0
    
    # 转换格式
    for item in tqdm(to_convert, desc="转换格式"):
        source = item['source']
        target = item['target']
        
        # 如果目标文件已存在，跳过
        if target.exists():
            print(f"⚠️  目标文件已存在，跳过: {target.name}")
            continue
        
        if convert_image_to_png(source, target):
            # 删除原文件
            source.unlink()
            converted_count += 1
        else:
            failed_count += 1
    
    # 重命名
    for item in tqdm(to_rename, desc="重命名"):
        source = item['source']
        target = item['target']
        
        # 如果目标文件已存在，跳过
        if target.exists():
            print(f"⚠️  目标文件已存在，跳过: {target.name}")
            continue
        
        try:
            source.rename(target)
            renamed_count += 1
        except Exception as e:
            print(f"❌ 重命名失败 {source.name}: {e}")
            failed_count += 1
    
    print("")
    print("==========================================")
    print("✅ 转换完成！")
    print(f"   - 格式转换: {converted_count} 个")
    print(f"   - 重命名: {renamed_count} 个")
    if failed_count > 0:
        print(f"   - 失败: {failed_count} 个")
    print("==========================================")


def main():
    parser = argparse.ArgumentParser(
        description="规范化图片文件格式：将JPG转换为PNG，确保文件名为 *_edited.png 格式"
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='图片目录路径'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='JSON标注文件路径'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式：只显示需要转换的文件，不实际执行'
    )
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.isdir(args.image_dir):
        print(f"❌ 图像目录不存在: {args.image_dir}")
        exit(1)
    
    if not os.path.isfile(args.json_path):
        print(f"❌ JSON文件不存在: {args.json_path}")
        exit(1)
    
    normalize_image_files(args.image_dir, args.json_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()

