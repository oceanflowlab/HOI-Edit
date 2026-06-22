import os
import sys
import tqdm
import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加 GroundingDINO 路径到 sys.path
GROUNDING_DINO_PATH = os.environ.get("GROUNDING_DINO_ROOT", "")
GROUNDING_DINO_CHECKPOINT = os.environ.get(
    "GROUNDING_DINO_CHECKPOINT",
    os.path.join(GROUNDING_DINO_PATH, "weights/groundingdino_swint_ogc.pth"),
)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, GROUNDING_DINO_PATH)

# transformers 5.x 需补丁后再 import groundingdino
from gdino_transformers_compat import apply_gdino_transformers_compat

apply_gdino_transformers_compat()

# 导入 GroundingDINO 相关模块
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行 GroundingDINO 推理脚本（优化版：只加载一次模型）")
    
    parser.add_argument(
        "--json-path", 
        required=True, 
        type=str, 
        help="包含图片名和对应 prompt 的 JSON 文件绝对路径"
    )
    
    parser.add_argument(
        "--image-root", 
        required=True, 
        type=str, 
        help="存放图片文件的根目录绝对路径"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=os.path.join(os.environ.get("GROUNDING_DINO_ROOT", ""), "Dino_out_human"),
        help="推理结果输出目录"
    )
    
    parser.add_argument(
        "--gpu-id", 
        type=str, 
        default="5",
        help="指定使用的 GPU ID（默认：5）"
    )
    
    parser.add_argument(
        "--origin_prompt", 
        type=str, 
        default="human",
        help="检测类型：human 或 object（默认：human）"
    )
    
    parser.add_argument(
        "--not_edited",
        type=bool,
        default=False,
        help="是否使用未编辑的图片（默认：False）"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        default=os.path.join(os.environ.get("GROUNDING_DINO_ROOT", ""), "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        help="模型配置文件路径"
    )
    
    parser.add_argument(
        "--weights-file",
        type=str,
        default=GROUNDING_DINO_CHECKPOINT,
        help="模型权重文件路径"
    )
    
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.3,
        help="box threshold"
    )
    
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="text threshold"
    )
    
    args = parser.parse_args()
    return args


def load_image(image_path):
    """加载并预处理图片"""
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    """加载模型（只加载一次）"""
    print(f"🔄 正在加载模型...")
    print(f"   📄 配置文件: {model_config_path}")
    print(f"   💾 权重文件: {model_checkpoint_path}")
    print(f"   🖥️  设备: {device}")
    
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"   模型加载结果: {load_res}")
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成！")
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cuda"):
    """执行推理"""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
    # 过滤输出
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    
    # 获取短语
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
    
    return boxes_filt, pred_phrases


def plot_boxes_to_image(image_pil, tgt):
    """绘制检测框到图片"""
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    
    box_xys = []
    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        box_xys.append([x0, y0, x1, y1])
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    
    return image_pil, mask, box_xys


def plot_single_box_to_image(image_pil, box, label, H, W):
    """为单个检测框生成可视化图片"""
    image_copy = image_pil.copy()
    draw = ImageDraw.Draw(image_copy)
    
    box = box * torch.Tensor([W, H, W, H])
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    
    draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
    font = ImageFont.load_default()
    if hasattr(font, "getbbox"):
        bbox = draw.textbbox((x0, y0), str(label), font)
    else:
        w, h = draw.textsize(str(label), font)
        bbox = (x0, y0, w + x0, y0 + h)
    draw.rectangle(bbox, fill=color)
    draw.text((x0, y0), str(label), fill="white")
    
    return image_copy, [x0, y0, x1, y1]


def process_single_image(model, image_path, prompt, output_dir, box_threshold, text_threshold, device="cuda"):
    """处理单张图片"""
    # 加载图片
    image_pil, image = load_image(image_path)
    
    # 执行推理
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, prompt, box_threshold, text_threshold, device
    )
    
    # 保存原始图片
    base_image_name = image_path.split("/")[-1].split(".png")[0].split(".jpg")[0]
    os.makedirs(output_dir, exist_ok=True)
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))
    
    # 可视化所有检测框
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],
        "labels": pred_phrases,
    }
    
    image_with_box, _, box_xys = plot_boxes_to_image(image_pil.copy(), pred_dict)
    image_with_box.save(os.path.join(output_dir, "pred_all.jpg"))
    
    # 保存检测结果文本
    with open(os.path.join(output_dir, f"detection_results_{base_image_name}.txt"), "w") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Total detections: {len(boxes_filt)}\n\n")
        for i, (box_xy, label) in enumerate(zip(box_xys, pred_phrases)):
            f.write(f"Detection {i}: {label}\n")
            f.write(f"Box coordinates: {box_xy}\n\n")
    
    # 为每个检测结果单独生成可视化图片
    H, W = size[1], size[0]
    for i, (box, label) in enumerate(zip(boxes_filt, pred_phrases)):
        single_image, box_xy = plot_single_box_to_image(image_pil, box, label, H, W)
        
        if "(" in label and ")" in label:
            score_str = label.split("(")[-1].replace(")", "")
            label_clean = label.split("(")[0]
        else:
            score_str = "0.00"
            label_clean = label
        
        output_filename = f"pred_{base_image_name}_{i:03d}_{label_clean}_{score_str}.jpg"
        single_image.save(os.path.join(output_dir, output_filename))
    
    return len(boxes_filt)


def main():
    args = parse_args()
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载 JSON 文件
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            prompt_list = json.load(f)
        print(f"✅ 成功加载 JSON 文件：{args.json_path}")
        print(f"✅ 共包含 {len(prompt_list)} 个图片推理任务")
    except FileNotFoundError:
        print(f"❌ 错误：找不到 JSON 文件 -> {args.json_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误：JSON 文件格式不正确 -> {args.json_path}")
        return
    
    # ⭐ 关键优化：只加载一次模型
    print("\n" + "="*60)
    print("🚀 开始加载模型（仅此一次）")
    print("="*60)
    model = load_model(args.config_file, args.weights_file, device)
    print("="*60 + "\n")
    
    # 批量处理图片
    print("📊 开始批量推理...")
    total_detections = 0
    processed_count = 0
    
    # 导入glob用于文件名匹配
    import glob
    
    def find_image_file(image_root, image_name, not_edited=False):
        """灵活查找图像文件，支持多种文件名格式"""
        if not_edited:
            # 原图模式：直接使用JSON中的文件名
            candidates = [os.path.join(image_root, image_name)]
        else:
            # 编辑后图片模式：尝试多种可能的文件名组合
            base_name = os.path.splitext(image_name)[0]
            candidates = []
            
            # 1. _edited.png 格式
            candidates.append(os.path.join(image_root, f"{base_name}_edited.png"))
            # 2. _edited.jpg 格式
            candidates.append(os.path.join(image_root, f"{base_name}_edited.jpg"))
            # 3. 原文件名（保持原扩展名）
            candidates.append(os.path.join(image_root, image_name))
            # 4. 原文件名但扩展名改为 .png
            if not image_name.endswith(".png"):
                candidates.append(os.path.join(image_root, f"{base_name}.png"))
            # 5. 原文件名但扩展名改为 .jpg
            if not image_name.endswith(".jpg"):
                candidates.append(os.path.join(image_root, f"{base_name}.jpg"))
            
            # 6. 尝试匹配带哈希后缀的文件（如 base_name_hash.jpg）
            if os.path.isdir(image_root):
                pattern_png = os.path.join(image_root, f"{base_name}_*.png")
                pattern_jpg = os.path.join(image_root, f"{base_name}_*.jpg")
                matching_files = glob.glob(pattern_png) + glob.glob(pattern_jpg)
                # 排除 _edited 格式（已经在前面尝试过了）
                matching_files = [f for f in matching_files 
                                 if not os.path.basename(f).endswith("_edited.png") 
                                 and not os.path.basename(f).endswith("_edited.jpg")]
                candidates.extend(matching_files)
        
        # 返回第一个存在的文件
        return next((p for p in candidates if os.path.isfile(p)), None)
    
    for image_name, data in tqdm.tqdm(prompt_list.items(), desc="📊 批量推理进度"):
        # 查找图片文件
        image_full_path = find_image_file(args.image_root, image_name, args.not_edited)
        
        # 检查图片是否存在
        if not image_full_path:
            print(f"\n⚠️  警告：图片文件不存在，已跳过 -> {image_name}")
            print(f"   尝试过的路径（前3个）: {[os.path.join(args.image_root, image_name.replace('.png', '_edited.png'))]}")
            continue
        
        # 获取 prompt
        if args.origin_prompt == "human":
            prompt = "human"
        else:
            if "object" not in data:
                print(f"\n⚠️  警告：图片 '{image_name}' 的 JSON 数据中无 'object' 字段，已跳过")
                continue
            else:
                prompt = data["object"].strip()
        
        if not prompt:
            print(f"\n⚠️  警告：图片 '{image_name}' 的 prompt 为空，已跳过")
            continue
        
        # 为每张图片创建单独的输出目录
        base_image_name = image_name.replace(".png", "").replace(".jpg", "")
        image_output_dir = os.path.join(args.output_dir, base_image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        try:
            # 处理图片（模型已加载，直接推理）
            num_detections = process_single_image(
                model, image_full_path, prompt, image_output_dir,
                args.box_threshold, args.text_threshold, device
            )
            total_detections += num_detections
            processed_count += 1
            
        except Exception as e:
            print(f"\n❌ 处理失败！-> {image_name}")
            print(f"   错误信息：{str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 所有推理任务处理完毕！")
    print(f"📊 统计信息：")
    print(f"   - 处理图片数: {processed_count}/{len(prompt_list)}")
    print(f"   - 总检测数: {total_detections}")
    print(f"📁 最终结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
