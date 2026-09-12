"""
USVTrack数据集转换为YOLO格式
GT格式: frame,id,x,y,w,h,conf,class,vis
YOLO格式: class_id x_center y_center width height (归一化)
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# USVTrack数据集路径
USVTRACK_ROOT = Path("/home/xds/dataset/USVTrack-Published")
OUTPUT_ROOT = Path("/home/xds/dataset/USVTrack-YOLO")

# 类别映射（USVTrack使用1-based索引）
# 根据waterscenes_benchmark.txt
CLASS_NAMES = ['pier', 'buoy', 'sailor', 'ship', 'boat', 'vessel', 'kayak']
USVTRACK_TO_YOLO = {
    1: 0,  # pier
    2: 1,  # buoy
    3: 2,  # sailor
    4: 3,  # ship
    5: 4,  # boat
    6: 5,  # vessel
    7: 6   # kayak
}

def convert_bbox_to_yolo(x, y, w, h, img_width, img_height):
    """
    转换bbox格式
    输入: x,y,w,h (左上角坐标 + 宽高)
    输出: x_center,y_center,w,h (归一化到[0,1])
    """
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    # 裁剪到[0,1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_sequence(seq_id, split='train'):
    """转换单个序列"""

    # GT文件路径
    gt_file = USVTRACK_ROOT / 'images' / split / seq_id / 'gt' / 'gt.txt'

    if not gt_file.exists():
        print(f"  ⚠️  跳过 {seq_id}: 未找到GT文件")
        return 0, 0

    # 读取GT
    gt_df = pd.read_csv(gt_file, header=None,
                        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])

    # 图像目录
    img_dir = USVTRACK_ROOT / 'images' / split / seq_id / 'img1'

    if not img_dir.exists():
        print(f"  ⚠️  跳过 {seq_id}: 未找到图像目录")
        return 0, 0

    # 创建输出目录
    out_img_dir = OUTPUT_ROOT / 'images' / split / seq_id
    out_label_dir = OUTPUT_ROOT / 'labels' / split / seq_id
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    converted_frames = 0
    total_boxes = 0

    # 按帧处理
    for frame_id, frame_gt in gt_df.groupby('frame'):
        # 图像文件名（USVTrack使用时间戳作为文件名）
        # 先查找对应的图像文件
        img_files = sorted(img_dir.glob("*.jpg"))

        if frame_id > len(img_files):
            continue

        # 使用frame_id作为索引（1-based）
        img_file = img_files[int(frame_id) - 1]

        if not img_file.exists():
            continue

        # 获取图像尺寸
        try:
            img = Image.open(img_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"    ⚠️  无法读取图像 {img_file.name}: {e}")
            continue

        # 输出文件名（保持原始文件名）
        out_img_file = out_img_dir / img_file.name
        out_label_file = out_label_dir / (img_file.stem + '.txt')

        # 复制图像
        shutil.copy2(img_file, out_img_file)

        # 生成YOLO标注
        yolo_labels = []
        for _, row in frame_gt.iterrows():
            class_usvtrack = int(row['class'])

            # 类别映射（跳过类别0，这通常是"DontCare"或背景）
            if class_usvtrack not in USVTRACK_TO_YOLO:
                # 静默跳过类别0（忽略区域）
                if class_usvtrack != 0:
                    print(f"    ⚠️  未知类别: {class_usvtrack}")
                continue

            class_yolo = USVTRACK_TO_YOLO[class_usvtrack]

            # 转换bbox
            x, y, w, h = row['x'], row['y'], row['w'], row['h']

            # 过滤无效框
            if w <= 0 or h <= 0:
                continue

            x_center, y_center, width, height = convert_bbox_to_yolo(
                x, y, w, h, img_width, img_height
            )

            # 过滤过小的框
            if width < 0.001 or height < 0.001:
                continue

            yolo_labels.append(
                f"{class_yolo} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
            total_boxes += 1

        # 保存标注文件
        if len(yolo_labels) > 0:
            with open(out_label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
            converted_frames += 1

    return converted_frames, total_boxes


def main():
    print("=" * 80)
    print("USVTrack → YOLO 格式转换")
    print("=" * 80)

    # 获取所有序列
    train_sequences = sorted([d.name for d in (USVTRACK_ROOT / 'images' / 'train').iterdir() if d.is_dir()])

    # 分配训练集和验证集（80/20分割）
    num_train = int(len(train_sequences) * 0.8)
    sequences_train = train_sequences[:num_train]
    sequences_val = train_sequences[num_train:]

    print(f"\n数据集分割:")
    print(f"  训练集: {len(sequences_train)} 序列")
    print(f"  验证集: {len(sequences_val)} 序列")
    print(f"\n训练序列: {sequences_train}")
    print(f"验证序列: {sequences_val}")

    # 转换训练集
    print(f"\n{'='*80}")
    print("转换训练集")
    print(f"{'='*80}")

    total_train_frames = 0
    total_train_boxes = 0

    for seq_id in tqdm(sequences_train, desc="训练集"):
        frames, boxes = convert_sequence(seq_id, 'train')
        total_train_frames += frames
        total_train_boxes += boxes

    print(f"\n训练集统计:")
    print(f"  转换帧数: {total_train_frames}")
    print(f"  标注框数: {total_train_boxes}")

    # 转换验证集
    print(f"\n{'='*80}")
    print("转换验证集")
    print(f"{'='*80}")

    total_val_frames = 0
    total_val_boxes = 0

    for seq_id in tqdm(sequences_val, desc="验证集"):
        frames, boxes = convert_sequence(seq_id, 'train')  # 注意：验证集也来自train目录
        total_val_frames += frames
        total_val_boxes += boxes

    print(f"\n验证集统计:")
    print(f"  转换帧数: {total_val_frames}")
    print(f"  标注框数: {total_val_boxes}")

    # 总结
    print(f"\n{'='*80}")
    print("转换完成")
    print(f"{'='*80}")
    print(f"\n总计:")
    print(f"  帧数: {total_train_frames + total_val_frames}")
    print(f"  标注框: {total_train_boxes + total_val_boxes}")
    print(f"\n输出目录: {OUTPUT_ROOT}")
    print(f"\n下一步:")
    print(f"  1. 检查转换结果: ls {OUTPUT_ROOT}/images/train/")
    print(f"  2. 开始训练: python scripts/train_yolov11.py")
    print("="*80)


if __name__ == "__main__":
    main()
