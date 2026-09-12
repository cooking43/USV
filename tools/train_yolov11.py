"""
YOLOv11s训练脚本 - USVTrack数据集
模型: YOLOv11-small (平衡速度和精度)
"""

from ultralytics import YOLO
import torch
from pathlib import Path

print("=" * 80)
print("YOLOv11 训练 - USVTrack水面目标检测")
print("=" * 80)

# 检查CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# 配置
PROJECT_DIR = Path("/home/xds/VLM_USV")
DATASET_YAML = PROJECT_DIR / "scripts" / "dataset.yaml"
OUTPUT_DIR = PROJECT_DIR / "runs" / "detect"
PRETRAINED_MODEL = PROJECT_DIR / "models" / "YOLO" / "yolo11s.pt"

print(f"\n配置:")
print(f"  数据集配置: {DATASET_YAML}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  预训练模型: {PRETRAINED_MODEL}")

# 加载预训练模型（使用本地文件，无需下载）
print(f"\n加载YOLOv11s预训练模型（本地）...")
model = YOLO(str(PRETRAINED_MODEL))  # 使用本地预训练权重

print(f"模型加载成功！")
print(f"\n开始训练...")
print("=" * 80)

# 训练参数
results = model.train(
    # 数据
    data=str(DATASET_YAML),

    # 训练配置
    epochs=100,              # 训练轮数
    patience=20,             # Early stopping patience
    batch=64,                # Batch size (根据GPU显存调整)
    imgsz=640,               # 输入图像尺寸

    # 设备
    device=0,                # GPU 0
    workers=8,               # 数据加载线程数

    # 优化器
    optimizer='AdamW',       # 优化器
    lr0=0.001,              # 初始学习率
    lrf=0.01,               # 最终学习率 (lr0 * lrf)
    momentum=0.937,          # SGD momentum/Adam beta1
    weight_decay=0.0005,     # 权重衰减
    warmup_epochs=3.0,       # Warmup epochs
    warmup_momentum=0.8,     # Warmup momentum
    warmup_bias_lr=0.1,      # Warmup bias learning rate

    # 数据增强
    hsv_h=0.015,            # HSV-Hue增强 (fraction)
    hsv_s=0.7,              # HSV-Saturation增强 (fraction)
    hsv_v=0.4,              # HSV-Value增强 (fraction)
    degrees=0.0,            # 旋转增强 (deg)
    translate=0.1,          # 平移增强 (fraction)
    scale=0.5,              # 缩放增强 (fraction)
    shear=0.0,              # 剪切增强 (deg)
    perspective=0.0,        # 透视增强 (fraction)
    flipud=0.0,             # 上下翻转概率
    fliplr=0.5,             # 左右翻转概率
    mosaic=1.0,             # Mosaic增强概率
    mixup=0.0,              # MixUp增强概率
    copy_paste=0.0,         # Copy-paste增强概率

    # 保存和日志
    project=str(OUTPUT_DIR),
    name='yolov11s_usvtrack',
    exist_ok=False,          # 不覆盖现有目录
    pretrained=True,         # 使用预训练权重
    save=True,               # 保存检查点
    save_period=-1,          # 每N轮保存一次 (-1=仅保存last和best)

    # 验证
    val=True,                # 训练时验证

    # 其他
    verbose=True,            # 详细输出
    seed=0,                  # 随机种子
    deterministic=True,      # 确定性训练
    single_cls=False,        # 多类别
    rect=False,              # 矩形训练
    cos_lr=False,            # 余弦学习率调度
    close_mosaic=10,         # 最后N轮关闭mosaic
    amp=True,                # 自动混合精度训练
    fraction=1.0,            # 使用的数据集比例
    profile=False,           # 性能分析
    freeze=None,             # 冻结层

    # 多尺度训练
    multi_scale=False,       # 多尺度训练 (±50%变化)
)

print("\n" + "=" * 80)
print("训练完成！")
print("=" * 80)

# 验证
print("\n运行验证...")
metrics = model.val()

print(f"\n最终性能:")
print(f"  mAP@0.5: {metrics.box.map50:.3f}")
print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"  Precision: {metrics.box.mp:.3f}")
print(f"  Recall: {metrics.box.mr:.3f}")

# 显示结果路径
results_dir = OUTPUT_DIR / "yolov11s_usvtrack"
print(f"\n结果保存在:")
print(f"  目录: {results_dir}")
print(f"  最佳模型: {results_dir}/weights/best.pt")
print(f"  最后模型: {results_dir}/weights/last.pt")
print(f"  训练曲线: {results_dir}/results.png")
print(f"  混淆矩阵: {results_dir}/confusion_matrix.png")

print("\n下一步:")
print(f"  1. 查看训练曲线: open {results_dir}/results.png")
print(f"  2. 测试模型: python scripts/test_yolo_model.py")
print(f"  3. 集成到VLM_USV: python scripts/integrate_yolo.py")

print("=" * 80)
