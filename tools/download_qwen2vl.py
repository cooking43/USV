"""
下载Qwen2-VL模型的辅助脚本
"""
import os
import sys
from pathlib import Path


def download_from_huggingface(model_id, local_dir):
    """从HuggingFace下载模型"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("错误: 需要安装 huggingface_hub")
        print("运行: pip install huggingface_hub")
        return False

    print(f"从HuggingFace下载 {model_id}...")
    print(f"保存位置: {local_dir}")
    print("提示: 如果下载较慢或失败，可以使用 --modelscope 参数切换到国内镜像\n")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n✓ 模型已下载到: {local_dir}")
        return True
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n如果是网络问题，建议:")
        print("1. 使用 --modelscope 参数切换到国内镜像")
        print("2. 或手动下载后放置到指定目录")
        return False


def download_from_modelscope(model_id, local_dir):
    """从ModelScope下载模型（国内镜像）"""
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        print("错误: 需要安装 modelscope")
        print("运行: pip install modelscope")
        return False

    print(f"从ModelScope下载 {model_id}...")
    print(f"保存位置: {local_dir}\n")

    try:
        ms_snapshot_download(
            model_id,
            cache_dir=str(local_dir),
            revision='master'
        )
        print(f"\n✓ 模型已下载到: {local_dir}")
        return True
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Qwen2-VL-7B-Instruct 模型下载工具")
    print("=" * 60)

    # 解析参数
    use_modelscope = "--modelscope" in sys.argv or "--ms" in sys.argv

    # 模型配置
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    local_dir = Path("models/qwen2-vl-7b-instruct")

    # 创建目录
    local_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已下载
    if (local_dir / "config.json").exists():
        print(f"\n⚠ 模型已存在于: {local_dir}")
        response = input("是否重新下载？(y/N): ")
        if response.lower() != 'y':
            print("取消下载")
            return

    print(f"\n模型ID: {model_id}")
    print(f"下载方式: {'ModelScope (国内镜像)' if use_modelscope else 'HuggingFace'}")
    print(f"保存路径: {local_dir.absolute()}")
    print(f"\n预计大小: ~15 GB")
    print("=" * 60)
    print()

    # 下载
    if use_modelscope:
        success = download_from_modelscope(model_id, local_dir)
    else:
        success = download_from_huggingface(model_id, local_dir)

    if success:
        print("\n" + "=" * 60)
        print("下载完成！")
        print("=" * 60)
        print(f"\n模型位置: {local_dir.absolute()}")
        print("\n下一步:")
        print("1. 运行 test_vlm_setup.py 验证环境")
        print("2. 使用 src/vlm/vlm_inference.py 进行推理")
        print("=" * 60)
    else:
        print("\n下载失败。请检查网络连接或尝试其他下载方式。")


if __name__ == "__main__":
    main()
