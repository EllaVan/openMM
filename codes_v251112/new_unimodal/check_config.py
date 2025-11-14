#!/usr/bin/env python
"""
配置检查脚本
检查所有配置文件是否正确，模型和数据集路径是否存在
"""

import os
import json
import sys


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_status(item, status, message=""):
    """打印状态"""
    symbol = "✓" if status else "✗"
    color = "\033[0;32m" if status else "\033[0;31m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {item}: {message}")


def check_file_exists(path, name):
    """检查文件是否存在"""
    exists = os.path.exists(path) and os.path.isfile(path)
    if exists:
        size = os.path.getsize(path)
        print_status(name, True, f"存在 ({size / 1024:.1f} KB)")
    else:
        print_status(name, False, f"不存在: {path}")
    return exists


def check_dir_exists(path, name):
    """检查目录是否存在"""
    exists = os.path.exists(path) and os.path.isdir(path)
    if exists:
        print_status(name, True, f"存在")
    else:
        print_status(name, False, f"不存在: {path}")
    return exists


def check_config_files():
    """检查配置文件"""
    print_section("配置文件检查")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_ok = True

    # 检查主配置文件
    config_files = {
        "config_hybrid.json": "主配置文件",
        "dataset_paths.json": "数据集路径配置",
        "extraction_config.json": "提取参数配置"
    }

    for filename, description in config_files.items():
        path = os.path.join(base_dir, filename)
        if not check_file_exists(path, description):
            all_ok = False

    return all_ok


def check_model_paths():
    """检查模型路径"""
    print_section("模型路径检查")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config_hybrid.json")

    if not os.path.exists(config_path):
        print_status("配置文件", False, "config_hybrid.json 不存在")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    all_ok = True
    models = {
        "text": ("MiniLM-L6-v2", config['text']['model_path']),
        "audio": ("HuBERT-base", config['audio']['model_path']),
        "video": ("ViT-small", config['video']['model_path'])
    }

    for modality, (name, path) in models.items():
        if path == "/path/to/" or "/path/to/" in path:
            print_status(f"{name} ({modality})", False, f"需要配置路径: {path}")
            all_ok = False
        elif not os.path.exists(path):
            print_status(f"{name} ({modality})", False, f"路径不存在: {path}")
            all_ok = False
        else:
            # 检查是否包含必要文件
            required_files = ['config.json', 'pytorch_model.bin']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(path, f))]

            if missing_files:
                print_status(f"{name} ({modality})", False, f"缺少文件: {', '.join(missing_files)}")
                all_ok = False
            else:
                print_status(f"{name} ({modality})", True, f"正确配置")

    return all_ok


def check_dataset_paths():
    """检查数据集路径"""
    print_section("数据集路径检查")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "dataset_paths.json")

    if not os.path.exists(config_path):
        print_status("配置文件", False, "dataset_paths.json 不存在")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    all_ok = True

    # 检查 MOSEI
    print("\n[MOSEI]")
    mosei = config['mosei']
    mosei_ok = True

    if mosei['base_dir'] == "/path/to/MOSEI" or "/path/to/" in mosei['base_dir']:
        print_status("MOSEI 根目录", False, f"需要配置: {mosei['base_dir']}")
        mosei_ok = False
    elif not os.path.exists(mosei['base_dir']):
        print_status("MOSEI 根目录", False, f"不存在: {mosei['base_dir']}")
        mosei_ok = False
    else:
        print_status("MOSEI 根目录", True, mosei['base_dir'])

        # 检查子目录
        audio_dir = os.path.join(mosei['base_dir'], 'audio')
        video_dir = os.path.join(mosei['base_dir'], 'video')

        if not check_dir_exists(audio_dir, "  audio 目录"):
            mosei_ok = False
        if not check_dir_exists(video_dir, "  video 目录"):
            mosei_ok = False

    # 检查标签文件
    if mosei['label_file'] == "/path/to/" or "/path/to/" in mosei['label_file']:
        print_status("  label 文件", False, f"需要配置: {mosei['label_file']}")
        mosei_ok = False
    elif not os.path.exists(mosei['label_file']):
        print_status("  label 文件", False, f"不存在: {mosei['label_file']}")
        mosei_ok = False
    else:
        print_status("  label 文件", True, mosei['label_file'])

    all_ok = all_ok and mosei_ok

    # 检查 MELD
    print("\n[MELD]")
    meld = config['meld']
    meld_ok = True

    if meld['base_dir'] == "/path/to/MELD" or "/path/to/" in meld['base_dir']:
        print_status("MELD 根目录", False, f"需要配置: {meld['base_dir']}")
        meld_ok = False
    elif not os.path.exists(meld['base_dir']):
        print_status("MELD 根目录", False, f"不存在: {meld['base_dir']}")
        meld_ok = False
    else:
        print_status("MELD 根目录", True, meld['base_dir'])

        # 检查各划分目录
        for split in ['train', 'dev', 'test']:
            split_dir = os.path.join(meld['base_dir'], split)
            if not os.path.exists(split_dir):
                print_status(f"  {split} 目录", False, f"不存在")
                meld_ok = False
            else:
                print_status(f"  {split} 目录", True, "")

                # 检查子目录和标签文件
                audio_dir = os.path.join(split_dir, 'audio')
                video_dir = os.path.join(split_dir, 'video')
                label_file = os.path.join(split_dir, 'label.csv')

                if not os.path.exists(audio_dir):
                    print_status(f"    audio 目录", False, "")
                    meld_ok = False
                if not os.path.exists(video_dir):
                    print_status(f"    video 目录", False, "")
                    meld_ok = False
                if not os.path.exists(label_file):
                    print_status(f"    label.csv", False, "")
                    meld_ok = False

    all_ok = all_ok and meld_ok

    return all_ok


def check_dependencies():
    """检查依赖包"""
    print_section("依赖包检查")

    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'tqdm'),
        ('librosa', 'Librosa'),
        ('cv2', 'OpenCV')
    ]

    all_ok = True
    for package, name in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print_status(name, True, "已安装")
        except ImportError:
            print_status(name, False, "未安装")
            all_ok = False

    return all_ok


def check_gpu():
    """检查 GPU 可用性"""
    print_section("GPU 检查")

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print_status("CUDA", True, f"可用，{device_count} 个设备")

            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name} ({memory:.1f} GB)")

            return True
        else:
            print_status("CUDA", False, "不可用")
            return False
    except ImportError:
        print_status("PyTorch", False, "未安装")
        return False


def print_summary(results):
    """打印总结"""
    print_section("检查总结")

    all_ok = all(results.values())

    for name, status in results.items():
        print_status(name, status, "通过" if status else "失败")

    print("\n" + "=" * 60)
    if all_ok:
        print("\033[0;32m✓ 所有检查通过！可以开始特征提取\033[0m")
    else:
        print("\033[0;31m✗ 部分检查失败，请修复后再运行\033[0m")
    print("=" * 60 + "\n")

    return all_ok


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("特征提取配置检查工具")
    print("=" * 60)

    results = {
        "配置文件": check_config_files(),
        "模型路径": check_model_paths(),
        "数据集路径": check_dataset_paths(),
        "依赖包": check_dependencies(),
        "GPU": check_gpu()
    }

    success = print_summary(results)

    if not success:
        print("\n修复建议:")
        if not results["配置文件"]:
            print("  1. 确保所有 JSON 配置文件存在")
        if not results["模型路径"]:
            print("  2. 编辑 config_hybrid.json，设置正确的模型路径")
            print("     或运行下载脚本下载所需模型")
        if not results["数据集路径"]:
            print("  3. 编辑 dataset_paths.json，设置正确的数据集路径")
        if not results["依赖包"]:
            print("  4. 安装缺失的依赖包:")
            print("     pip install torch transformers pandas numpy scikit-learn tqdm librosa opencv-python")
        if not results["GPU"]:
            print("  5. 检查 CUDA 安装，或使用 CPU 模式（较慢）")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
