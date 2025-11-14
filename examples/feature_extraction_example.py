"""
多模态特征提取示例代码

演示如何使用 MultimodalFeatureExtractor 从原始数据提取特征并进行时间对齐
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction_demo import MultimodalFeatureExtractor
import json
import numpy as np


def example_1_basic_extraction():
    """
    示例 1: 基础特征提取
    使用默认配置提取对齐的多模态特征
    """
    print("\n" + "=" * 60)
    print("示例 1: 基础特征提取")
    print("=" * 60)

    # 初始化提取器
    extractor = MultimodalFeatureExtractor()

    # 示例文件路径（请替换为实际文件路径）
    text_file = "../example_data/sample.txt"
    audio_file = "../example_data/sample.wav"
    video_file = "../example_data/sample.mp4"

    # 检查文件是否存在
    if not all([os.path.exists(f) for f in [text_file, audio_file, video_file] if f]):
        print("⚠ 示例数据文件不存在，请准备数据文件:")
        print("  - example_data/sample.txt")
        print("  - example_data/sample.wav")
        print("  - example_data/sample.wav")
        print("\n创建示例文本文件...")
        os.makedirs("../example_data", exist_ok=True)
        with open("../example_data/sample.txt", "w") as f:
            f.write("This is a sample text for feature extraction demonstration.")
        print("✓ 已创建示例文本文件")
        print("\n请准备对应的音频和视频文件后重新运行")
        return

    # 提取特征
    features = extractor.extract_from_files(
        text_file=text_file,
        audio_file=audio_file,
        video_file=video_file,
        output_file="../output/example1_features.pkl"
    )

    # 显示结果
    if features:
        print("\n提取结果:")
        print(f"  总帧数: {features.get('num_frames', 'N/A')}")
        if features.get('audio') is not None:
            print(f"  音频特征: {features['audio'].shape}")
        if features.get('text') is not None:
            print(f"  文本特征: {features['text'].shape}")
        if features.get('video') is not None:
            print(f"  视频特征: {features['video'].shape}")


def example_2_custom_config():
    """
    示例 2: 使用自定义配置
    从 JSON 配置文件加载配置
    """
    print("\n" + "=" * 60)
    print("示例 2: 使用自定义配置 (Librosa)")
    print("=" * 60)

    # 加载配置文件
    config_file = "../config_librosa.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    print(f"\n使用配置: {config['description']}")

    # 初始化提取器
    extractor = MultimodalFeatureExtractor(config=config)

    # 示例：只提取音频和文本
    text_file = "../example_data/sample.txt"
    audio_file = "../example_data/sample.wav"

    if os.path.exists(text_file) and os.path.exists(audio_file):
        features = extractor.extract_from_files(
            text_file=text_file,
            audio_file=audio_file,
            # video_file=None,  # 不提供视频
            output_file="../output/example2_features.pkl"
        )
    else:
        print("⚠ 示例数据文件不存在")


def example_3_audio_only():
    """
    示例 3: 仅提取音频特征
    演示如何提取单个模态的特征
    """
    print("\n" + "=" * 60)
    print("示例 3: 仅提取音频特征")
    print("=" * 60)

    # 配置：只启用音频
    config = {
        'text': {'enabled': False},
        'audio': {
            'model': 'librosa',
            'sample_rate': 16000,
            'enabled': True
        },
        'video': {'enabled': False},
        'alignment': {'enabled': False}
    }

    extractor = MultimodalFeatureExtractor(config=config)

    audio_file = "../example_data/sample.wav"

    if os.path.exists(audio_file):
        # 直接调用音频提取方法
        audio_features = extractor.extract_audio_features(audio_file)

        print("\n音频特征:")
        print(f"  形状: {audio_features['shape']}")
        print(f"  时长: {audio_features['duration']:.2f}s")
        print(f"  采样率: {audio_features['sample_rate']}Hz")
        print(f"  时间步数: {len(audio_features['timestamps'])}")

        # 保存
        extractor.save_features(
            {'audio': audio_features},
            "../output/example3_audio_only.pkl"
        )
    else:
        print("⚠ 音频文件不存在")


def example_4_batch_processing():
    """
    示例 4: 批量处理
    处理多个文件
    """
    print("\n" + "=" * 60)
    print("示例 4: 批量处理多个文件")
    print("=" * 60)

    extractor = MultimodalFeatureExtractor()

    # 定义文件列表
    file_list = [
        {
            'id': 'sample1',
            'text': '../example_data/sample1.txt',
            'audio': '../example_data/sample1.wav',
            'video': '../example_data/sample1.mp4'
        },
        {
            'id': 'sample2',
            'text': '../example_data/sample2.txt',
            'audio': '../example_data/sample2.wav',
            'video': '../example_data/sample2.mp4'
        }
    ]

    all_features = {}
    processed_count = 0

    for item in file_list:
        print(f"\n处理: {item['id']}")

        # 检查文件是否存在
        files_exist = all([
            os.path.exists(item[k]) for k in ['text', 'audio', 'video']
            if item.get(k)
        ])

        if files_exist:
            features = extractor.extract_from_files(
                text_file=item.get('text'),
                audio_file=item.get('audio'),
                video_file=item.get('video'),
                output_file=f"../output/{item['id']}_features.pkl"
            )
            all_features[item['id']] = features
            processed_count += 1
        else:
            print(f"  ⚠ 文件不存在，跳过")

    print(f"\n✓ 批量处理完成: {processed_count}/{len(file_list)} 个文件")


def example_5_load_and_use():
    """
    示例 5: 加载已提取的特征
    演示如何加载和使用保存的特征
    """
    print("\n" + "=" * 60)
    print("示例 5: 加载已提取的特征")
    print("=" * 60)

    extractor = MultimodalFeatureExtractor()

    # 加载特征
    feature_file = "../output/example1_features.pkl"

    if os.path.exists(feature_file):
        features = extractor.load_features(feature_file)

        print(f"\n✓ 特征已加载: {feature_file}")
        print("\n特征信息:")

        # 显示特征信息
        if isinstance(features, dict):
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif key == 'timestamps' and value is not None:
                    print(f"  {key}: {len(value)} 个时间戳")

        # 演示如何使用特征
        print("\n使用示例:")
        print("  # 获取对齐后的特征")
        print("  audio_feat = features['audio']  # [T, audio_dim]")
        print("  text_feat = features['text']    # [T, 768]")
        print("  video_feat = features['video']  # [T, video_dim]")
        print("  ")
        print("  # 可以输入到多模态模型")
        print("  # output = your_model(audio_feat, text_feat, video_feat)")

    else:
        print(f"⚠ 特征文件不存在: {feature_file}")
        print("请先运行示例 1 生成特征文件")


def example_6_create_sample_data():
    """
    示例 6: 创建示例数据
    生成简单的示例文件用于测试
    """
    print("\n" + "=" * 60)
    print("示例 6: 创建示例数据")
    print("=" * 60)

    os.makedirs("../example_data", exist_ok=True)

    # 创建示例文本文件
    sample_texts = [
        "This is a happy conversation about the beautiful weather today.",
        "I am feeling sad because I lost my favorite book yesterday.",
        "The exciting news made everyone in the room very surprised."
    ]

    for i, text in enumerate(sample_texts, 1):
        text_file = f"../example_data/sample{i}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ 已创建: {text_file}")

    print("\n⚠ 注意: 音频和视频文件需要手动准备")
    print("建议使用以下工具:")
    print("  - FFmpeg: 从视频中提取音频")
    print("    ffmpeg -i video.mp4 -ar 16000 audio.wav")
    print("  - 录制短视频或使用公开数据集")


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs("../output", exist_ok=True)
    os.makedirs("../example_data", exist_ok=True)

    print("=" * 60)
    print("多模态特征提取示例")
    print("=" * 60)
    print("\n可用示例:")
    print("  1. 基础特征提取 (默认配置)")
    print("  2. 使用自定义配置 (Librosa)")
    print("  3. 仅提取音频特征")
    print("  4. 批量处理多个文件")
    print("  5. 加载已提取的特征")
    print("  6. 创建示例数据")
    print("  0. 运行所有示例")

    choice = input("\n请选择示例 (0-6): ").strip()

    if choice == '1':
        example_1_basic_extraction()
    elif choice == '2':
        example_2_custom_config()
    elif choice == '3':
        example_3_audio_only()
    elif choice == '4':
        example_4_batch_processing()
    elif choice == '5':
        example_5_load_and_use()
    elif choice == '6':
        example_6_create_sample_data()
    elif choice == '0':
        example_6_create_sample_data()
        example_1_basic_extraction()
        example_2_custom_config()
        example_3_audio_only()
        example_4_batch_processing()
        example_5_load_and_use()
    else:
        print("无效选择")

    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
