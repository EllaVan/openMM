"""
不同 Backbone 的特征提取对比示例

演示如何使用不同的特征提取器（BERT vs RoBERTa, Wav2vec2 vs HuBERT, MediaPipe vs ViT）
并分析它们对时间对齐的影响
"""

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction_demo import MultimodalFeatureExtractor
import numpy as np


def compare_text_backbones():
    """
    对比 BERT 和 RoBERTa 的文本特征提取
    """
    print("\n" + "=" * 70)
    print("对比 1: BERT vs RoBERTa 文本特征提取")
    print("=" * 70)

    # 创建示例文本
    os.makedirs("../example_data", exist_ok=True)
    text_file = "../example_data/sample.txt"
    if not os.path.exists(text_file):
        with open(text_file, 'w') as f:
            f.write("This is a test sentence for comparing different text feature extractors.")

    # BERT 配置
    bert_config = {
        'text': {'model': 'bert-base-uncased', 'enabled': True},
        'audio': {'enabled': False},
        'video': {'enabled': False},
        'alignment': {'enabled': False}
    }

    # RoBERTa 配置
    roberta_config = {
        'text': {'model': 'roberta-base', 'enabled': True},
        'audio': {'enabled': False},
        'video': {'enabled': False},
        'alignment': {'enabled': False}
    }

    print("\n提取 BERT 特征...")
    bert_extractor = MultimodalFeatureExtractor(config=bert_config)
    bert_features = bert_extractor.extract_text_features(text_file)

    print("\n提取 RoBERTa 特征...")
    roberta_extractor = MultimodalFeatureExtractor(config=roberta_config)
    roberta_features = roberta_extractor.extract_text_features(text_file)

    print("\n" + "-" * 70)
    print("对比结果:")
    print("-" * 70)
    print(f"文本: {bert_features['text']}")
    print(f"\nBERT 特征:")
    print(f"  - 形状: {bert_features['shape']}")
    print(f"  - Token 数量: {bert_features['shape'][1]}")
    print(f"  - 特征维度: {bert_features['shape'][2]}")
    print(f"  - 单词数: {len(bert_features['words'])}")

    print(f"\nRoBERTa 特征:")
    print(f"  - 形状: {roberta_features['shape']}")
    print(f"  - Token 数量: {roberta_features['shape'][1]}")
    print(f"  - 特征维度: {roberta_features['shape'][2]}")
    print(f"  - 单词数: {len(roberta_features['words'])}")

    print("\n分析:")
    print(f"  ✓ 特征维度: {'相同' if bert_features['shape'][2] == roberta_features['shape'][2] else '不同'}")
    print(f"  ✓ Token 数量: {'相同' if bert_features['shape'][1] == roberta_features['shape'][1] else '不同'}")
    print(f"    (BERT: {bert_features['shape'][1]}, RoBERTa: {roberta_features['shape'][1]})")
    print("\n  注意: RoBERTa 使用不同的 tokenizer,可能产生不同数量的 token")
    print("        但线性插值对齐会自动处理这个差异")


def compare_audio_backbones():
    """
    对比 Wav2vec2 和 HuBERT 的音频特征提取
    """
    print("\n" + "=" * 70)
    print("对比 2: Wav2vec2 vs HuBERT 音频特征提取")
    print("=" * 70)

    audio_file = "../example_data/sample.wav"

    if not os.path.exists(audio_file):
        print("\n⚠ 音频文件不存在，跳过此对比")
        print(f"请准备音频文件: {audio_file}")
        return

    # Wav2vec2 配置
    wav2vec2_config = {
        'text': {'enabled': False},
        'audio': {
            'model': 'wav2vec2',
            'model_name': 'facebook/wav2vec2-base-960h',
            'sample_rate': 16000,
            'enabled': True
        },
        'video': {'enabled': False},
        'alignment': {'enabled': False}
    }

    # HuBERT 配置
    hubert_config = {
        'text': {'enabled': False},
        'audio': {
            'model': 'hubert',
            'model_name': 'facebook/hubert-base-ls960',
            'sample_rate': 16000,
            'enabled': True
        },
        'video': {'enabled': False},
        'alignment': {'enabled': False}
    }

    print("\n提取 Wav2vec2 特征...")
    wav2vec2_extractor = MultimodalFeatureExtractor(config=wav2vec2_config)
    wav2vec2_features = wav2vec2_extractor.extract_audio_features(audio_file)

    print("\n提取 HuBERT 特征...")
    hubert_extractor = MultimodalFeatureExtractor(config=hubert_config)
    hubert_features = hubert_extractor.extract_audio_features(audio_file)

    print("\n" + "-" * 70)
    print("对比结果:")
    print("-" * 70)
    print(f"Wav2vec2 特征:")
    print(f"  - 形状: {wav2vec2_features['shape']}")
    print(f"  - 时间步数: {wav2vec2_features['shape'][1]}")
    print(f"  - 特征维度: {wav2vec2_features['shape'][2]}")
    print(f"  - 音频时长: {wav2vec2_features['duration']:.2f}s")
    print(f"  - 时间分辨率: {wav2vec2_features['duration'] / wav2vec2_features['shape'][1] * 1000:.1f}ms/帧")

    print(f"\nHuBERT 特征:")
    print(f"  - 形状: {hubert_features['shape']}")
    print(f"  - 时间步数: {hubert_features['shape'][1]}")
    print(f"  - 特征维度: {hubert_features['shape'][2]}")
    print(f"  - 音频时长: {hubert_features['duration']:.2f}s")
    print(f"  - 时间分辨率: {hubert_features['duration'] / hubert_features['shape'][1] * 1000:.1f}ms/帧")

    print("\n分析:")
    print(f"  ✓ 时间步数: {'相同' if wav2vec2_features['shape'][1] == hubert_features['shape'][1] else '不同'}")
    print(f"    (Wav2vec2: {wav2vec2_features['shape'][1]}, HuBERT: {hubert_features['shape'][1]})")
    print(f"  ✓ 特征维度: {'相同' if wav2vec2_features['shape'][2] == hubert_features['shape'][2] else '不同'}")
    print("\n  注意: Wav2vec2 和 HuBERT 使用相同的卷积架构")
    print("        下采样率相同 (320x),因此时间步数应该一致")
    print(f"        对齐后的总帧数由音频时间步数决定: {wav2vec2_features['shape'][1]}")


def compare_video_backbones():
    """
    对比 MediaPipe 和 ViT 的视频特征提取
    """
    print("\n" + "=" * 70)
    print("对比 3: MediaPipe vs ViT-16 视频特征提取")
    print("=" * 70)

    video_file = "../example_data/sample.mp4"

    if not os.path.exists(video_file):
        print("\n⚠ 视频文件不存在，跳过此对比")
        print(f"请准备视频文件: {video_file}")
        return

    # MediaPipe 配置
    mediapipe_config = {
        'text': {'enabled': False},
        'audio': {'enabled': False},
        'video': {
            'model': 'mediapipe',
            'fps': 25,
            'enabled': True
        },
        'alignment': {'enabled': False}
    }

    # ViT 配置
    vit_config = {
        'text': {'enabled': False},
        'audio': {'enabled': False},
        'video': {
            'model': 'vit',
            'model_name': 'google/vit-base-patch16-224',
            'fps': 25,
            'enabled': True,
            'feature_mode': 'cls'
        },
        'alignment': {'enabled': False}
    }

    print("\n提取 MediaPipe 特征...")
    mediapipe_extractor = MultimodalFeatureExtractor(config=mediapipe_config)
    mediapipe_features = mediapipe_extractor.extract_video_features(video_file)

    print("\n提取 ViT 特征...")
    vit_extractor = MultimodalFeatureExtractor(config=vit_config)
    vit_features = vit_extractor.extract_video_features(video_file)

    print("\n" + "-" * 70)
    print("对比结果:")
    print("-" * 70)
    print(f"MediaPipe 特征:")
    print(f"  - 形状: {mediapipe_features['shape']}")
    print(f"  - 帧数: {mediapipe_features['frame_count']}")
    print(f"  - 特征维度: {mediapipe_features['shape'][1]} (468个关键点 × 3维)")
    print(f"  - FPS: {mediapipe_features['fps']}")
    print(f"  - 特征类型: 面部关键点坐标")

    print(f"\nViT-16 特征:")
    print(f"  - 形状: {vit_features['shape']}")
    print(f"  - 帧数: {vit_features['frame_count']}")
    print(f"  - 特征维度: {vit_features['shape'][1]} (ViT hidden size)")
    print(f"  - FPS: {vit_features['fps']}")
    print(f"  - 特征类型: 图像全局特征 ([CLS] token)")

    print("\n分析:")
    print(f"  ✓ 帧数: {'相同' if mediapipe_features['frame_count'] == vit_features['frame_count'] else '不同'}")
    print(f"    (MediaPipe: {mediapipe_features['frame_count']}, ViT: {vit_features['frame_count']})")
    print(f"  ✓ 特征维度: 不同")
    print(f"    - MediaPipe: {mediapipe_features['shape'][1]} (几何特征)")
    print(f"    - ViT: {vit_features['shape'][1]} (语义特征)")
    print("\n  注意: 特征维度不同不影响对齐")
    print("        两者都是 [num_frames, feature_dim] 格式")
    print("        对齐时使用时间戳匹配,与特征维度无关")


def test_alignment_with_different_backbones():
    """
    测试不同 backbone 组合的对齐效果
    """
    print("\n" + "=" * 70)
    print("对比 4: 不同 Backbone 组合的时间对齐")
    print("=" * 70)

    text_file = "../example_data/sample.txt"
    audio_file = "../example_data/sample.wav"
    video_file = "../example_data/sample.mp4"

    # 检查文件
    files_exist = all([
        os.path.exists(text_file),
        os.path.exists(audio_file),
        os.path.exists(video_file)
    ])

    if not files_exist:
        print("\n⚠ 缺少必要的数据文件，跳过此对比")
        print("需要的文件:")
        print(f"  - {text_file}")
        print(f"  - {audio_file}")
        print(f"  - {video_file}")
        return

    # 配置 1: BERT + Wav2vec2 + MediaPipe
    config1 = {
        'text': {'model': 'bert-base-uncased', 'enabled': True},
        'audio': {'model': 'wav2vec2', 'sample_rate': 16000, 'enabled': True},
        'video': {'model': 'mediapipe', 'fps': 25, 'enabled': True},
        'alignment': {'enabled': True, 'reference': 'audio'}
    }

    # 配置 2: RoBERTa + HuBERT + ViT
    config2 = {
        'text': {'model': 'roberta-base', 'enabled': True},
        'audio': {'model': 'hubert', 'model_name': 'facebook/hubert-base-ls960',
                  'sample_rate': 16000, 'enabled': True},
        'video': {'model': 'vit', 'model_name': 'google/vit-base-patch16-224',
                  'fps': 25, 'enabled': True, 'feature_mode': 'cls'},
        'alignment': {'enabled': True, 'reference': 'audio'}
    }

    print("\n配置 1: BERT + Wav2vec2 + MediaPipe")
    extractor1 = MultimodalFeatureExtractor(config=config1)
    features1 = extractor1.extract_from_files(
        text_file=text_file,
        audio_file=audio_file,
        video_file=video_file
    )

    print("\n配置 2: RoBERTa + HuBERT + ViT-16")
    extractor2 = MultimodalFeatureExtractor(config=config2)
    features2 = extractor2.extract_from_files(
        text_file=text_file,
        audio_file=audio_file,
        video_file=video_file
    )

    print("\n" + "-" * 70)
    print("对齐结果对比:")
    print("-" * 70)

    print("\n配置 1 对齐结果:")
    print(f"  - 总帧数: {features1['num_frames']}")
    print(f"  - 音频: {features1['audio'].shape if features1['audio'] is not None else 'N/A'}")
    print(f"  - 文本: {features1['text'].shape if features1['text'] is not None else 'N/A'}")
    print(f"  - 视频: {features1['video'].shape if features1['video'] is not None else 'N/A'}")

    print("\n配置 2 对齐结果:")
    print(f"  - 总帧数: {features2['num_frames']}")
    print(f"  - 音频: {features2['audio'].shape if features2['audio'] is not None else 'N/A'}")
    print(f"  - 文本: {features2['text'].shape if features2['text'] is not None else 'N/A'}")
    print(f"  - 视频: {features2['video'].shape if features2['video'] is not None else 'N/A'}")

    print("\n" + "-" * 70)
    print("关键发现:")
    print("-" * 70)
    print(f"1. 总帧数: {'相同' if features1['num_frames'] == features2['num_frames'] else '不同'}")
    print(f"   - 配置 1: {features1['num_frames']}")
    print(f"   - 配置 2: {features2['num_frames']}")
    print(f"\n   原因: {'Wav2vec2 和 HuBERT 下采样率相同 (320x)' if features1['num_frames'] == features2['num_frames'] else '音频模型下采样率不同'}")

    print(f"\n2. 文本特征维度:")
    if features1['text'] is not None and features2['text'] is not None:
        print(f"   - BERT: {features1['text'].shape}")
        print(f"   - RoBERTa: {features2['text'].shape}")
        print(f"   - 第一维(时间步): {'相同' if features1['text'].shape[0] == features2['text'].shape[0] else '不同'}")
        print(f"   - 第二维(特征维): {'相同' if features1['text'].shape[1] == features2['text'].shape[1] else '不同'}")

    print(f"\n3. 视频特征维度:")
    if features1['video'] is not None and features2['video'] is not None:
        print(f"   - MediaPipe: {features1['video'].shape} (关键点坐标)")
        print(f"   - ViT: {features2['video'].shape} (语义特征)")
        print(f"   - 第一维(时间步): {'相同' if features1['video'].shape[0] == features2['video'].shape[0] else '不同'}")

    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)
    print("✓ 更换 backbone 不影响时间对齐")
    print("✓ 对齐后的总帧数由音频模型决定")
    print("✓ 只要音频模型的下采样率相同,总帧数就相同")
    print("✓ 文本和视频特征会被插值/匹配到音频的时间步")
    print("✓ 特征维度的变化不影响对齐逻辑")


def main():
    print("=" * 70)
    print("多模态特征提取 Backbone 对比")
    print("=" * 70)
    print("\n可用对比:")
    print("  1. BERT vs RoBERTa (文本)")
    print("  2. Wav2vec2 vs HuBERT (音频)")
    print("  3. MediaPipe vs ViT-16 (视频)")
    print("  4. 不同 Backbone 组合的时间对齐")
    print("  0. 运行所有对比")

    choice = input("\n请选择对比 (0-4): ").strip()

    if choice == '1':
        compare_text_backbones()
    elif choice == '2':
        compare_audio_backbones()
    elif choice == '3':
        compare_video_backbones()
    elif choice == '4':
        test_alignment_with_different_backbones()
    elif choice == '0':
        compare_text_backbones()
        compare_audio_backbones()
        compare_video_backbones()
        test_alignment_with_different_backbones()
    else:
        print("无效选择")

    print("\n" + "=" * 70)
    print("对比完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
