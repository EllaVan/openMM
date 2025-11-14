#!/usr/bin/env python
"""
批量提取数据集特征 - 混合版本（音频 PCA 降维）
文本: MiniLM-L6 (384d)
音频: HuBERT (768d) → PCA (384d)
视频: ViT-small (384d)
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_unimodal.feature_extractor_hybrid import HybridFeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'extraction_hybrid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def train_audio_pca_model(
    extractor: HybridFeatureExtractor,
    base_dir: str,
    label_file: str,
    num_samples: int = 1000,
    save_path: str = None
):
    """
    训练音频 PCA 模型

    Args:
        extractor: 特征提取器
        base_dir: 数据集根目录
        label_file: 标签文件
        num_samples: 用于训练 PCA 的样本数量
        save_path: PCA 模型保存路径
    """
    logger.info("="*60)
    logger.info("阶段 1: 训练音频 PCA 模型")
    logger.info("="*60)

    # 读取标签文件
    df = pd.read_csv(label_file)
    logger.info(f"数据集总样本数: {len(df)}")

    # 随机采样
    if len(df) > num_samples:
        df_sample = df.sample(n=num_samples, random_state=42)
        logger.info(f"随机采样 {num_samples} 个样本用于训练 PCA")
    else:
        df_sample = df
        logger.info(f"使用全部 {len(df)} 个样本训练 PCA")

    audio_dir = os.path.join(base_dir, 'audio')

    # 收集音频文件路径
    audio_paths = []
    for index, row in df_sample.iterrows():
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        audio_path = os.path.join(audio_dir, video_id, f"{clip_id}.wav")

        if os.path.exists(audio_path):
            audio_paths.append(audio_path)

    logger.info(f"找到 {len(audio_paths)} 个有效音频文件")

    # 训练 PCA
    extractor.train_audio_pca(audio_paths, save_path=save_path)

    logger.info("="*60)
    logger.info("音频 PCA 模型训练完成")
    logger.info("="*60 + "\n")


def extract_mosei_hybrid(
    config_path: str,
    base_dir: str,
    label_file: str,
    output_dir: str,
    pca_model_path: str = None,
    train_pca: bool = False,
    pca_training_samples: int = 1000
):
    """
    提取 MOSEI 数据集（混合版本）

    Args:
        config_path: 配置文件路径
        base_dir: MOSEI 根目录
        label_file: 标签文件
        output_dir: 输出目录
        pca_model_path: PCA 模型路径（如果已训练）
        train_pca: 是否训练 PCA
        pca_training_samples: 用于训练 PCA 的样本数
    """
    logger.info("="*60)
    logger.info("开始提取 MOSEI 数据集（混合版本）")
    logger.info("="*60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"数据集目录: {base_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*60 + "\n")

    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 合并内存管理配置
    if 'memory_management' in config:
        for key, value in config['memory_management'].items():
            config[key] = value

    # 如果提供了 PCA 模型路径，更新配置
    if pca_model_path and os.path.exists(pca_model_path):
        config['audio']['pca_reduction']['model_path'] = pca_model_path
        logger.info(f"使用预训练 PCA 模型: {pca_model_path}")

    # 初始化提取器
    extractor = HybridFeatureExtractor(config=config)

    # 训练 PCA（如果需要）
    if train_pca and extractor.use_audio_pca and not extractor.audio_pca.is_fitted:
        pca_save_path = os.path.join(output_dir, 'audio_pca_model.pkl')
        train_audio_pca_model(
            extractor=extractor,
            base_dir=base_dir,
            label_file=label_file,
            num_samples=pca_training_samples,
            save_path=pca_save_path
        )

    # 开始提取特征
    logger.info("="*60)
    logger.info("阶段 2: 批量提取特征")
    logger.info("="*60 + "\n")

    # 读取标签
    df = pd.read_csv(label_file)
    logger.info(f"总样本数: {len(df)}")

    audio_dir = os.path.join(base_dir, 'audio')
    video_dir = os.path.join(base_dir, 'video')

    emotion_mapping = {
        'happy': 0, 'happiness': 0,
        'sad': 1, 'sadness': 1,
        'anger': 2,
        'disgust': 3,
        'surprise': 4,
        'fear': 5,
        'neutral': 6
    }

    # 按情感分组
    emotion_data = {emotion: [] for emotion in emotion_mapping.keys()}

    # 处理每个样本
    success_count = 0
    fail_count = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        text = row['text']
        emotion = row['emotion'].lower()

        # 文件路径
        audio_path = os.path.join(audio_dir, video_id, f"{clip_id}.wav")
        video_path = os.path.join(video_dir, video_id, f"{clip_id}.mp4")

        # 检查文件
        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            fail_count += 1
            continue

        if emotion not in emotion_mapping:
            fail_count += 1
            continue

        # 提取特征
        sample_id = f"{video_id}_{clip_id}"
        try:
            features = extractor.extract_multimodal_features(text, audio_path, video_path)

            # 保存数据
            sample_data = {
                'audio_features': features['audio_features'],      # [T, 384]
                'text_features': features['text_features'],        # [T, 384]
                'video_features': features['video_features'],      # [T, 384]
                'label': emotion_mapping[emotion],
                'emotion': emotion,
                'sample_id': sample_id,
                'num_frames': features['num_frames']
            }

            emotion_data[emotion].append(sample_data)
            success_count += 1

        except Exception as e:
            logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
            fail_count += 1

    # 保存特征文件
    os.makedirs(output_dir, exist_ok=True)
    stats = {}

    for emotion, samples in emotion_data.items():
        if len(samples) > 0:
            label_id = emotion_mapping[emotion]
            filename = f"MOSEI{emotion}label{label_id}.pkl"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(samples, f)

            stats[emotion] = len(samples)
            logger.info(f"✓ 已保存 {emotion}: {len(samples)} 样本")

    logger.info("="*60)
    logger.info("MOSEI 混合提取完成")
    logger.info("="*60)
    logger.info(f"成功: {success_count} 样本")
    logger.info(f"失败: {fail_count} 样本")
    logger.info(f"统计: {stats}")
    logger.info("="*60 + "\n")

    return stats


def train_audio_pca_model_meld(
    extractor: HybridFeatureExtractor,
    base_dir: str,
    num_samples: int = 1000,
    save_path: str = None
):
    """
    训练音频 PCA 模型（MELD 数据集）

    Args:
        extractor: 特征提取器
        base_dir: MELD 根目录
        num_samples: 用于训练 PCA 的样本数量
        save_path: PCA 模型保存路径
    """
    logger.info("="*60)
    logger.info("阶段 1: 训练音频 PCA 模型（MELD）")
    logger.info("="*60)

    # 收集所有划分的音频文件
    audio_paths = []
    for split in ['train', 'dev', 'test']:
        split_dir = os.path.join(base_dir, split)
        audio_dir = os.path.join(split_dir, 'audio')
        label_file = os.path.join(split_dir, 'label.csv')

        if not os.path.exists(label_file):
            logger.warning(f"标签文件不存在: {label_file}")
            continue

        df = pd.read_csv(label_file)
        for index, row in df.iterrows():
            file_id = row['file_id']
            audio_path = os.path.join(audio_dir, f"{file_id}.wav")
            if os.path.exists(audio_path):
                audio_paths.append(audio_path)

    logger.info(f"找到 {len(audio_paths)} 个音频文件")

    # 随机采样
    import random
    if len(audio_paths) > num_samples:
        audio_paths = random.sample(audio_paths, num_samples)
        logger.info(f"随机采样 {num_samples} 个样本用于训练 PCA")
    else:
        logger.info(f"使用全部 {len(audio_paths)} 个样本训练 PCA")

    # 训练 PCA
    extractor.train_audio_pca(audio_paths, save_path=save_path)

    logger.info("="*60)
    logger.info("音频 PCA 模型训练完成")
    logger.info("="*60 + "\n")


def extract_meld_hybrid(
    config_path: str,
    base_dir: str,
    output_dir: str,
    split: str = 'all',
    pca_model_path: str = None,
    train_pca: bool = False,
    pca_training_samples: int = 1000
):
    """
    提取 MELD 数据集（混合版本）

    Args:
        config_path: 配置文件路径
        base_dir: MELD 根目录
        output_dir: 输出目录
        split: 'train', 'dev', 'test', 或 'all'
        pca_model_path: PCA 模型路径（如果已训练）
        train_pca: 是否训练 PCA
        pca_training_samples: 用于训练 PCA 的样本数
    """
    logger.info("="*60)
    logger.info("开始提取 MELD 数据集（混合版本）")
    logger.info("="*60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"数据集目录: {base_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"划分: {split}")
    logger.info("="*60 + "\n")

    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 合并内存管理配置
    if 'memory_management' in config:
        for key, value in config['memory_management'].items():
            config[key] = value

    # 如果提供了 PCA 模型路径，更新配置
    if pca_model_path and os.path.exists(pca_model_path):
        config['audio']['pca_reduction']['model_path'] = pca_model_path
        logger.info(f"使用预训练 PCA 模型: {pca_model_path}")

    # 初始化提取器
    extractor = HybridFeatureExtractor(config=config)

    # 训练 PCA（如果需要）
    if train_pca and extractor.use_audio_pca and not extractor.audio_pca.is_fitted:
        pca_save_path = os.path.join(output_dir, 'audio_pca_model.pkl')
        train_audio_pca_model_meld(
            extractor=extractor,
            base_dir=base_dir,
            num_samples=pca_training_samples,
            save_path=pca_save_path
        )

    # 确定要处理的划分
    if split == 'all':
        splits = ['train', 'dev', 'test']
    else:
        splits = [split]

    # 情感映射
    emotion_mapping = {
        'happy': 0, 'happiness': 0,
        'sad': 1, 'sadness': 1,
        'anger': 2,
        'disgust': 3,
        'surprise': 4,
        'fear': 5,
        'neutral': 6
    }

    # 处理每个划分
    all_stats = {}

    for current_split in splits:
        logger.info("="*60)
        logger.info(f"处理 MELD {current_split} 集")
        logger.info("="*60 + "\n")

        split_dir = os.path.join(base_dir, current_split)
        audio_dir = os.path.join(split_dir, 'audio')
        video_dir = os.path.join(split_dir, 'video')
        label_file = os.path.join(split_dir, 'label.csv')

        if not os.path.exists(label_file):
            logger.error(f"标签文件不存在: {label_file}")
            continue

        # 读取标签
        df = pd.read_csv(label_file)
        logger.info(f"{current_split} 集总样本数: {len(df)}")

        # 按情感分组
        emotion_data = {emotion: [] for emotion in emotion_mapping.keys()}

        # 处理每个样本
        success_count = 0
        fail_count = 0

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"提取 {current_split} 特征"):
            file_id = row['file_id']  # dia0_utt0
            text = row['text']
            emotion = row['emotion'].lower()

            # 文件路径
            audio_path = os.path.join(audio_dir, f"{file_id}.wav")
            video_path = os.path.join(video_dir, f"{file_id}.mp4")

            # 检查文件
            if not os.path.exists(audio_path) or not os.path.exists(video_path):
                fail_count += 1
                continue

            if emotion not in emotion_mapping:
                fail_count += 1
                continue

            # 提取特征
            sample_id = f"{current_split}_{file_id}"
            try:
                features = extractor.extract_multimodal_features(text, audio_path, video_path)

                # 保存数据
                sample_data = {
                    'audio_features': features['audio_features'],      # [T, 384]
                    'text_features': features['text_features'],        # [T, 384]
                    'video_features': features['video_features'],      # [T, 384]
                    'label': emotion_mapping[emotion],
                    'emotion': emotion,
                    'sample_id': sample_id,
                    'num_frames': features['num_frames']
                }

                emotion_data[emotion].append(sample_data)
                success_count += 1

            except Exception as e:
                logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
                fail_count += 1

        # 保存特征文件
        os.makedirs(output_dir, exist_ok=True)
        stats = {}

        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                label_id = emotion_mapping[emotion]
                filename = f"MELD_{current_split}{emotion}label{label_id}.pkl"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'wb') as f:
                    pickle.dump(samples, f)

                stats[emotion] = len(samples)
                logger.info(f"✓ 已保存 {emotion}: {len(samples)} 样本")

        logger.info("="*60)
        logger.info(f"MELD {current_split} 集提取完成")
        logger.info("="*60)
        logger.info(f"成功: {success_count} 样本")
        logger.info(f"失败: {fail_count} 样本")
        logger.info(f"统计: {stats}")
        logger.info("="*60 + "\n")

        all_stats[current_split] = stats

    logger.info("="*60)
    logger.info("MELD 混合提取全部完成")
    logger.info("="*60)
    logger.info(f"总统计: {all_stats}")
    logger.info("="*60 + "\n")

    return all_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量提取数据集特征（混合版本）')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['mosei', 'meld'],
        help='数据集名称'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='new_unimodal/config_hybrid.json',
        help='配置文件路径'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='数据集根目录'
    )
    parser.add_argument(
        '--label_file',
        type=str,
        default=None,
        help='标签文件路径（仅 MOSEI 需要）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'dev', 'test', 'all'],
        help='MELD 数据集划分（仅 MELD 使用）'
    )
    parser.add_argument(
        '--pca_model_path',
        type=str,
        default=None,
        help='预训练 PCA 模型路径（可选）'
    )
    parser.add_argument(
        '--train_pca',
        action='store_true',
        help='是否训练音频 PCA 模型'
    )
    parser.add_argument(
        '--pca_training_samples',
        type=int,
        default=1000,
        help='用于训练 PCA 的样本数量'
    )

    args = parser.parse_args()

    # 根据数据集类型提取特征
    if args.dataset == 'mosei':
        if not args.label_file:
            logger.error("MOSEI 数据集需要提供 --label_file 参数")
            return

        extract_mosei_hybrid(
            config_path=args.config,
            base_dir=args.base_dir,
            label_file=args.label_file,
            output_dir=args.output_dir,
            pca_model_path=args.pca_model_path,
            train_pca=args.train_pca,
            pca_training_samples=args.pca_training_samples
        )

    elif args.dataset == 'meld':
        extract_meld_hybrid(
            config_path=args.config,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            split=args.split,
            pca_model_path=args.pca_model_path,
            train_pca=args.train_pca,
            pca_training_samples=args.pca_training_samples
        )


if __name__ == '__main__':
    main()
