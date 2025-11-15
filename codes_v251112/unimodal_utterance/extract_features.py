#!/usr/bin/env python
"""
Utterance-Level 特征提取脚本
使用 RoBERTa/HuBERT/ViT-Base 提取固定维度特征

使用方法：
1. 编辑 extraction_settings.json 配置文件
2. 运行：python extract_features.py
"""

import os
import json
import logging
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import datetime
from collections import defaultdict

from feature_extractor import create_extractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path='extraction_settings.json'):
    """加载配置文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_path)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def extract_mosei(config):
    """提取 MOSEI 数据集特征"""
    logger.info("\n" + "="*60)
    logger.info("开始提取 MOSEI 数据集")
    logger.info("="*60)

    mosei_config = config['mosei']
    base_dir = mosei_config['base_dir']
    label_file = mosei_config['label_file']
    output_dir = mosei_config['output_dir']

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建提取器
    extractor = create_extractor('extraction_settings.json')

    # 读取标签文件
    df = pd.read_csv(label_file)
    logger.info(f"数据集总样本数: {len(df)}")

    # 情感映射（支持 angry/anger, surprise/surprised 等变体）
    emotion_mapping = {
        'happy': 0, 'sad': 1, 'angry': 2, 'anger': 2,
        'surprise': 3, 'surprised': 3,
        'disgust': 4, 'disgusted': 4,
        'fear': 5, 'fearful': 5,
        'neutral': 6
    }

    # 情感标准化映射（统一为标准名称）
    emotion_normalize = {
        'happy': 'happy', 'sad': 'sad',
        'angry': 'angry', 'anger': 'angry',
        'surprise': 'surprise', 'surprised': 'surprise',
        'disgust': 'disgust', 'disgusted': 'disgust',
        'fear': 'fear', 'fearful': 'fear',
        'neutral': 'neutral'
    }

    # 按情感分组存储
    emotion_data = defaultdict(list)
    success_count = 0
    fail_count = 0

    # 提取特征
    for index, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        emotion = row['emotion'].lower()
        text = row['text']

        # 文件路径
        video_path = os.path.join(base_dir, 'video', video_id, f"{clip_id}.mp4")
        audio_path = os.path.join(base_dir, 'audio', video_id, f"{clip_id}.wav")

        # 检查文件是否存在
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            fail_count += 1
            continue

        sample_id = f"{video_id}_{clip_id}"
        try:
            # 标准化情感名称
            emotion_std = emotion_normalize.get(emotion, emotion)

            features = extractor.extract_multimodal_features(text, audio_path, video_path)

            sample_data = {
                'text_features': features['text_features'],
                'audio_features': features['audio_features'],
                'video_features': features['video_features'],
                'label': emotion_mapping[emotion],
                'emotion': emotion_std,
                'sample_id': sample_id
            }

            emotion_data[emotion_std].append(sample_data)
            success_count += 1

        except Exception as e:
            logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
            fail_count += 1

    # 保存特征文件
    stats = {}
    for emotion, samples in emotion_data.items():
        if len(samples) > 0:
            output_file = os.path.join(output_dir, f"MOSEI{emotion}label{emotion_mapping[emotion]}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f)
            stats[emotion] = len(samples)
            logger.info(f"✓ {emotion}: {len(samples)} 样本 -> {output_file}")

    logger.info("\n" + "="*60)
    logger.info("MOSEI 特征提取完成")
    logger.info(f"成功: {success_count} | 失败: {fail_count}")
    logger.info("="*60 + "\n")

    return stats


def extract_meld(config):
    """提取 MELD 数据集特征"""
    logger.info("\n" + "="*60)
    logger.info("开始提取 MELD 数据集")
    logger.info("="*60)

    meld_config = config['meld']
    base_dir = meld_config['base_dir']
    output_dir = meld_config['output_dir']
    split_mode = meld_config.get('split', 'all')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建提取器
    extractor = create_extractor('extraction_settings.json')

    # 情感映射（支持 angry/anger, surprise/surprised 等变体）
    emotion_mapping = {
        'happy': 0, 'sad': 1, 'angry': 2, 'anger': 2,
        'surprise': 3, 'surprised': 3,
        'disgust': 4, 'disgusted': 4,
        'fear': 5, 'fearful': 5,
        'neutral': 6
    }

    # 情感标准化映射（统一为标准名称）
    emotion_normalize = {
        'happy': 'happy', 'sad': 'sad',
        'angry': 'angry', 'anger': 'angry',
        'surprise': 'surprise', 'surprised': 'surprise',
        'disgust': 'disgust', 'disgusted': 'disgust',
        'fear': 'fear', 'fearful': 'fear',
        'neutral': 'neutral'
    }

    # 确定要处理的splits
    if split_mode == 'all':
        splits = ['train', 'dev', 'test']
    else:
        splits = [split_mode]

    total_stats = {}

    for split in splits:
        logger.info(f"\n处理 {split} split...")
        split_dir = os.path.join(base_dir, split)
        label_file = os.path.join(split_dir, 'label.csv')

        if not os.path.exists(label_file):
            logger.warning(f"标签文件不存在: {label_file}")
            continue

        df = pd.read_csv(label_file)
        logger.info(f"{split} 样本数: {len(df)}")

        # 按情感分组存储
        emotion_data = defaultdict(list)
        success_count = 0
        fail_count = 0

        # 提取特征
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"提取 {split}"):
            video_name = row['video_name']
            emotion = row['emotion'].lower()
            text = row['text']

            # 文件路径
            video_path = os.path.join(split_dir, 'video', f"{video_name}.mp4")
            audio_path = os.path.join(split_dir, 'audio', f"{video_name}.wav")

            # 检查文件是否存在
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                fail_count += 1
                continue

            sample_id = f"{split}_{video_name}"
            try:
                # 标准化情感名称
                emotion_std = emotion_normalize.get(emotion, emotion)

                features = extractor.extract_multimodal_features(text, audio_path, video_path)

                sample_data = {
                    'text_features': features['text_features'],
                    'audio_features': features['audio_features'],
                    'video_features': features['video_features'],
                    'label': emotion_mapping[emotion],
                    'emotion': emotion_std,
                    'sample_id': sample_id
                }

                emotion_data[emotion_std].append(sample_data)
                success_count += 1

            except Exception as e:
                logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
                fail_count += 1

        # 保存特征文件
        stats = {}
        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                output_file = os.path.join(output_dir, f"MELD_{split}{emotion}label{emotion_mapping[emotion]}.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(samples, f)
                stats[emotion] = len(samples)
                logger.info(f"✓ {emotion}: {len(samples)} 样本 -> {output_file}")

        logger.info(f"{split} 完成 - 成功: {success_count} | 失败: {fail_count}")
        total_stats[split] = stats

    logger.info("\n" + "="*60)
    logger.info("MELD 特征提取完成")
    logger.info("="*60 + "\n")

    return total_stats


def main():
    """主函数 - 完全自动化"""
    config = load_config()

    logger.info("\n" + "="*60)
    logger.info("Utterance-Level 特征提取系统")
    logger.info(f"配置版本: {config.get('version', 'unknown')}")
    logger.info("="*60)

    # 检查哪些数据集已启用
    enabled_datasets = []
    if config['mosei'].get('enabled', False):
        enabled_datasets.append('mosei')
    if config['meld'].get('enabled', False):
        enabled_datasets.append('meld')

    if not enabled_datasets:
        logger.error("错误：没有启用任何数据集！")
        logger.error("请在 extraction_settings.json 中设置至少一个数据集的 enabled=true")
        return

    logger.info(f"启用的数据集: {', '.join(enabled_datasets)}\n")

    # 提取特征
    all_stats = {}
    for dataset in enabled_datasets:
        if dataset == 'mosei':
            stats = extract_mosei(config)
            all_stats['mosei'] = stats
        elif dataset == 'meld':
            stats = extract_meld(config)
            all_stats['meld'] = stats

    # 输出总结
    logger.info("\n" + "="*60)
    logger.info("所有数据集提取完成！")
    logger.info("="*60)
    for dataset, stats in all_stats.items():
        logger.info(f"\n{dataset.upper()}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                logger.info(f"  {key}: {value if isinstance(value, int) else sum(value.values()) if isinstance(value, dict) else value}")


if __name__ == '__main__':
    main()
