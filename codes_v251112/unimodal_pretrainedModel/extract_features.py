#!/usr/bin/env python
"""
自动化特征提取脚本 - 从 JSON 配置文件读取所有参数
无需命令行参数，直接运行即可

使用方法：
1. 编辑 extraction_settings.json 配置文件
2. 运行：python extract_features.py
"""

import os
import sys
import json
import logging
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codes_v251112.unimodal_pretrainedModel.feature_extractor_hybrid import HybridFeatureExtractor

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


def train_audio_pca_model(extractor, base_dir, label_file, num_samples, save_path):
    """训练音频 PCA 模型（MOSEI）"""
    logger.info("="*60)
    logger.info("阶段 1: 训练音频 PCA 模型")
    logger.info("="*60)

    df = pd.read_csv(label_file)
    logger.info(f"数据集总样本数: {len(df)}")

    if len(df) > num_samples:
        df_sample = df.sample(n=num_samples, random_state=42)
        logger.info(f"随机采样 {num_samples} 个样本用于训练 PCA")
    else:
        df_sample = df
        logger.info(f"使用全部 {len(df)} 个样本训练 PCA")

    audio_dir = os.path.join(base_dir, 'audio')
    audio_paths = []

    for index, row in df_sample.iterrows():
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        audio_path = os.path.join(audio_dir, video_id, f"{clip_id}.wav")
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)

    logger.info(f"找到 {len(audio_paths)} 个有效音频文件")
    extractor.train_audio_pca(audio_paths, save_path=save_path)

    logger.info("="*60)
    logger.info("音频 PCA 模型训练完成")
    logger.info("="*60 + "\n")


def train_audio_pca_model_meld(extractor, base_dir, num_samples, save_path):
    """训练音频 PCA 模型（MELD）"""
    logger.info("="*60)
    logger.info("阶段 1: 训练音频 PCA 模型（MELD）")
    logger.info("="*60)

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

    import random
    if len(audio_paths) > num_samples:
        audio_paths = random.sample(audio_paths, num_samples)
        logger.info(f"随机采样 {num_samples} 个样本用于训练 PCA")
    else:
        logger.info(f"使用全部 {len(audio_paths)} 个样本训练 PCA")

    extractor.train_audio_pca(audio_paths, save_path=save_path)

    logger.info("="*60)
    logger.info("音频 PCA 模型训练完成")
    logger.info("="*60 + "\n")


def extract_mosei(config):
    """提取 MOSEI 数据集"""
    mosei_config = config['mosei']
    extraction_config = config['extraction']
    models_config = config['models']

    base_dir = mosei_config['base_dir']
    label_file = mosei_config['label_file']
    output_dir = mosei_config['output_dir']

    logger.info("="*60)
    logger.info("开始提取 MOSEI 数据集")
    logger.info("="*60)
    logger.info(f"数据集目录: {base_dir}")
    logger.info(f"标签文件: {label_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"采样率: {extraction_config['sampling_rate_fps']} fps")
    logger.info("="*60 + "\n")

    # 构建模型配置
    model_config = {
        'text': models_config['text'],
        'audio': models_config['audio'],
        'video': models_config['video'],
        'memory_management': {
            'max_frames': extraction_config['max_frames'],
            'video_batch_size': extraction_config['video_batch_size'],
            'enable_memory_cleanup': True
        }
    }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录已创建: {output_dir}\n")

    # 初始化提取器
    extractor = HybridFeatureExtractor(config=model_config)

    # 训练 PCA
    if extraction_config['train_pca'] and extractor.use_audio_pca and not extractor.audio_pca.is_fitted:
        pca_save_path = os.path.join(output_dir, 'audio_pca_model.pkl')
        train_audio_pca_model(
            extractor=extractor,
            base_dir=base_dir,
            label_file=label_file,
            num_samples=extraction_config['pca_training_samples'],
            save_path=pca_save_path
        )

    # 提取特征
    logger.info("="*60)
    logger.info("阶段 2: 批量提取特征")
    logger.info("="*60 + "\n")

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

    emotion_data = {emotion: [] for emotion in emotion_mapping.keys()}
    success_count = 0
    fail_count = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        text = row['text']
        emotion = row['emotion'].lower()

        audio_path = os.path.join(audio_dir, video_id, f"{clip_id}.wav")
        video_path = os.path.join(video_dir, video_id, f"{clip_id}.mp4")

        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            fail_count += 1
            continue

        if emotion not in emotion_mapping:
            fail_count += 1
            continue

        sample_id = f"{video_id}_{clip_id}"
        try:
            features = extractor.extract_multimodal_features(text, audio_path, video_path)

            sample_data = {
                'audio_features': features['audio_features'],
                'text_features': features['text_features'],
                'video_features': features['video_features'],
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
    logger.info("MOSEI 提取完成")
    logger.info("="*60)
    logger.info(f"成功: {success_count} 样本")
    logger.info(f"失败: {fail_count} 样本")
    logger.info(f"统计: {stats}")
    logger.info("="*60 + "\n")

    return stats


def extract_meld(config):
    """提取 MELD 数据集"""
    meld_config = config['meld']
    extraction_config = config['extraction']
    models_config = config['models']

    base_dir = meld_config['base_dir']
    output_dir = meld_config['output_dir']
    split = meld_config['split']

    logger.info("="*60)
    logger.info("开始提取 MELD 数据集")
    logger.info("="*60)
    logger.info(f"数据集目录: {base_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"划分: {split}")
    logger.info(f"采样率: {extraction_config['sampling_rate_fps']} fps")
    logger.info("="*60 + "\n")

    # 构建模型配置
    model_config = {
        'text': models_config['text'],
        'audio': models_config['audio'],
        'video': models_config['video'],
        'memory_management': {
            'max_frames': extraction_config['max_frames'],
            'video_batch_size': extraction_config['video_batch_size'],
            'enable_memory_cleanup': True
        }
    }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录已创建: {output_dir}\n")

    # 初始化提取器
    extractor = HybridFeatureExtractor(config=model_config)

    # 训练 PCA
    if extraction_config['train_pca'] and extractor.use_audio_pca and not extractor.audio_pca.is_fitted:
        pca_save_path = os.path.join(output_dir, 'audio_pca_model.pkl')
        train_audio_pca_model_meld(
            extractor=extractor,
            base_dir=base_dir,
            num_samples=extraction_config['pca_training_samples'],
            save_path=pca_save_path
        )

    # 确定要处理的划分
    if split == 'all':
        splits = ['train', 'dev', 'test']
    else:
        splits = [split]

    emotion_mapping = {
        'happy': 0, 'happiness': 0,
        'sad': 1, 'sadness': 1,
        'anger': 2,
        'disgust': 3,
        'surprise': 4,
        'fear': 5,
        'neutral': 6
    }

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

        df = pd.read_csv(label_file)
        logger.info(f"{current_split} 集总样本数: {len(df)}")

        emotion_data = {emotion: [] for emotion in emotion_mapping.keys()}
        success_count = 0
        fail_count = 0

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"提取 {current_split} 特征"):
            file_id = row['file_id']
            text = row['text']
            emotion = row['emotion'].lower()

            audio_path = os.path.join(audio_dir, f"{file_id}.wav")
            video_path = os.path.join(video_dir, f"{file_id}.mp4")

            if not os.path.exists(audio_path) or not os.path.exists(video_path):
                fail_count += 1
                continue

            if emotion not in emotion_mapping:
                fail_count += 1
                continue

            sample_id = f"{current_split}_{file_id}"
            try:
                features = extractor.extract_multimodal_features(text, audio_path, video_path)

                sample_data = {
                    'audio_features': features['audio_features'],
                    'text_features': features['text_features'],
                    'video_features': features['video_features'],
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
    logger.info("MELD 提取全部完成")
    logger.info("="*60)
    logger.info(f"总统计: {all_stats}")
    logger.info("="*60 + "\n")

    return all_stats


def main():
    """主函数 - 完全自动化，支持多数据集提取"""
    logger.info("="*60)
    logger.info("自动化特征提取系统 v1.1")
    logger.info("="*60 + "\n")

    # 加载配置
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        return

    # 检查哪些数据集已启用
    enabled_datasets = []
    if config['mosei'].get('enabled', False):
        enabled_datasets.append('mosei')
    if config['meld'].get('enabled', False):
        enabled_datasets.append('meld')

    if not enabled_datasets:
        logger.error("没有启用任何数据集！")
        logger.error("请在 extraction_settings.json 中设置 mosei.enabled=true 或 meld.enabled=true")
        return

    # 显示配置信息
    logger.info(f"配置加载成功")
    logger.info(f"启用的数据集: {', '.join([d.upper() for d in enabled_datasets])}")
    logger.info(f"采样率: {config['extraction']['sampling_rate_fps']} fps")
    logger.info(f"最大帧数: {config['extraction']['max_frames']}")
    logger.info(f"训练 PCA: {config['extraction']['train_pca']}")
    logger.info(f"PCA 模型共享: {config['extraction'].get('share_pca_model', True)}")
    logger.info("")

    # PCA 模型路径管理
    shared_pca_path = None
    share_pca = config['extraction'].get('share_pca_model', True)

    # 按顺序提取所有启用的数据集
    all_results = {}
    for idx, dataset in enumerate(enabled_datasets):
        logger.info("="*60)
        logger.info(f"[{idx+1}/{len(enabled_datasets)}] 开始提取数据集: {dataset.upper()}")
        logger.info("="*60 + "\n")

        # 如果是后续数据集且启用了 PCA 共享，使用已训练的 PCA 模型
        if idx > 0 and share_pca and shared_pca_path and os.path.exists(shared_pca_path):
            logger.info(f"使用已训练的 PCA 模型: {shared_pca_path}")
            config['extraction']['train_pca'] = False
            config['extraction']['pca_model_path'] = shared_pca_path
        else:
            # 第一个数据集训练 PCA
            config['extraction']['train_pca'] = True
            config['extraction']['pca_model_path'] = None

        # 提取特征
        try:
            if dataset == 'mosei':
                stats = extract_mosei(config)
                all_results['mosei'] = stats
                # 保存 PCA 模型路径供后续数据集使用
                if share_pca and idx == 0:
                    shared_pca_path = os.path.join(config['mosei']['output_dir'], 'audio_pca_model.pkl')
            elif dataset == 'meld':
                stats = extract_meld(config)
                all_results['meld'] = stats
                # 保存 PCA 模型路径供后续数据集使用
                if share_pca and idx == 0:
                    shared_pca_path = os.path.join(config['meld']['output_dir'], 'audio_pca_model.pkl')
        except Exception as e:
            logger.error(f"提取数据集 {dataset.upper()} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # 显示总结
    logger.info("="*60)
    logger.info("全部提取完成！")
    logger.info("="*60)
    logger.info(f"已完成的数据集: {len(all_results)}/{len(enabled_datasets)}")
    for dataset, stats in all_results.items():
        logger.info(f"\n{dataset.upper()} 统计:")
        logger.info(f"  {stats}")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
