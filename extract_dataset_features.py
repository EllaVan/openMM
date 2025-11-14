"""
数据集特征提取器
支持 MOSEI 和 MELD 数据集的批量特征提取，使用统一的多模态特征提取框架
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction_demo import MultimodalFeatureExtractor


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetFeatureExtractor:
    """
    数据集特征提取器基类
    """

    def __init__(
        self,
        dataset_name: str,
        base_dir: str,
        output_dir: str,
        config: Optional[Dict] = None
    ):
        """
        Args:
            dataset_name: 数据集名称 ('MOSEI' 或 'MELD')
            base_dir: 数据集根目录
            output_dir: 输出目录
            config: 特征提取配置
        """
        self.dataset_name = dataset_name.upper()
        self.base_dir = base_dir
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化特征提取器
        self.extractor = MultimodalFeatureExtractor(config=config)

        # 情感标签映射
        self.emotion_mapping = {
            'happy': 0,
            'happiness': 0,
            'sad': 1,
            'sadness': 1,
            'anger': 2,
            'disgust': 3,
            'surprise': 4,
            'fear': 5,
            'neutral': 6
        }

        logger.info(f"初始化 {dataset_name} 数据集提取器")
        logger.info(f"数据集目录: {base_dir}")
        logger.info(f"输出目录: {output_dir}")

    def extract_single_sample(
        self,
        text: str,
        audio_path: str,
        video_path: str,
        sample_id: str
    ) -> Optional[Dict]:
        """
        提取单个样本的特征

        Args:
            text: 文本内容
            audio_path: 音频文件路径
            video_path: 视频文件路径
            sample_id: 样本ID

        Returns:
            特征字典或 None
        """
        try:
            # 创建临时文本文件
            temp_text_file = os.path.join(self.output_dir, f"temp_{sample_id}.txt")
            with open(temp_text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            # 检查文件是否存在
            if not os.path.exists(audio_path):
                logger.warning(f"音频文件不存在: {audio_path}")
                return None

            if not os.path.exists(video_path):
                logger.warning(f"视频文件不存在: {video_path}")
                return None

            # 提取特征
            features = self.extractor.extract_from_files(
                text_file=temp_text_file,
                audio_file=audio_path,
                video_file=video_path,
                output_file=None  # 不单独保存每个样本
            )

            # 删除临时文件
            if os.path.exists(temp_text_file):
                os.remove(temp_text_file)

            return features

        except Exception as e:
            logger.error(f"提取样本 {sample_id} 特征时出错: {str(e)}")
            return None

    def process_dataset(self, split: str = 'all') -> Dict[str, List]:
        """
        处理整个数据集

        Args:
            split: 数据集划分 ('train', 'dev', 'test', 'all')

        Returns:
            处理后的数据字典
        """
        raise NotImplementedError("子类必须实现 process_dataset 方法")

    def save_features(
        self,
        features_list: List[Dict],
        emotion: str,
        label_id: int,
        split: Optional[str] = None
    ):
        """
        保存提取的特征

        Args:
            features_list: 特征列表
            emotion: 情感类型
            label_id: 标签ID
            split: 数据集划分 (MELD需要)
        """
        # 生成文件名
        if self.dataset_name == 'MELD' and split:
            filename = f"MELD_{split}{emotion}label{label_id}.pkl"
        else:
            filename = f"{self.dataset_name}{emotion}label{label_id}.pkl"

        filepath = os.path.join(self.output_dir, filename)

        # 保存为 pickle
        with open(filepath, 'wb') as f:
            pickle.dump(features_list, f)

        logger.info(f"✓ 已保存 {len(features_list)} 个样本到 {filepath}")


class MOSEIFeatureExtractor(DatasetFeatureExtractor):
    """
    MOSEI 数据集特征提取器

    数据集组织结构:
    base_dir/
        ├── audio/
        │   └── video_id/
        │       └── clip_id.wav
        ├── video/
        │   └── video_id/
        │       └── clip_id.mp4
        ├── text/
        │   └── ... (可选)
        └── label/
            └── label.csv  (包含 video_id, clip_id, text, emotion 等)
    """

    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        label_file: str,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_dir: MOSEI 数据集根目录
            output_dir: 输出目录
            label_file: 标签CSV文件路径
            config: 特征提取配置
        """
        super().__init__('MOSEI', base_dir, output_dir, config)

        self.audio_dir = os.path.join(base_dir, 'audio')
        self.video_dir = os.path.join(base_dir, 'video')
        self.text_dir = os.path.join(base_dir, 'text')
        self.label_file = label_file

        # 加载标签文件
        self.df = pd.read_csv(label_file)
        logger.info(f"加载 MOSEI 标签文件: {label_file}")
        logger.info(f"总样本数: {len(self.df)}")

    def process_dataset(self, split: str = 'all') -> Dict[str, List]:
        """
        处理 MOSEI 数据集

        Returns:
            按情感分组的特征字典
        """
        # 按情感分组
        emotion_data = {emotion: [] for emotion in self.emotion_mapping.keys()}

        logger.info("开始处理 MOSEI 数据集...")

        # 遍历所有样本
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="处理 MOSEI"):
            try:
                video_id = row['video_id']
                clip_id = row['clip_id']
                text = row['text'] if 'text' in row else ""

                # 获取情感标签
                if 'emotion' in row:
                    emotion = row['emotion'].lower()
                elif 'voted_emotion' in row:
                    emotion = row['voted_emotion'].lower()
                else:
                    logger.warning(f"样本 {video_id}/{clip_id} 缺少情感标签")
                    continue

                # 跳过 neutral
                if emotion == 'neutral' or emotion == 'neural':
                    continue

                if emotion not in self.emotion_mapping:
                    logger.warning(f"未知情感类型: {emotion}")
                    continue

                label_id = self.emotion_mapping[emotion]

                # 构建文件路径
                audio_path = os.path.join(self.audio_dir, video_id, f"{clip_id}.wav")
                video_path = os.path.join(self.video_dir, video_id, f"{clip_id}.mp4")

                # 提取特征
                sample_id = f"{video_id}_{clip_id}"
                features = self.extract_single_sample(text, audio_path, video_path, sample_id)

                if features is not None:
                    # 添加标签和元数据
                    sample_data = {
                        'audio_features': features['audio'],
                        'text_features': features['text'],
                        'video_features': features['video'],
                        'label': label_id,
                        'emotion': emotion,
                        'sample_id': sample_id,
                        'num_frames': features['num_frames']
                    }

                    emotion_data[emotion].append(sample_data)

            except Exception as e:
                logger.error(f"处理样本 {index} 时出错: {str(e)}")
                continue

        # 保存每个情感类别的特征
        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                label_id = self.emotion_mapping[emotion]
                self.save_features(samples, emotion, label_id)
                logger.info(f"情感 '{emotion}': {len(samples)} 个样本")

        return emotion_data


class MELDFeatureExtractor(DatasetFeatureExtractor):
    """
    MELD 数据集特征提取器

    数据集组织结构:
    base_dir/
        ├── train/
        │   ├── audio/
        │   │   └── video_id/
        │   │       └── clip_id.wav
        │   ├── video/
        │   │   └── video_id/
        │   │       └── clip_id.mp4
        │   └── label/
        │       └── merged_label_new.csv
        ├── dev/
        │   └── ... (同上)
        └── test/
            └── ... (同上)
    """

    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_dir: MELD 数据集根目录
            output_dir: 输出目录
            config: 特征提取配置
        """
        super().__init__('MELD', base_dir, output_dir, config)

    def process_split(self, split: str) -> Dict[str, List]:
        """
        处理单个数据集划分

        Args:
            split: 'train', 'dev', 或 'test'

        Returns:
            按情感分组的特征字典
        """
        split_dir = os.path.join(self.base_dir, split)
        audio_dir = os.path.join(split_dir, 'audio')
        video_dir = os.path.join(split_dir, 'video')
        label_file = os.path.join(split_dir, 'label', 'merged_label_new.csv')

        if not os.path.exists(label_file):
            logger.warning(f"标签文件不存在: {label_file}")
            return {}

        # 加载标签文件
        df = pd.read_csv(label_file)
        logger.info(f"处理 MELD {split} 集: {len(df)} 个样本")

        # 按情感分组
        emotion_data = {emotion: [] for emotion in self.emotion_mapping.keys()}

        # 遍历所有样本
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 MELD {split}"):
            try:
                video_id = row['video_id']
                clip_id = row['clip_id']
                text = row['text'] if 'text' in row else ""

                # 获取情感标签
                if 'emotion' in row:
                    emotion = row['emotion'].lower()
                elif 'voted_emotion' in row:
                    emotion = row['voted_emotion'].lower()
                else:
                    logger.warning(f"样本 {video_id}/{clip_id} 缺少情感标签")
                    continue

                # 跳过 neutral
                if emotion == 'neutral' or emotion == 'neural':
                    continue

                if emotion not in self.emotion_mapping:
                    logger.warning(f"未知情感类型: {emotion}")
                    continue

                label_id = self.emotion_mapping[emotion]

                # 构建文件路径
                audio_path = os.path.join(audio_dir, video_id, f"{clip_id}.wav")
                video_path = os.path.join(video_dir, video_id, f"{clip_id}.mp4")

                # 提取特征
                sample_id = f"{split}_{video_id}_{clip_id}"
                features = self.extract_single_sample(text, audio_path, video_path, sample_id)

                if features is not None:
                    # 添加标签和元数据
                    sample_data = {
                        'audio_features': features['audio'],
                        'text_features': features['text'],
                        'video_features': features['video'],
                        'label': label_id,
                        'emotion': emotion,
                        'sample_id': sample_id,
                        'num_frames': features['num_frames'],
                        'split': split
                    }

                    emotion_data[emotion].append(sample_data)

            except Exception as e:
                logger.error(f"处理样本 {index} 时出错: {str(e)}")
                continue

        # 保存每个情感类别的特征
        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                label_id = self.emotion_mapping[emotion]
                self.save_features(samples, emotion, label_id, split=split)
                logger.info(f"{split} - 情感 '{emotion}': {len(samples)} 个样本")

        return emotion_data

    def process_dataset(self, split: str = 'all') -> Dict[str, Dict[str, List]]:
        """
        处理 MELD 数据集

        Args:
            split: 'train', 'dev', 'test', 或 'all'

        Returns:
            按划分和情感分组的特征字典
        """
        results = {}

        if split == 'all':
            splits_to_process = ['train', 'dev', 'test']
        else:
            splits_to_process = [split]

        for s in splits_to_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"处理 MELD {s.upper()} 集")
            logger.info(f"{'='*60}")

            results[s] = self.process_split(s)

        return results


def main():
    """
    主函数 - 示例用法
    """
    # 配置
    # 选择特征提取器配置
    config_options = {
        '1': 'default',  # BERT + Wav2vec2 + MediaPipe
        '2': 'config_roberta_hubert_vit.json',  # RoBERTa + HuBERT + ViT
        '3': 'config_hubert.json',  # HuBERT
        '4': 'config_vit.json',  # ViT
    }

    print("\n" + "="*60)
    print("数据集特征提取工具")
    print("="*60)
    print("\n请选择特征提取器配置:")
    print("  1. 默认配置 (BERT + Wav2vec2 + MediaPipe)")
    print("  2. RoBERTa + HuBERT + ViT-16")
    print("  3. HuBERT 音频")
    print("  4. ViT-16 视频")

    config_choice = input("\n选择配置 (1-4): ").strip()

    if config_choice in config_options:
        config_name = config_options[config_choice]
        if config_name == 'default':
            config = None
        else:
            config_path = os.path.join(os.path.dirname(__file__), config_name)
            with open(config_path, 'r') as f:
                config = json.load(f)
        print(f"✓ 使用配置: {config_name}")
    else:
        config = None
        print("✓ 使用默认配置")

    # 选择数据集
    print("\n请选择数据集:")
    print("  1. MOSEI")
    print("  2. MELD")
    print("  3. 两者都处理")

    dataset_choice = input("\n选择数据集 (1-3): ").strip()

    # 处理 MOSEI
    if dataset_choice in ['1', '3']:
        print("\n" + "="*60)
        print("MOSEI 数据集配置")
        print("="*60)

        mosei_base_dir = input("MOSEI 数据集根目录: ").strip()
        mosei_label_file = input("标签CSV文件路径: ").strip()
        mosei_output_dir = input("输出目录 (默认: ./output/mosei): ").strip() or "./output/mosei"

        extractor = MOSEIFeatureExtractor(
            base_dir=mosei_base_dir,
            output_dir=mosei_output_dir,
            label_file=mosei_label_file,
            config=config
        )

        extractor.process_dataset()

    # 处理 MELD
    if dataset_choice in ['2', '3']:
        print("\n" + "="*60)
        print("MELD 数据集配置")
        print("="*60)

        meld_base_dir = input("MELD 数据集根目录: ").strip()
        meld_output_dir = input("输出目录 (默认: ./output/meld): ").strip() or "./output/meld"

        split = input("处理哪个划分? (train/dev/test/all, 默认: all): ").strip() or "all"

        extractor = MELDFeatureExtractor(
            base_dir=meld_base_dir,
            output_dir=meld_output_dir,
            config=config
        )

        extractor.process_dataset(split=split)

    print("\n" + "="*60)
    print("✓ 特征提取完成！")
    print("="*60)


if __name__ == "__main__":
    main()
