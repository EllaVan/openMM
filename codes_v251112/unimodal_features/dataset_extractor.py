"""
数据集特征提取器 - MOSEI 和 MELD
"""

import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unimodal_features.feature_extractor import MultimodalFeatureExtractor


class DatasetFeatureExtractor:
    """数据集特征提取器基类"""

    def __init__(self, base_dir: str, output_dir: str, config: Dict):
        """
        初始化提取器

        Args:
            base_dir: 数据集根目录
            output_dir: 输出目录
            config: 特征提取配置
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.config = config

        os.makedirs(output_dir, exist_ok=True)

        # 初始化特征提取器
        self.extractor = MultimodalFeatureExtractor(config=config)

        # 情感标签映射
        self.emotion_mapping = {
            'happy': 0, 'happiness': 0,
            'sad': 1, 'sadness': 1,
            'anger': 2,
            'disgust': 3,
            'surprise': 4,
            'fear': 5,
            'neutral': 6
        }

    def extract_single_sample(
        self,
        text: str,
        audio_path: str,
        video_path: str,
        sample_id: str
    ) -> Dict:
        """提取单个样本的特征"""
        try:
            features = self.extractor.extract_multimodal_features(text, audio_path, video_path)
            return features
        except Exception as e:
            print(f"  ⚠ 样本 {sample_id} 提取失败: {str(e)}")
            return None

    def save_features(self, samples: List[Dict], emotion: str, label_id: int):
        """保存特征到文件"""
        filename = f"{self.dataset_name}{emotion}label{label_id}.pkl"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(samples, f)

        print(f"✓ 已保存 {len(samples)} 个样本到 {filename}")


class MOSEIFeatureExtractor(DatasetFeatureExtractor):
    """MOSEI 数据集特征提取器"""

    def __init__(self, base_dir: str, output_dir: str, label_file: str, config: Dict):
        """
        初始化 MOSEI 提取器

        Args:
            base_dir: MOSEI 根目录
            output_dir: 输出目录
            label_file: 标签文件路径
            config: 特征提取配置
        """
        super().__init__(base_dir, output_dir, config)
        self.dataset_name = "MOSEI"
        self.label_file = label_file

        # 数据目录
        self.audio_dir = os.path.join(base_dir, 'audio')
        self.video_dir = os.path.join(base_dir, 'video')

    def process_dataset(self) -> Dict:
        """
        处理整个 MOSEI 数据集

        Returns:
            stats: 处理统计信息
        """
        print(f"\n{'='*60}")
        print(f"开始处理 MOSEI 数据集")
        print(f"{'='*60}")
        print(f"标签文件: {self.label_file}")
        print(f"音频目录: {self.audio_dir}")
        print(f"视频目录: {self.video_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")

        # 读取标签文件
        df = pd.read_csv(self.label_file)
        print(f"总样本数: {len(df)}")

        # 按情感分组
        emotion_data = {emotion: [] for emotion in self.emotion_mapping.keys()}

        # 处理每个样本
        for index, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
            video_id = row['video_id']
            clip_id = str(row['clip_id'])
            text = row['text']
            emotion = row['emotion'].lower()

            # 文件路径
            audio_path = os.path.join(self.audio_dir, video_id, f"{clip_id}.wav")
            video_path = os.path.join(self.video_dir, video_id, f"{clip_id}.mp4")

            # 检查文件是否存在
            if not os.path.exists(audio_path) or not os.path.exists(video_path):
                continue

            # 提取特征
            sample_id = f"{video_id}_{clip_id}"
            features = self.extract_single_sample(text, audio_path, video_path, sample_id)

            if features is None:
                continue

            # 获取标签
            if emotion not in self.emotion_mapping:
                continue

            label_id = self.emotion_mapping[emotion]

            # 保存数据
            sample_data = {
                'audio_features': features['audio_features'],
                'text_features': features['text_features'],
                'video_features': features['video_features'],
                'label': label_id,
                'emotion': emotion,
                'sample_id': sample_id,
                'num_frames': features['num_frames']
            }

            emotion_data[emotion].append(sample_data)

        # 保存每个情感的特征
        stats = {}
        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                label_id = self.emotion_mapping[emotion]
                self.save_features(samples, emotion, label_id)
                stats[emotion] = len(samples)

        print(f"\n{'='*60}")
        print(f"MOSEI 数据集处理完成")
        print(f"{'='*60}")
        for emotion, count in stats.items():
            print(f"{emotion}: {count} 样本")
        print(f"{'='*60}\n")

        return stats


class MELDFeatureExtractor(DatasetFeatureExtractor):
    """MELD 数据集特征提取器"""

    def __init__(self, base_dir: str, output_dir: str, config: Dict):
        """
        初始化 MELD 提取器

        Args:
            base_dir: MELD 根目录
            output_dir: 输出目录
            config: 特征提取配置
        """
        super().__init__(base_dir, output_dir, config)
        self.dataset_name = "MELD"

    def process_split(self, split: str) -> Dict:
        """
        处理单个数据集划分

        Args:
            split: 'train', 'dev', 或 'test'

        Returns:
            stats: 处理统计信息
        """
        print(f"\n{'='*60}")
        print(f"处理 MELD {split} 集")
        print(f"{'='*60}")

        split_dir = os.path.join(self.base_dir, split)
        audio_dir = os.path.join(split_dir, 'audio')
        video_dir = os.path.join(split_dir, 'video')
        label_file = os.path.join(split_dir, 'label.csv')

        print(f"标签文件: {label_file}")
        print(f"音频目录: {audio_dir}")
        print(f"视频目录: {video_dir}")

        # 读取标签文件
        df = pd.read_csv(label_file)
        print(f"总样本数: {len(df)}")

        # 按情感分组
        emotion_data = {emotion: [] for emotion in self.emotion_mapping.keys()}

        # 处理每个样本
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"提取 {split} 特征"):
            file_id = row['file_id']  # 例如: dia0_utt0
            text = row['text']
            emotion = row['emotion'].lower()

            # 文件路径（直接在 audio/video 目录下）
            audio_path = os.path.join(audio_dir, f"{file_id}.wav")
            video_path = os.path.join(video_dir, f"{file_id}.mp4")

            # 检查文件是否存在
            if not os.path.exists(audio_path) or not os.path.exists(video_path):
                continue

            # 提取特征
            features = self.extract_single_sample(text, audio_path, video_path, file_id)

            if features is None:
                continue

            # 获取标签
            if emotion not in self.emotion_mapping:
                continue

            label_id = self.emotion_mapping[emotion]

            # 保存数据
            sample_data = {
                'audio_features': features['audio_features'],
                'text_features': features['text_features'],
                'video_features': features['video_features'],
                'label': label_id,
                'emotion': emotion,
                'sample_id': f"{split}_{file_id}",
                'num_frames': features['num_frames']
            }

            emotion_data[emotion].append(sample_data)

        # 保存每个情感的特征
        stats = {}
        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                label_id = self.emotion_mapping[emotion]
                # 文件名包含 split
                filename = f"MELD_{split}{emotion}label{label_id}.pkl"
                filepath = os.path.join(self.output_dir, filename)

                with open(filepath, 'wb') as f:
                    pickle.dump(samples, f)

                print(f"✓ 已保存 {len(samples)} 个样本到 {filename}")
                stats[emotion] = len(samples)

        print(f"{'='*60}")
        for emotion, count in stats.items():
            print(f"{emotion}: {count} 样本")
        print(f"{'='*60}\n")

        return stats

    def process_dataset(self, split: str = 'all') -> Dict:
        """
        处理 MELD 数据集

        Args:
            split: 'train', 'dev', 'test', 或 'all'

        Returns:
            stats: 处理统计信息
        """
        all_stats = {}

        if split == 'all':
            splits = ['train', 'dev', 'test']
        else:
            splits = [split]

        for s in splits:
            stats = self.process_split(s)
            all_stats[s] = stats

        print(f"\n{'='*60}")
        print(f"MELD 数据集处理完成")
        print(f"{'='*60}\n")

        return all_stats


if __name__ == "__main__":
    import json

    # 加载配置
    with open('unimodal_features/config.json', 'r') as f:
        config = json.load(f)

    # 示例: 提取 MOSEI
    mosei_extractor = MOSEIFeatureExtractor(
        base_dir='/path/to/MOSEI',
        output_dir='./output/mosei_features',
        label_file='/path/to/MOSEI/label/label.csv',
        config=config
    )
    mosei_extractor.process_dataset()

    # 示例: 提取 MELD
    meld_extractor = MELDFeatureExtractor(
        base_dir='/path/to/MELD',
        output_dir='./output/meld_features',
        config=config
    )
    meld_extractor.process_dataset(split='all')
