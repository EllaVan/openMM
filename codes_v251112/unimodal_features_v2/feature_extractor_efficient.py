"""
多模态特征提取器 - 内存优化版本
支持本地模型路径 + 解决 CUDA OOM 问题
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
import gc
warnings.filterwarnings('ignore')


class MultimodalFeatureExtractor:
    """多模态特征提取器（内存优化版）"""

    def __init__(self, config: Dict):
        """
        初始化提取器

        Args:
            config: 配置字典，包含本地模型路径
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 内存管理配置
        self.max_frames = config.get('max_frames', 500)  # 最大帧数限制
        self.video_batch_size = config.get('video_batch_size', 32)  # 视频帧批处理大小
        self.enable_memory_cleanup = config.get('enable_memory_cleanup', True)

        print(f"\n{'='*60}")
        print(f"初始化多模态特征提取器（内存优化版）")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"最大帧数限制: {self.max_frames}")
        print(f"视频批处理大小: {self.video_batch_size}")
        print(f"自动内存清理: {self.enable_memory_cleanup}")

        # 初始化各模态提取器
        self._init_text_extractor()
        self._init_audio_extractor()
        self._init_video_extractor()

        print(f"{'='*60}\n")

    def _init_text_extractor(self):
        """初始化文本特征提取器 (RoBERTa)"""
        from transformers import AutoTokenizer, AutoModel

        text_config = self.config['text']
        model_path = text_config['model_path']

        print(f"✓ 加载文本模型: {model_path}")

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.text_model.eval()

        self.text_max_length = text_config.get('max_length', 512)
        print(f"  最大长度: {self.text_max_length}")

    def _init_audio_extractor(self):
        """初始化音频特征提取器 (HuBERT)"""
        from transformers import AutoProcessor, AutoModel

        audio_config = self.config['audio']
        model_path = audio_config['model_path']

        print(f"✓ 加载音频模型: {model_path}")

        self.audio_processor = AutoProcessor.from_pretrained(model_path)
        self.audio_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.audio_model.eval()

        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.max_audio_duration = audio_config.get('max_duration', 30.0)  # 最大音频时长（秒）
        print(f"  采样率: {self.sample_rate} Hz")
        print(f"  最大时长: {self.max_audio_duration}s")

    def _init_video_extractor(self):
        """初始化视频特征提取器 (ViT)"""
        from transformers import AutoImageProcessor, AutoModel

        video_config = self.config['video']
        model_path = video_config['model_path']

        print(f"✓ 加载视频模型: {model_path}")

        self.video_processor = AutoImageProcessor.from_pretrained(model_path)
        self.video_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.video_model.eval()

        self.video_fps = video_config.get('fps', 25)
        self.feature_mode = video_config.get('feature_mode', 'cls')
        print(f"  帧率: {self.video_fps} fps, 特征模式: {self.feature_mode}")

    def _cleanup_memory(self):
        """清理 GPU 内存"""
        if self.enable_memory_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def extract_text_features(self, text: str, target_frames: Optional[int] = None) -> torch.Tensor:
        """
        提取文本特征

        Args:
            text: 输入文本
            target_frames: 目标帧数（用于对齐）

        Returns:
            text_features: [num_frames, hidden_dim]
        """
        # Tokenize
        inputs = self.text_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.text_max_length
        ).to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            features = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

        # 移到 CPU 并清理
        features = features.cpu()
        self._cleanup_memory()

        # 对齐到目标帧数
        if target_frames is not None:
            features = self._align_features(features, target_frames)

        return features

    def extract_audio_features(self, audio_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        提取音频特征（带时长限制）

        Args:
            audio_path: 音频文件路径

        Returns:
            audio_features: [num_frames, hidden_dim]
            timestamps: 时间戳数组 [num_frames]
        """
        import librosa

        # 加载音频（限制最大时长）
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_audio_duration)

        # 如果音频为空
        if len(waveform) == 0:
            raise ValueError(f"音频文件为空: {audio_path}")

        # 处理音频
        inputs = self.audio_processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding=True
        ).to(self.device)

        # 提取特征
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state[0]  # [time_steps, hidden_dim]

        # 移到 CPU 并清理
        features = features.cpu()
        self._cleanup_memory()

        # 计算时间戳
        num_frames = features.shape[0]
        duration = len(waveform) / sr
        timestamps = np.linspace(0, duration, num_frames)

        return features, timestamps

    def extract_video_features(self, video_path: str, timestamps: np.ndarray) -> torch.Tensor:
        """
        提取视频特征（分批处理，内存优化）

        Args:
            video_path: 视频文件路径
            timestamps: 音频时间戳（用于对齐）

        Returns:
            video_features: [num_frames, hidden_dim]
        """
        import cv2

        # 限制最大帧数
        if len(timestamps) > self.max_frames:
            # 下采样时间戳
            indices = np.linspace(0, len(timestamps) - 1, self.max_frames, dtype=int)
            timestamps = timestamps[indices]

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise ValueError(f"视频文件无法读取或为空: {video_path}")

        # 提取关键帧
        frames = []
        for timestamp in timestamps:
            frame_idx = int(timestamp * fps)
            frame_idx = min(frame_idx, total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 使用黑帧
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        # 分批处理帧（关键优化！）
        all_features = []
        num_frames = len(frames)

        for i in range(0, num_frames, self.video_batch_size):
            batch_frames = frames[i:i + self.video_batch_size]

            # 处理当前批次
            inputs = self.video_processor(images=batch_frames, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.video_model(**inputs)

                if self.feature_mode == 'cls':
                    batch_features = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
                else:
                    batch_features = outputs.last_hidden_state.mean(dim=1)

            # 移到 CPU
            all_features.append(batch_features.cpu())

            # 清理当前批次的 GPU 内存
            del inputs, outputs, batch_features
            self._cleanup_memory()

        # 拼接所有批次
        features = torch.cat(all_features, dim=0)  # [num_frames, hidden_dim]

        return features

    def _align_features(self, features: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        将特征对齐到目标帧数（线性插值）

        Args:
            features: [current_frames, hidden_dim]
            target_frames: 目标帧数

        Returns:
            aligned_features: [target_frames, hidden_dim]
        """
        current_frames = features.shape[0]

        if current_frames == target_frames:
            return features

        # 使用线性插值
        features = features.unsqueeze(0).unsqueeze(0)  # [1, 1, current_frames, hidden_dim]
        aligned = torch.nn.functional.interpolate(
            features.permute(0, 3, 1, 2),  # [1, hidden_dim, 1, current_frames]
            size=(1, target_frames),
            mode='bilinear',
            align_corners=False
        )
        aligned = aligned.permute(0, 2, 3, 1).squeeze(0).squeeze(0)  # [target_frames, hidden_dim]

        return aligned

    def extract_multimodal_features(
        self,
        text: str,
        audio_path: str,
        video_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        提取多模态特征（自动对齐 + 内存优化）

        Args:
            text: 文本内容
            audio_path: 音频文件路径
            video_path: 视频文件路径

        Returns:
            features: 包含对齐后的特征字典
        """
        try:
            # 1. 提取音频特征（作为对齐基准）
            audio_features, timestamps = self.extract_audio_features(audio_path)
            num_frames = len(timestamps)

            # 检查帧数
            if num_frames > self.max_frames:
                print(f"  ⚠ 警告: 帧数 {num_frames} 超过限制 {self.max_frames}，将下采样")
                indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
                audio_features = audio_features[indices]
                timestamps = timestamps[indices]
                num_frames = self.max_frames

            # 2. 提取文本特征（对齐到音频帧数）
            text_features = self.extract_text_features(text, target_frames=num_frames)

            # 3. 提取视频特征（根据音频时间戳）
            video_features = self.extract_video_features(video_path, timestamps)

            return {
                'audio_features': audio_features,
                'text_features': text_features,
                'video_features': video_features,
                'num_frames': num_frames
            }

        except Exception as e:
            # 发生错误时清理内存
            self._cleanup_memory()
            raise e

    def extract_from_files(
        self,
        text_file: str,
        audio_file: str,
        video_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        从文件提取特征

        Args:
            text_file: 文本文件路径
            audio_file: 音频文件路径
            video_file: 视频文件路径
            output_file: 可选的输出文件路径（.pkl）

        Returns:
            features: 特征字典
        """
        # 读取文本
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # 提取特征
        features = self.extract_multimodal_features(text, audio_file, video_file)

        # 保存（如果指定）
        if output_file:
            import pickle
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump(features, f)
            print(f"✓ 特征已保存: {output_file}")

        return features


def load_config(config_file: str) -> Dict:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 添加默认内存管理配置
    if 'max_frames' not in config:
        config['max_frames'] = 500
    if 'video_batch_size' not in config:
        config['video_batch_size'] = 32
    if 'enable_memory_cleanup' not in config:
        config['enable_memory_cleanup'] = True

    print(f"\n{'='*60}")
    print(f"配置加载成功")
    print(f"{'='*60}")
    print(f"文本模型: {config['text']['model_path']}")
    print(f"音频模型: {config['audio']['model_path']}")
    print(f"视频模型: {config['video']['model_path']}")
    print(f"{'='*60}\n")

    return config


if __name__ == "__main__":
    # 示例用法
    config = load_config('unimodal_features/config.json')

    extractor = MultimodalFeatureExtractor(config=config)

    # 提取特征
    features = extractor.extract_multimodal_features(
        text="This is a test sentence.",
        audio_path="example_data/sample.wav",
        video_path="example_data/sample.mp4"
    )

    print(f"音频特征: {features['audio_features'].shape}")
    print(f"文本特征: {features['text_features'].shape}")
    print(f"视频特征: {features['video_features'].shape}")
    print(f"总帧数: {features['num_frames']}")
