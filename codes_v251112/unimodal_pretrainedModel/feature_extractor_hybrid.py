"""
混合特征提取器 - 支持音频即时 PCA 降维
文本: MiniLM-L6 (384d)
音频: HuBERT (768d) → PCA (384d)
视频: ViT-small (384d)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
import gc
import pickle
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


class AudioPCAReducer:
    """音频特征 PCA 降维器"""

    def __init__(self, target_dim: int = 384):
        """
        初始化 PCA 降维器

        Args:
            target_dim: 目标维度
        """
        self.target_dim = target_dim
        self.pca = None
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """
        训练 PCA 模型

        Args:
            features: [num_samples, 768] 或 [num_samples, time_steps, 768]
        """
        # 如果是 3D，展平为 2D
        if len(features.shape) == 3:
            num_samples, time_steps, feat_dim = features.shape
            features = features.reshape(-1, feat_dim)

        print(f"\n训练音频 PCA 降维模型...")
        print(f"  输入: {features.shape}")
        print(f"  目标维度: {self.target_dim}")

        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(features)
        self.is_fitted = True

        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"  ✓ PCA 训练完成")
        print(f"  保留方差: {explained_variance:.2%}\n")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        应用 PCA 降维

        Args:
            features: [time_steps, 768]

        Returns:
            reduced_features: [time_steps, target_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("PCA 模型未训练，请先调用 fit() 或 load()")

        return self.pca.transform(features)

    def save(self, save_path: str):
        """保存 PCA 模型"""
        if not self.is_fitted:
            raise RuntimeError("PCA 模型未训练")

        with open(save_path, 'wb') as f:
            pickle.dump({
                'target_dim': self.target_dim,
                'pca': self.pca
            }, f)
        print(f"✓ PCA 模型已保存: {save_path}")

    def load(self, load_path: str):
        """加载 PCA 模型"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.target_dim = data['target_dim']
        self.pca = data['pca']
        self.is_fitted = True
        print(f"✓ PCA 模型已加载: {load_path}")


class HybridFeatureExtractor:
    """混合特征提取器（音频即时 PCA 降维）"""

    def __init__(self, config: Dict):
        """
        初始化提取器

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 内存管理配置
        self.max_frames = config.get('max_frames', 500)
        self.video_batch_size = config.get('video_batch_size', 32)
        self.enable_memory_cleanup = config.get('enable_memory_cleanup', True)

        print(f"\n{'='*60}")
        print(f"初始化混合特征提取器（音频 PCA 降维）")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"最大帧数限制: {self.max_frames}")
        print(f"视频批处理大小: {self.video_batch_size}")

        # 音频 PCA 降维器
        audio_config = config.get('audio', {})
        pca_config = audio_config.get('pca_reduction', {})

        if pca_config.get('enabled', False):
            target_dim = pca_config.get('target_dim', 384)
            self.audio_pca = AudioPCAReducer(target_dim=target_dim)
            self.use_audio_pca = True

            # 如果提供了预训练模型路径，加载它
            pca_model_path = pca_config.get('model_path', None)
            if pca_model_path and os.path.exists(pca_model_path):
                self.audio_pca.load(pca_model_path)
            else:
                print(f"⚠ 音频 PCA 模型未预训练，将在第一次运行时训练")
                self.audio_pca_need_training = True
        else:
            self.use_audio_pca = False
            self.audio_pca = None

        print(f"音频 PCA 降维: {'启用' if self.use_audio_pca else '禁用'}")

        # 初始化各模态提取器
        self._init_text_extractor()
        self._init_audio_extractor()
        self._init_video_extractor()

        print(f"{'='*60}\n")

    def _init_text_extractor(self):
        """初始化文本特征提取器 (MiniLM-L6)"""
        from transformers import AutoTokenizer, AutoModel

        text_config = self.config['text']
        model_path = text_config['model_path']

        print(f"✓ 加载文本模型: {model_path}")
        print(f"  输出维度: {text_config.get('output_dim', 384)}")

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.text_model.eval()

        self.text_max_length = text_config.get('max_length', 512)

    def _init_audio_extractor(self):
        """初始化音频特征提取器 (HuBERT)"""
        from transformers import AutoProcessor, AutoModel

        audio_config = self.config['audio']
        model_path = audio_config['model_path']

        print(f"✓ 加载音频模型: {model_path}")
        print(f"  原始输出维度: {audio_config.get('output_dim', 768)}")

        if self.use_audio_pca:
            pca_dim = audio_config['pca_reduction']['target_dim']
            print(f"  PCA 降维后: {pca_dim}")

        self.audio_processor = AutoProcessor.from_pretrained(model_path)
        self.audio_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.audio_model.eval()

        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.max_audio_duration = audio_config.get('max_duration', 30.0)

    def _init_video_extractor(self):
        """初始化视频特征提取器 (ViT-small)"""
        from transformers import AutoImageProcessor, AutoModel

        video_config = self.config['video']
        model_path = video_config['model_path']

        print(f"✓ 加载视频模型: {model_path}")
        print(f"  输出维度: {video_config.get('output_dim', 384)}")

        self.video_processor = AutoImageProcessor.from_pretrained(model_path)
        self.video_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.video_model.eval()

        self.video_fps = video_config.get('fps', 25)
        self.feature_mode = video_config.get('feature_mode', 'cls')

    def _cleanup_memory(self):
        """清理 GPU 内存"""
        if self.enable_memory_cleanup and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def extract_text_features(self, text: str, target_frames: int) -> torch.Tensor:
        """
        提取文本特征（MiniLM-L6，384 维）

        Args:
            text: 输入文本
            target_frames: 目标帧数

        Returns:
            text_features: [target_frames, 384]
        """
        with torch.no_grad():
            inputs = self.text_tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.text_max_length
            ).to(self.device)

            outputs = self.text_model(**inputs)
            features = outputs.last_hidden_state[0]  # [seq_len, 384]

            # 线性插值到目标帧数
            if features.shape[0] != target_frames:
                features_t = features.T.unsqueeze(0)  # [1, 384, seq_len]
                aligned = torch.nn.functional.interpolate(
                    features_t,
                    size=target_frames,
                    mode='linear',
                    align_corners=False
                )
                features = aligned.squeeze(0).T  # [target_frames, 384]

        return features.cpu()

    def extract_audio_features(self, audio_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        提取音频特征（HuBERT 768 维 → PCA 384 维）

        Args:
            audio_path: 音频文件路径

        Returns:
            audio_features: [time_steps, 384] (如果启用 PCA) 或 [time_steps, 768]
            timestamps: 时间戳数组
        """
        import librosa

        # 加载音频
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 可选：限制音频最大时长（会丢失后半段信息，不推荐）
        # 更好的方案是在 extract_multimodal_features 中对特征降采样
        # max_samples = int(self.max_audio_duration * sr)
        # if len(waveform) > max_samples:
        #     waveform = waveform[:max_samples]

        # HuBERT 推理
        with torch.no_grad():
            inputs = self.audio_processor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors='pt'
            ).to(self.device)

            outputs = self.audio_model(**inputs)
            features = outputs.last_hidden_state[0].cpu().numpy()  # [time_steps, 768]

        # 立即应用 PCA 降维
        if self.use_audio_pca:
            if not self.audio_pca.is_fitted:
                raise RuntimeError(
                    "音频 PCA 未训练。请先使用 train_audio_pca() 训练，"
                    "或在配置中提供预训练模型路径"
                )
            features = self.audio_pca.transform(features)  # [time_steps, 384]

        features = torch.from_numpy(features)

        # 计算时间戳
        num_frames = features.shape[0]
        duration = len(waveform) / sr
        timestamps = np.linspace(0, duration, num_frames)

        return features, timestamps

    def extract_video_features(self, video_path: str, timestamps: np.ndarray) -> torch.Tensor:
        """
        提取视频特征（ViT-small，384 维）

        Args:
            video_path: 视频文件路径
            timestamps: 音频时间戳（用于对齐）

        Returns:
            video_features: [num_frames, 384]
        """
        import cv2

        # 限制最大帧数
        if len(timestamps) > self.max_frames:
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
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        # 分批处理帧
        all_features = []
        num_frames = len(frames)

        for i in range(0, num_frames, self.video_batch_size):
            batch_frames = frames[i:i + self.video_batch_size]

            inputs = self.video_processor(images=batch_frames, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.video_model(**inputs)

                if self.feature_mode == 'cls':
                    batch_features = outputs.last_hidden_state[:, 0, :]  # [batch, 384]
                else:
                    batch_features = outputs.last_hidden_state.mean(dim=1)

            all_features.append(batch_features.cpu())

            del inputs, outputs, batch_features
            self._cleanup_memory()

        features = torch.cat(all_features, dim=0)  # [num_frames, 384]
        return features

    def extract_multimodal_features(
        self,
        text: str,
        audio_path: str,
        video_path: str
    ) -> Dict:
        """
        提取对齐的多模态特征（全部 384 维）

        Args:
            text: 文本内容
            audio_path: 音频文件路径
            video_path: 视频文件路径

        Returns:
            features: 包含对齐特征的字典
        """
        # 1. 提取音频特征（作为基准）
        audio_features, timestamps = self.extract_audio_features(audio_path)
        num_frames = len(timestamps)

        # 降采样音频特征（加速优化：减少整体帧数）
        # HuBERT 输出约 50 帧/秒，降采样可同时减少音频和视频处理量
        audio_downsample_factor = 10  # 降采样倍数：10=5帧/秒, 5=10帧/秒, 2=25帧/秒
        if audio_downsample_factor > 1:
            indices = np.arange(0, num_frames, audio_downsample_factor)
            audio_features = audio_features[indices]
            timestamps = timestamps[indices]
            num_frames = len(timestamps)

        # 检查帧数
        if num_frames > self.max_frames:
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            audio_features = audio_features[indices]
            timestamps = timestamps[indices]
            num_frames = self.max_frames

        # 2. 提取文本特征（对齐到音频帧数）
        text_features = self.extract_text_features(text, target_frames=num_frames)

        # 3. 提取视频特征（对齐到音频时间戳）
        video_features = self.extract_video_features(video_path, timestamps)

        return {
            'audio_features': audio_features,      # [T, 384]
            'text_features': text_features,        # [T, 384]
            'video_features': video_features,      # [T, 384]
            'num_frames': num_frames,
            'timestamps': timestamps
        }

    def train_audio_pca(self, audio_paths: list, save_path: Optional[str] = None):
        """
        训练音频 PCA 模型

        Args:
            audio_paths: 音频文件路径列表（用于训练 PCA）
            save_path: PCA 模型保存路径
        """
        if not self.use_audio_pca:
            print("⚠ 音频 PCA 未启用")
            return

        if self.audio_pca.is_fitted:
            print("⚠ 音频 PCA 已训练")
            return

        print(f"\n{'='*60}")
        print(f"训练音频 PCA 模型")
        print(f"{'='*60}")
        print(f"使用 {len(audio_paths)} 个音频文件")
        print(f"{'='*60}\n")

        import librosa
        from tqdm import tqdm

        all_features = []

        for audio_path in tqdm(audio_paths, desc="收集音频特征", ncols=80):
            try:
                # 加载音频
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate)

                # HuBERT 推理
                with torch.no_grad():
                    inputs = self.audio_processor(
                        waveform,
                        sampling_rate=self.sample_rate,
                        return_tensors='pt'
                    ).to(self.device)

                    outputs = self.audio_model(**inputs)
                    features = outputs.last_hidden_state[0].cpu().numpy()  # [time_steps, 768]

                all_features.append(features)

                # 清理
                del inputs, outputs
                self._cleanup_memory()

            except Exception as e:
                print(f"  ⚠ 跳过 {audio_path}: {str(e)}")

        # 合并所有特征
        combined_features = np.concatenate(all_features, axis=0)  # [N, 768]
        print(f"\n收集到 {combined_features.shape[0]} 帧特征")

        # 训练 PCA
        self.audio_pca.fit(combined_features)

        # 保存模型
        if save_path:
            self.audio_pca.save(save_path)

        print(f"{'='*60}\n")


def create_hybrid_extractor(config_path: str) -> HybridFeatureExtractor:
    """
    创建混合特征提取器

    Args:
        config_path: 配置文件路径

    Returns:
        extractor: 混合特征提取器
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 合并内存管理配置
    if 'memory_management' in config:
        for key, value in config['memory_management'].items():
            config[key] = value

    return HybridFeatureExtractor(config=config)
