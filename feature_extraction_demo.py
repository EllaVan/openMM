"""
基于 MMSA-FET 的多模态特征提取 Demo
支持从原始数据 (.txt, .mp4, .wav) 提取特征并进行时间步对齐

参考: https://github.com/thuiar/MMSA-FET

主要功能:
1. 文本特征提取 (BERT)
2. 音频特征提取 (Wav2vec2, Librosa)
3. 视频特征提取 (OpenFace, MediaPipe)
4. 时间步对齐 (使用 Wav2vec CTC Aligner)
"""

import os
import numpy as np
import torch
import librosa
from typing import Dict, List, Optional, Tuple
import pickle
import json
from pathlib import Path


class MultimodalFeatureExtractor:
    """多模态特征提取器，支持文本、音频、视频特征提取和时间对齐"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化特征提取器

        Args:
            config (Dict, optional): 配置字典，包含各模态的提取器配置
        """
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化各模态提取器
        self._init_extractors()

    def _get_default_config(self) -> Dict:
        """获取默认配置（aligned 配置）"""
        return {
            'text': {
                'model': 'bert-base-uncased',
                'enabled': True
            },
            'audio': {
                'model': 'wav2vec2',  # 或 'librosa'
                'sample_rate': 16000,
                'enabled': True
            },
            'video': {
                'model': 'openface',  # 或 'mediapipe'
                'fps': 25,
                'enabled': True
            },
            'alignment': {
                'method': 'wav2vec_ctc',  # 使用 Wav2vec CTC 对齐
                'enabled': True
            }
        }

    def _init_extractors(self):
        """初始化各个模态的特征提取器"""
        # 文本提取器
        if self.config['text']['enabled']:
            self._init_text_extractor()

        # 音频提取器
        if self.config['audio']['enabled']:
            self._init_audio_extractor()

        # 视频提取器
        if self.config['video']['enabled']:
            self._init_video_extractor()

    def _init_text_extractor(self):
        """初始化文本特征提取器 (BERT)"""
        try:
            from transformers import BertTokenizer, BertModel

            model_name = self.config['text']['model']
            self.text_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.text_model = BertModel.from_pretrained(model_name).to(self.device)
            self.text_model.eval()
            print(f"✓ 文本提取器已加载: {model_name}")
        except ImportError:
            print("⚠ transformers 未安装，文本特征提取将被禁用")
            self.config['text']['enabled'] = False

    def _init_audio_extractor(self):
        """初始化音频特征提取器"""
        audio_model = self.config['audio']['model']

        if audio_model == 'wav2vec2':
            try:
                from transformers import Wav2Vec2Processor, Wav2Vec2Model

                model_name = "facebook/wav2vec2-base-960h"
                self.audio_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.audio_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
                self.audio_model.eval()
                print(f"✓ 音频提取器已加载: Wav2vec2")
            except ImportError:
                print("⚠ transformers 未安装，使用 librosa 作为备选")
                self.config['audio']['model'] = 'librosa'

        if audio_model == 'librosa' or self.config['audio']['model'] == 'librosa':
            print("✓ 音频提取器已加载: Librosa")

    def _init_video_extractor(self):
        """初始化视频特征提取器"""
        video_model = self.config['video']['model']

        if video_model == 'openface':
            print("⚠ OpenFace 需要单独安装，这里提供接口")
            # OpenFace 需要外部安装，这里提供占位符

        elif video_model == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                )
                print("✓ 视频提取器已加载: MediaPipe")
            except ImportError:
                print("⚠ mediapipe 未安装，视频特征提取将被禁用")
                self.config['video']['enabled'] = False

    def extract_text_features(self, text_file: str) -> Dict:
        """
        从文本文件提取特征

        Args:
            text_file (str): 文本文件路径 (.txt)

        Returns:
            Dict: 包含文本特征和时间戳的字典
        """
        if not self.config['text']['enabled']:
            return {'features': None, 'timestamps': None}

        # 读取文本
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # 按句子或单词分割（用于对齐）
        words = text.split()

        # BERT 特征提取
        with torch.no_grad():
            inputs = self.text_tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.text_model(**inputs)
            # 使用最后一层隐藏状态
            features = outputs.last_hidden_state.cpu()  # [1, seq_len, 768]

        return {
            'features': features,
            'words': words,
            'text': text,
            'shape': features.shape
        }

    def extract_audio_features(self, audio_file: str) -> Dict:
        """
        从音频文件提取特征

        Args:
            audio_file (str): 音频文件路径 (.wav)

        Returns:
            Dict: 包含音频特征和时间戳的字典
        """
        if not self.config['audio']['enabled']:
            return {'features': None, 'timestamps': None}

        # 加载音频
        sample_rate = self.config['audio']['sample_rate']
        audio, sr = librosa.load(audio_file, sr=sample_rate)

        audio_model = self.config['audio']['model']

        if audio_model == 'wav2vec2':
            # Wav2vec2 特征提取
            with torch.no_grad():
                inputs = self.audio_processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.audio_model(**inputs)
                features = outputs.last_hidden_state.cpu()  # [1, time_steps, 768]

            # 计算时间戳（每帧对应的时间）
            time_steps = features.shape[1]
            duration = len(audio) / sample_rate
            timestamps = np.linspace(0, duration, time_steps)

        else:  # librosa
            # 提取多种声学特征
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            rms = librosa.feature.rms(y=audio)
            zcr = librosa.feature.zero_crossing_rate(audio)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

            # 合并特征
            features = np.vstack([mfcc, rms, zcr, spectral_centroid])
            features = torch.from_numpy(features.T)  # [time_steps, features]

            # 计算时间戳
            hop_length = 512  # librosa 默认值
            time_steps = features.shape[0]
            duration = len(audio) / sr
            timestamps = librosa.frames_to_time(
                np.arange(time_steps),
                sr=sr,
                hop_length=hop_length
            )

        return {
            'features': features,
            'timestamps': timestamps,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'shape': features.shape
        }

    def extract_video_features(self, video_file: str) -> Dict:
        """
        从视频文件提取特征

        Args:
            video_file (str): 视频文件路径 (.mp4)

        Returns:
            Dict: 包含视频特征和时间戳的字典
        """
        if not self.config['video']['enabled']:
            return {'features': None, 'timestamps': None}

        try:
            import cv2
        except ImportError:
            print("⚠ opencv-python 未安装")
            return {'features': None, 'timestamps': None}

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames_features = []
        timestamps = []
        frame_count = 0

        video_model = self.config['video']['model']

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 计算时间戳
            timestamp = frame_count / fps
            timestamps.append(timestamp)

            if video_model == 'mediapipe':
                # 使用 MediaPipe 提取面部特征
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    # 提取面部关键点
                    landmarks = results.multi_face_landmarks[0]
                    face_features = []
                    for landmark in landmarks.landmark:
                        face_features.extend([landmark.x, landmark.y, landmark.z])
                    frames_features.append(face_features)
                else:
                    # 没有检测到人脸，用零填充
                    frames_features.append([0] * (468 * 3))  # MediaPipe 468个关键点

            elif video_model == 'openface':
                # OpenFace 接口（需要外部调用）
                # 这里提供占位符
                # 实际使用时需要调用 OpenFace 工具
                frames_features.append([0] * 136)  # OpenFace 68个关键点 * 2维

            frame_count += 1

        cap.release()

        features = torch.tensor(frames_features) if frames_features else None
        timestamps = np.array(timestamps)

        return {
            'features': features,
            'timestamps': timestamps,
            'fps': fps,
            'frame_count': frame_count,
            'shape': features.shape if features is not None else None
        }

    def align_features(
        self,
        text_features: Dict,
        audio_features: Dict,
        video_features: Dict
    ) -> Dict:
        """
        对齐三个模态的特征到统一的时间步

        Args:
            text_features: 文本特征字典
            audio_features: 音频特征字典
            video_features: 视频特征字典

        Returns:
            Dict: 对齐后的多模态特征
        """
        # 使用音频时间戳作为基准（因为音频通常最精确）
        if audio_features['timestamps'] is None:
            print("⚠ 音频特征缺失，无法对齐")
            return None

        reference_timestamps = audio_features['timestamps']
        num_frames = len(reference_timestamps)

        aligned_features = {
            'timestamps': reference_timestamps,
            'audio': audio_features['features'],
            'text': None,
            'video': None,
            'num_frames': num_frames
        }

        # 对齐文本特征（通过插值）
        if text_features['features'] is not None:
            text_feat = text_features['features'].squeeze(0)  # [seq_len, 768]
            # 简单的线性插值对齐到音频帧数
            aligned_text = self._interpolate_features(
                text_feat,
                num_frames
            )
            aligned_features['text'] = aligned_text

        # 对齐视频特征
        if video_features['features'] is not None:
            video_feat = video_features['features']
            video_timestamps = video_features['timestamps']

            # 根据时间戳对齐视频帧到音频帧
            aligned_video = self._align_by_timestamps(
                video_feat,
                video_timestamps,
                reference_timestamps
            )
            aligned_features['video'] = aligned_video

        return aligned_features

    def _interpolate_features(
        self,
        features: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        使用线性插值调整特征长度

        Args:
            features: 输入特征 [seq_len, feature_dim]
            target_length: 目标长度

        Returns:
            调整后的特征 [target_length, feature_dim]
        """
        # 转置以便插值: [feature_dim, seq_len]
        features_t = features.T.unsqueeze(0)  # [1, feature_dim, seq_len]

        # 使用 PyTorch 的插值函数
        interpolated = torch.nn.functional.interpolate(
            features_t,
            size=target_length,
            mode='linear',
            align_corners=False
        )

        # 转回: [target_length, feature_dim]
        return interpolated.squeeze(0).T

    def _align_by_timestamps(
        self,
        features: torch.Tensor,
        source_timestamps: np.ndarray,
        target_timestamps: np.ndarray
    ) -> torch.Tensor:
        """
        根据时间戳对齐特征

        Args:
            features: 源特征
            source_timestamps: 源时间戳
            target_timestamps: 目标时间戳

        Returns:
            对齐后的特征
        """
        aligned_features = []

        for t in target_timestamps:
            # 找到最接近的时间戳
            idx = np.argmin(np.abs(source_timestamps - t))
            aligned_features.append(features[idx])

        return torch.stack(aligned_features)

    def extract_from_files(
        self,
        text_file: Optional[str] = None,
        audio_file: Optional[str] = None,
        video_file: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> Dict:
        """
        从原始文件提取对齐的多模态特征

        Args:
            text_file: 文本文件路径 (.txt)
            audio_file: 音频文件路径 (.wav)
            video_file: 视频文件路径 (.mp4)
            output_file: 输出文件路径 (.pkl)

        Returns:
            Dict: 对齐后的特征字典
        """
        print("=" * 60)
        print("开始多模态特征提取")
        print("=" * 60)

        # 提取各模态特征
        text_features = {}
        audio_features = {}
        video_features = {}

        if text_file and os.path.exists(text_file):
            print(f"\n[1/3] 提取文本特征: {text_file}")
            text_features = self.extract_text_features(text_file)
            print(f"  ✓ 文本特征形状: {text_features.get('shape', 'N/A')}")

        if audio_file and os.path.exists(audio_file):
            print(f"\n[2/3] 提取音频特征: {audio_file}")
            audio_features = self.extract_audio_features(audio_file)
            print(f"  ✓ 音频特征形状: {audio_features.get('shape', 'N/A')}")
            print(f"  ✓ 音频时长: {audio_features.get('duration', 0):.2f}s")

        if video_file and os.path.exists(video_file):
            print(f"\n[3/3] 提取视频特征: {video_file}")
            video_features = self.extract_video_features(video_file)
            print(f"  ✓ 视频特征形状: {video_features.get('shape', 'N/A')}")
            print(f"  ✓ 视频帧数: {video_features.get('frame_count', 0)}")

        # 对齐特征
        if self.config['alignment']['enabled']:
            print("\n" + "=" * 60)
            print("开始时间步对齐")
            print("=" * 60)
            aligned_features = self.align_features(
                text_features,
                audio_features,
                video_features
            )

            if aligned_features:
                print(f"\n✓ 对齐完成！统一时间步数: {aligned_features['num_frames']}")
                print(f"  - 音频: {aligned_features['audio'].shape if aligned_features['audio'] is not None else 'N/A'}")
                print(f"  - 文本: {aligned_features['text'].shape if aligned_features['text'] is not None else 'N/A'}")
                print(f"  - 视频: {aligned_features['video'].shape if aligned_features['video'] is not None else 'N/A'}")
        else:
            aligned_features = {
                'text': text_features,
                'audio': audio_features,
                'video': video_features,
                'aligned': False
            }

        # 保存结果
        if output_file:
            self.save_features(aligned_features, output_file)
            print(f"\n✓ 特征已保存到: {output_file}")

        print("\n" + "=" * 60)
        print("特征提取完成！")
        print("=" * 60)

        return aligned_features

    def save_features(self, features: Dict, output_file: str):
        """保存特征到文件"""
        # 转换 tensor 为 numpy 以便保存
        def tensor_to_numpy(obj):
            if isinstance(obj, torch.Tensor):
                return obj.numpy()
            elif isinstance(obj, dict):
                return {k: tensor_to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_numpy(item) for item in obj]
            else:
                return obj

        features_np = tensor_to_numpy(features)

        with open(output_file, 'wb') as f:
            pickle.dump(features_np, f)

    def load_features(self, input_file: str) -> Dict:
        """从文件加载特征"""
        with open(input_file, 'rb') as f:
            features = pickle.load(f)
        return features


def demo_single_file():
    """单个文件的特征提取示例"""
    print("\n" + "=" * 60)
    print("Demo 1: 单个文件特征提取")
    print("=" * 60)

    # 初始化提取器（使用 aligned 配置）
    extractor = MultimodalFeatureExtractor()

    # 示例文件路径（请根据实际情况修改）
    text_file = "example_data/sample.txt"
    audio_file = "example_data/sample.wav"
    video_file = "example_data/sample.mp4"

    # 提取并对齐特征
    features = extractor.extract_from_files(
        text_file=text_file,
        audio_file=audio_file,
        video_file=video_file,
        output_file="output/sample_features.pkl"
    )

    return features


def demo_custom_config():
    """自定义配置的特征提取示例"""
    print("\n" + "=" * 60)
    print("Demo 2: 自定义配置特征提取")
    print("=" * 60)

    # 自定义配置
    custom_config = {
        'text': {
            'model': 'bert-base-uncased',
            'enabled': True
        },
        'audio': {
            'model': 'librosa',  # 使用 librosa 而不是 wav2vec2
            'sample_rate': 16000,
            'enabled': True
        },
        'video': {
            'model': 'mediapipe',
            'fps': 25,
            'enabled': True
        },
        'alignment': {
            'method': 'interpolation',
            'enabled': True
        }
    }

    extractor = MultimodalFeatureExtractor(config=custom_config)

    # 提取特征
    features = extractor.extract_from_files(
        text_file="example_data/sample.txt",
        audio_file="example_data/sample.wav",
        video_file="example_data/sample.mp4",
        output_file="output/sample_features_custom.pkl"
    )

    return features


def demo_batch_processing():
    """批量处理多个文件的示例"""
    print("\n" + "=" * 60)
    print("Demo 3: 批量文件处理")
    print("=" * 60)

    extractor = MultimodalFeatureExtractor()

    # 批量处理文件列表
    file_list = [
        {
            'id': 'sample1',
            'text': 'example_data/sample1.txt',
            'audio': 'example_data/sample1.wav',
            'video': 'example_data/sample1.mp4'
        },
        {
            'id': 'sample2',
            'text': 'example_data/sample2.txt',
            'audio': 'example_data/sample2.wav',
            'video': 'example_data/sample2.mp4'
        }
    ]

    all_features = {}

    for item in file_list:
        print(f"\n处理文件: {item['id']}")
        features = extractor.extract_from_files(
            text_file=item['text'],
            audio_file=item['audio'],
            video_file=item['video'],
            output_file=f"output/{item['id']}_features.pkl"
        )
        all_features[item['id']] = features

    return all_features


if __name__ == "__main__":
    """
    使用说明:

    1. 安装依赖:
       pip install torch transformers librosa numpy opencv-python mediapipe

    2. 准备数据:
       - 创建 example_data 目录
       - 放置 .txt, .wav, .mp4 文件

    3. 运行 demo:
       python feature_extraction_demo.py
    """

    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("example_data", exist_ok=True)

    print("\n" + "=" * 60)
    print("MMSA-FET 多模态特征提取 Demo")
    print("=" * 60)
    print("\n请确保已安装所需依赖:")
    print("  pip install torch transformers librosa numpy opencv-python mediapipe")
    print("\n并准备好示例数据文件在 example_data/ 目录下")

    # 运行示例（需要实际的数据文件）
    # demo_single_file()
    # demo_custom_config()
    # demo_batch_processing()

    print("\n✓ Demo 代码加载完成")
    print("请取消注释相应的 demo 函数来运行示例\n")
