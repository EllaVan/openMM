"""
Utterance-Level 多模态特征提取器
使用 RoBERTa/HuBERT/ViT-Base 提取固定维度特征
"""

import os
import gc
import json
import torch
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class UtteranceFeatureExtractor:
    """
    Utterance-Level 特征提取器
    - 文本: RoBERTa [CLS] token → [768]
    - 音频: HuBERT → mean pooling → [768]
    - 视频: ViT-Base → mean pooling → [768]
    """

    def __init__(self, config: Dict):
        """
        初始化提取器

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 视频处理配置
        extraction_config = config.get('extraction', {})
        self.max_frames = extraction_config.get('max_frames', 300)
        self.video_batch_size = extraction_config.get('video_batch_size', 64)
        self.video_sample_rate = extraction_config.get('video_sample_rate_fps', 5)

        print(f"\n{'='*60}")
        print(f"初始化 Utterance-Level 特征提取器")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"视频采样: {self.video_sample_rate} fps")
        print(f"最大帧数: {self.max_frames}")

        # 初始化各模态提取器
        self._init_text_extractor()
        self._init_audio_extractor()
        self._init_video_extractor()
        self._init_openface()

        print(f"{'='*60}\n")

    def _init_text_extractor(self):
        """初始化文本特征提取器 (RoBERTa)"""
        from transformers import AutoTokenizer, AutoModel

        text_config = self.config['models']['text']
        model_path = text_config.get('model_path', '')
        model_name = text_config.get('model_name', 'roberta-base')

        # 优先使用本地路径，如果不存在则使用HuggingFace model_name
        if model_path and os.path.exists(model_path):
            model_source = model_path
            print(f"✓ 加载文本模型（本地）: {model_path}")
        else:
            model_source = model_name
            if model_path:
                print(f"⚠ 本地路径不存在: {model_path}")
            print(f"✓ 从 HuggingFace 加载文本模型: {model_name}")

        print(f"  输出维度: {text_config.get('output_dim', 768)}")

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.text_model = AutoModel.from_pretrained(model_source).to(self.device)
        self.text_model.eval()

        self.text_max_length = text_config.get('max_length', 512)

    def _init_audio_extractor(self):
        """初始化音频特征提取器 (HuBERT)"""
        from transformers import AutoProcessor, AutoModel

        audio_config = self.config['models']['audio']
        model_path = audio_config.get('model_path', '')
        model_name = audio_config.get('model_name', 'facebook/hubert-base-ls960')

        # 优先使用本地路径
        if model_path and os.path.exists(model_path):
            model_source = model_path
            print(f"✓ 加载音频模型（本地）: {model_path}")
        else:
            model_source = model_name
            if model_path:
                print(f"⚠ 本地路径不存在: {model_path}")
            print(f"✓ 从 HuggingFace 加载音频模型: {model_name}")

        print(f"  输出维度: {audio_config.get('output_dim', 768)}")

        self.audio_processor = AutoProcessor.from_pretrained(model_source)
        self.audio_model = AutoModel.from_pretrained(model_source).to(self.device)
        self.audio_model.eval()

        self.sample_rate = audio_config.get('sample_rate', 16000)

    def _init_video_extractor(self):
        """初始化视频特征提取器 (ViT-Base)"""
        from transformers import AutoImageProcessor, AutoModel

        video_config = self.config['models']['video']
        model_path = video_config.get('model_path', '')
        model_name = video_config.get('model_name', 'google/vit-base-patch16-224')

        # 优先使用本地路径
        if model_path and os.path.exists(model_path):
            model_source = model_path
            print(f"✓ 加载视频模型（本地）: {model_path}")
        else:
            model_source = model_name
            if model_path:
                print(f"⚠ 本地路径不存在: {model_path}")
            print(f"✓ 从 HuggingFace 加载视频模型: {model_name}")

        print(f"  输出维度: {video_config.get('output_dim', 768)}")

        self.video_processor = AutoImageProcessor.from_pretrained(model_source)
        self.video_model = AutoModel.from_pretrained(model_source).to(self.device)
        self.video_model.eval()

    def _init_openface(self):
        """初始化 OpenFace AU 特征提取器配置"""
        openface_config = self.config['models']['openface']

        self.openface_executable = openface_config.get('executable_path', '')
        self.openface_output_dir = openface_config.get('output_dir', './temp_openface_output')

        # 从 au_emo_prior.json 加载 AU 名称列表
        au_prior_path = openface_config.get('au_prior_path', '../materials/au_emo_prior.json')
        au_prior_full_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            au_prior_path
        )

        if os.path.exists(au_prior_full_path):
            with open(au_prior_full_path, 'r', encoding='utf-8') as f:
                au_prior_data = json.load(f)
                self.au_names = au_prior_data.get('au_names', [])
                print(f"✓ 加载 OpenFace 配置")
                print(f"  可执行文件: {self.openface_executable}")
                print(f"  AU 数量: {len(self.au_names)}")
                print(f"  输出维度: {openface_config.get('output_dim', 20)}")
        else:
            print(f"⚠ AU prior 文件不存在: {au_prior_full_path}")
            self.au_names = []

        # 创建临时输出目录
        os.makedirs(self.openface_output_dir, exist_ok=True)

    def _cleanup_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def extract_text_features(self, text: str) -> torch.Tensor:
        """
        提取文本特征（RoBERTa，768 维）- Utterance级别

        Args:
            text: 输入文本

        Returns:
            text_features: [768] - 使用 [CLS] token 作为句子表示
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
            # 使用 [CLS] token (第一个位置) 作为句子级别表示
            features = outputs.last_hidden_state[0, 0, :]  # [768]

        return features.cpu()

    def extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """
        提取音频特征（HuBERT 768 维）- Utterance级别

        Args:
            audio_path: 音频文件路径

        Returns:
            audio_features: [768] - 使用 mean pooling 聚合所有时间步
        """
        import librosa

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

        # Mean pooling: 对所有时间步取平均
        features = np.mean(features, axis=0)  # [768]

        return torch.from_numpy(features).float()

    def extract_video_features(self, video_path: str) -> torch.Tensor:
        """
        提取视频特征（ViT-Base，768 维）- Utterance级别

        Args:
            video_path: 视频文件路径

        Returns:
            video_features: [768] - 使用 mean pooling 聚合所有帧
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return torch.zeros(768)  # 如果视频为空，返回零向量

        # 计算采样间隔
        sample_interval = max(1, int(fps / self.video_sample_rate))

        # 渐进式处理：边抽帧边处理，使用 running average 避免累积所有帧特征
        running_sum = None
        total_processed_frames = 0
        current_batch_frames = []

        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_batch_frames.append(frame)

            # 当累积了足够的帧，或到达最后一帧时，处理当前batch
            if len(current_batch_frames) >= self.video_batch_size or frame_idx + sample_interval >= total_frames:
                if len(current_batch_frames) > 0:
                    inputs = self.video_processor(images=current_batch_frames, return_tensors='pt').to(self.device)

                    with torch.no_grad():
                        outputs = self.video_model(**inputs)
                        # 使用 [CLS] token
                        batch_features = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

                    # 累积特征和（而非累积特征张量）
                    batch_sum = batch_features.sum(dim=0).cpu()  # [768]
                    if running_sum is None:
                        running_sum = batch_sum
                    else:
                        running_sum += batch_sum

                    total_processed_frames += len(current_batch_frames)

                    # 清理内存
                    del inputs, outputs, batch_features, batch_sum
                    current_batch_frames = []
                    self._cleanup_memory()

        cap.release()

        if total_processed_frames == 0:
            return torch.zeros(768)  # 如果没有帧，返回零向量

        # Mean pooling: 计算平均特征
        features = running_sum / total_processed_frames  # [768]

        return features

    def extract_au_features(self, video_path: str) -> torch.Tensor:
        """
        使用 OpenFace 提取 AU 概率特征 - Utterance级别

        Args:
            video_path: 视频文件路径

        Returns:
            au_features: [20] - 20个AU的概率（基于intensity的mean pooling归一化）
        """
        import subprocess
        import pandas as pd
        import tempfile
        import shutil

        # 检查 OpenFace 可执行文件是否存在
        if not os.path.exists(self.openface_executable):
            print(f"⚠ OpenFace 可执行文件不存在: {self.openface_executable}")
            return torch.zeros(len(self.au_names))

        # 为当前视频创建临时输出目录
        temp_dir = tempfile.mkdtemp(dir=self.openface_output_dir)

        try:
            # 调用 OpenFace FeatureExtraction
            # 使用与 ViT 相同的采样率确保帧一致性
            cmd = [
                self.openface_executable,
                '-f', video_path,
                '-out_dir', temp_dir,
                '-aus',  # 只提取 AU 特征
                '-fps', str(self.video_sample_rate)  # 使用相同的5fps采样率
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60  # 60秒超时
            )

            if result.returncode != 0:
                print(f"⚠ OpenFace 处理失败: {result.stderr.decode()}")
                return torch.zeros(len(self.au_names))

            # 查找输出的 CSV 文件
            csv_files = list(Path(temp_dir).glob('*.csv'))
            if len(csv_files) == 0:
                print(f"⚠ OpenFace 未生成 CSV 文件")
                return torch.zeros(len(self.au_names))

            csv_file = csv_files[0]

            # 读取 CSV 文件
            df = pd.read_csv(csv_file)

            # 提取 AU intensity 列并计算平均值
            au_probabilities = []

            for au_name in self.au_names:
                # au_name 格式: "AU1_raise_inner_brow"
                # OpenFace CSV 列名格式: "AU01_r" (intensity)
                au_number = au_name.split('_')[0].replace('AU', '')
                au_col_name = f'AU{au_number.zfill(2)}_r'

                if au_col_name in df.columns:
                    # 计算该 AU 的时间平均强度
                    au_intensity = df[au_col_name].mean()

                    # OpenFace intensity 范围通常是 [0, 5]
                    # 归一化到 [0, 1] 作为概率
                    au_prob = np.clip(au_intensity / 5.0, 0.0, 1.0)
                    au_probabilities.append(au_prob)
                else:
                    # 如果该 AU 列不存在，使用 0
                    au_probabilities.append(0.0)

            au_features = torch.tensor(au_probabilities, dtype=torch.float32)

        except subprocess.TimeoutExpired:
            print(f"⚠ OpenFace 处理超时")
            au_features = torch.zeros(len(self.au_names))
        except Exception as e:
            print(f"⚠ AU 特征提取失败: {str(e)}")
            au_features = torch.zeros(len(self.au_names))
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return au_features

    def extract_multimodal_features(
        self,
        text: str,
        audio_path: str,
        video_path: str
    ) -> Dict:
        """
        提取多模态特征（包括AU特征）- Utterance级别

        Args:
            text: 文本内容
            audio_path: 音频文件路径
            video_path: 视频文件路径

        Returns:
            features: 包含特征的字典
                - text_features: [768]
                - audio_features: [768]
                - video_features: [768]
                - au_features: [20] - 20个AU的概率
        """
        # 1. 提取文本特征
        text_features = self.extract_text_features(text)

        # 2. 提取音频特征
        audio_features = self.extract_audio_features(audio_path)

        # 3. 提取视频特征
        video_features = self.extract_video_features(video_path)

        # 4. 提取 AU 特征
        au_features = self.extract_au_features(video_path)

        return {
            'text_features': text_features,        # [768]
            'audio_features': audio_features,      # [768]
            'video_features': video_features,      # [768]
            'au_features': au_features,            # [20]
        }


def create_extractor(config_path: str) -> UtteranceFeatureExtractor:
    """
    创建特征提取器

    Args:
        config_path: 配置文件路径

    Returns:
        UtteranceFeatureExtractor 实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return UtteranceFeatureExtractor(config)
