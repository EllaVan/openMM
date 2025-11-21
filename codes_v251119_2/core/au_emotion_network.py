"""
AU-based情绪识别网络

核心架构：
1. 多模态特征编码
2. 超图融合
3. AU预测分支
4. 通过AU-EMO矩阵预测情绪
5. 直接情绪预测（辅助分支）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion.hypergraph_fusion import UnimodalEncoder, MultimodalHypergraphLayer
from core.learnable_matrix import LearnableAUEMOMatrix


class AUPredictor(nn.Module):
    """
    AU预测器

    从融合特征预测23个Action Units的激活概率
    使用多标签分类（sigmoid），因为多个AU可以同时激活
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_aus: int = 23,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_aus = num_aus

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_aus)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim]
        Returns:
            au_probs: [batch_size, num_aus] - Sigmoid概率
        """
        logits = self.predictor(features)
        au_probs = torch.sigmoid(logits)
        return au_probs


class DirectEmotionClassifier(nn.Module):
    """
    直接情绪分类器

    用于辅助训练和对比
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_emotions: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim]
        Returns:
            logits: [batch_size, num_emotions]
        """
        return self.classifier(features)


class AUEmotionNetwork(nn.Module):
    """
    完整的AU-based情绪识别网络

    网络流程：
    1. 文本/音频/视频 -> 单模态编码器 -> 编码特征
    2. 编码特征 -> 超图融合 -> 融合特征
    3. 融合特征 -> AU预测器 -> AU概率
    4. AU概率 -> AU-EMO矩阵 -> 情绪预测（主路径）
    5. 融合特征 -> 直接分类器 -> 情绪预测（辅助路径）
    """

    def __init__(
        self,
        # 输入维度
        text_input_dim: int = 768,
        audio_input_dim: int = 768,
        video_input_dim: int = 768,
        # AU和情绪
        num_aus: int = 23,
        num_emotions: int = 6,
        # 网络架构
        encoder_hidden_dim: int = 256,
        encoder_output_dim: int = 256,
        hypergraph_hidden_dim: int = 256,
        num_hyperedges: int = 64,
        num_conv_layers: int = 2,
        dropout: float = 0.1,
        # AU-EMO矩阵
        au_emo_prior: Optional[np.ndarray] = None,
        prior_strength: float = 0.1,
        # 设备
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.device_str = device

        # 1. 单模态编码器
        self.text_encoder = UnimodalEncoder(
            input_dim=text_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.audio_encoder = UnimodalEncoder(
            input_dim=audio_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.video_encoder = UnimodalEncoder(
            input_dim=video_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        # 2. 多模态超图融合
        self.hypergraph = MultimodalHypergraphLayer(
            text_dim=encoder_output_dim,
            audio_dim=encoder_output_dim,
            video_dim=encoder_output_dim,
            hidden_dim=hypergraph_hidden_dim,
            num_hyperedges=num_hyperedges,
            num_layers=num_conv_layers,
            dropout=dropout
        )

        # 3. AU预测分支
        self.au_predictor = AUPredictor(
            input_dim=hypergraph_hidden_dim,
            num_aus=num_aus,
            hidden_dim=256,
            dropout=dropout
        )

        # 4. 直接情绪分类器（辅助）
        self.emotion_classifier = DirectEmotionClassifier(
            input_dim=hypergraph_hidden_dim,
            num_emotions=num_emotions,
            hidden_dim=256,
            dropout=dropout
        )

        # 5. AU-EMO可学习矩阵
        self.au_emo_matrix = LearnableAUEMOMatrix(
            num_aus=num_aus,
            num_emotions=num_emotions,
            prior_p_au_given_emo=au_emo_prior,
            prior_strength=prior_strength,
            device=device
        )

    def expand_classifiers(self, new_num_emotions: int):
        """
        动态扩展分类器以适应新的情绪类别数

        用于持续学习中，当新任务引入新类别时调用

        Args:
            new_num_emotions: 新的总类别数
        """
        if new_num_emotions <= self.num_emotions:
            return  # 无需扩展

        old_num = self.num_emotions
        self.num_emotions = new_num_emotions

        # 1. 扩展DirectEmotionClassifier的最后一层
        old_weight = self.emotion_classifier.classifier[-1].weight.data
        old_bias = self.emotion_classifier.classifier[-1].bias.data

        # 创建新的分类层
        hidden_dim = old_weight.shape[1]
        new_classifier_layer = nn.Linear(hidden_dim, new_num_emotions).to(self.device_str)

        # 复制旧权重
        with torch.no_grad():
            new_classifier_layer.weight.data[:old_num] = old_weight
            new_classifier_layer.bias.data[:old_num] = old_bias
            # 新类别的权重用小随机值初始化
            nn.init.xavier_uniform_(new_classifier_layer.weight.data[old_num:])
            new_classifier_layer.bias.data[old_num:] = 0

        # 替换最后一层
        self.emotion_classifier.classifier[-1] = new_classifier_layer

        # 2. 扩展AU-EMO矩阵
        self.au_emo_matrix.expand_emotions(new_num_emotions)

        print(f"分类器已扩展: {old_num} -> {new_num_emotions} 个类别")

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            text: [batch_size, text_dim]
            audio: [batch_size, audio_dim]
            video: [batch_size, video_dim]
            masks: 可选的mask字典

        Returns:
            outputs: {
                'au_probs': [batch_size, num_aus],
                'emo_from_au': [batch_size, num_emotions],  # 通过AU-EMO矩阵
                'emo_direct': [batch_size, num_emotions],   # 直接分类
                'fused_features': [batch_size, hidden_dim]
            }
        """
        # 1. 单模态编码
        text_encoded = self.text_encoder(text)
        audio_encoded = self.audio_encoder(audio)
        video_encoded = self.video_encoder(video)

        # 2. 多模态融合
        fused_features = self.hypergraph(text_encoded, audio_encoded, video_encoded)

        # 3. AU预测
        au_probs = self.au_predictor(fused_features)

        # 4. 通过AU-EMO矩阵预测情绪（主路径）
        emo_from_au = self.au_emo_matrix(au_probs)

        # 5. 直接情绪分类（辅助路径）
        emo_direct = self.emotion_classifier(fused_features)

        return {
            'au_probs': au_probs,
            'emo_from_au': emo_from_au,
            'emo_direct': emo_direct,
            'fused_features': fused_features
        }


if __name__ == "__main__":
    # 测试代码
    print("测试 AUEmotionNetwork...")

    device = 'cpu'
    batch_size = 4

    # 创建网络
    model = AUEmotionNetwork(
        text_input_dim=768,
        audio_input_dim=768,
        video_input_dim=768,
        num_aus=23,
        num_emotions=6,
        device=device
    ).to(device)

    # 创建测试输入
    text = torch.randn(batch_size, 768).to(device)
    audio = torch.randn(batch_size, 768).to(device)
    video = torch.randn(batch_size, 768).to(device)

    # 前向传播
    outputs = model(text, audio, video)

    print(f"AU probs shape: {outputs['au_probs'].shape}")
    print(f"Emo from AU shape: {outputs['emo_from_au'].shape}")
    print(f"Emo direct shape: {outputs['emo_direct'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")

    print("\n✓ 网络测试通过!")
