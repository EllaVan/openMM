"""
多模态融合网络模块

包含：
1. 单模态编码器 (UnimodalEncoder)
2. 多模态超图融合层 (MultimodalHypergraphLayer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class UnimodalEncoder(nn.Module):
    """
    单模态特征编码器

    将原始特征映射到统一的嵌入空间
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            encoded: [batch_size, output_dim]
        """
        return self.encoder(x)


class MultimodalHypergraphLayer(nn.Module):
    """
    多模态超图融合层

    通过超图卷积融合文本、音频、视频特征
    """

    def __init__(
        self,
        text_dim: int = 256,
        audio_dim: int = 256,
        video_dim: int = 256,
        hidden_dim: int = 256,
        num_hyperedges: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_hyperedges = num_hyperedges
        self.num_layers = num_layers

        # 超边生成器
        self.hyperedge_generator = nn.Sequential(
            nn.Linear(text_dim + audio_dim + video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_hyperedges)
        )

        # 超图卷积层
        self.hypergraph_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim + audio_dim + video_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_features: [batch_size, text_dim]
            audio_features: [batch_size, audio_dim]
            video_features: [batch_size, video_dim]
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        # 拼接多模态特征
        multimodal_features = torch.cat([text_features, audio_features, video_features], dim=1)

        # 生成超边权重
        hyperedge_weights = torch.sigmoid(self.hyperedge_generator(multimodal_features))

        # 超图卷积
        x = multimodal_features
        for conv in self.hypergraph_convs:
            # 超边加权
            weighted_features = x * hyperedge_weights.unsqueeze(-1).expand(-1, -1, x.size(-1)).mean(dim=1)
            # 卷积
            x = conv(x) + weighted_features  # 残差连接

        # 最终融合
        fused = self.fusion(x)

        return fused
