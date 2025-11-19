"""
多模态一致性检查器

用于为unseen类生成可靠的伪标签
通过检查多个模态预测的一致性来过滤噪声样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from enum import Enum


class ConsistencyStrategy(Enum):
    """一致性检查策略"""
    ALL_AGREE = "all_agree"          # 所有模态必须一致
    MAJORITY = "majority"            # 多数投票（推荐）
    WEIGHTED_VOTE = "weighted_vote"  # 加权投票
    ENTROPY_THRESHOLD = "entropy_threshold"  # 熵阈值
    COMBINED = "combined"            # 组合策略


class MultimodalConsistencyChecker:
    """
    多模态一致性检查器

    检查从AU路径得到的情绪预测是否一致可靠
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        min_confidence: float = 0.8,
        entropy_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        self.model = model
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.entropy_threshold = entropy_threshold
        self.device = device

    def check_consistency(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        masks: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        检查多模态一致性

        Args:
            text: [batch_size, text_dim]
            audio: [batch_size, audio_dim]
            video: [batch_size, video_dim]
            masks: 可选的mask

        Returns:
            {
                'is_consistent': [batch_size] bool,
                'consensus_label': [batch_size] int,
                'confidence': [batch_size] float
            }
        """
        with torch.no_grad():
            # 获取完整预测
            outputs = self.model(text, audio, video, masks)

            # 从AU路径得到的情绪预测
            emo_probs = F.softmax(outputs['emo_from_au'], dim=1)
            emo_preds = emo_probs.argmax(dim=1)

            # 直接路径的预测（用于交叉验证）
            direct_probs = F.softmax(outputs['emo_direct'], dim=1)
            direct_preds = direct_probs.argmax(dim=1)

            # 计算置信度
            confidence, _ = emo_probs.max(dim=1)

            # 根据策略检查一致性
            if self.strategy == ConsistencyStrategy.MAJORITY:
                # 简化版：AU路径和直接路径一致 + 高置信度
                is_consistent = (emo_preds == direct_preds) & (confidence >= self.min_confidence)

            elif self.strategy == ConsistencyStrategy.ENTROPY_THRESHOLD:
                # 基于熵的检查
                entropy = -(emo_probs * torch.log(emo_probs + 1e-10)).sum(dim=1)
                is_consistent = (entropy < self.entropy_threshold) & (confidence >= self.min_confidence)

            elif self.strategy == ConsistencyStrategy.COMBINED:
                # 组合策略
                entropy = -(emo_probs * torch.log(emo_probs + 1e-10)).sum(dim=1)
                is_consistent = (
                    (emo_preds == direct_preds) &
                    (confidence >= self.min_confidence) &
                    (entropy < self.entropy_threshold)
                )

            else:  # ALL_AGREE
                # 默认最严格策略
                is_consistent = (emo_preds == direct_preds) & (confidence >= self.min_confidence)

        return {
            'is_consistent': is_consistent,
            'consensus_label': emo_preds,
            'confidence': confidence
        }
