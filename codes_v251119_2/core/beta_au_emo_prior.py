"""
Beta分布管理的AU-EMO先验更新模块

用于zero-shot学习中，通过Beta分布建模P(AU|EMO)，并用贝叶斯方法在线更新。

核心思想：
1. 每个P(AU_i|EMO_j)用Beta(α, β)分布建模
2. 观测unseen样本时，累积统计量：α += p(au|x), β += (1-p(au|x))
3. 定期进行贝叶斯更新
4. 当前估计：P(AU|EMO) = α / (α + β)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BetaAUEMOPrior(nn.Module):
    """
    Beta分布管理的AU-EMO先验

    维护：
    - alpha: [num_emotions, num_aus] - Beta分布的α参数
    - beta: [num_emotions, num_aus] - Beta分布的β参数
    - accumulated_alpha: 累积的观测统计量
    - accumulated_beta: 累积的观测统计量
    """

    def __init__(
        self,
        num_emotions: int = 7,
        num_aus: int = 23,
        prior_json_path: Optional[str] = None,
        pseudo_count: float = 2.0,
        device: str = 'cuda'
    ):
        """
        Args:
            num_emotions: 情绪类别数
            num_aus: AU数量（实际使用前20个）
            prior_json_path: 先验矩阵JSON文件路径
            pseudo_count: 伪计数（控制先验强度）
            device: 设备
        """
        super().__init__()

        self.num_emotions = num_emotions
        self.num_aus = num_aus
        self.pseudo_count = pseudo_count
        self.device_str = device

        # 加载先验
        if prior_json_path is not None and Path(prior_json_path).exists():
            prior_matrix = self._load_prior_from_json(prior_json_path)
        else:
            # 默认均匀先验
            prior_matrix = np.ones((num_emotions, num_aus)) * 0.5

        # 初始化Beta分布参数
        # α = prior * pseudo_count, β = (1-prior) * pseudo_count
        alpha_init = prior_matrix * pseudo_count
        beta_init = (1 - prior_matrix) * pseudo_count

        # 注册为buffer（不参与梯度更新，但会保存到checkpoint）
        self.register_buffer('alpha', torch.tensor(alpha_init, dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(beta_init, dtype=torch.float32))

        # 累积的观测统计量（用于批量更新）
        self.register_buffer('accumulated_alpha', torch.zeros_like(self.alpha))
        self.register_buffer('accumulated_beta', torch.zeros_like(self.beta))

        # 观测计数
        self.register_buffer('observation_count', torch.zeros(num_emotions, dtype=torch.long))

        # 情绪名称映射（用于调试和可视化）
        self.emotion_names = None
        self.au_names = None

    def _load_prior_from_json(self, json_path: str) -> np.ndarray:
        """
        从JSON文件加载先验矩阵

        Args:
            json_path: JSON文件路径

        Returns:
            prior_matrix: [num_emotions, num_aus]
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 提取先验矩阵
        prior_matrix = np.array(data['prior_matrix'], dtype=np.float32)

        # 只使用前20个AU
        prior_matrix = prior_matrix[:, :20]

        # 保存名称映射
        self.emotion_names = data.get('emotion_names', [])
        self.au_names = data.get('au_names', [])[:20]

        print(f"从 {json_path} 加载先验矩阵:")
        print(f"  形状: {prior_matrix.shape}")
        print(f"  情绪: {self.emotion_names}")
        print(f"  AU数: {len(self.au_names)}")

        return prior_matrix

    def get_p_au_given_emo(self, emotion_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取当前的P(AU|EMO)估计

        Args:
            emotion_indices: 可选，只获取特定情绪的P(AU|EMO)

        Returns:
            p_au_emo: [num_emotions, num_aus] or [len(emotion_indices), num_aus]
        """
        # P(AU|EMO) = α / (α + β)
        p_au_emo = self.alpha / (self.alpha + self.beta)

        if emotion_indices is not None:
            return p_au_emo[emotion_indices]

        return p_au_emo

    def accumulate_observations(
        self,
        emotion_indices: torch.Tensor,
        au_probs: torch.Tensor
    ):
        """
        累积观测统计量（不立即更新Beta参数）

        Args:
            emotion_indices: [batch_size] - 伪标签（情绪类别）
            au_probs: [batch_size, num_aus] - AU检测概率 p(au|x)
        """
        batch_size = emotion_indices.shape[0]

        for i in range(batch_size):
            emo_idx = emotion_indices[i].item()
            au_prob = au_probs[i]  # [num_aus]

            # 累积 α += p(au), β += (1-p(au))
            self.accumulated_alpha[emo_idx] += au_prob
            self.accumulated_beta[emo_idx] += (1.0 - au_prob)

            # 计数
            self.observation_count[emo_idx] += 1

    def update_beta_parameters(self, emotion_indices: Optional[List[int]] = None):
        """
        从累积的统计量更新Beta参数（贝叶斯更新）

        Args:
            emotion_indices: 可选，只更新特定情绪的参数
        """
        if emotion_indices is None:
            # 更新所有情绪
            emotion_indices = list(range(self.num_emotions))

        for emo_idx in emotion_indices:
            # 贝叶斯更新：后验 = 先验 + 观测
            self.alpha[emo_idx] += self.accumulated_alpha[emo_idx]
            self.beta[emo_idx] += self.accumulated_beta[emo_idx]

            # 清空累积量
            self.accumulated_alpha[emo_idx].zero_()
            self.accumulated_beta[emo_idx].zero_()

        print(f"更新了 {len(emotion_indices)} 个情绪的Beta参数")
        for emo_idx in emotion_indices:
            if self.observation_count[emo_idx] > 0:
                print(f"  情绪 {emo_idx}: 观测数={self.observation_count[emo_idx].item()}")

    def reset_observations(self, emotion_indices: Optional[List[int]] = None):
        """
        重置观测计数（用于新任务）

        Args:
            emotion_indices: 可选，只重置特定情绪
        """
        if emotion_indices is None:
            self.observation_count.zero_()
        else:
            for emo_idx in emotion_indices:
                self.observation_count[emo_idx] = 0

    def get_statistics(self, emotion_idx: Optional[int] = None) -> Dict:
        """
        获取统计信息

        Args:
            emotion_idx: 可选，只获取特定情绪的统计

        Returns:
            stats: 统计字典
        """
        if emotion_idx is not None:
            # 单个情绪
            p_au_emo = self.get_p_au_given_emo(torch.tensor([emotion_idx]))

            return {
                'emotion_idx': emotion_idx,
                'emotion_name': self.emotion_names[emotion_idx] if self.emotion_names else str(emotion_idx),
                'observation_count': self.observation_count[emotion_idx].item(),
                'p_au_emo': p_au_emo.cpu().numpy(),
                'alpha': self.alpha[emotion_idx].cpu().numpy(),
                'beta': self.beta[emotion_idx].cpu().numpy(),
                'mean_alpha': self.alpha[emotion_idx].mean().item(),
                'mean_beta': self.beta[emotion_idx].mean().item()
            }
        else:
            # 所有情绪
            p_au_emo = self.get_p_au_given_emo()

            return {
                'num_emotions': self.num_emotions,
                'num_aus': self.num_aus,
                'observation_counts': self.observation_count.cpu().numpy(),
                'p_au_emo': p_au_emo.cpu().numpy(),
                'mean_p_au_emo': p_au_emo.mean().item(),
                'total_observations': self.observation_count.sum().item()
            }

    def save(self, filepath: str):
        """
        保存Beta参数到文件

        Args:
            filepath: 保存路径（.npz格式）
        """
        np.savez(
            filepath,
            alpha=self.alpha.cpu().numpy(),
            beta=self.beta.cpu().numpy(),
            observation_count=self.observation_count.cpu().numpy(),
            p_au_emo=self.get_p_au_given_emo().cpu().numpy(),
            emotion_names=self.emotion_names,
            au_names=self.au_names
        )
        print(f"Beta参数已保存到: {filepath}")

    def load(self, filepath: str):
        """
        从文件加载Beta参数

        Args:
            filepath: 文件路径（.npz格式）
        """
        data = np.load(filepath, allow_pickle=True)

        self.alpha = torch.tensor(data['alpha'], dtype=torch.float32).to(self.device_str)
        self.beta = torch.tensor(data['beta'], dtype=torch.float32).to(self.device_str)
        self.observation_count = torch.tensor(data['observation_count'], dtype=torch.long).to(self.device_str)

        if 'emotion_names' in data:
            self.emotion_names = data['emotion_names'].tolist()
        if 'au_names' in data:
            self.au_names = data['au_names'].tolist()

        print(f"Beta参数已从 {filepath} 加载")
        print(f"  总观测数: {self.observation_count.sum().item()}")


if __name__ == "__main__":
    """测试代码"""
    print("="*80)
    print("测试 BetaAUEMOPrior")
    print("="*80)

    # 创建模块
    prior_path = "../materials/au_emo_prior.json"
    beta_prior = BetaAUEMOPrior(
        num_emotions=7,
        num_aus=20,
        prior_json_path=prior_path,
        pseudo_count=2.0,
        device='cpu'
    )

    print("\n初始P(AU|EMO):")
    p_au_emo = beta_prior.get_p_au_given_emo()
    print(f"形状: {p_au_emo.shape}")
    print(f"均值: {p_au_emo.mean().item():.4f}")

    # 模拟观测
    print("\n模拟观测unseen样本...")
    emotion_indices = torch.tensor([2, 2, 3, 3, 3])  # surprise和disgust
    au_probs = torch.rand(5, 20)  # 随机AU概率

    beta_prior.accumulate_observations(emotion_indices, au_probs)

    print(f"累积观测数: {beta_prior.observation_count[2:4]}")

    # 更新
    print("\n执行贝叶斯更新...")
    beta_prior.update_beta_parameters(emotion_indices=[2, 3])

    # 查看更新后的结果
    print("\n更新后的统计:")
    stats = beta_prior.get_statistics(emotion_idx=2)
    print(f"情绪: {stats['emotion_name']}")
    print(f"观测数: {stats['observation_count']}")
    print(f"P(AU|EMO)均值: {stats['p_au_emo'].mean():.4f}")

    # 保存
    save_path = "/tmp/beta_prior_test.npz"
    beta_prior.save(save_path)

    # 加载
    beta_prior2 = BetaAUEMOPrior(num_emotions=7, num_aus=20, device='cpu')
    beta_prior2.load(save_path)

    print("\n✓ 测试完成!")
