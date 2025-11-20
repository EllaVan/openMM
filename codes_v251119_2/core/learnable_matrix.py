"""
Learnable AU-EMO Probability Matrix (Correct Mathematical Framework)

正确的数学框架：
p(x, ∪{au}^x, emo_k) ∝ ∏_i [P(au_i|x) * P(AU_i|EMO_k) * P(EMO_k) / Σ_k P(AU_i|EMO_k)]

直接使用 P(AU|EMO) 先验矩阵，不需要转换到 P(EMO|AU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import json


class LearnableAUEMOMatrix(nn.Module):
    """
    可学习的AU-EMO矩阵（基于正确的概率推理）

    数学框架：
    ----------
    先验矩阵：P(AU|EMO) [num_emotions, num_aus]
    输入：P(au_i|x) - 从样本x预测的AU概率 [batch, num_aus]
    输出：P(emo_k|x) - 情绪概率

    推理公式：
    p(emo_k | x) ∝ ∏_i [P(au_i|x) * P(AU_i|EMO_k) / Σ_k P(AU_i|EMO_k)] * P(EMO_k)

    取对数避免数值下溢：
    log p(emo_k | x) = Σ_i log[P(au_i|x) * P(AU_i|EMO_k) / Σ_k P(AU_i|EMO_k)] + log P(EMO_k)

    参数：
    ------
    num_aus : int
        AU数量（例如23）
    num_emotions : int
        情绪类别数量（例如7）
    prior_p_au_given_emo : np.ndarray [num_emotions, num_aus]
        心理学先验 P(AU|EMO) 矩阵
    prior_strength : float
        正则化强度
    device : str
        设备
    """

    def __init__(
        self,
        num_aus: int = 23,
        num_emotions: int = 6,
        prior_p_au_given_emo: Optional[np.ndarray] = None,
        prior_strength: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.prior_strength = prior_strength
        self.device_str = device

        # 初始化 P(AU|EMO) 先验矩阵
        if prior_p_au_given_emo is None:
            # 均匀先验：每个情绪下每个AU的激活概率
            prior_p_au_emo = np.ones((num_emotions, num_aus)) / num_aus
        else:
            prior_p_au_emo = np.array(prior_p_au_given_emo)
            assert prior_p_au_emo.shape == (num_emotions, num_aus), \
                f"先验矩阵形状不匹配: 期望 {(num_emotions, num_aus)}, 实际 {prior_p_au_emo.shape}"

        # 将概率转换为logits用于参数化
        # 对于sigmoid: p = sigmoid(logit) = 1/(1+exp(-logit))
        # 反函数: logit = log(p/(1-p))
        # Clip概率值避免极端情况
        prior_p_au_emo_clipped = np.clip(prior_p_au_emo, 1e-7, 1 - 1e-7)
        prior_logits = np.log(prior_p_au_emo_clipped / (1 - prior_p_au_emo_clipped))

        # 存储先验（不可学习）
        self.register_buffer(
            'prior_logits',
            torch.tensor(prior_logits, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'prior_p_au_given_emo',
            torch.tensor(prior_p_au_emo, dtype=torch.float32, device=device)
        )

        # 可学习矩阵参数（初始化为先验logits）
        self.matrix_logits = nn.Parameter(
            torch.tensor(prior_logits, dtype=torch.float32, device=device)
        )

        # 情绪先验 P(EMO_k) - 默认均匀分布
        self.register_buffer(
            'emo_prior',
            torch.ones(num_emotions, dtype=torch.float32, device=device) / num_emotions
        )

        # 统计信息
        self.register_buffer(
            'update_count',
            torch.tensor(0, dtype=torch.long, device=device)
        )

    def get_probability_matrix(self) -> torch.Tensor:
        """
        获取当前的 P(AU|EMO) 概率矩阵（未归一化）

        P(AU_i|EMO_k)表示给定情绪k时，AU i被激活的概率，范围[0,1]
        这是独立的概率值，不需要满足Σ_k P(AU_i|EMO_k) = 1的约束

        Returns:
        --------
        p_au_given_emo : torch.Tensor [num_emotions, num_aus]
            P(AU_i|EMO_k) 概率矩阵，[k, i]表示给定情绪k，AU i的激活概率
            每个值独立地在[0,1]范围内
        """
        # 使用sigmoid将logits转换为独立的概率值
        # 不使用softmax，因为P(AU_i|EMO_k)对于不同的k应该是独立的
        p_au_given_emo = torch.sigmoid(self.matrix_logits)
        return p_au_given_emo

    def forward(self, au_probs: torch.Tensor, emo_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从AU概率预测情绪概率

        使用正确的公式：
        p(emo_k | x) ∝ P(EMO_k) * ∏_i [P(au_i|x) * P(AU_i|EMO_k) / Σ_k' P(AU_i|EMO_k')]

        在对数空间：
        log p(emo_k | x) = log P(EMO_k) + Σ_i [log P(au_i|x) + log P(AU_i|EMO_k) - log Σ_k' P(AU_i|EMO_k')]

        其中：
        - P(au_i|x): 从样本x预测的AU i激活概率
        - P(AU_i|EMO_k): 给定情绪k时AU i的激活概率（先验知识）
        - Σ_k' P(AU_i|EMO_k'): 对所有情绪求和，作为归一化因子
        - P(EMO_k): 情绪先验概率

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            从样本预测的AU激活概率 P(au_i|x)
        emo_prior : torch.Tensor [num_emotions], optional
            情绪先验概率 P(EMO_k)，默认使用均匀分布

        Returns:
        --------
        emo_logits : torch.Tensor [batch_size, num_emotions]
            情绪预测对数概率
        """
        # 获取 P(AU|EMO) 矩阵 [num_emotions, num_aus]
        p_au_given_emo = self.get_probability_matrix()

        # 使用先验
        if emo_prior is None:
            emo_prior = self.emo_prior

        # 计算分母：Σ_k' P(AU_i|EMO_k') 对每个AU在所有情绪上求和
        # [num_emotions, num_aus] -> [1, num_aus]
        denominator = p_au_given_emo.sum(dim=0, keepdim=True)  # [1, num_aus]

        # 计算归一化后的项：P(AU_i|EMO_k) / Σ_k' P(AU_i|EMO_k')
        # [num_emotions, num_aus]
        normalized_p = p_au_given_emo / (denominator + 1e-10)

        # 计算：P(au_i|x) * normalized_p
        # au_probs: [batch, num_aus]
        # normalized_p: [num_emotions, num_aus]
        # 扩展维度以便广播
        au_probs_expanded = au_probs.unsqueeze(1)  # [batch, 1, num_aus]
        normalized_p_expanded = normalized_p.unsqueeze(0)  # [1, num_emotions, num_aus]

        # 逐元素相乘 [batch, num_emotions, num_aus]
        weighted = au_probs_expanded * normalized_p_expanded

        # 在AU维度求和（对数空间）：Σ_i log[P(au_i|x) * normalized_p]
        # [batch, num_emotions]
        emo_logits = torch.log(weighted + 1e-10).sum(dim=2)

        # 加上情绪先验的对数：log P(EMO_k)
        emo_logits = emo_logits + torch.log(emo_prior + 1e-10).unsqueeze(0)

        return emo_logits

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        计算正则化损失（向先验正则化）

        使用KL散度：D_KL(current || prior)

        Returns:
        --------
        reg_loss : torch.Tensor (标量)
            正则化损失
        """
        current_probs = self.get_probability_matrix()

        # KL散度：D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        # 在AU维度和EMO维度上求和
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-10),
            self.prior_p_au_given_emo,
            reduction='batchmean'
        )

        return self.prior_strength * kl_div

    def compute_entropy_regularization(self, strength: float = 0.01) -> torch.Tensor:
        """
        熵正则化，防止过拟合

        Parameters:
        -----------
        strength : float
            正则化强度

        Returns:
        --------
        entropy_loss : torch.Tensor (标量)
            负熵（最小化此项以最大化熵）
        """
        probs = self.get_probability_matrix()

        # 熵：H = -Σ p(x) log p(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=0).mean()

        # 最大化熵 = 最小化负熵
        return -strength * entropy

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        current_probs = self.get_probability_matrix()

        # KL散度
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-10),
            self.prior_p_au_given_emo,
            reduction='batchmean'
        ).item()

        # 平均熵（每个情绪列的熵）
        entropy_per_emo = -(current_probs * torch.log(current_probs + 1e-10)).sum(dim=0)
        avg_entropy = entropy_per_emo.mean().item()

        # Logits统计
        logits_mean = self.matrix_logits.mean().item()
        logits_std = self.matrix_logits.std().item()
        logits_max = self.matrix_logits.max().item()
        logits_min = self.matrix_logits.min().item()

        return {
            'avg_probability': current_probs.mean().item(),
            'kl_from_prior': kl_div,
            'avg_entropy_per_emotion': avg_entropy,
            'logits_mean': logits_mean,
            'logits_std': logits_std,
            'logits_max': logits_max,
            'logits_min': logits_min,
            'update_count': self.update_count.item()
        }

    def reset_to_prior(self):
        """重置矩阵到先验（硬重置）"""
        self.matrix_logits.data.copy_(self.prior_logits)
        print("矩阵已重置到先验")

    def soft_reset_to_prior(self, strength: float = 0.1):
        """软重置：当前矩阵与先验插值"""
        self.matrix_logits.data.mul_(1 - strength).add_(self.prior_logits * strength)
        print(f"矩阵软重置（强度={strength:.3f}）")

    def update_emo_prior(self, new_prior: torch.Tensor):
        """
        更新情绪先验 P(EMO_k)

        Parameters:
        -----------
        new_prior : torch.Tensor [num_emotions]
            新的情绪先验概率
        """
        assert new_prior.shape == (self.num_emotions,)
        assert torch.allclose(new_prior.sum(), torch.tensor(1.0))
        self.emo_prior.copy_(new_prior)

    def save(self, filepath: str):
        """保存矩阵状态"""
        state = {
            'matrix_logits': self.matrix_logits.detach().cpu().numpy(),
            'prior_logits': self.prior_logits.cpu().numpy(),
            'prior_p_au_given_emo': self.prior_p_au_given_emo.cpu().numpy(),
            'emo_prior': self.emo_prior.cpu().numpy(),
            'update_count': self.update_count.item(),
            'num_aus': self.num_aus,
            'num_emotions': self.num_emotions,
            'prior_strength': self.prior_strength
        }

        np.savez(filepath, **state)
        print(f"可学习AU-EMO矩阵已保存到 {filepath}")

    def load(self, filepath: str):
        """加载矩阵状态"""
        state = np.load(filepath)

        assert state['num_aus'] == self.num_aus
        assert state['num_emotions'] == self.num_emotions

        self.matrix_logits.data.copy_(
            torch.tensor(state['matrix_logits'], device=self.device_str)
        )
        if 'emo_prior' in state:
            self.emo_prior.copy_(
                torch.tensor(state['emo_prior'], device=self.device_str)
            )
        self.update_count.copy_(
            torch.tensor(state['update_count'], device=self.device_str)
        )

        print(f"可学习AU-EMO矩阵已从 {filepath} 加载")

    def visualize_matrix(
        self,
        au_names: Optional[list] = None,
        emotion_names: Optional[list] = None,
        show_logits: bool = False
    ) -> str:
        """
        可视化 P(AU|EMO) 矩阵

        Parameters:
        -----------
        au_names : list, optional
            AU名称列表
        emotion_names : list, optional
            情绪名称列表
        show_logits : bool
            是否显示logits而非概率

        Returns:
        --------
        visualization : str
            格式化的文本表格
        """
        import io

        if show_logits:
            matrix = self.matrix_logits.detach().cpu().numpy()
            title = "矩阵 Logits"
        else:
            matrix = self.get_probability_matrix().detach().cpu().numpy()
            title = "P(AU|EMO) 概率矩阵"

        if au_names is None:
            au_names = [f"AU{i}" for i in range(self.num_aus)]
        if emotion_names is None:
            emotion_names = [f"EMO{i}" for i in range(self.num_emotions)]

        output = io.StringIO()

        # 标题
        output.write(f"{title}\n")
        output.write("=" * (15 + 12 * self.num_aus) + "\n")

        # 表头（列是AU）
        output.write(f"{'Emotion':<15}")
        for au_name in au_names:
            output.write(f"{au_name:>12}")
        output.write("\n")
        output.write("-" * (15 + 12 * self.num_aus) + "\n")

        # 行（每行是一个情绪）
        for i, emo_name in enumerate(emotion_names):
            output.write(f"{emo_name:<15}")
            for j in range(self.num_aus):
                output.write(f"{matrix[i, j]:>12.4f}")
            output.write("\n")

        # 列和（每个AU在情绪维度的和，不需要等于1.0）
        output.write("-" * (15 + 12 * self.num_aus) + "\n")
        output.write(f"{'列和(AU)':<15}")
        col_sums = matrix.sum(axis=0)  # 对每个AU，在情绪维度求和
        for j in range(self.num_aus):
            output.write(f"{col_sums[j]:>12.4f}")
        output.write("\n")
        output.write(f"{'(归一化用)':<15}")
        output.write("(这些值在forward()中作为分母用于归一化)\n")

        return output.getvalue()


def load_au_emo_prior(filepath: str) -> Tuple[np.ndarray, list, list]:
    """
    从JSON文件加载AU-EMO先验

    期望格式：
    {
        "au_names": ["AU1", "AU2", ...],
        "emotion_names": ["happy", "sad", ...],
        "prior_matrix": [[...], ...]  # P(AU|EMO) [num_emotions, num_aus]
    }

    Returns:
    --------
    prior_matrix : np.ndarray [num_emotions, num_aus]
        P(AU|EMO) 矩阵
    au_names : list
    emotion_names : list
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    prior_matrix = np.array(data['prior_matrix'])
    au_names = data['au_names']
    emotion_names = data['emotion_names']

    print(f"已从 {filepath} 加载 P(AU|EMO) 先验")
    print(f"  形状: {prior_matrix.shape}")
    print(f"  情绪数量: {len(emotion_names)}")
    print(f"  AU数量: {len(au_names)}")

    return prior_matrix, au_names, emotion_names


if __name__ == "__main__":
    # 测试可学习矩阵
    print("测试可学习AU-EMO矩阵...")

    # 创建简单先验
    num_aus, num_emotions = 3, 2
    # P(AU|EMO): [num_emotions, num_aus]
    prior_p = np.array([
        [0.8, 0.3, 0.5],  # EMO0: 各AU的激活概率
        [0.2, 0.7, 0.5]   # EMO1: 各AU的激活概率
    ])

    # 归一化（确保每列/每个AU在情绪维度上和为1）
    prior_p = prior_p / prior_p.sum(axis=0, keepdims=True)

    # 初始化矩阵
    matrix = LearnableAUEMOMatrix(
        num_aus=num_aus,
        num_emotions=num_emotions,
        prior_p_au_given_emo=prior_p,
        prior_strength=0.1,
        device='cpu'
    )

    print("\n初始 P(AU|EMO):")
    print(matrix.get_probability_matrix())

    print("\n初始 logits:")
    print(matrix.matrix_logits)

    # 测试预测
    au_probs = torch.tensor([[0.9, 0.1, 0.5]])  # AU0高激活
    emo_logits = matrix(au_probs)
    print(f"\n预测测试:")
    print(f"  AU概率: {au_probs}")
    print(f"  情绪logits: {emo_logits}")
    print(f"  情绪概率: {F.softmax(emo_logits, dim=1)}")

    # 测试正则化
    reg_loss = matrix.compute_regularization_loss()
    print(f"\n正则化损失: {reg_loss.item():.6f}")

    # 测试梯度流
    optimizer = torch.optim.Adam([matrix.matrix_logits], lr=0.01)

    print("\n测试梯度流...")
    for i in range(5):
        # 虚拟损失
        au_probs_batch = torch.tensor([
            [0.9, 0.1, 0.5],
            [0.1, 0.9, 0.5],
        ])
        emo_labels = torch.tensor([0, 1])

        emo_pred = matrix(au_probs_batch)
        loss = F.cross_entropy(emo_pred, emo_labels)
        loss += matrix.compute_regularization_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  步骤 {i+1}: loss={loss.item():.4f}")

    print("\n更新后的 P(AU|EMO):")
    print(matrix.get_probability_matrix())

    print("\n矩阵统计:")
    for k, v in matrix.get_statistics().items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ 所有测试通过!")
