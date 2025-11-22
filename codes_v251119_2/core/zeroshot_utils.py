"""
Zero-shot学习的辅助函数

用于构建zeroshotExpander所需的：
1. 情绪转换矩阵（边权）
2. 情绪语义特征（节点特征）
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional


def get_transition_prob(x1: np.ndarray, x2: np.ndarray, num_emotions: int = 6) -> tuple:
    """
    计算两个情绪之间的转换概率

    Args:
        x1: P(AU|EMO1), shape [num_aus]
        x2: P(AU|EMO2), shape [num_aus]
        num_emotions: 情绪总数

    Returns:
        p_x1_given_x2: P(EMO1|EMO2)
        p_x2_given_x1: P(EMO2|EMO1)
    """
    prob_sum = np.sum(x1 * x2)
    p_x1_given_x2 = prob_sum / np.sum(x2) / num_emotions  # P(EMO1|EMO2)
    p_x2_given_x1 = prob_sum / np.sum(x1) / num_emotions  # P(EMO2|EMO1)
    return p_x1_given_x2, p_x2_given_x1


def get_transition_matrix(p_au_given_emo: np.ndarray) -> np.ndarray:
    """
    根据P(AU|EMO)构建情绪之间的转换矩阵（用于图卷积的邻接矩阵）

    Args:
        p_au_given_emo: [num_emotions, num_aus] - P(AU|EMO)矩阵

    Returns:
        trans_matrix: [num_emotions, num_emotions] - 归一化的转换矩阵
    """
    num_emotions, num_aus = p_au_given_emo.shape

    # 初始化转换矩阵
    trans_matrix = np.zeros((num_emotions, num_emotions))

    # 计算每对情绪之间的转换概率
    for i in range(num_emotions - 1):
        for j in range(i + 1, num_emotions):
            p_i_given_j, p_j_given_i = get_transition_prob(
                p_au_given_emo[i],
                p_au_given_emo[j],
                num_emotions
            )
            trans_matrix[i, j] = p_i_given_j
            trans_matrix[j, i] = p_j_given_i

    # 行归一化（每行和为1）
    for i in range(num_emotions):
        row_sum = np.sum(trans_matrix[i])
        if row_sum > 0:
            trans_matrix[i] = trans_matrix[i] / row_sum

    # 添加自连接
    self_connection = np.identity(num_emotions)
    trans_matrix = trans_matrix + self_connection

    return torch.from_numpy(trans_matrix).float()


def get_class_embedding(
    au_embedding: Dict[str, torch.Tensor],
    p_au_given_emo: np.ndarray
) -> torch.Tensor:
    """
    根据AU embedding和P(AU|EMO)构建情绪的语义特征

    语义特征 = 加权平均的AU embedding
    EMO_i = Σ_j P(AU_j|EMO_i) * AU_j_embedding

    Args:
        au_embedding: AU embedding字典, {'AU1': tensor, 'AU2': tensor, ...}
        p_au_given_emo: [num_emotions, num_aus] - P(AU|EMO)矩阵

    Returns:
        class_embeddings: [num_emotions, embedding_dim] - 归一化的情绪语义特征
    """
    num_emotions, num_aus = p_au_given_emo.shape

    # 获取embedding维度（从第一个AU）
    first_au_key = list(au_embedding.keys())[0]
    embedding_dim = au_embedding[first_au_key].shape[0]

    # 初始化
    class_embeddings = []

    for emo_idx in range(num_emotions):
        # 累积加权的AU embedding
        emo_vector = torch.zeros(embedding_dim)

        for au_idx in range(num_aus):
            au_key = f'AU{au_idx + 1}'

            if au_key in au_embedding:
                # 加权累加: P(AU|EMO) * AU_embedding
                weight = p_au_given_emo[emo_idx, au_idx]
                emo_vector += weight * au_embedding[au_key]

        # 归一化（确保不为零向量）
        if torch.norm(emo_vector) > 0:
            emo_vector = F.normalize(emo_vector.unsqueeze(0), dim=1).squeeze(0)

        class_embeddings.append(emo_vector)

    # 堆叠并归一化
    class_embeddings = torch.stack(class_embeddings)
    class_embeddings = F.normalize(class_embeddings, dim=1)

    return class_embeddings


def check_convergence(
    emo_from_classifier: torch.Tensor,
    emo_from_au: torch.Tensor,
    threshold: float = 0.95
) -> bool:
    """
    检查是否收敛：直接分类器推导的EMO与AU推导的EMO是否一致

    收敛条件：两种方式预测的标签一致率 >= threshold

    Args:
        emo_from_classifier: [batch_size, num_emotions] - 直接分类器logits
        emo_from_au: [batch_size, num_emotions] - AU路径logits
        threshold: 一致率阈值

    Returns:
        converged: 是否收敛
    """
    # 获取预测标签
    pred_classifier = emo_from_classifier.argmax(dim=1)
    pred_au = emo_from_au.argmax(dim=1)

    # 计算一致率
    agreement = (pred_classifier == pred_au).float().mean().item()

    return agreement >= threshold


def l2_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L2损失

    Args:
        a: tensor
        b: tensor

    Returns:
        loss: L2距离
    """
    return ((a - b) ** 2).sum() / (len(a) * 2)


def mask_l2_loss(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    带mask的L2损失（只计算mask=1的位置）

    Args:
        a: [num_classes, dim] - 预测的分类器权重
        b: [num_classes, dim] - 目标分类器权重
        mask: [num_classes] - seen mask (1=seen, 0=unseen)

    Returns:
        loss: masked L2距离
    """
    # 扩展mask维度 [num_classes] -> [num_classes, 1]
    mask = mask.unsqueeze(1)

    # 只计算mask位置的损失
    return l2_loss(a * mask, b * mask)


if __name__ == "__main__":
    """测试代码"""
    print("="*80)
    print("测试 zeroshot_utils")
    print("="*80)

    # 测试1: 转换矩阵
    print("\n【测试1: 构建转换矩阵】")
    num_emotions = 6
    num_aus = 23
    p_au_emo = np.random.rand(num_emotions, num_aus)

    trans_matrix = get_transition_matrix(p_au_emo)
    print(f"转换矩阵形状: {trans_matrix.shape}")
    print(f"对角线元素（自连接）: {np.diag(trans_matrix)}")
    print(f"行和: {trans_matrix.sum(axis=1)}")

    # 测试2: 类embedding
    print("\n【测试2: 构建类embedding】")
    au_embeddings = {f'AU{i+1}': torch.randn(768) for i in range(num_aus)}
    class_embeddings = get_class_embedding(au_embeddings, p_au_emo)
    print(f"类embedding形状: {class_embeddings.shape}")
    print(f"L2范数: {torch.norm(class_embeddings, dim=1)}")

    # 测试3: 收敛检查
    print("\n【测试3: 收敛检查】")
    batch_size = 32
    logits1 = torch.randn(batch_size, num_emotions)
    logits2 = logits1 + torch.randn(batch_size, num_emotions) * 0.1  # 接近但不完全相同

    converged = check_convergence(logits1, logits2, threshold=0.8)
    print(f"收敛状态: {converged}")

    pred1 = logits1.argmax(dim=1)
    pred2 = logits2.argmax(dim=1)
    agreement = (pred1 == pred2).float().mean().item()
    print(f"实际一致率: {agreement:.4f}")

    # 测试4: mask_l2_loss
    print("\n【测试4: Mask L2 Loss】")
    a = torch.randn(num_emotions, 768)
    b = torch.randn(num_emotions, 768)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # 前2个seen

    loss = mask_l2_loss(a, b, mask)
    print(f"Masked L2 Loss: {loss.item():.4f}")

    print("\n✓ 测试完成!")
