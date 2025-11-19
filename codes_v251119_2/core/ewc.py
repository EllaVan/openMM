"""
Elastic Weight Consolidation (EWC) for Preventing Catastrophic Forgetting

EWC通过惩罚重要参数的变化来防止遗忘之前任务的知识
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import copy


class EWC:
    """
    Elastic Weight Consolidation

    在每个任务后计算Fisher信息矩阵，
    在新任务训练时惩罚重要参数的变化

    数学公式：
        L_EWC = Σ_i (λ/2) * F_i * (θ_i - θ*_i)^2

    其中：
        F_i: 参数i的Fisher信息
        θ_i: 当前参数
        θ*_i: 之前任务的最优参数
        λ: EWC强度
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        ewc_lambda: float = 1000.0
    ):
        """
        Args:
            model: 神经网络模型
            device: 设备
            ewc_lambda: EWC正则化强度
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda

        # 存储每个任务的Fisher信息和最优参数
        self.fisher_dict = {}
        self.optimal_params_dict = {}

        # 是否已经合并过Fisher信息
        self.is_consolidated = False

    def consolidate(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ):
        """
        计算Fisher信息矩阵并保存当前参数

        Args:
            dataloader: 当前任务的数据加载器
            num_samples: 用于计算Fisher的样本数（None表示全部）
        """
        print(f"计算Fisher信息矩阵...")

        self.model.eval()

        # 初始化Fisher字典
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        # 计算Fisher信息
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_samples is not None and sample_count >= num_samples:
                break

            # 获取数据
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            outputs = self.model(text, audio, video)

            # 计算损失
            loss = nn.functional.cross_entropy(outputs['emo_from_au'], labels)

            # 清除之前的梯度
            self.model.zero_grad()

            # 反向传播
            loss.backward()

            # 累加平方梯度作为Fisher信息的近似
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2

            sample_count += labels.size(0)

        # 平均化Fisher信息
        for name in fisher_dict:
            fisher_dict[name] /= sample_count

        # 保存当前Fisher信息（累加）
        if not self.fisher_dict:
            self.fisher_dict = fisher_dict
        else:
            for name in fisher_dict:
                self.fisher_dict[name] += fisher_dict[name]

        # 保存当前最优参数
        self.optimal_params_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params_dict[name] = param.data.clone()

        self.is_consolidated = True

        print(f"Fisher信息已更新 (使用 {sample_count} 个样本)")

    def penalty(self) -> torch.Tensor:
        """
        计算EWC惩罚项

        Returns:
            ewc_loss: EWC损失
        """
        if not self.is_consolidated:
            return torch.tensor(0.0).to(self.device)

        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                # Fisher加权的参数变化惩罚
                fisher = self.fisher_dict[name]
                optimal_param = self.optimal_params_dict[name]

                ewc_loss += (fisher * (param - optimal_param) ** 2).sum()

        return (self.ewc_lambda / 2) * ewc_loss

    def reset(self):
        """重置EWC状态"""
        self.fisher_dict = {}
        self.optimal_params_dict = {}
        self.is_consolidated = False

    def get_importance_dict(self) -> Dict[str, float]:
        """
        获取每个参数的重要性得分

        Returns:
            importance_dict: {param_name: importance_score}
        """
        importance_dict = {}

        for name in self.fisher_dict:
            importance_dict[name] = self.fisher_dict[name].sum().item()

        return importance_dict


if __name__ == "__main__":
    # 测试代码
    print("测试 EWC...")

    # 创建一个简单的模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    ).to('cpu')

    # 创建EWC
    ewc = EWC(model, device='cpu', ewc_lambda=1000.0)

    # 创建虚拟数据
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                'text': torch.randn(10),
                'audio': torch.randn(10),
                'video': torch.randn(10),
                'label': torch.randint(0, 2, (1,)).item()
            }

    # 模拟网络前向传播
    original_forward = model.forward

    def mock_forward(text, audio, video, masks=None):
        x = text  # 简化：只使用text
        return {'emo_from_au': original_forward(x)}

    model.forward = mock_forward

    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

    # 合并Fisher信息
    ewc.consolidate(dataloader)

    # 计算惩罚
    penalty = ewc.penalty()
    print(f"EWC penalty: {penalty.item():.6f}")

    # 获取重要性
    importance = ewc.get_importance_dict()
    print(f"Parameter importance: {importance}")

    print("\n✓ EWC测试通过!")
