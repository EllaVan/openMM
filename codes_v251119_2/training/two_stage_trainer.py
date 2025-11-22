"""
二阶段持续学习训练器

每个任务分为两个阶段：
1. 阶段1：Seen训练 - 用seen样本训练backbone + AU分支 + seen分类器
2. 阶段2：Unseen Zero-shot - EM迭代训练unseen分类器权重

阶段2的EM迭代：
- E步：固定P(AU|EMO)，更新zeroshotExpander
- M步：固定zeroshotExpander，更新P(AU|EMO)
- 收敛：直接分类 ≈ AU路径分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    AUEmotionNetwork,
    zeroshotExpander,
    BetaAUEMOPrior,
    zeroshot_utils,
    EWC
)
from utils import save_checkpoint


class TwoStageTrainer:
    """
    二阶段持续学习训练器

    阶段1: Seen训练（持续学习）
    阶段2: Unseen Zero-shot（EM迭代）
    """

    def __init__(
        self,
        model: AUEmotionNetwork,
        config: dict,
        logger,
        device: str = 'cuda'
    ):
        """
        Args:
            model: AU情绪识别网络
            config: 配置字典
            logger: 日志记录器
            device: 设备
        """
        self.model = model.to(device)
        self.config = config
        self.logger = logger
        self.device = device

        # 优化器（阶段1用）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        # Beta分布先验管理器
        prior_path = config.get('au_emo_prior_path', 'materials/au_emo_prior.json')
        self.beta_prior = BetaAUEMOPrior(
            num_emotions=config['model']['num_emotions'],
            num_aus=config['model']['num_aus'],
            prior_json_path=prior_path,
            pseudo_count=config.get('pseudo_count', 2.0),
            device=device
        ).to(device)

        # AU embeddings（用于构建类语义特征）
        au_embedding_path = config.get('au_embedding_path', 'materials/au_embedding.pt')
        self.au_embeddings = torch.load(au_embedding_path)
        self.logger.info(f"加载AU embeddings: {len(self.au_embeddings)} AUs")

        # Zero-shot expander（阶段2创建）
        self.zeroshot_expander = None
        self.zeroshot_optimizer = None

        # EWC
        self.ewc = None
        if config['continual_learning'].get('use_ewc', False):
            self.ewc = EWC(
                model=self.model,
                device=device,
                ewc_lambda=config['continual_learning']['ewc_lambda']
            )

        # 统计
        self.training_stats = {
            'tasks': []
        }

        # 保存目录
        self.save_dir = Path(config['output']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_task(
        self,
        task_id: int,
        task_name: str,
        task_info: Dict,
        train_loaders: Dict[str, DataLoader],
        test_loaders: Dict[str, DataLoader],
        num_epochs_stage1: int,
        num_em_iterations: int = 10,
        num_epochs_per_em: int = 5
    ) -> Dict:
        """
        训练单个任务（二阶段）

        Args:
            task_id: 任务ID
            task_name: 任务名称
            task_info: 任务信息
            train_loaders: {'seen': DataLoader, 'unseen': DataLoader}
            test_loaders: {'seen': DataLoader, 'unseen': DataLoader}
            num_epochs_stage1: 阶段1的训练轮数
            num_em_iterations: EM迭代次数
            num_epochs_per_em: 每次EM的E步训练轮数

        Returns:
            task_stats: 任务统计
        """
        self.logger.info(f"开始训练 Task {task_id}: {task_name}")
        self.logger.info(f"  Seen情绪: {task_info['seen_emotions']}")
        self.logger.info(f"  Unseen情绪: {task_info['unseen_emotions']}")

        # 扩展分类器以适应新的类别数
        num_classes_so_far = task_info['num_classes_so_far']
        self.logger.info(f"\n检查分类器维度...")
        self.logger.info(f"  当前类别数: {self.model.num_emotions}")
        self.logger.info(f"  需要类别数: {num_classes_so_far}")

        if num_classes_so_far > self.model.num_emotions:
            self.model.expand_classifiers(num_classes_so_far)
            # 更新优化器以包含新参数
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            self.logger.info(f"  优化器已更新")

        # 激活当前任务的所有情绪（seen + unseen）
        self.logger.info(f"\n激活当前任务情绪...")

        # 激活seen情绪
        for emotion_name, global_id in task_info['seen_emotions'].items():
            # 获取增量标签
            incremental_label = task_info['mapping_info']['seen_mapping'][emotion_name]
            self.model.au_emo_matrix.add_emotion(emotion_name, incremental_label)

        # 激活unseen情绪
        for emotion_name, global_id in task_info['unseen_emotions'].items():
            # 获取增量标签
            incremental_label = task_info['mapping_info']['unseen_mapping'][emotion_name]
            self.model.au_emo_matrix.add_emotion(emotion_name, incremental_label)

        self.logger.info(f"  当前激活情绪数: {self.model.au_emo_matrix.num_active_emotions}")

        task_stats = {
            'task_id': task_id,
            'task_name': task_name,
            'stage1_epochs': [],
            'stage2_em_iterations': []
        }

        # 阶段1: Seen训练
        self.logger.info(f"# 阶段1: Seen训练 ({num_epochs_stage1} epochs)")

        stage1_stats = self._stage1_seen_training(
            train_loaders['seen'],
            test_loaders['seen'],
            num_epochs_stage1,
            task_id
        )
        task_stats['stage1_epochs'] = stage1_stats

        # 提取seen分类器权重
        seen_classifier_weights = self._extract_seen_classifier_weights(task_info)

        # 阶段2: Unseen Zero-shot (EM迭代)
        if 'unseen' in train_loaders:
            self.logger.info(f"# 阶段2: Unseen Zero-shot (EM迭代)")

            stage2_stats = self._stage2_unseen_zeroshot(
                train_loaders['unseen'],
                test_loaders['unseen'],
                task_info,
                seen_classifier_weights,
                num_em_iterations,
                num_epochs_per_em
            )
            task_stats['stage2_em_iterations'] = stage2_stats
        else:
            self.logger.info(f"\n# 阶段2: 无unseen数据，跳过")

        # EWC合并
        if self.ewc is not None:
            self.logger.info(f"\n[EWC] 合并Fisher信息...")
            self.ewc.consolidate(train_loaders['seen'])

        # 保存任务检查点
        self._save_task_checkpoint(task_id, task_name, task_stats)

        return task_stats

    def _stage1_seen_training(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        task_id: int
    ) -> List[Dict]:
        """
        阶段1: Seen训练

        训练backbone + AU分支 + seen分类器
        """
        epoch_stats = []

        for epoch in range(num_epochs):
            self.model.train()

            total_loss = 0
            correct = 0
            total = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f'Stage1 Epoch {epoch+1}/{num_epochs}', ncols=80)

            for batch in progress_bar:
                # 获取数据
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)  # 全局增量标签

                # 前向传播
                outputs = self.model(text, audio, video)

                # 损失：AU路径 + 直接分类
                # 输出是[batch, num_emotions]，非激活情绪已mask为-inf
                loss_au_path = F.cross_entropy(outputs['emo_from_au'], labels)
                loss_direct = F.cross_entropy(outputs['emo_direct'], labels)

                loss = loss_au_path + loss_direct # 0.1*loss_direct

                # EWC惩罚
                if task_id!= 0 and self.ewc is not None and self.ewc.is_consolidated:
                    loss += self.ewc.penalty()

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if 'gradient_clip' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()

                # 统计
                total_loss += loss.item()
                num_batches += 1

                preds = outputs['emo_from_au'].argmax(dim=1)
                correc += (preds == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct / total,
                })

            # 评估
            avg_loss = total_loss / num_batches
            train_acc = correct / total
            test_acc = self._evaluate_seen(test_loader)

            epoch_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })

            self.logger.info(f"Stage1 Epoch {epoch+1}: loss={avg_loss:.4f}, "
                           f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        return epoch_stats

    def _stage2_unseen_zeroshot(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        task_info: Dict,
        seen_classifier_weights: torch.Tensor,
        num_em_iterations: int,
        num_epochs_per_em: int
    ) -> List[Dict]:
        """
        阶段2: Unseen Zero-shot (EM迭代)

        Args:
            train_loader: Unseen训练数据
            test_loader: Unseen测试数据
            task_info: 任务信息
            seen_classifier_weights: Seen分类器权重 [num_seen, weight_dim]
            num_em_iterations: EM迭代次数
            num_epochs_per_em: 每次E步的训练轮数
        """
        em_stats = []

        # 提取seen和unseen的情绪索引
        seen_indices = list(task_info['mapping_info']['seen_mapping'].values())
        unseen_indices = list(task_info['mapping_info']['unseen_mapping'].values())
        num_emotions = task_info['num_classes_so_far']

        self.logger.info(f"Seen indices: {seen_indices}")
        self.logger.info(f"Unseen indices: {unseen_indices}")

        # 创建seen mask
        seen_mask = torch.zeros(num_emotions, device=self.device)
        for idx in seen_indices:
            seen_mask[idx] = 1.0

        # EM迭代
        for em_iter in range(num_em_iterations):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"EM Iteration {em_iter+1}/{num_em_iterations}")
            self.logger.info(f"{'='*70}")

            # E步：更新zeroshotExpander
            self.logger.info(f"\n[E-Step] 训练 zeroshotExpander...")
            e_step_loss = self._em_e_step(
                seen_classifier_weights,
                seen_mask,
                num_emotions,
                num_epochs_per_em
            )

            # 生成所有情绪的权重
            all_weights = self._generate_all_weights()

            # M步：更新P(AU|EMO)
            self.logger.info(f"\n[M-Step] 更新 P(AU|EMO)...")
            m_step_stats = self._em_m_step(
                train_loader,
                all_weights,
                unseen_indices
            )

            # 检查收敛
            converged = m_step_stats['agreement'] >= self.config.get('convergence_threshold', 0.95)

            # 评估
            test_acc = self._evaluate_unseen(test_loader, all_weights, unseen_indices)

            em_stats.append({
                'em_iteration': em_iter + 1,
                'e_step_loss': e_step_loss,
                'm_step_agreement': m_step_stats['agreement'],
                'm_step_observations': m_step_stats['num_observations'],
                'test_acc': test_acc,
                'converged': converged
            })

            self.logger.info(f"EM Iter {em_iter+1}: e_loss={e_step_loss:.4f}, "
                           f"agreement={m_step_stats['agreement']:.4f}, "
                           f"test_acc={test_acc:.4f}, converged={converged}")

            if converged:
                self.logger.info(f"✓ 收敛于第 {em_iter+1} 次迭代")
                break

        # 保存最终的所有权重（包括seen和unseen）
        final_all_weights = self._generate_all_weights()
        self._save_classifier_weights(task_info['task_id'], final_all_weights, unseen_indices)

        return em_stats

    def _em_e_step(
        self,
        seen_classifier_weights: torch.Tensor,
        seen_mask: torch.Tensor,
        num_emotions: int,
        num_epochs: int
    ) -> float:
        """
        EM的E步：固定P(AU|EMO)，训练zeroshotExpander

        Returns:
            final_loss: 最终的训练损失
        """
        # # 获取当前P(AU|EMO)
        # p_au_emo = self.beta_prior.get_p_au_given_emo().cpu().numpy()

        # 获取当前P(AU|EMO)，只取前num_emotions个（即num_classes_so_far）
        p_au_emo_all = self.beta_prior.get_p_au_given_emo().cpu().numpy()
        p_au_emo = p_au_emo_all[:num_emotions]  # [num_classes_so_far, num_aus]

        # 构建转换矩阵
        trans_matrix = zeroshot_utils.get_transition_matrix(p_au_emo)

        # 构建类语义特征
        class_embeddings = zeroshot_utils.get_class_embedding(
            self.au_embeddings,
            p_au_emo
        ).to(self.device)

        # 创建或重新初始化zeroshotExpander
        weight_dim = seen_classifier_weights.shape[1]
        embedding_dim = class_embeddings.shape[1]

        hidden_layers = self.config.get('zeroshot_hidden_layers', 'd512,d1024,d512,d')

        self.zeroshot_expander = zeroshotExpander(
            n=num_emotions,
            edges=trans_matrix,
            in_channels=embedding_dim,
            out_channels=weight_dim,
            hidden_layers=hidden_layers
        ).to(self.device)

        self.current_num_emotions = num_emotions

        # 优化器
        self.zeroshot_optimizer = optim.Adam(
            self.zeroshot_expander.parameters(),
            lr=self.config.get('zeroshot_lr', 0.001)
        )

        # 准备目标权重 [num_emotions, weight_dim]
        # 将seen权重填入对应位置
        target_weights = torch.zeros(num_emotions, weight_dim, device=self.device)
        seen_indices = (seen_mask == 1.0).nonzero(as_tuple=True)[0]
        target_weights[seen_indices] = seen_classifier_weights

        # 训练zeroshotExpander
        final_loss = 0.0
        self.zeroshot_expander.train()

        for epoch in range(num_epochs):
            self.zeroshot_optimizer.zero_grad()

            # 前向传播
            output_weights = self.zeroshot_expander(class_embeddings)

            # 只在seen位置计算损失
            loss = zeroshot_utils.mask_l2_loss(
                output_weights,
                target_weights,
                seen_mask
            )

            # 反向传播
            loss.backward()
            self.zeroshot_optimizer.step()

            final_loss = loss.item()

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"  E-step epoch {epoch+1}/{num_epochs}: loss={loss.item():.6f}")

        return final_loss

    def _em_m_step(
        self,
        train_loader: DataLoader,
        all_weights: torch.Tensor,
        unseen_indices: List[int]
    ) -> Dict:
        """
        EM的M步：固定zeroshotExpander，更新P(AU|EMO)

        用unseen样本进行推理，累积统计量并更新Beta参数

        Args:
            train_loader: Unseen训练数据
            all_weights: 所有情绪的分类器权重 [num_classes_so_far, weight_dim]
            unseen_indices: Unseen情绪的全局索引列表

        Returns:
            stats: {'agreement': float, 'num_observations': int}
        """
        self.model.eval()
        self.zeroshot_expander.eval()

        # 重置观测计数
        self.beta_prior.reset_observations(unseen_indices)

        total_agreement = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(train_loader, desc='M-step', ncols=80):
                # 获取数据
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)

                batch_size = text.size(0)

                # 前向传播获取AU概率
                outputs = self.model(text, audio, video)
                au_probs = outputs['au_probs']  # [batch_size, num_aus]
                fused_features = outputs['fused_features']  # [batch_size, weight_dim]

                # 直接分类：用所有权重计算logits（包括seen+unseen）
                # all_weights: [num_classes_so_far, weight_dim]
                # fused_features: [batch_size, weight_dim]
                # logits: [batch_size, num_classes_so_far]
                logits_all = torch.mm(fused_features, all_weights.t())
                pseudo_labels = logits_all.argmax(dim=1)  # [batch_size] - 全局索引

                # AU路径分类：通过AU-EMO矩阵
                emo_from_au_logits = outputs['emo_from_au']  # [batch_size, num_classes_so_far]
                au_labels = emo_from_au_logits.argmax(dim=1)  # [batch_size] - 全局索引

                # 一致性检查（两种方式的预测是否一致）
                agreement = (pseudo_labels == au_labels).float().mean().item()
                total_agreement += agreement * batch_size
                total_samples += batch_size

                # 累积观测统计量（只对unseen情绪）
                # 过滤出被预测为unseen类别的样本
                is_unseen_pred = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                for idx in unseen_indices:
                    is_unseen_pred |= (pseudo_labels == idx)

                if is_unseen_pred.any():
                    self.beta_prior.accumulate_observations(
                        emotion_indices=pseudo_labels[is_unseen_pred],
                        au_probs=au_probs[is_unseen_pred]
                    )

        # 批量更新Beta参数
        self.beta_prior.update_beta_parameters(unseen_indices)

        avg_agreement = total_agreement / total_samples if total_samples > 0 else 0.0

        stats = {
            'agreement': avg_agreement,
            'num_observations': total_samples
        }

        return stats

    def _generate_all_weights(self) -> torch.Tensor:
        """
        用当前的zeroshotExpander生成所有情绪的分类器权重

        Returns:
            all_weights: [num_classes_so_far, weight_dim]
        """
        self.zeroshot_expander.eval()

        with torch.no_grad():
            # # 获取当前P(AU|EMO)
            # p_au_emo = self.beta_prior.get_p_au_given_emo().cpu().numpy()

            # 获取当前P(AU|EMO)，只取前current_num_emotions个（与zeroshotExpander的n一致）
            p_au_emo_all = self.beta_prior.get_p_au_given_emo().cpu().numpy()
            p_au_emo = p_au_emo_all[:self.current_num_emotions]  # [current_num_emotions, num_aus]

            # 构建类语义特征
            class_embeddings = zeroshot_utils.get_class_embedding(
                self.au_embeddings,
                p_au_emo
            ).to(self.device)

            # 生成所有情绪的权重
            all_weights = self.zeroshot_expander(class_embeddings)  # [num_classes_so_far, weight_dim]

        return all_weights

    def _extract_seen_classifier_weights(self, task_info: Dict) -> torch.Tensor:
        """
        从DirectEmotionClassifier提取seen情绪的分类器权重

        Returns:
            seen_weights: [num_seen, weight_dim]
        """
        # 获取分类器最后一层的权重矩阵
        classifier = self.model.emotion_classifier.classifier
        weight_matrix = classifier[-1].weight.data  # [num_total_emotions, weight_dim]

        # 提取seen情绪对应的权重
        seen_indices = list(task_info['mapping_info']['seen_mapping'].values())
        seen_weights = weight_matrix[seen_indices]  # [num_seen, weight_dim]

        return seen_weights

    def _evaluate_seen(self, test_loader: DataLoader) -> float:
        """评估seen样本的准确率"""
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(text, audio, video)
                preds = outputs['emo_from_au'].argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def _evaluate_unseen(
        self,
        test_loader: DataLoader,
        all_weights: torch.Tensor,
        unseen_indices: List[int]
    ) -> float:
        """
        评估unseen样本的准确率

        Args:
            test_loader: Unseen测试数据
            all_weights: 所有情绪的分类器权重 [num_classes_so_far, weight_dim]
            unseen_indices: Unseen情绪的全局索引

        Returns:
            accuracy: 准确率
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)  # 全局增量标签

                # 获取特征
                outputs = self.model(text, audio, video)
                fused_features = outputs['fused_features']  # [batch_size, weight_dim]

                # 用所有权重分类
                # all_weights: [num_classes_so_far, weight_dim]
                logits = torch.mm(fused_features, all_weights.t())  # [batch_size, num_classes_so_far]
                preds = logits.argmax(dim=1)  # [batch_size] - 全局索引

                # 计算准确率（预测和真实标签都是全局索引）
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def _save_classifier_weights(
        self,
        task_id: int,
        all_weights: torch.Tensor,
        unseen_indices: List[int]
    ):
        """保存分类器权重（seen + unseen）"""
        save_path = self.save_dir / f'task{task_id}_classifier_weights.pt'

        torch.save({
            'all_weights': all_weights.cpu(),
            'unseen_indices': unseen_indices,
            'num_classes': all_weights.shape[0]
        }, save_path)

        self.logger.info(f"分类器权重已保存: {save_path}")

    def _save_task_checkpoint(self, task_id: int, task_name: str, task_stats: Dict):
        """保存任务检查点"""
        checkpoint_path = self.save_dir / f'task{task_id}_final.pt'

        checkpoint = {
            'task_id': task_id,
            'task_name': task_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'task_stats': task_stats
        }

        if self.zeroshot_expander is not None:
            checkpoint['zeroshot_expander_state_dict'] = self.zeroshot_expander.state_dict()

        if self.ewc is not None:
            checkpoint['ewc_state'] = {
                'fisher_dict': self.ewc.fisher_dict,
                'optimal_params_dict': self.ewc.optimal_params_dict
            }

        torch.save(checkpoint, checkpoint_path)

        # 保存Beta先验
        beta_path = self.save_dir / f'task{task_id}_beta_prior.npz'
        self.beta_prior.save(str(beta_path))

        self.logger.info(f"\n任务检查点已保存:")
        self.logger.info(f"  模型: {checkpoint_path}")
        self.logger.info(f"  Beta先验: {beta_path}")
