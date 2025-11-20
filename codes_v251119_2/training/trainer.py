"""
跨域零样本持续学习训练器

核心训练流程：
1. Task 0: 预热 + Seen训练 + Unseen伪标签训练 + EWC合并
2. Task 1-N: Seen训练(带EWC) + Unseen伪标签训练 + EWC合并
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    AUEmotionNetwork,
    LearnableAUEMOMatrix,
    MultimodalConsistencyChecker,
    ConsistencyStrategy,
    EWC
)
from utils import save_checkpoint


class ContinualLearningTrainer:
    """
    持续学习训练器

    负责整个3任务持续学习流程的训练
    """

    def __init__(
        self,
        model: AUEmotionNetwork,
        optimizer: optim.Optimizer,
        config: dict,
        logger,
        device: str = 'cuda'
    ):
        """
        Args:
            model: AU情绪识别网络（已集成AU-EMO矩阵）
            optimizer: 优化器
            config: 配置字典
            logger: 日志记录器
            device: 设备
        """
        self.model = model.to(device)
        self.au_emo_matrix = model.au_emo_matrix  # 从模型中获取矩阵引用
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.device = device

        # 持续学习组件
        self.ewc = None
        if config['continual_learning']['use_ewc']:
            self.ewc = EWC(
                model=self.model,
                device=device,
                ewc_lambda=config['continual_learning']['ewc_lambda']
            )

        # 一致性检查器
        strategy_map = {
            'all_agree': ConsistencyStrategy.ALL_AGREE,
            'majority': ConsistencyStrategy.MAJORITY,
            'weighted_vote': ConsistencyStrategy.WEIGHTED_VOTE,
            'entropy_threshold': ConsistencyStrategy.ENTROPY_THRESHOLD,
            'combined': ConsistencyStrategy.COMBINED
        }
        self.consistency_checker = MultimodalConsistencyChecker(
            model=self.model,
            strategy=strategy_map[config['continual_learning']['consistency_strategy']],
            min_confidence=config['continual_learning']['min_confidence'],
            device=device
        )

        # 训练统计
        self.training_stats = {
            'tasks': [],
            'matrix_evolution': []
        }

        # 保存目录
        self.save_dir = Path(config['output']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_task(
        self,
        task_id: int,
        task_name: str,
        task_info: Dict,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int
    ) -> Dict:
        """
        训练单个任务

        Args:
            task_id: 任务ID
            task_name: 任务名称
            task_info: 任务信息
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            num_epochs: 训练轮数

        Returns:
            task_stats: 任务训练统计
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"开始训练 Task {task_id}: {task_name}")
        self.logger.info(f"  数据集: {task_info['dataset_name']}")
        self.logger.info(f"  Seen情绪: {task_info['seen_emotions']}")
        self.logger.info(f"  Unseen情绪: {task_info['unseen_emotions']}")
        self.logger.info(f"{'='*80}")

        task_stats = {
            'task_id': task_id,
            'task_name': task_name,
            'epochs': []
        }

        # Task 0需要预热
        if task_id == 0:
            warmup_epochs = self.config['training'].get('warmup_epochs', 3)
            self.logger.info(f"\n[Phase 1] 预热训练 ({warmup_epochs} epochs)...")
            self._warmup_phase(train_loader, warmup_epochs)

        # 主训练循环
        for epoch in range(num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            self.logger.info(f"{'='*60}")

            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader, task_id, epoch)

            # 评估
            eval_metrics = self._evaluate(test_loader, task_id)

            # 记录
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['avg_loss'],
                'train_acc': train_metrics['accuracy'],
                'test_acc': eval_metrics['accuracy']
            }
            task_stats['epochs'].append(epoch_stats)

            self.logger.info(f"Epoch {epoch+1} 结果:")
            self.logger.info(f"  训练损失: {train_metrics['avg_loss']:.4f}")
            self.logger.info(f"  训练准确率: {train_metrics['accuracy']:.4f}")
            self.logger.info(f"  测试准确率: {eval_metrics['accuracy']:.4f}")

            # 保存矩阵统计
            matrix_stats = self.au_emo_matrix.get_statistics()
            self.logger.info(f"  矩阵MSE距离: {matrix_stats['mse_from_prior']:.4f}")
            self.logger.info(f"  矩阵MAE距离: {matrix_stats['mae_from_prior']:.4f}")

            # 定期保存
            if (epoch + 1) % self.config['output'].get('save_frequency', 5) == 0:
                self._save_checkpoint(task_id, epoch + 1)

        # EWC合并
        if self.ewc is not None:
            self.logger.info(f"\n[Phase 2] EWC Fisher信息合并...")
            self.ewc.consolidate(train_loader)

        # 保存任务检查点
        self._save_task_checkpoint(task_id, task_name)

        # 更新统计
        self.training_stats['tasks'].append(task_stats)

        return task_stats

    def _warmup_phase(self, train_loader: DataLoader, num_epochs: int):
        """
        预热阶段（仅Task 0）

        训练整个网络（包括矩阵）到合理状态: unimodal维度映射+ hypergraph融合+AU预测器+AU-EMO路径预测+情绪预测器
        """
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f'Warmup Epoch {epoch+1}/{num_epochs}', ncols=80)

            for batch in progress_bar:
                # 获取数据
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                outputs = self.model(text, audio, video)

                # 损失
                loss_au_path = F.cross_entropy(outputs['emo_from_au'], labels)
                loss_direct = F.cross_entropy(outputs['emo_direct'], labels)

                # 组合损失
                loss = loss_au_path + 0.1 * loss_direct

                # 轻量级矩阵正则化
                loss += 0.01 * self.au_emo_matrix.compute_regularization_loss()

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

                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / num_batches
            self.logger.info(f"Warmup Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    def _train_epoch(self, train_loader: DataLoader, task_id: int, epoch: int) -> Dict:
        """
        训练一个epoch

        分别处理seen和unseen样本
        """
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        seen_count = 0
        unseen_count = 0
        consistent_count = 0

        progress_bar = tqdm(train_loader, desc=f'Training', ncols=80)

        for batch in progress_bar:
            # 获取数据
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            is_seen = batch['is_seen'].to(self.device)

            # 分离seen和unseen
            seen_mask = is_seen
            unseen_mask = ~is_seen

            # 前向传播
            outputs = self.model(text, audio, video)

            # 初始化损失
            loss = 0.0

            # 1. Seen样本损失（有标签，高权重）
            if seen_mask.any():
                loss_seen = F.cross_entropy(
                    outputs['emo_from_au'][seen_mask],
                    labels[seen_mask]
                )
                loss += self.config['continual_learning']['seen_loss_weight'] * loss_seen
                seen_count += seen_mask.sum().item()

            # 2. Unseen样本损失（伪标签，低权重）
            if unseen_mask.any():
                # 一致性检查
                consistency_results = self.consistency_checker.check_consistency(
                    text[unseen_mask],
                    audio[unseen_mask],
                    video[unseen_mask]
                )

                is_consistent = consistency_results['is_consistent']
                pseudo_labels = consistency_results['consensus_label']
                confidence = consistency_results['confidence']

                if is_consistent.any():
                    # 只对一致的样本计算损失
                    loss_unseen = F.cross_entropy(
                        outputs['emo_from_au'][unseen_mask][is_consistent],
                        pseudo_labels[is_consistent],
                        reduction='none'
                    )

                    # 置信度加权
                    loss_unseen = (loss_unseen * confidence[is_consistent]).mean()
                    loss += self.config['continual_learning']['unseen_loss_weight'] * loss_unseen

                    consistent_count += is_consistent.sum().item()

                unseen_count += unseen_mask.sum().item()

            # 3. 矩阵正则化
            if self.config['continual_learning']['matrix_reg_lambda'] > 0:
                loss += (
                    self.config['continual_learning']['matrix_reg_lambda'] *
                    self.au_emo_matrix.compute_regularization_loss()
                )

            # 4. EWC惩罚
            if self.ewc is not None and self.ewc.is_consolidated:
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

            # 准确率（只计算seen样本）
            if seen_mask.any():
                preds = outputs['emo_from_au'][seen_mask].argmax(dim=1)
                correct += (preds == labels[seen_mask]).sum().item()
                total += seen_mask.sum().item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'seen': seen_count,
                'unseen': unseen_count,
                'consistent': consistent_count
            })

        # 计算平均指标
        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0

        self.logger.info(f"  Seen样本: {seen_count}, Unseen样本: {unseen_count}, 一致样本: {consistent_count}")

        return {
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'seen_count': seen_count,
            'unseen_count': unseen_count,
            'consistent_count': consistent_count
        }

    def _evaluate(self, test_loader: DataLoader, task_id: int) -> Dict:
        """
        评估模型

        Args:
            test_loader: 测试数据加载器
            task_id: 任务ID

        Returns:
            metrics: 评估指标
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(text, audio, video)
                preds = outputs['emo_from_au'].argmax(dim=1)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        accuracy = (all_preds == all_labels).float().mean().item()

        return {
            'accuracy': accuracy,
            'predictions': all_preds.cpu().numpy(),
            'labels': all_labels.cpu().numpy()
        }

    def _save_checkpoint(self, task_id: int, epoch: int):
        """保存检查点"""
        checkpoint_path = self.save_dir / f'task{task_id}_epoch{epoch}.pt'

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            filepath=str(checkpoint_path),
            task_id=task_id,
            matrix_stats=self.au_emo_matrix.get_statistics()
        )

        self.logger.info(f"  检查点已保存: {checkpoint_path}")

    def _save_task_checkpoint(self, task_id: int, task_name: str):
        """保存任务完成后的检查点"""
        checkpoint_path = self.save_dir / f'task{task_id}_final.pt'

        checkpoint = {
            'task_id': task_id,
            'task_name': task_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'matrix_stats': self.au_emo_matrix.get_statistics(),
            'training_stats': self.training_stats
        }

        if self.ewc is not None:
            checkpoint['ewc_state'] = {
                'fisher_dict': self.ewc.fisher_dict,
                'optimal_params_dict': self.ewc.optimal_params_dict
            }

        torch.save(checkpoint, checkpoint_path)

        # 保存矩阵
        matrix_path = self.save_dir / f'task{task_id}_matrix.npz'
        self.au_emo_matrix.save(str(matrix_path))

        self.logger.info(f"\n任务检查点已保存:")
        self.logger.info(f"  模型: {checkpoint_path}")
        self.logger.info(f"  矩阵: {matrix_path}")

    def save_final_model(self):
        """保存最终模型"""
        final_path = self.save_dir / 'final_model.pt'

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'training_stats': self.training_stats,
            'matrix_stats': self.au_emo_matrix.get_statistics()
        }

        torch.save(checkpoint, final_path)

        # 保存最终矩阵
        matrix_path = self.save_dir / 'final_matrix.npz'
        self.au_emo_matrix.save(str(matrix_path))

        self.logger.info(f"\n最终模型已保存:")
        self.logger.info(f"  模型: {final_path}")
        self.logger.info(f"  矩阵: {matrix_path}")
