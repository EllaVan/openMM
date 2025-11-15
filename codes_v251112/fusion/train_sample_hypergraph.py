#!/usr/bin/env python
"""
样本级别超图网络训练脚本
"""

import os
import sys
import argparse
import logging
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config_utils import load_config
from dataloader import create_multi_emotion_dataloaders
from sample_network import SampleHypergraphClassifier


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch

    Returns:
        metrics: 训练指标
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        # 移动数据到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # 前向传播
        output = model(batch)
        loss = output['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()

        predictions = torch.argmax(output['logits'], dim=1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {100.*correct/total:.2f}%"
            )

    # 计算平均指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }

    return metrics


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """
    评估模型

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        metrics: 评估指标
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            output = model(batch)

            # 统计
            total_loss += output['loss'].item()

            predictions = torch.argmax(output['logits'], dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    # 计算平均指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练样本级别超图网络')
    parser.add_argument(
        '--config',
        type=str,
        default='./codes_v251112/fusion/config_sample_hypergraph.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    config.print_config()

    # 设置设备
    device_name = config.system.get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    seed = config.system.get('random_seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 创建保存目录
    save_dir = config.system.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = config.system.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)

    # 加载数据
    logger.info("加载数据...")
    dataloaders = create_multi_emotion_dataloaders(config)

    train_loader = dataloaders['train_seen']
    test_loader = dataloaders['test_seen']

    # 获取特征维度
    batch = next(iter(train_loader))
    feature_dims = {
        'text': batch['text_features'].shape[-1],
        'audio': batch['audio_features'].shape[-1],
        'video': batch['video_features'].shape[-1]
    }

    logger.info(f"特征维度: {feature_dims}")

    # 获取分类类别数
    num_classes = config.get_num_classes()
    logger.info(f"分类类别数: {num_classes}")

    # 创建模型
    logger.info("创建样本级别超图模型...")
    model_config = {
        'encoder_hidden_dim': config.model.get('encoder', {}).get('hidden_dim', 256),
        'encoder_output_dim': config.model.get('encoder', {}).get('output_dim', 256),
        'hypergraph_hidden_dim': config.model.get('hypergraph', {}).get('hidden_dim', 256),
        'num_conv_layers': config.model.get('hypergraph', {}).get('num_conv_layers', 2),
        'dropout': config.model.get('encoder', {}).get('dropout', 0.1),
        'pooling_type': config.model.get('pooling', {}).get('pooling_type', 'masked_mean'),
        'use_edge_weights': config.model.get('sample_hypergraph', {}).get('use_edge_weights', True),
        'similarity_temperature': config.model.get('sample_hypergraph', {}).get('similarity_temperature', 1.0)
    }

    logger.info(f"模型配置:")
    logger.info(f"  编码器隐藏层维度: {model_config['encoder_hidden_dim']}")
    logger.info(f"  超图隐藏层维度: {model_config['hypergraph_hidden_dim']}")
    logger.info(f"  超图卷积层数: {model_config['num_conv_layers']}")
    logger.info(f"  池化类型: {model_config['pooling_type']}")
    logger.info(f"  使用边权重: {model_config['use_edge_weights']}")
    logger.info(f"  相似度温度: {model_config['similarity_temperature']}")

    model = SampleHypergraphClassifier(
        feature_dims=feature_dims,
        num_classes=num_classes,
        config=model_config
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {num_params:,}")

    # 创建优化器
    lr = config.training.get('learning_rate', 1e-4)
    weight_decay = config.training.get('weight_decay', 1e-4)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 学习率调度器
    use_scheduler = config.training.get('scheduler', {}).get('use_scheduler', True)
    if use_scheduler:
        scheduler_mode = config.training.get('scheduler', {}).get('mode', 'max')
        scheduler_factor = config.training.get('scheduler', {}).get('factor', 0.5)
        scheduler_patience = config.training.get('scheduler', {}).get('patience', 5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True
        )
    else:
        scheduler = None

    # 训练
    epochs = config.training.get('epochs', 50)
    logger.info("开始训练...")
    logger.info(f"{'='*70}")

    best_acc = 0.0
    best_epoch = 0

    # 早停
    use_early_stopping = config.training.get('early_stopping', {}).get('use_early_stopping', False)
    early_stopping_patience = config.training.get('early_stopping', {}).get('patience', 10)
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # 评估
        test_metrics = evaluate(model, test_loader, device)

        # 打印结果
        logger.info(f"{'='*70}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {100.*train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Test  - Loss: {test_metrics['loss']:.4f}, "
            f"Acc: {100.*test_metrics['accuracy']:.2f}%"
        )

        # 学习率调度
        if scheduler is not None:
            scheduler.step(test_metrics['accuracy'])

        # 保存最佳模型
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_epoch = epoch
            epochs_without_improvement = 0

            if config.experiment.get('save_best_model', True):
                experiment_name = config.experiment.get('name', 'sample_hypergraph')
                save_path = os.path.join(
                    save_dir,
                    f"best_model_{experiment_name}.pth"
                )

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'config': model_config,
                    'feature_dims': feature_dims,
                    'num_classes': num_classes,
                    'seen_emotions': config.get_seen_emotions(),
                    'unseen_emotions': config.get_unseen_emotions()
                }, save_path)

                logger.info(f"✓ 保存最佳模型 (Acc: {100.*best_acc:.2f}%) -> {save_path}")
        else:
            epochs_without_improvement += 1

        # 检查点保存
        if config.experiment.get('save_checkpoints', False):
            checkpoint_freq = config.experiment.get('checkpoint_frequency', 10)
            if epoch % checkpoint_freq == 0:
                experiment_name = config.experiment.get('name', 'sample_hypergraph')
                checkpoint_path = os.path.join(
                    save_dir,
                    f"checkpoint_{experiment_name}_epoch{epoch}.pth"
                )

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_metrics['accuracy'],
                    'config': model_config,
                    'feature_dims': feature_dims,
                    'num_classes': num_classes
                }, checkpoint_path)

                logger.info(f"✓ 保存检查点 -> {checkpoint_path}")

        # 早停检查
        if use_early_stopping and epochs_without_improvement >= early_stopping_patience:
            logger.info(f"\n早停触发！{early_stopping_patience} 个epoch没有改进。")
            break

        logger.info(f"{'='*70}\n")

    # 训练完成
    logger.info(f"{'='*70}")
    logger.info(f"训练完成！")
    logger.info(f"最佳测试准确率: {100.*best_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
