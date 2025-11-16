#!/usr/bin/env python
"""
样本级别超图网络训练脚本 - 使用 Hydra 配置
"""

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from pathlib import Path

# 保存执行目录
exc_dir = os.getcwd()

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from fusion.dataloader import create_multi_emotion_dataloaders_hydra
from fusion.sample_network import SampleHypergraphClassifier


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
    """训练一个epoch"""
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
    """评估模型"""
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


@hydra.main(config_path="config", config_name="config_sample_hypergraph", version_base=None)
def run_main(cfg: DictConfig):
    """主函数"""
    # Hydra 会改变工作目录，切换回执行目录
    os.chdir(exc_dir)

    # 打印配置
    logger.info("=" * 70)
    logger.info("配置信息:")
    logger.info("=" * 70)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 70)

    # 设置设备
    device = torch.device(cfg.system.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(cfg.system.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.system.random_seed)

    # 创建保存目录
    os.makedirs(cfg.system.save_dir, exist_ok=True)
    os.makedirs(cfg.system.log_dir, exist_ok=True)

    # 加载数据
    logger.info("加载数据...")
    dataloaders = create_multi_emotion_dataloaders_hydra(cfg)

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
    num_classes = len(cfg.dataset.seen_emotions)
    logger.info(f"分类类别数: {num_classes}")

    # 创建模型
    logger.info("创建样本级别超图模型...")
    model_config = {
        'encoder_hidden_dim': cfg.model.encoder.hidden_dim,
        'encoder_output_dim': cfg.model.encoder.output_dim,
        'hypergraph_hidden_dim': cfg.model.hypergraph.hidden_dim,
        'num_conv_layers': cfg.model.hypergraph.num_conv_layers,
        'dropout': cfg.model.encoder.dropout,
        'pooling_type': cfg.model.pooling.pooling_type,
        'use_edge_weights': cfg.model.sample_hypergraph.use_edge_weights,
        'similarity_temperature': cfg.model.sample_hypergraph.similarity_temperature
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # 学习率调度器
    scheduler = None
    if cfg.training.scheduler.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.scheduler.mode,
            factor=cfg.training.scheduler.factor,
            patience=cfg.training.scheduler.patience,
            verbose=True
        )

    # 训练
    logger.info("开始训练...")
    logger.info(f"{'='*70}")

    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.training.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # 评估
        test_metrics = evaluate(model, test_loader, device)

        # 打印结果
        logger.info(f"{'='*70}")
        logger.info(f"Epoch {epoch}/{cfg.training.epochs}")
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

            if cfg.experiment.save_best_model:
                save_path = os.path.join(
                    cfg.system.save_dir,
                    f"best_model_{cfg.experiment.name}.pth"
                )

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'feature_dims': feature_dims,
                    'num_classes': num_classes
                }, save_path)

                logger.info(f"✓ 保存最佳模型 (Acc: {100.*best_acc:.2f}%) -> {save_path}")
        else:
            epochs_without_improvement += 1

        # 早停检查
        if cfg.training.early_stopping.use_early_stopping:
            if epochs_without_improvement >= cfg.training.early_stopping.patience:
                logger.info(f"\n早停触发！{cfg.training.early_stopping.patience} 个epoch没有改进。")
                break

        logger.info(f"{'='*70}\n")

    # 训练完成
    logger.info(f"{'='*70}")
    logger.info(f"训练完成！")
    logger.info(f"最佳测试准确率: {100.*best_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    run_main()
