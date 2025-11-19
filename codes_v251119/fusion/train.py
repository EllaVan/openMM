#!/usr/bin/env python
"""
样本级别超图网络训练脚本
"""
import os
import sys
from pathlib import Path
# 保存执行目录
exc_dir = str(Path(__file__).parent) #os.getcwd()

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils.core_tools import seed_init

from dataloader import create_emotion_dataloaders
from sample_network import SampleHypergraphClassifier


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        output = model(batch)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(output['logits'], dim=1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

        '''
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%"
            )
        '''
    logger.info(f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    return {'loss': total_loss / len(dataloader), 'accuracy': correct / total}


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            output = model(batch)
            total_loss += output['loss'].item()
            predictions = torch.argmax(output['logits'], dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    return {'loss': total_loss / len(dataloader), 'accuracy': correct / total}


@hydra.main(config_path="./config", config_name="config_fusion", version_base=None)
def run_main(cfg: DictConfig):
    os.chdir(exc_dir)  # Hydra会改变工作目录

    logger.info("="*70)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("="*70)

    device = torch.device(cfg.system.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    seed_init(cfg.system.random_seed)

    os.makedirs(cfg.system.save_dir, exist_ok=True)

    # 加载数据
    logger.info("加载数据...")
    dataloaders = create_emotion_dataloaders(cfg)

    batch = next(iter(dataloaders['train']))
    feature_dims = {
        'text': batch['text_features'].shape[-1],
        'audio': batch['audio_features'].shape[-1],
        'video': batch['video_features'].shape[-1]
    }
    logger.info(f"特征维度: {feature_dims}")

    # 计算总类别数 (seen + unseen)
    num_seen = len(cfg.dataset.seen_emotions)
    num_unseen = len(cfg.dataset.unseen_emotions) if cfg.dataset.unseen_emotions else 0
    num_classes = num_seen + num_unseen
    logger.info(f"分类类别数: {num_classes} (seen={num_seen}, unseen={num_unseen})")

    # 创建模型
    model_config = {
        'hypergraph_hidden_dim': cfg.model.hypergraph.hidden_dim,
        'num_conv_layers': cfg.model.hypergraph.num_conv_layers,
        'dropout': cfg.model.dropout,
        'use_edge_weights': cfg.model.sample_hypergraph.use_edge_weights,
        'similarity_temperature': cfg.model.sample_hypergraph.similarity_temperature
    }

    model = SampleHypergraphClassifier(feature_dims, num_classes, model_config).to(device)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    scheduler = None
    if cfg.training.scheduler.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg.training.scheduler.mode,
            factor=cfg.training.scheduler.factor,
            patience=cfg.training.scheduler.patience,
            # verbose=True,
        )

    # 训练
    best_acc, best_epoch = 0.0, 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_epoch(model, dataloaders['train'], optimizer, device, epoch)
        test_metrics = evaluate(model, dataloaders['test'], device)

        logger.info(f"Epoch {epoch}/{cfg.training.epochs}")
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {100.*train_metrics['accuracy']:.2f}%")
        logger.info(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {100.*test_metrics['accuracy']:.2f}%")

        if scheduler:
            scheduler.step(test_metrics['accuracy'])

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_epoch = epoch
            epochs_without_improvement = 0

            if cfg.experiment.save_best_model:
                save_path = os.path.join(cfg.system.save_dir, f"best_model_{cfg.experiment.name}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'num_classes': num_classes
                }, save_path)
                logger.info(f"✓ 保存最佳模型 (Acc: {100.*best_acc:.2f}%)")
        else:
            epochs_without_improvement += 1

        if cfg.training.early_stopping.use_early_stopping:
            if epochs_without_improvement >= cfg.training.early_stopping.patience:
                logger.info(f"早停触发！")
                break

    logger.info(f"训练完成！最佳准确率: {100.*best_acc:.2f}% (Epoch {best_epoch})")


if __name__ == '__main__':
    run_main()
