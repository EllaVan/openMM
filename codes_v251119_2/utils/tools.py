"""
工具函数
"""

import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import datetime


def seed_init(seed: int = 2025):
    """
    设置随机种子以保证可复现性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了完全可复现，需要设置以下参数（但会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str,
    log_file: str,
    level=logging.INFO
) -> logging.Logger:
    """
    创建logger

    Args:
        name: logger名称
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        logger: 配置好的logger
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的handlers
    logger.handlers.clear()

    # 文件handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def make_saving_folder_and_logger(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    father_folder_name = os.path.join(cfg.output.log_dir, cfg.task.type) # 实验总文件夹名称
    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name, exist_ok=True)
    folder_name = f"{timestamp}" # 本次实验文件夹名称

    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_file = f'{father_folder_name}/{folder_name}/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # return father_folder_name, folder_name, logger
    return logger, log_file


def create_save_directory(base_dir: str, exp_name: str = None) -> Path:
    """
    创建保存目录

    Args:
        base_dir: 基础目录
        exp_name: 实验名称（可选）

    Returns:
        save_dir: 保存目录路径
    """
    if exp_name is None:
        exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_dir = Path(base_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir


def count_parameters(model: torch.nn.Module) -> dict:
    """
    统计模型参数量

    Args:
        model: PyTorch模型

    Returns:
        stats: 参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    **kwargs
):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        filepath: 保存路径
        **kwargs: 其他要保存的内容
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> dict:
    """
    加载训练检查点

    Args:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）

    Returns:
        checkpoint: 检查点内容
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
