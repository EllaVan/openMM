"""
批量数据集特征提取脚本
使用配置文件批量处理多个数据集
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging

from extract_dataset_features import MOSEIFeatureExtractor, MELDFeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_extraction_config(config_file: str) -> dict:
    """加载提取配置文件"""
    with open(config_file, 'r') as f:
        return json.load(f)


def extract_mosei(config: dict):
    """提取 MOSEI 数据集特征"""
    logger.info("\n" + "="*70)
    logger.info("开始提取 MOSEI 数据集特征")
    logger.info("="*70)

    mosei_config = config['mosei']

    # 加载特征提取器配置
    if 'feature_config' in mosei_config and mosei_config['feature_config']:
        feature_config_path = mosei_config['feature_config']
        with open(feature_config_path, 'r') as f:
            feature_config = json.load(f)
    else:
        feature_config = None

    # 创建提取器
    extractor = MOSEIFeatureExtractor(
        base_dir=mosei_config['base_dir'],
        output_dir=mosei_config['output_dir'],
        label_file=mosei_config['label_file'],
        config=feature_config
    )

    # 处理数据集
    results = extractor.process_dataset()

    logger.info("\n✓ MOSEI 数据集特征提取完成")
    return results


def extract_meld(config: dict):
    """提取 MELD 数据集特征"""
    logger.info("\n" + "="*70)
    logger.info("开始提取 MELD 数据集特征")
    logger.info("="*70)

    meld_config = config['meld']

    # 加载特征提取器配置
    if 'feature_config' in meld_config and meld_config['feature_config']:
        feature_config_path = meld_config['feature_config']
        with open(feature_config_path, 'r') as f:
            feature_config = json.load(f)
    else:
        feature_config = None

    # 创建提取器
    extractor = MELDFeatureExtractor(
        base_dir=meld_config['base_dir'],
        output_dir=meld_config['output_dir'],
        config=feature_config
    )

    # 处理数据集
    split = meld_config.get('split', 'all')
    results = extractor.process_dataset(split=split)

    logger.info("\n✓ MELD 数据集特征提取完成")
    return results


def main():
    parser = argparse.ArgumentParser(description='批量数据集特征提取')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='提取配置文件路径'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mosei', 'meld', 'all'],
        default='all',
        help='要处理的数据集'
    )

    args = parser.parse_args()

    # 加载配置
    config = load_extraction_config(args.config)
    logger.info(f"加载配置文件: {args.config}")

    # 处理数据集
    if args.dataset in ['mosei', 'all'] and 'mosei' in config:
        try:
            extract_mosei(config)
        except Exception as e:
            logger.error(f"MOSEI 提取失败: {str(e)}")

    if args.dataset in ['meld', 'all'] and 'meld' in config:
        try:
            extract_meld(config)
        except Exception as e:
            logger.error(f"MELD 提取失败: {str(e)}")

    logger.info("\n" + "="*70)
    logger.info("✓ 所有数据集特征提取完成！")
    logger.info("="*70)


if __name__ == "__main__":
    main()
