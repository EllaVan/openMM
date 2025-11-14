#!/usr/bin/env python
"""
批量提取数据集特征
支持 MOSEI 和 MELD 数据集
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unimodal_features.dataset_extractor import MOSEIFeatureExtractor, MELDFeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def extract_mosei(extraction_config: dict, feature_config: dict):
    """提取 MOSEI 数据集特征"""
    logger.info("="*60)
    logger.info("开始提取 MOSEI 数据集")
    logger.info("="*60)

    mosei_config = extraction_config['mosei']

    extractor = MOSEIFeatureExtractor(
        base_dir=mosei_config['base_dir'],
        output_dir=mosei_config['output_dir'],
        label_file=mosei_config['label_file'],
        config=feature_config
    )

    stats = extractor.process_dataset()

    logger.info("MOSEI 提取完成")
    logger.info(f"统计: {stats}")

    return stats


def extract_meld(extraction_config: dict, feature_config: dict):
    """提取 MELD 数据集特征"""
    logger.info("="*60)
    logger.info("开始提取 MELD 数据集")
    logger.info("="*60)

    meld_config = extraction_config['meld']

    extractor = MELDFeatureExtractor(
        base_dir=meld_config['base_dir'],
        output_dir=meld_config['output_dir'],
        config=feature_config
    )

    split = meld_config.get('split', 'all')
    stats = extractor.process_dataset(split=split)

    logger.info("MELD 提取完成")
    logger.info(f"统计: {stats}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='批量提取数据集特征')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['mosei', 'meld', 'all'],
        help='要提取的数据集'
    )
    parser.add_argument(
        '--extraction_config',
        type=str,
        default='unimodal_features/extraction_config.json',
        help='数据集配置文件路径'
    )
    parser.add_argument(
        '--feature_config',
        type=str,
        default='unimodal_features/config.json',
        help='特征提取配置文件路径'
    )

    args = parser.parse_args()

    # 加载配置
    logger.info(f"加载提取配置: {args.extraction_config}")
    with open(args.extraction_config, 'r', encoding='utf-8') as f:
        extraction_config = json.load(f)

    logger.info(f"加载特征配置: {args.feature_config}")
    with open(args.feature_config, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)

    # 提取特征
    results = {}

    if args.dataset in ['mosei', 'all']:
        try:
            results['mosei'] = extract_mosei(extraction_config, feature_config)
        except Exception as e:
            logger.error(f"MOSEI 提取失败: {str(e)}", exc_info=True)

    if args.dataset in ['meld', 'all']:
        try:
            results['meld'] = extract_meld(extraction_config, feature_config)
        except Exception as e:
            logger.error(f"MELD 提取失败: {str(e)}", exc_info=True)

    logger.info("="*60)
    logger.info("所有提取任务完成")
    logger.info("="*60)
    logger.info(f"结果: {results}")


if __name__ == '__main__':
    main()
