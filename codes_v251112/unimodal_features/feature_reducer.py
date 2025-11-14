#!/usr/bin/env python
"""
特征降维工具
支持多种降维方法：PCA, SVD, Autoencoder, Linear Projection
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, TruncatedSVD
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse


class FeatureReducer:
    """特征降维器"""

    def __init__(self, method: str = 'pca', target_dim: int = 384):
        """
        初始化降维器

        Args:
            method: 降维方法 ['pca', 'svd', 'autoencoder', 'linear']
            target_dim: 目标维度
        """
        self.method = method
        self.target_dim = target_dim
        self.reducer = None

        print(f"\n{'='*60}")
        print(f"特征降维器")
        print(f"{'='*60}")
        print(f"方法: {method}")
        print(f"目标维度: {target_dim}")
        print(f"{'='*60}\n")

    def fit(self, features: np.ndarray):
        """
        训练降维模型

        Args:
            features: [num_samples, original_dim] 或 [num_samples, seq_len, original_dim]
        """
        print(f"训练降维模型...")
        print(f"输入形状: {features.shape}")

        # 如果是3D，展平为2D
        original_shape = features.shape
        if len(features.shape) == 3:
            num_samples, seq_len, feat_dim = features.shape
            features = features.reshape(-1, feat_dim)
            print(f"展平为: {features.shape}")

        if self.method == 'pca':
            self.reducer = PCA(n_components=self.target_dim)
            self.reducer.fit(features)
            explained_variance = np.sum(self.reducer.explained_variance_ratio_)
            print(f"✓ PCA 训练完成")
            print(f"  保留方差: {explained_variance:.2%}")

        elif self.method == 'svd':
            self.reducer = TruncatedSVD(n_components=self.target_dim)
            self.reducer.fit(features)
            explained_variance = np.sum(self.reducer.explained_variance_ratio_)
            print(f"✓ SVD 训练完成")
            print(f"  保留方差: {explained_variance:.2%}")

        elif self.method == 'linear':
            # 简单的线性投影（随机初始化）
            original_dim = features.shape[1]
            self.reducer = np.random.randn(original_dim, self.target_dim).astype(np.float32)
            # 标准化
            self.reducer = self.reducer / np.linalg.norm(self.reducer, axis=0, keepdims=True)
            print(f"✓ 线性投影初始化完成")

        elif self.method == 'autoencoder':
            print("⚠ Autoencoder 降维需要单独训练，这里提供接口")
            self.reducer = None

        else:
            raise ValueError(f"不支持的降维方法: {self.method}")

        print()

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        应用降维

        Args:
            features: [num_samples, original_dim] 或 [num_samples, seq_len, original_dim]

        Returns:
            reduced_features: 降维后的特征
        """
        original_shape = features.shape

        # 如果是3D，记住原始形状
        if len(features.shape) == 3:
            num_samples, seq_len, feat_dim = features.shape
            features = features.reshape(-1, feat_dim)
            is_3d = True
        else:
            is_3d = False

        # 降维
        if self.method in ['pca', 'svd']:
            reduced = self.reducer.transform(features)
        elif self.method == 'linear':
            reduced = features @ self.reducer
        else:
            raise ValueError(f"Reducer not fitted")

        # 恢复3D形状
        if is_3d:
            reduced = reduced.reshape(num_samples, seq_len, self.target_dim)

        return reduced

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """训练并应用降维"""
        self.fit(features)
        return self.transform(features)

    def save(self, save_path: str):
        """保存降维模型"""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'target_dim': self.target_dim,
                'reducer': self.reducer
            }, f)
        print(f"✓ 降维模型已保存: {save_path}")

    def load(self, load_path: str):
        """加载降维模型"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.method = data['method']
        self.target_dim = data['target_dim']
        self.reducer = data['reducer']
        print(f"✓ 降维模型已加载: {load_path}")


def reduce_dataset_features(
    input_dir: str,
    output_dir: str,
    method: str = 'pca',
    target_dim: int = 384,
    modalities: List[str] = ['text', 'audio', 'video']
):
    """
    对整个数据集的特征进行降维

    Args:
        input_dir: 输入特征目录（768维）
        output_dir: 输出目录（降维后）
        method: 降维方法
        target_dim: 目标维度
        modalities: 要降维的模态列表
    """
    print(f"\n{'='*60}")
    print(f"批量降维数据集特征")
    print(f"{'='*60}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"降维方法: {method}")
    print(f"目标维度: {target_dim}")
    print(f"模态: {modalities}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 获取所有特征文件
    feature_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    print(f"找到 {len(feature_files)} 个特征文件\n")

    # 为每个模态创建降维器
    reducers = {}
    for modality in modalities:
        reducers[modality] = FeatureReducer(method=method, target_dim=target_dim)

    # 第一阶段：收集所有特征用于训练降维模型
    print("阶段 1: 收集特征训练降维模型")
    print("-" * 60)

    all_features = {modality: [] for modality in modalities}

    for filename in tqdm(feature_files, desc="收集特征"):
        filepath = os.path.join(input_dir, filename)

        with open(filepath, 'rb') as f:
            samples = pickle.load(f)

        for sample in samples:
            for modality in modalities:
                key = f"{modality}_features"
                if key in sample:
                    features = sample[key]
                    if isinstance(features, torch.Tensor):
                        features = features.numpy()
                    all_features[modality].append(features)

    # 训练降维模型
    print("\n训练降维模型:")
    for modality in modalities:
        if len(all_features[modality]) > 0:
            # 合并所有样本
            combined = np.concatenate(all_features[modality], axis=0)
            print(f"\n{modality.upper()}:")
            print(f"  总特征数: {combined.shape[0]}")

            # 训练
            reducers[modality].fit(combined)

            # 保存降维模型
            reducer_path = os.path.join(output_dir, f'reducer_{modality}_{method}_{target_dim}.pkl')
            reducers[modality].save(reducer_path)

    del all_features  # 释放内存

    # 第二阶段：应用降维并保存
    print(f"\n{'='*60}")
    print("阶段 2: 应用降维并保存")
    print("-" * 60)

    total_size_before = 0
    total_size_after = 0

    for filename in tqdm(feature_files, desc="降维特征"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'rb') as f:
            samples = pickle.load(f)

        # 计算原始大小
        import sys
        total_size_before += sys.getsizeof(pickle.dumps(samples))

        # 降维每个样本
        for sample in samples:
            for modality in modalities:
                key = f"{modality}_features"
                if key in sample:
                    features = sample[key]
                    if isinstance(features, torch.Tensor):
                        features = features.numpy()

                    # 应用降维
                    reduced = reducers[modality].transform(features)
                    sample[key] = reduced

        # 保存降维后的特征
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)

        # 计算降维后大小
        total_size_after += sys.getsizeof(pickle.dumps(samples))

    # 统计
    print(f"\n{'='*60}")
    print("降维完成")
    print(f"{'='*60}")
    print(f"原始大小: {total_size_before / 1e9:.2f} GB")
    print(f"降维后大小: {total_size_after / 1e9:.2f} GB")
    print(f"压缩率: {total_size_after / total_size_before:.2%}")
    print(f"减少: {(1 - total_size_after / total_size_before):.2%}")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征降维工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入特征目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--method', type=str, default='pca',
                        choices=['pca', 'svd', 'linear'],
                        help='降维方法')
    parser.add_argument('--target_dim', type=int, default=384,
                        help='目标维度')
    parser.add_argument('--modalities', type=str, default='text,audio,video',
                        help='要降维的模态，逗号分隔')

    args = parser.parse_args()

    modalities = args.modalities.split(',')

    reduce_dataset_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        target_dim=args.target_dim,
        modalities=modalities
    )


if __name__ == '__main__':
    main()
