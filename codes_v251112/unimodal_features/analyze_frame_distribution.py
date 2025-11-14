#!/usr/bin/env python
"""
分析数据集帧数分布，帮助设置最优 max_frames
"""

import os
import sys
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def analyze_mosei_frames(base_dir: str, label_file: str, sample_rate: int = 16000):
    """分析 MOSEI 数据集的帧数分布"""

    df = pd.read_csv(label_file)
    frame_counts = []
    durations = []

    print(f"分析 MOSEI 数据集...")
    print(f"总样本数: {len(df)}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="分析帧数"):
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        audio_path = os.path.join(base_dir, 'audio', video_id, f"{clip_id}.wav")

        if not os.path.exists(audio_path):
            continue

        try:
            # 加载音频获取时长
            waveform, sr = librosa.load(audio_path, sr=sample_rate, duration=None)
            duration = len(waveform) / sr

            # 估算帧数（HuBERT 约 50 帧/秒）
            estimated_frames = int(duration * 50)

            frame_counts.append(estimated_frames)
            durations.append(duration)

        except Exception as e:
            continue

    frame_counts = np.array(frame_counts)
    durations = np.array(durations)

    return frame_counts, durations


def print_statistics(frame_counts: np.ndarray, durations: np.ndarray):
    """打印统计信息"""

    print(f"\n{'='*60}")
    print(f"帧数统计分析")
    print(f"{'='*60}")
    print(f"有效样本数: {len(frame_counts)}")
    print(f"\n时长统计:")
    print(f"  平均时长: {np.mean(durations):.2f} 秒")
    print(f"  中位数: {np.median(durations):.2f} 秒")
    print(f"  最小值: {np.min(durations):.2f} 秒")
    print(f"  最大值: {np.max(durations):.2f} 秒")
    print(f"  标准差: {np.std(durations):.2f} 秒")

    print(f"\n帧数统计:")
    print(f"  平均帧数: {np.mean(frame_counts):.0f}")
    print(f"  中位数: {np.median(frame_counts):.0f}")
    print(f"  最小值: {np.min(frame_counts):.0f}")
    print(f"  最大值: {np.max(frame_counts):.0f}")
    print(f"  标准差: {np.std(frame_counts):.0f}")

    print(f"\n百分位数:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(frame_counts, p)
        percentage = (frame_counts <= value).sum() / len(frame_counts) * 100
        print(f"  {p}%: {value:.0f} 帧 (覆盖 {percentage:.1f}% 样本)")

    print(f"\n不同 max_frames 的影响:")
    max_frames_options = [300, 400, 500, 600, 800, 1000]
    for mf in max_frames_options:
        affected = (frame_counts > mf).sum()
        percentage = affected / len(frame_counts) * 100
        avg_loss = np.mean(np.maximum(0, frame_counts - mf))
        print(f"  max_frames={mf:4d}: {affected:5d} 样本受影响 ({percentage:5.2f}%), 平均损失 {avg_loss:.0f} 帧")

    print(f"{'='*60}\n")


def recommend_max_frames(frame_counts: np.ndarray):
    """推荐 max_frames 设置"""

    print(f"\n{'='*60}")
    print(f"推荐设置")
    print(f"{'='*60}")

    # 计算不同目标覆盖率下的 max_frames
    targets = [0.90, 0.95, 0.98, 0.99]

    for target in targets:
        recommended = int(np.percentile(frame_counts, target * 100))
        affected = (frame_counts > recommended).sum()
        percentage = (1 - target) * 100

        print(f"\n覆盖 {target*100:.0f}% 样本:")
        print(f"  推荐 max_frames = {recommended}")
        print(f"  受影响样本: {affected} ({percentage:.1f}%)")

        if target == 0.95:
            print(f"  ⭐ 推荐使用此设置（平衡性能和内存）")

    print(f"\n{'='*60}\n")


def plot_distribution(frame_counts: np.ndarray, output_file: str = 'frame_distribution.png'):
    """绘制帧数分布图"""

    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(frame_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.median(frame_counts), color='r', linestyle='--', label=f'中位数: {np.median(frame_counts):.0f}')
    plt.axvline(np.percentile(frame_counts, 95), color='g', linestyle='--', label=f'95%: {np.percentile(frame_counts, 95):.0f}')
    plt.xlabel('帧数')
    plt.ylabel('样本数')
    plt.title('帧数分布直方图')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 累积分布
    plt.subplot(1, 2, 2)
    sorted_frames = np.sort(frame_counts)
    cumulative = np.arange(1, len(sorted_frames) + 1) / len(sorted_frames) * 100
    plt.plot(sorted_frames, cumulative)
    plt.axhline(95, color='g', linestyle='--', label='95% 覆盖率')
    plt.axvline(np.percentile(frame_counts, 95), color='g', linestyle='--')
    plt.xlabel('帧数')
    plt.ylabel('累积百分比 (%)')
    plt.title('帧数累积分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 分布图已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析数据集帧数分布')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='MOSEI 数据集根目录')
    parser.add_argument('--label_file', type=str, required=True,
                       help='标签文件路径')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='音频采样率')
    parser.add_argument('--plot', action='store_true',
                       help='是否生成分布图')

    args = parser.parse_args()

    # 分析帧数分布
    frame_counts, durations = analyze_mosei_frames(
        args.base_dir,
        args.label_file,
        args.sample_rate
    )

    # 打印统计信息
    print_statistics(frame_counts, durations)

    # 推荐设置
    recommend_max_frames(frame_counts)

    # 绘图
    if args.plot:
        try:
            plot_distribution(frame_counts)
        except Exception as e:
            print(f"绘图失败: {str(e)}")


if __name__ == '__main__':
    main()
