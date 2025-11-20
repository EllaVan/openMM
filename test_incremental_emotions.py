#!/usr/bin/env python3
"""测试增量情绪处理的mask方案"""

import torch
import numpy as np
import sys
sys.path.append('/home/user/openMM/codes_v251119_2')

from core.learnable_matrix import LearnableAUEMOMatrix

print("=" * 70)
print("测试增量情绪处理（Mask方案）")
print("=" * 70)

# 创建一个简单的先验矩阵用于测试
num_aus = 23
num_emotions = 7

# 创建随机先验矩阵
np.random.seed(42)
prior_matrix = np.random.rand(num_emotions, num_aus) * 0.5 + 0.2  # 范围[0.2, 0.7]

print(f"\n初始化矩阵: {num_emotions}个情绪, {num_aus}个AUs")

# 初始化矩阵
matrix = LearnableAUEMOMatrix(
    num_aus=num_aus,
    num_emotions=num_emotions,
    prior_p_au_given_emo=prior_matrix,
    prior_strength=0.1,
    device='cpu'
)

print(f"✓ 矩阵初始化完成")
print(f"  活跃情绪数: {matrix.num_active_emotions}/{matrix.num_emotions}")

# 模拟Task 0：激活happy, sad, surprise, disgust
print("\n" + "=" * 70)
print("模拟 Task 0: 激活 happy(0), sad(1), surprise(2), disgust(3)")
print("=" * 70)

emotion_mapping = {
    "happy": 0,
    "sad": 1,
    "surprise": 2,
    "disgust": 3,
    "anger": 4,
    "fear": 5,
    "joy": 6
}

task0_emotions = [("happy", 0), ("sad", 1), ("surprise", 2), ("disgust", 3)]
for emo_name, emo_id in task0_emotions:
    matrix.add_emotion(emo_name, emo_id)

print(f"\n活跃情绪: {matrix.get_active_emotions()}")

# 测试forward pass
print("\n测试 forward pass (Task 0):")
batch_size = 4
au_probs = torch.rand(batch_size, num_aus)  # 随机AU概率
print(f"  输入: au_probs shape = {au_probs.shape}")

emo_logits = matrix(au_probs)
print(f"  输出: emo_logits shape = {emo_logits.shape}")
print(f"  预期: [{batch_size}, {matrix.num_active_emotions}] = [{batch_size}, 4]")

if emo_logits.shape == (batch_size, 4):
    print("  ✓ 输出维度正确！")
else:
    print(f"  ✗ 输出维度错误！预期 ({batch_size}, 4)，实际 {emo_logits.shape}")

# 检查softmax后的概率和是否为1
emo_probs = torch.softmax(emo_logits, dim=1)
prob_sums = emo_probs.sum(dim=1)
print(f"\n  Softmax后的概率和: {prob_sums}")
print(f"  是否接近1.0: {torch.allclose(prob_sums, torch.ones_like(prob_sums))}")

# 模拟Task 1：新增anger, fear
print("\n" + "=" * 70)
print("模拟 Task 1: 新增 anger(4), fear(5)")
print("=" * 70)

task1_new_emotions = [("anger", 4), ("fear", 5)]
for emo_name, emo_id in task1_new_emotions:
    matrix.add_emotion(emo_name, emo_id)

print(f"\n活跃情绪: {matrix.get_active_emotions()}")

# 测试forward pass
print("\n测试 forward pass (Task 1):")
emo_logits = matrix(au_probs)
print(f"  输出: emo_logits shape = {emo_logits.shape}")
print(f"  预期: [{batch_size}, {matrix.num_active_emotions}] = [{batch_size}, 6]")

if emo_logits.shape == (batch_size, 6):
    print("  ✓ 输出维度正确！")
else:
    print(f"  ✗ 输出维度错误！预期 ({batch_size}, 6)，实际 {emo_logits.shape}")

# 模拟Task 2：新增joy
print("\n" + "=" * 70)
print("模拟 Task 2: 新增 joy(6)")
print("=" * 70)

matrix.add_emotion("joy", 6)
print(f"\n活跃情绪: {matrix.get_active_emotions()}")

# 测试forward pass
print("\n测试 forward pass (Task 2):")
emo_logits = matrix(au_probs)
print(f"  输出: emo_logits shape = {emo_logits.shape}")
print(f"  预期: [{batch_size}, {matrix.num_active_emotions}] = [{batch_size}, 7]")

if emo_logits.shape == (batch_size, 7):
    print("  ✓ 输出维度正确！")
else:
    print(f"  ✗ 输出维度错误！预期 ({batch_size}, 7)，实际 {emo_logits.shape}")

# 测试重复激活
print("\n" + "=" * 70)
print("测试重复激活（应该跳过）")
print("=" * 70)
matrix.add_emotion("happy", 0)  # 已经激活，应该跳过

# 测试保存和加载
print("\n" + "=" * 70)
print("测试保存和加载")
print("=" * 70)

save_path = "/tmp/test_matrix_with_mask.npz"
matrix.save(save_path)

# 创建新矩阵并加载
matrix2 = LearnableAUEMOMatrix(
    num_aus=num_aus,
    num_emotions=num_emotions,
    prior_p_au_given_emo=prior_matrix,
    prior_strength=0.1,
    device='cpu'
)

print(f"\n加载前: 活跃情绪数 = {matrix2.num_active_emotions}")
matrix2.load(save_path)
print(f"加载后: 活跃情绪数 = {matrix2.num_active_emotions}")
print(f"活跃情绪列表: {matrix2.get_active_emotions()}")

# 验证加载后的forward pass
emo_logits2 = matrix2(au_probs)
print(f"\n加载后的forward pass:")
print(f"  输出shape: {emo_logits2.shape}")
print(f"  与原矩阵输出相同: {torch.allclose(emo_logits, emo_logits2)}")

# 测试重置
print("\n" + "=" * 70)
print("测试重置活跃情绪")
print("=" * 70)

matrix.reset_active_emotions()
print(f"重置后活跃情绪数: {matrix.num_active_emotions}")
emo_logits_empty = matrix(au_probs)
print(f"重置后forward pass输出shape: {emo_logits_empty.shape}")
print(f"预期: [{batch_size}, 0]")

if emo_logits_empty.shape == (batch_size, 0):
    print("✓ 重置后返回空tensor，符合预期！")
else:
    print(f"✗ 重置后输出维度错误！")

print("\n" + "=" * 70)
print("所有测试完成！")
print("=" * 70)
