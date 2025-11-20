#!/usr/bin/env python3
"""测试初始KL散度问题"""

import json
import sys
# sys.path.insert(0, '/home/user/openMM/codes_v251119_2')

# 使用纯Python实现关键函数
def sigmoid(x):
    """Sigmoid函数"""
    import math
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

def logit(p):
    """Sigmoid的反函数"""
    import math
    # Clip to avoid log(0) or log(negative)
    p_clipped = max(1e-7, min(p, 1 - 1e-7))
    return math.log(p_clipped / (1 - p_clipped))

def kl_divergence(p, q):
    """KL散度 KL(p||q) = sum(p * log(p/q))"""
    import math
    kl = 0.0
    eps = 1e-10
    for pi, qi in zip(p, q):
        if pi > eps:
            kl += pi * math.log((pi + eps) / (qi + eps))
    return kl

# 加载prior
with open('/media/sda/wf/openMM/codes_v251119_2/materials/au_emo_prior.json', 'r') as f:
    data = json.load(f)

prior_matrix = data['prior_matrix']

print("=" * 70)
print("测试初始化过程")
print("=" * 70)

# 测试sigmoid往返精度
print("\n1. 测试sigmoid(logit(p)) = p 的精度:")
test_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45]
print(f"   {'原始':>8} {'logit':>10} {'恢复':>8} {'误差':>12}")
max_error = 0
for p in test_values:
    l = logit(p)
    p_recovered = sigmoid(l)
    error = abs(p - p_recovered)
    max_error = max(max_error, error)
    print(f"   {p:>8.4f} {l:>10.4f} {p_recovered:>8.4f} {error:>12.10f}")

print(f"\n   最大误差: {max_error:.12f}")
print(f"   结论: sigmoid初始化精度{'正常' if max_error < 1e-6 else '有问题'}")

# 模拟完整初始化过程
print("\n2. 模拟完整矩阵的初始化:")
total_kl = 0.0
num_elements = 0

for i in range(len(prior_matrix)):
    for j in range(len(prior_matrix[0])):
        p_original = prior_matrix[i][j]

        # 初始化过程：p -> logit -> sigmoid
        l = logit(p_original)
        p_recovered = sigmoid(l)

        # 计算单个元素的KL散度贡献
        # 注意：完整KL需要在分布上计算，这里只是示意
        error = abs(p_original - p_recovered)

        if error > 1e-6:
            print(f"   [{i},{j}] {p_original:.4f} -> {p_recovered:.4f}, 误差={error:.10f}")

        num_elements += 1

print(f"\n   共{num_elements}个元素")

# 计算整体KL散度（将矩阵flatten后计算）
flat_original = [val for row in prior_matrix for val in row]
flat_recovered = [sigmoid(logit(val)) for row in prior_matrix for val in row]

# 归一化为概率分布（用于KL散度计算）
sum_original = sum(flat_original)
sum_recovered = sum(flat_recovered)

dist_original = [v/sum_original for v in flat_original]
dist_recovered = [v/sum_recovered for v in flat_recovered]

kl = kl_divergence(dist_original, dist_recovered)

print(f"\n3. 整体KL散度测试:")
print(f"   原始分布归一化和: {sum([v/sum_original for v in flat_original]):.10f}")
print(f"   恢复分布归一化和: {sum([v/sum_recovered for v in flat_recovered]):.10f}")
print(f"   KL(original || recovered): {kl:.10f}")
print(f"   ")
print(f"   结论: 如果KL接近0，说明初始化没问题")
print(f"         如果KL很大，说明有其他问题")

print("\n4. 可能的问题:")
print("   如果初始KL=2.48，可能原因：")
print("   (1) KL散度的计算方式不对（不是在归一化分布上计算）")
print("   (2) PyTorch的F.kl_div使用不当")
print("   (3) 其他代码逻辑问题")

print("\n" + "=" * 70)
