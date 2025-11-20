#!/usr/bin/env python3
"""深入分析loss=47的原因"""

import math

print("=" * 70)
print("分析loss=47的数值问题")
print("=" * 70)

# 模拟当前实现
num_aus = 23
num_emotions = 7

# 假设场景
print("\n场景1: 所有AU概率都较低（如0.1）")
au_prob = 0.1
p_au_emo = 0.14  # 平均 P(AU|EMO)
normalized_p = 0.14  # P(AU|EMO) / Σ_k P(AU|EMO) ≈ 1/7

# 当前实现：log(P(au|x) * normalized_p) 然后求和
weighted = au_prob * normalized_p  # 0.1 * 0.14 = 0.014
log_weighted = math.log(weighted)
total_log = log_weighted * num_aus

print(f"  weighted = {weighted:.6f}")
print(f"  log(weighted) = {log_weighted:.2f}")
print(f"  sum over {num_aus} AUs = {total_log:.2f}")
print(f"  加上log P(EMO) ≈ {math.log(1/7):.2f}")
final_logit = total_log + math.log(1/7)
print(f"  最终logit ≈ {final_logit:.2f}")

print("\n如果所有情绪的logits都是-99.7：")
print("  softmax后每个情绪概率 ≈ 1/7")
print("  cross-entropy = -log(1/7) ≈ 1.95")
print("  ⚠️  这不等于47！")

print("\n" + "=" * 70)
print("那么47是从哪来的？")
print("=" * 70)

# 可能的原因：不同情绪的logits差异很大
print("\n假设：不同情绪的P(AU|EMO)差异导致logits范围很大")
print()

# 情绪1：大部分AU的P(AU|EMO)较高
weighted_high = 0.3 * 0.3  # 更高的激活
log_high = math.log(weighted_high) * num_aus
print(f"情绪1 (高激活): logit ≈ {log_high:.2f}")

# 情绪2：大部分AU的P(AU|EMO)较低
weighted_low = 0.1 * 0.05  # 更低的激活
log_low = math.log(weighted_low) * num_aus
print(f"情绪2 (低激活): logit ≈ {log_low:.2f}")

print(f"\nlogit差异 = {log_high - log_low:.2f}")

# 计算cross-entropy
# 如果真实标签是情绪2（低激活的）
print("\n如果真实标签是低激活情绪：")
logits = [log_high, log_low]
max_logit = max(logits)
print(f"  logits = [{log_high:.1f}, {log_low:.1f}]")
print(f"  max_logit = {max_logit:.1f}")

# Softmax（数值稳定版本）
import math
exp_sum = sum(math.exp(l - max_logit) for l in logits)
prob_low = math.exp(log_low - max_logit) / exp_sum

print(f"  P(情绪2) = {prob_low:.10f}")
ce_loss = -math.log(prob_low)
print(f"  Cross-entropy = -log({prob_low:.10f}) = {ce_loss:.2f}")

if ce_loss > 40:
    print(f"  ✓ 找到了！当真实标签的logit远小于其他时，CE会爆炸")

print("\n" + "=" * 70)
print("问题根源")
print("=" * 70)
print("1. 连乘23个小概率导致极小的值（10^-42）")
print("2. 不同情绪的logit范围可能从-50到-150")
print("3. 当真实标签的logit=-150，而其他是-50时：")
print("   P(true) ≈ exp(-150) / [exp(-50) + exp(-150)] ≈ exp(-100) ≈ 10^-44")
print("   CE = -log(10^-44) = 44 * log(10) ≈ 101")
print()
print("但loss=47暗示logit差异约为47/log(e) ≈ 108")

print("\n" + "=" * 70)
print("可能的解决方案")
print("=" * 70)

print("\n方案1: 温度缩放")
print("  emo_logits = emo_logits / temperature")
print(f"  temperature = {num_aus} (AU数量)")
print(f"  缩放后logit: -99.7 / 23 ≈ -4.3")
print("  这会使logits回到合理范围")

print("\n方案2: 归一化常数")
print("  在连乘前除以某个常数，使数值保持在合理范围")
print("  类似于log-sum-exp trick")

print("\n方案3: 稀疏化（只用高激活AU）")
print("  只对P(au|x) > threshold的AU进行计算")
print("  减少连乘项数")

print("\n方案4: 改用平均而非连乘")
print("  score = exp(Σ_i log(...) / num_aus)")
print("  等价于几何平均")

print("\n推荐方案：温度缩放 (最简单且保持数学意义)")
