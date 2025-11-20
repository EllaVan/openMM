#!/usr/bin/env python3
"""诊断AU-EMO推理中的数值问题"""

import sys
sys.path.insert(0, '/home/user/openMM/codes_v251119_2')

# 模拟计算过程
print("=" * 70)
print("诊断AU-EMO推理的数值稳定性")
print("=" * 70)

# 模拟场景
num_aus = 23
num_emotions = 7

# 场景1：AU概率较小（常见情况）
print("\n场景1：AU概率较小（如0.1）")
au_prob_avg = 0.1
normalized_p_avg = 1.0 / num_emotions  # 约0.14

weighted_avg = au_prob_avg * normalized_p_avg
print(f"  平均 weighted = {weighted_avg:.6f}")

import math
log_weighted = math.log(weighted_avg + 1e-10)
print(f"  log(weighted) = {log_weighted:.2f}")

total_log_sum = log_weighted * num_aus
print(f"  在{num_aus}个AU上求和 = {total_log_sum:.2f}")
print(f"  ⚠️  这个值太负了！")

# 场景2：正确的对数空间计算
print("\n场景2：正确的对数空间计算")
log_au_prob = math.log(au_prob_avg + 1e-10)
log_normalized_p = math.log(normalized_p_avg + 1e-10)

log_product = log_au_prob + log_normalized_p
print(f"  log P(au|x) = {log_au_prob:.2f}")
print(f"  log P(AU|EMO) = {log_normalized_p:.2f}")
print(f"  log(乘积) = {log_product:.2f}")

total_correct = log_product * num_aus
print(f"  在{num_aus}个AU上求和 = {total_correct:.2f}")
print(f"  结果相同！数值问题依然存在")

# 问题根源
print("\n" + "=" * 70)
print("问题根源分析：")
print("=" * 70)
print("1. 对数空间累加23个负数会产生非常大的负值")
print("2. 即使AU概率=0.1（合理值），log(0.1)=-2.3，23个AU累加=-52")
print("3. Cross-entropy对极大负logits会产生巨大loss")
print()
print("这说明：**公式本身可能有问题**，或者需要不同的归一化方式")
print()

# 检查公式
print("检查原始公式：")
print("p(emo_k | x) ∝ ∏_i [P(au_i|x) * P(AU_i|EMO_k) / Σ_k P(AU_i|EMO_k)]")
print()
print("如果23个AU的概率都是0.1，那么：")
print("∏_i [0.1 * 0.14] = (0.014)^23 ≈ 1e-42")
print("这个值极其小，在对数空间就是 23*log(0.014) ≈ -97")
print()
print("⚠️  可能的问题：")
print("1. 公式中的连乘可能不合适（应该是某种形式的加权平均？）")
print("2. 或者需要额外的归一化/缩放")
print("3. 或者P(au_i|x)不应该直接参与连乘")

print("\n" + "=" * 70)
print("可能的解决方案：")
print("=" * 70)
print("方案1: 使用加权平均而非连乘")
print("  emo_score = Σ_i w_i * [P(au_i|x) * P(AU_i|EMO_k)]")
print()
print("方案2: 只对激活的AU计算（阈值>0.5）")
print("  只考虑显著激活的AU")
print()
print("方案3: 对logits进行温度缩放")
print("  emo_logits = log_sum / temperature")
print()
print("方案4: 重新审视公式的物理意义")
print("  确认是否应该是连乘")
