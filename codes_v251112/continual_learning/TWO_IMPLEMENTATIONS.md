# Two Implementation Approaches / 两种实现方式

[English](#english) | [中文](#中文)

---

## English

### Overview

The continual learning framework now provides **two complete implementations** of the AU-EMO probability matrix updating mechanism:

1. **Whitebox Bayesian** (`whitebox_bayesian/`) - Interpretable Beta-Bernoulli approach
2. **Blackbox Learnable** (`blackbox_learnable/`) - End-to-end gradient descent approach

Both approaches implement the complete Task 0 to Task T training pipeline with:
- ✓ AU predictor network
- ✓ AU-EMO probability matrix
- ✓ Multimodal consistency checking
- ✓ EWC anti-forgetting
- ✓ Seen/unseen update strategies

### Quick Start

#### Whitebox Bayesian
```bash
cd codes_v251112/continual_learning/whitebox_bayesian

python whitebox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --num_epochs 10 \
    --prior_strength 100.0 \
    --seen_update_weight 1.0 \
    --unseen_update_weight 0.8 \
    --save_dir ../../checkpoints/whitebox
```

#### Blackbox Learnable
```bash
cd codes_v251112/continual_learning/blackbox_learnable

python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --num_epochs 10 \
    --prior_strength 0.1 \
    --seen_loss_weight 1.0 \
    --unseen_loss_weight 0.3 \
    --save_dir ../../checkpoints/blackbox
```

### Which One Should I Use?

**Use Whitebox if you need**:
- Full interpretability (explain every probability)
- Uncertainty quantification (know when model is uncertain)
- Limited data (< 1000 samples per class)
- Working with domain experts
- Academic research

**Use Blackbox if you need**:
- Best performance (potentially higher accuracy)
- Faster training (3x faster than whitebox)
- Simpler code (no EM algorithm)
- Large datasets (> 5000 samples per class)
- Production deployment

**Not sure?** Read the [detailed comparison](FRAMEWORK_COMPARISON.md) or try both!

### File Structure

```
continual_learning/
├── whitebox_bayesian/
│   ├── beta_bernoulli_matrix.py    # Beta-Bernoulli AU-EMO matrix
│   ├── em_trainer.py               # EM algorithm trainer
│   ├── whitebox_main.py            # Complete execution script
│   └── README.md                   # Detailed documentation
│
├── blackbox_learnable/
│   ├── learnable_matrix.py         # Learnable AU-EMO matrix
│   ├── gradient_trainer.py         # Gradient descent trainer
│   ├── blackbox_main.py            # Complete execution script
│   └── README.md                   # Detailed documentation
│
├── FRAMEWORK_COMPARISON.md         # Detailed comparison guide
├── TWO_IMPLEMENTATIONS.md          # This file
└── [shared modules...]              # Common utilities
```

### Key Differences

| Aspect | Whitebox | Blackbox |
|--------|----------|----------|
| **Matrix Type** | Beta(α, β) parameters | Learnable logits |
| **Training** | EM algorithm | Gradient descent |
| **Speed** | Slower (3 EM iterations) | Faster (single pass) |
| **Interpretability** | Full (α, β have meaning) | Limited (logits) |
| **Uncertainty** | Yes (Beta variance) | No |
| **Code Complexity** | Higher | Lower |
| **Performance** | Good with limited data | Better with large data |

### Next Steps

1. **Prepare your AU-EMO prior** (JSON format with P(AU|EMO) matrix)
2. **Choose an approach** (whitebox or blackbox)
3. **Run training** following the quickstart above
4. **Evaluate results** in the generated checkpoint directories
5. **Compare approaches** if needed using validation performance

For detailed usage instructions, see:
- [Whitebox README](whitebox_bayesian/README.md)
- [Blackbox README](blackbox_learnable/README.md)
- [Framework Comparison](FRAMEWORK_COMPARISON.md)

---

## 中文

### 概述

持续学习框架现在提供了AU-EMO概率矩阵更新机制的**两种完整实现**：

1. **白盒贝叶斯** (`whitebox_bayesian/`) - 可解释的Beta-Bernoulli方法
2. **黑盒可学习** (`blackbox_learnable/`) - 端到端梯度下降方法

两种方法都实现了从Task 0到Task T的完整训练流程，包括：
- ✓ AU预测器网络
- ✓ AU-EMO概率矩阵
- ✓ 多模态一致性检查
- ✓ EWC防遗忘机制
- ✓ Seen/unseen更新策略

### 快速开始

#### 白盒贝叶斯
```bash
cd codes_v251112/continual_learning/whitebox_bayesian

python whitebox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --num_epochs 10 \
    --prior_strength 100.0 \
    --seen_update_weight 1.0 \
    --unseen_update_weight 0.8 \
    --save_dir ../../checkpoints/whitebox
```

#### 黑盒可学习
```bash
cd codes_v251112/continual_learning/blackbox_learnable

python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --num_epochs 10 \
    --prior_strength 0.1 \
    --seen_loss_weight 1.0 \
    --unseen_loss_weight 0.3 \
    --save_dir ../../checkpoints/blackbox
```

### 我应该选择哪一个？

**选择白盒如果你需要**：
- 完全可解释性（解释每个概率）
- 不确定性量化（知道模型何时不确定）
- 有限数据（< 1000样本每类）
- 与领域专家合作
- 学术研究

**选择黑盒如果你需要**：
- 最佳性能（可能更高的准确率）
- 更快训练（比白盒快3倍）
- 更简单代码（无EM算法）
- 大型数据集（> 5000样本每类）
- 生产部署

**不确定？** 阅读[详细对比](FRAMEWORK_COMPARISON.md)或两者都尝试！

### 文件结构

```
continual_learning/
├── whitebox_bayesian/
│   ├── beta_bernoulli_matrix.py    # Beta-Bernoulli AU-EMO矩阵
│   ├── em_trainer.py               # EM算法训练器
│   ├── whitebox_main.py            # 完整执行脚本
│   └── README.md                   # 详细文档
│
├── blackbox_learnable/
│   ├── learnable_matrix.py         # 可学习AU-EMO矩阵
│   ├── gradient_trainer.py         # 梯度下降训练器
│   ├── blackbox_main.py            # 完整执行脚本
│   └── README.md                   # 详细文档
│
├── FRAMEWORK_COMPARISON.md         # 详细对比指南
├── TWO_IMPLEMENTATIONS.md          # 本文件
└── [共享模块...]                    # 公共工具
```

### 核心差异

| 方面 | 白盒 | 黑盒 |
|------|------|------|
| **矩阵类型** | Beta(α, β)参数 | 可学习logits |
| **训练方式** | EM算法 | 梯度下降 |
| **速度** | 较慢（3次EM迭代） | 较快（单次遍历） |
| **可解释性** | 完全（α, β有意义） | 有限（logits） |
| **不确定性** | 有（Beta方差） | 无 |
| **代码复杂度** | 较高 | 较低 |
| **性能** | 少量数据好 | 大量数据更好 |

### 下一步

1. **准备AU-EMO先验**（JSON格式，包含P(AU|EMO)矩阵）
2. **选择方法**（白盒或黑盒）
3. **运行训练**（按照上面的快速开始）
4. **评估结果**（在生成的checkpoint目录中）
5. **对比方法**（如需要，使用验证集性能对比）

详细使用说明请参见：
- [白盒README](whitebox_bayesian/README.md)
- [黑盒README](blackbox_learnable/README.md)
- [框架对比](FRAMEWORK_COMPARISON.md)

---

## Technical Details / 技术细节

### Whitebox Bayesian Approach / 白盒贝叶斯方法

**Mathematical Foundation / 数学基础**:
```
P(AU_i|EMO_j) ~ Beta(α_ij, β_ij)

Point Estimate / 点估计:
P(AU_i=1|EMO_j) = α_ij / (α_ij + β_ij)

Update Rule / 更新规则:
- Observe AU_i active: α_ij += weight
- Observe AU_i inactive: β_ij += weight

Uncertainty / 不确定性:
Var[P] = αβ / [(α+β)²(α+β+1)]
```

**Advantages / 优势**:
- ✓ Full statistical interpretation / 完全统计解释
- ✓ Built-in uncertainty / 内置不确定性
- ✓ Principled Bayesian updates / 原则性贝叶斯更新
- ✓ Never completely forgets prior / 永不完全忘记先验

**Disadvantages / 劣势**:
- ✗ Slower training (EM) / 训练较慢（EM算法）
- ✗ More complex code / 代码更复杂
- ✗ Fixed update assumptions / 固定更新假设

### Blackbox Learnable Approach / 黑盒可学习方法

**Mathematical Foundation / 数学基础**:
```
Matrix M ∈ ℝ^(num_aus × num_emotions)
M_ij = learnable logit (nn.Parameter)

Probability / 概率:
P(EMO_j|AU_i) = softmax_j(M_ij)

Update Rule / 更新规则:
∂L/∂M_ij via backpropagation
M_ij ← M_ij - lr × ∂L/∂M_ij

Regularization / 正则化:
L_reg = λ × KL(P_current || P_prior)
```

**Advantages / 优势**:
- ✓ Faster training / 训练更快
- ✓ Simpler implementation / 实现更简单
- ✓ More flexible / 更灵活
- ✓ Potential higher accuracy / 可能更高准确率

**Disadvantages / 劣势**:
- ✗ Less interpretable / 可解释性较差
- ✗ No uncertainty quantification / 无不确定性量化
- ✗ Sensitive to hyperparameters / 对超参数敏感
- ✗ May drift from prior / 可能偏离先验

---

## Experimental Recommendations / 实验建议

### Small Data Experiment / 小数据实验
```bash
# Test with 500 samples per class / 每类500个样本测试
# Expected: Whitebox performs better / 预期：白盒表现更好

# Whitebox
python whitebox_main.py --data_dir ./small_data --num_epochs 20

# Blackbox
python blackbox_main.py --data_dir ./small_data --num_epochs 20
```

### Large Data Experiment / 大数据实验
```bash
# Test with 5000+ samples per class / 每类5000+样本测试
# Expected: Blackbox performs better / 预期：黑盒表现更好

# Whitebox
python whitebox_main.py --data_dir ./large_data --num_epochs 10

# Blackbox
python blackbox_main.py --data_dir ./large_data --num_epochs 10
```

### Speed Comparison / 速度对比
```bash
# Time both approaches / 对比两种方法的时间

time python whitebox_main.py [...args]
# Typical: ~2-3 hours for 5 tasks / 典型：5个任务约2-3小时

time python blackbox_main.py [...args]
# Typical: ~1 hour for 5 tasks / 典型：5个任务约1小时
```

---

## Citation / 引用

If you use these implementations, please cite / 如果使用这些实现，请引用：

```bibtex
@software{continual_learning_dual_2024,
  title={Dual-Approach Continual Learning for Multimodal Emotion Recognition},
  author={openMM Team},
  year={2024},
  note={Whitebox Bayesian and Blackbox Learnable implementations}
}
```

---

## Support / 支持

For questions or issues / 如有问题：
- Read the detailed READMEs / 阅读详细README
- Check the comparison guide / 查看对比指南
- Open a GitHub issue / 提交GitHub issue

Happy experimenting! / 实验愉快！
