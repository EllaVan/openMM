# Multimodal Continual Learning for Emotion Recognition

## 项目概述

本项目实现了一个完整的**多模态跨域零样本情感持续学习**框架，基于**Action Units (AU)** 作为中间表示来进行情感识别。

### 核心特性

1. **AU-EMO概率关联机制**
   - 全局共享的AU-EMO概率矩阵
   - 基于贝叶斯框架的动态更新
   - 区分seen/unseen类别的不同更新策略

2. **零样本学习能力**
   - 通过AU-EMO矩阵预测未见过的情感类别
   - 多模态一致性验证确保伪标签质量

3. **防遗忘机制**
   - Elastic Weight Consolidation (EWC)
   - 无需额外存储历史样本
   - 支持标准、在线和选择性EWC

4. **多模态融合**
   - 文本 (Text) + 音频 (Audio) + 视频 (Video)
   - 基于超图卷积的融合架构
   - 支持单模态和多模态一致性检查

## 任务定义

在每个训练阶段，模型接收：
- **当前域**的seen class样本（有标签）
- **当前域**的unseen class样本（无标签）

模型目标：
1. 识别当前阶段及之前所有阶段的seen class和unseen class
2. 保持之前域的学习结果，避免灾难性遗忘
3. 通过AU-EMO矩阵实现跨域知识迁移

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│          多模态输入 (Text + Audio + Video)               │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│          单模态编码 (Bi-LSTM x3)                         │
│  text_encoded [B,T,256] + audio + video                 │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│          多模态超图融合                                   │
│  - 超图初始化                                            │
│  - 超图卷积 x2层                                         │
│  - Bottleneck压缩                                        │
│  Output: fused_features [B, 768]                         │
└──────────────────────┬──────────────────────────────────┘
                       ↓
        ┌──────────────┴──────────────┐
        ↓                              ↓
┌──────────────────┐         ┌──────────────────┐
│  AU预测分支(新)  │         │  直接情感分类    │
│  au_probs [B,23] │         │  emo [B,6]       │
└────────┬─────────┘         └──────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│          AU-EMO概率矩阵 (全局共享)                       │
│  P(EMO|AU) ∈ R^[23×6]                                    │
│                                                           │
│  更新策略:                                                │
│  - Seen:   真实标签, 高权重 (weight=10.0)                │
│  - Unseen: 一致性预测, 低权重 (weight=1.0), 置信度过滤  │
└────────┬────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│          基于AU的情感预测                                 │
│  emo_pred = au_probs @ P(EMO|AU)                         │
└─────────────────────────────────────────────────────────┘
```

## 已实现的模块

### 1. 核心网络 (`continual_learning/`)

#### `au_emo_matrix.py`
- **AUEMOMatrix**: Dirichlet-Multinomial贝叶斯更新
- 支持心理学先验初始化
- 自动统计追踪和可视化

#### `au_emotion_network.py`
- **AUEmotionNetwork**: 扩展超图融合网络的AU预测分支
- **AUPredictor**: 23个AU的多标签分类器
- 双路径预测：AU路径 + 直接路径

#### `consistency_checker.py`
- **MultimodalConsistencyChecker**: 多模态一致性验证
- 5种策略：All-agree, Majority, Weighted-vote, Entropy, Combined
- **AdaptiveConsistencyChecker**: 自适应阈值调整

#### `ewc.py`
- **EWC**: 标准弹性权重巩固
- **OnlineEWC**: 在线EWC（指数移动平均）
- **SelectiveEWC**: 选择性保护重要参数

### 2. 训练框架 (`continual_learning/`)

#### `trainer.py`
- **ContinualLearningTrainer**: 主训练循环
- 自动管理seen/unseen训练阶段
- 集成EWC和一致性检查
- 支持任务检查点保存

#### `domain_splitter.py`
- **DomainSplitter**: 数据集划分工具
- **TaskConfig**: 任务配置类
- 4种划分策略：small_unseen, incremental, disjoint, overlap
- 预定义任务序列

#### `metrics.py`
- **ContinualLearningMetrics**: 持续学习评估指标
- Average Accuracy, Forgetting, Forward/Backward Transfer
- 性能矩阵可视化
- 学习曲线绘制

### 3. 工具和示例

#### `train_continual.py`
- 完整的训练脚本
- 命令行参数配置
- 自动保存和可视化

#### `example_au_emo_prior.json`
- AU-EMO先验矩阵模板
- 23个AU的示例定义
- 基于FACS的心理学映射

## 快速开始

### 1. 准备AU-EMO先验

请提供您的23个AU定义和心理学先验概率矩阵：

```json
{
  "au_names": ["AU1_Inner_Brow_Raiser", ...],
  "emotion_names": ["happy", "sad", "angry", "surprise", "disgust", "fear"],
  "prior_matrix": [
    [0.1, 0.2, 0.0, 0.3, 0.0, 0.1],  # AU1的情感概率
    ...
  ]
}
```

### 2. 运行训练

```bash
cd codes_v251112

# 使用预定义任务序列
python continual_learning/train_continual.py \
    --data_dir ../output/mosei_features \
    --au_prior_path continual_learning/au_emo_prior.json \
    --task_sequence custom \
    --num_epochs 10 \
    --batch_size 32 \
    --use_ewc \
    --save_dir ../checkpoints/continual_demo

# 使用自定义任务配置
python continual_learning/train_continual.py \
    --data_dir ../output/mosei_features \
    --au_prior_path continual_learning/au_emo_prior.json \
    --task_config_path my_tasks.json \
    --num_epochs 20 \
    --ewc_lambda 5000.0 \
    --consistency_strategy majority \
    --min_confidence 0.8
```

### 3. 在代码中使用

```python
from continual_learning import (
    AUEmotionNetwork,
    ContinualLearningTrainer,
    create_predefined_task_sequence,
    load_au_emo_prior
)

# 加载先验
prior_matrix, au_names, emotion_names = load_au_emo_prior('au_emo_prior.json')

# 创建模型
model = AUEmotionNetwork(
    text_input_dim=768,
    audio_input_dim=768,
    video_input_dim=768,
    num_aus=23,
    num_emotions=6,
    au_emo_prior=prior_matrix
)

# 创建训练器
trainer = ContinualLearningTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    use_ewc=True,
    ewc_lambda=1000.0
)

# 定义任务
tasks = create_predefined_task_sequence('custom')

# 训练
for task_config in tasks:
    seen_loader, unseen_loader = splitter.create_task_dataloaders(task_config)
    trainer.train_task(
        task_id=task_config.task_id,
        task_name=task_config.task_name,
        seen_loader=seen_loader,
        unseen_loader=unseen_loader,
        num_epochs=10
    )

# 保存
trainer.save_final_model()
```

## 任务配置示例

### 示例1：自定义3任务序列

```python
from continual_learning import TaskConfig

tasks = [
    TaskConfig(
        task_id=0,
        task_name="MOSEI_Task0",
        dataset_name="MOSEI",
        seen_classes=[0, 1],  # happy, sad
        unseen_classes=[2]     # angry
    ),
    TaskConfig(
        task_id=1,
        task_name="MOSEI_Task1",
        dataset_name="MOSEI",
        seen_classes=[0],      # happy
        unseen_classes=[4, 5]  # disgust, fear
    ),
    TaskConfig(
        task_id=2,
        task_name="MOSEI_Task2",
        dataset_name="MOSEI",
        seen_classes=[0, 1],  # happy, sad
        unseen_classes=[3]     # surprise
    )
]
```

### 示例2：使用域划分器

```python
from continual_learning import DomainSplitter

splitter = DomainSplitter(dataset, exclude_neutral=True)

# 自动生成任务（小样本类作为unseen）
tasks = splitter.create_tasks_by_strategy(
    strategy='small_unseen',
    num_tasks=3,
    seen_classes_base=[0, 1]  # happy, sad固定为seen
)

# 保存配置
splitter.save_task_configs(tasks, 'my_tasks.json')
```

## 预期性能

基于框架设计，预期性能：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Seen类准确率 | 80-95% | 取决于数据集质量 |
| Unseen类准确率 | 60-75% | 零样本学习，基于AU-EMO矩阵 |
| 遗忘率 | <10% | 使用EWC防遗忘 |
| 一致性通过率 | 20-40% | unseen样本中通过多模态一致性检查的比例 |

## 关键参数调优指南

### AU-EMO矩阵参数

```python
# 先验强度：越高越难更新
prior_strength = 100.0  # 默认值，适合大多数情况

# 更新权重
seen_update_weight = 10.0    # seen类：高权重
unseen_update_weight = 1.0   # unseen类：低权重，防止噪声

# 正则化：向先验回归的强度
au_emo_regularization = 0.01  # 每5个epoch执行一次
```

### EWC参数

```python
# 正则化强度
ewc_lambda = 1000.0  # 基准值
# 如果遗忘严重，增加到5000-10000
# 如果新任务学不好，减少到500-1000

# EWC类型选择
ewc_type = 'online'  # 推荐：内存高效
# 'standard': 标准EWC，每个任务单独Fisher矩阵
# 'selective': 只保护重要参数，提高可塑性
```

### 一致性检查参数

```python
# 策略选择
consistency_strategy = 'majority'  # 推荐：3/4一致即可
# 'all_agree': 最严格，4/4一致
# 'weighted_vote': 基于置信度加权
# 'combined': 结合多种策略

# 置信度阈值
min_confidence = 0.8  # 默认值
# 如果一致性通过率太低 (<10%)，降低到0.6-0.7
# 如果unseen类性能差，提高到0.85-0.9
```

## 文件结构

```
codes_v251112/
├── continual_learning/
│   ├── __init__.py                      # 模块导出
│   ├── README.md                        # 详细文档
│   ├── au_emo_matrix.py                # AU-EMO概率矩阵
│   ├── au_emotion_network.py           # 神经网络（含AU分支）
│   ├── consistency_checker.py          # 多模态一致性验证
│   ├── ewc.py                          # 弹性权重巩固
│   ├── trainer.py                      # 主训练循环
│   ├── domain_splitter.py              # 域划分工具
│   ├── metrics.py                      # 评估指标
│   ├── train_continual.py              # 训练脚本
│   └── example_au_emo_prior.json       # AU-EMO先验模板
├── hyper_fusion/                        # 基础超图融合网络
└── ...
```

## 待补充

为完成训练，您需要提供：

1. **AU定义**: 23个AU的名称和描述
2. **AU-EMO先验矩阵**: 基于心理学研究的先验概率
3. **数据集准备**: 确保MOSEI/MELD特征已提取

## 下一步计划

1. ✅ 核心框架实现
2. ✅ 训练脚本和工具
3. ⏳ **获取AU-EMO先验**（需要您提供）
4. ⏳ **在MOSEI数据集上验证**
5. ⏳ 跨数据集测试（MOSEI→MELD）
6. ⏳ 性能优化和消融实验

## 贡献者

- openMM Team
- Claude (AI Assistant)

## 许可证

MIT License

## 引用

如使用本框架，请引用：

```bibtex
@software{openMM_continual_learning_2024,
  title={Multimodal Continual Learning Framework for Emotion Recognition},
  author={openMM Team},
  year={2024},
  url={https://github.com/EllaVan/openMM}
}
```

---

**联系方式**: 如有问题，请在GitHub上提issue。
