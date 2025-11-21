# Zero-shot Continual Learning for Emotion Recognition

基于AU-EMO关系的零样本持续学习框架

## 架构概述

### 核心思想
通过P(AU|EMO)和AU embeddings构建情绪语义特征，用图卷积网络生成unseen情绪的分类器权重。

### 二阶段训练流程

**阶段1: Seen训练**
- 用seen样本训练backbone、AU分支和seen分类器
- 支持EWC防止灾难性遗忘

**阶段2: Unseen Zero-shot (EM迭代)**
- **E步**: 固定P(AU|EMO)，训练zeroshotExpander
  - 从P(AU|EMO)构建转换矩阵（边权）和类语义特征（节点）
  - 图卷积生成分类器权重
  - 在seen位置用mask_l2_loss监督

- **M步**: 固定zeroshotExpander，更新P(AU|EMO)
  - 直接分类器预测unseen样本的伪标签
  - AU分支获取p(au|x)
  - 用Beta分布贝叶斯更新P(AU|EMO)

- **收敛**: 直接分类预测 ≈ AU路径预测

## 目录结构

```
codes_v251119_2/
├── main_zeroshot.py              # 主训练脚本
├── config/
│   ├── train_config.yaml         # 训练配置
│   └── tasks.json                # 任务配置
├── core/
│   ├── au_emotion_network.py     # AU-情绪识别网络
│   ├── zeroshot_expander.py      # 图卷积扩展器
│   ├── beta_au_emo_prior.py      # Beta分布先验管理
│   ├── zeroshot_utils.py         # Zero-shot工具函数
│   └── ...
├── data/
│   └── dataloader.py             # 数据加载器（seen/unseen分离）
├── training/
│   └── two_stage_trainer.py      # 二阶段训练器
├── materials/
│   ├── au_emo_prior.json         # AU-EMO先验矩阵
│   └── au_embedding.pt           # AU语义embeddings
└── output/                       # 输出目录
    ├── zeroshot_continual/       # 模型检查点
    └── logs/                     # 训练日志
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision pyyaml numpy tqdm

# 检查数据
# 确保以下文件存在：
# - codes_v251119_2/materials/au_emo_prior.json
# - codes_v251119_2/materials/au_embedding.pt
# - 数据集文件（根据tasks.json中的data_dir配置）
```

### 2. 配置修改

编辑 `config/train_config.yaml`：

```yaml
# 主要配置项
data:
  batch_size: 32              # 根据GPU调整
  num_workers: 4

training:
  stage1_epochs: 20           # 阶段1训练轮数
  em_iterations: 20           # EM迭代次数
  epochs_per_em: 10           # 每次E步的训练轮数
  convergence_threshold: 0.95 # 收敛阈值

device: "cuda"                # 或 "cpu"
```

### 3. 运行训练

```bash
cd /home/user/openMM
python codes_v251119_2/main_zeroshot.py
```

### 4. 监控训练

训练日志保存在 `output/logs/` 目录：
- 控制台实时输出
- 日志文件详细记录

检查点保存在 `output/zeroshot_continual/` 目录：
- `task{id}_final.pt`: 每个任务完成后的检查点
- `task{id}_classifier_weights.pt`: 分类器权重（seen+unseen）
- `task{id}_beta_prior.npz`: Beta先验参数
- `final_model.pt`: 最终模型

## 关键配置说明

### 任务配置 (tasks.json)

定义每个任务的seen和unseen情绪：

```json
{
  "tasks": [
    {
      "task_id": 0,
      "seen_emotions": {"happy": 0, "sad": 1},
      "unseen_emotions": {"surprise": 2, "disgust": 3}
    }
  ]
}
```

### 训练配置 (train_config.yaml)

#### 模型架构
- `encoder_hidden_dim`: 单模态编码器隐藏层维度
- `hypergraph_hidden_dim`: 超图融合层维度
- `num_hyperedges`: 超边数量
- `num_aus`: AU数量（建议20）

#### 训练超参数
- `learning_rate`: 主优化器学习率
- `zeroshot_lr`: zeroshotExpander学习率
- `gradient_clip`: 梯度裁剪阈值

#### EM迭代控制
- `em_iterations`: 最大EM迭代次数
- `epochs_per_em`: 每次E步训练轮数
- `convergence_threshold`: 一致性阈值（0-1）

#### Beta先验
- `pseudo_count`: 伪计数，控制先验强度（推荐2.0）

## 输出说明

### 训练日志
```
EM Iteration 1/20:
  [E-Step] 训练 zeroshotExpander...
    E-step epoch 10/10: loss=0.0234
  [M-Step] 更新 P(AU|EMO)...
    更新了 2 个情绪的Beta参数
    情绪 2: 观测数=128
    情绪 3: 观测数=145
  EM Iter 1: e_loss=0.0234, agreement=0.8567, test_acc=0.7234, converged=False
```

### 检查点内容

**task_final.pt**:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'zeroshot_expander_state_dict': ...,
    'task_stats': {
        'stage1_epochs': [...],
        'stage2_em_iterations': [...]
    }
}
```

**classifier_weights.pt**:
```python
{
    'all_weights': tensor([num_classes, weight_dim]),
    'unseen_indices': [2, 3, ...],
    'num_classes': 4
}
```

## 常见问题

### Q: 内存不足
A: 降低batch_size，减少num_hyperedges

### Q: EM不收敛
A:
- 增加em_iterations
- 降低convergence_threshold
- 调整zeroshot_lr

### Q: Unseen准确率低
A:
- 增加epochs_per_em (E步训练轮数)
- 调整pseudo_count (Beta先验强度)
- 检查au_emo_prior.json是否合理

### Q: 训练速度慢
A:
- 增加num_workers
- 使用更小的模型（降低hidden_dim）
- 减少em_iterations

## 实验建议

### 小规模测试
```yaml
data:
  batch_size: 16
training:
  stage1_epochs: 5
  em_iterations: 5
  epochs_per_em: 3
```

### 完整训练
```yaml
data:
  batch_size: 32
training:
  stage1_epochs: 20
  em_iterations: 20
  epochs_per_em: 10
```

## 引用

如果您使用此代码，请引用相关论文。

## 联系

如有问题，请提Issue。
