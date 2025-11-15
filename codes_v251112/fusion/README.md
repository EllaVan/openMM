# 多模态融合训练 - 支持 Seen/Unseen Emotions

## 📋 概述

这个目录包含基于超图融合网络的多模态情感识别系统，支持：
- ✅ 统一的 YAML 配置文件管理所有参数
- ✅ 支持 Seen/Unseen Emotions 的数据加载
- ✅ 灵活的标签映射和数据划分
- ✅ 完整的训练和评估流程

## 📁 文件结构

```
fusion/
├── config.yaml           # 统一配置文件 ⭐
├── config_utils.py       # 配置读取工具
├── dataloader.py         # 数据加载器（支持 seen/unseen emotions）
├── train.py              # 训练脚本
├── instruct.md           # 数据集格式说明
└── README.md             # 本文档
```

## ⚙️ 配置文件说明

配置文件 `config.yaml` 包含所有训练参数，分为以下几个部分：

### 1. 数据集配置

```yaml
dataset:
  name: "MELD"  # 数据集名称: MOSEI 或 MELD
  data_dir: "./output/meld_utterance_features"  # 数据目录

  # Seen emotions (有标签，用于训练)
  seen_emotions:
    happy: 0
    sad: 1

  # Unseen emotions (无标签，可选)
  unseen_emotions:
    fear: 5

  train_ratio: 0.7  # 训练集比例（MOSEI）
  merge_train_dev: true  # 是否合并 train 和 dev（MELD）
```

**说明：**
- `seen_emotions`: 指定用于训练的情感类型及其原始标签 ID
- `unseen_emotions`: 可选的未见过的情感类型（用于零样本学习或评估）
- 标签会自动重新映射：seen emotions 的标签会被映射到 0, 1, 2, ...

### 2. 数据加载配置

```yaml
dataloader:
  batch_size: 32
  num_workers: 4
  shuffle_train: true
  pin_memory: true
```

### 3. 模型配置

```yaml
model:
  encoder:
    hidden_dim: 256
    output_dim: 256
    dropout: 0.1

  hypergraph:
    hidden_dim: 256
    num_hyperedges: 64
    num_conv_layers: 2
    hyperedge_drop_rate: 0.2

  bottleneck:
    use_bottleneck: true
    bottleneck_dim: 128

  contrastive:
    use_contrastive: true
    contrastive_weight: 0.1
```

### 4. 训练配置

```yaml
training:
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001

  scheduler:
    use_scheduler: true
    mode: "max"
    factor: 0.5
    patience: 5

  early_stopping:
    use_early_stopping: true
    patience: 10
```

### 5. 系统配置

```yaml
system:
  device: "cuda"
  random_seed: 42
  save_dir: "./checkpoints"
  log_dir: "./logs"
  use_amp: false  # 混合精度训练
```

### 6. 实验配置

```yaml
experiment:
  name: "fusion_seen_happy_sad_unseen_fear"
  description: "训练happy和sad的二分类模型，使用fear作为unseen emotion"
  save_best_model: true
  save_checkpoints: true
  checkpoint_frequency: 10
```

## 🚀 使用方法

### 1. 准备数据

确保数据集已经按照 `instruct.md` 中的格式准备好：

**MOSEI 数据集:**
```
output/mosei_utterance_features/
├── MOSEIhappylabel0.pkl
├── MOSEIsadlabel1.pkl
├── MOSEIfearlabel5.pkl
└── ...
```

**MELD 数据集:**
```
output/meld_utterance_features/
├── MELD_trainhappylabel0.pkl
├── MELD_devhappylabel0.pkl
├── MELD_testhappylabel0.pkl
├── MELD_trainsadlabel1.pkl
├── MELD_devsadlabel1.pkl
├── MELD_testsadlabel1.pkl
└── ...
```

### 2. 修改配置文件

编辑 `config.yaml` 文件，设置你需要的参数：

```yaml
# 示例 1: 训练 happy vs sad 二分类
dataset:
  name: "MELD"
  seen_emotions:
    happy: 0
    sad: 1
  unseen_emotions: {}  # 不使用 unseen emotions

# 示例 2: 训练多分类，保留 fear 作为 unseen
dataset:
  name: "MELD"
  seen_emotions:
    happy: 0
    sad: 1
    angry: 2
  unseen_emotions:
    fear: 5
```

### 3. 运行训练

```bash
# 使用默认配置文件
python codes_v251112/fusion/train.py

# 使用自定义配置文件
python codes_v251112/fusion/train.py --config path/to/your/config.yaml
```

### 4. 测试配置和数据加载

```bash
# 测试配置读取
python codes_v251112/fusion/config_utils.py

# 测试数据加载器
python codes_v251112/fusion/dataloader.py
```

## 📊 数据加载器详解

### Seen Emotions

- **用途**: 用于训练和测试的情感类别
- **标签**: 自动重新映射到 0, 1, 2, ...
- **DataLoader**: `train_seen` 和 `test_seen`

### Unseen Emotions

- **用途**: 用于零样本学习、域适应或评估泛化能力
- **标签**: 保持原始标签，映射标签设为 -1
- **DataLoader**: `train_unseen` 和 `test_unseen`

### 标签映射示例

假设配置如下：
```yaml
seen_emotions:
  happy: 0    # 原始标签 0
  sad: 1      # 原始标签 1
unseen_emotions:
  fear: 5     # 原始标签 5
```

**映射结果:**
- `happy`: 原始标签 0 → 映射标签 0
- `sad`: 原始标签 1 → 映射标签 1
- `fear`: 原始标签 5 → 映射标签 -1 (unseen)

## 🔍 输出和日志

### 模型保存

训练过程中会保存以下文件：

```
checkpoints/
├── best_model_{experiment_name}.pth          # 最佳模型
└── checkpoint_{experiment_name}_epoch10.pth   # 定期检查点
```

### 保存的内容

每个模型文件包含：
- `model_state_dict`: 模型参数
- `optimizer_state_dict`: 优化器状态
- `accuracy`: 准确率
- `config`: 模型配置
- `feature_dims`: 特征维度
- `num_classes`: 类别数
- `seen_emotions`: Seen emotions 配置
- `unseen_emotions`: Unseen emotions 配置

## 📈 训练日志示例

```
==================================================================
配置文件: ./codes_v251112/fusion/config.yaml
==================================================================

【数据集配置】
  数据集名称: MELD
  数据目录: ./output/meld_utterance_features
  Seen emotions: {'happy': 0, 'sad': 1}
  Unseen emotions: {'fear': 5}
  训练集比例: 0.7

【数据加载配置】
  Batch size: 32
  Num workers: 4

==================================================================
加载数据集: MELD
==================================================================

标签映射: {0: 0, 1: 1}

加载 Seen Emotions: ['happy', 'sad']
  加载 happy (label_id=0)...
    训练集: 1500 样本, 测试集: 300 样本
  加载 sad (label_id=1)...
    训练集: 1200 样本, 测试集: 250 样本

【Seen Emotions 统计】
  训练集总样本数: 2700
  测试集总样本数: 550

==================================================================
Epoch 1/50
Train - Loss: 0.6523, Cls Loss: 0.6421, Contrastive Loss: 0.0102, Acc: 62.15%
Test  - Loss: 0.5834, Cls Loss: 0.5832, Acc: 68.36%
✓ 保存最佳模型 (Acc: 68.36%)
==================================================================
```

## 💡 常见使用场景

### 场景 1: 二分类任务

```yaml
seen_emotions:
  happy: 0
  sad: 1
unseen_emotions: {}
```

### 场景 2: 多分类任务

```yaml
seen_emotions:
  happy: 0
  sad: 1
  angry: 2
  surprise: 3
unseen_emotions: {}
```

### 场景 3: 零样本学习

```yaml
seen_emotions:
  happy: 0
  sad: 1
  angry: 2
unseen_emotions:
  fear: 5
  disgust: 4
```

训练时只使用 seen emotions，评估时可以测试模型在 unseen emotions 上的泛化能力。

## 🛠️ 自定义和扩展

### 修改模型架构

在 `config.yaml` 中调整模型参数：

```yaml
model:
  encoder:
    hidden_dim: 512  # 增大隐藏层维度
    output_dim: 512
  hypergraph:
    num_hyperedges: 128  # 增加超边数量
    num_conv_layers: 3   # 增加卷积层数
```

### 添加新的情感类别

在 `config.yaml` 中添加：

```yaml
seen_emotions:
  happy: 0
  sad: 1
  angry: 2
  surprise: 3
  disgust: 4
  fear: 5
  neutral: 6
```

### 使用不同的数据集

```yaml
dataset:
  name: "MOSEI"
  data_dir: "./output/mosei_utterance_features"
  train_ratio: 0.8  # MOSEI 需要手动划分
```

## 📝 注意事项

1. **标签映射**: Seen emotions 的标签会自动重新映射到从 0 开始的连续整数
2. **数据格式**: 确保数据文件遵循 `instruct.md` 中的格式
3. **设备选择**: 如果没有 GPU，配置文件会自动使用 CPU
4. **内存管理**: 调整 `batch_size` 和 `num_workers` 以适应你的硬件

## 🤝 依赖项

- Python 3.7+
- PyTorch 1.9+
- PyYAML
- NumPy

## 📧 问题反馈

如有问题，请查看：
1. 配置文件是否正确
2. 数据路径是否存在
3. 数据格式是否符合要求
