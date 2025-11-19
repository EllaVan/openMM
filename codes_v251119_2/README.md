# 跨域零样本持续学习框架

## 📋 项目概述

这是一个完整的**跨域零样本持续学习**训练框架，用于多模态情绪识别任务。

### 核心特性

✅ **3任务持续学习**：Task 0 (MOSEI) → Task 1 (MOSEI) → Task 2 (MELD)
✅ **跨数据集迁移**：支持在不同数据集间进行持续学习
✅ **零样本学习**：通过AU-EMO矩阵预测未见过的情绪类别
✅ **防遗忘机制**：使用EWC防止灾难性遗忘
✅ **类增量标签映射**：统一的标签映射跨所有任务
✅ **多模态一致性检查**：为unseen类生成可靠伪标签

---

## 🗂️ 项目结构

```
codes_v251119_2/
├── config/
│   ├── continual_learning.yaml    # 主配置文件（Hydra）
│   └── tasks.json                 # 任务序列配置
├── core/
│   ├── network.py                 # AU情绪识别网络
│   ├── learnable_matrix.py        # 可学习AU-EMO矩阵
│   ├── consistency_checker.py     # 多模态一致性检查
│   └── ewc.py                     # 防遗忘机制
├── data/
│   └── dataloader.py              # 持续学习数据加载器
├── training/
│   └── trainer.py                 # 训练器
├── fusion/
│   └── network.py                 # 多模态融合模块
├── utils/
│   └── tools.py                   # 工具函数
├── materials/
│   └── au_emo_prior.json          # AU-EMO先验矩阵
├── main.py                        # 主入口
└── README.md                      # 本文件
```

---

## 🚀 快速开始

### 1. 环境要求

```bash
Python >= 3.8
PyTorch >= 1.10
hydra-core >= 1.2
```

### 2. 修改配置

编辑 `config/continual_learning.yaml`：

```yaml
task:
  config_path: /home/user/openMM/codes_v251119_2/config/tasks.json

prior:
  au_prior_path: /home/user/openMM/codes_v251119_2/materials/au_emo_prior.json

system:
  device: cuda:0
  seed: 2025
```

编辑 `config/tasks.json` 中的数据路径：

```json
{
  "tasks": [
    {
      "task_id": 0,
      "dataset_name": "MOSEI",
      "data_dir": "/home/user/openMM/output/mosei_features",
      ...
    }
  ]
}
```

### 3. 运行训练

```bash
cd /home/user/openMM/codes_v251119_2

python main.py
```

### 4. 查看结果

训练完成后，结果保存在：

- **检查点**: `checkpoints/task{0,1,2}_final.pt`
- **矩阵**: `checkpoints/task{0,1,2}_matrix.npz`
- **日志**: `logs/training.log`
- **最终模型**: `checkpoints/final_model.pt`

---

## 📊 训练流程

### Task 0: MOSEI基础情绪

```
Seen: happy, sad
Unseen: surprise, disgust

流程：
1. Warmup (3 epochs) - 预热网络
2. Seen Training (20 epochs) - 训练seen类
3. Unseen Training - 一致性检查 + 伪标签
4. EWC Consolidation - 合并Fisher信息

标签映射：
  happy -> 0
  sad -> 1
  surprise -> 2
  disgust -> 3
```

### Task 1: MOSEI扩展情绪

```
Seen: happy, anger
Unseen: fear

流程：
1. Seen Training (20 epochs) - 带EWC正则化
2. Unseen Training - 一致性检查 + 伪标签
3. EWC Consolidation

标签映射：
  happy -> 0 (保持!)
  anger -> 4 (新分配)
  fear -> 5 (新分配)
```

### Task 2: MELD跨数据集

```
Seen: anger, surprise, joy
Unseen: disgust, fear

流程：
1. Seen Training (20 epochs) - 跨数据集迁移
2. Unseen Training
3. EWC Consolidation

标签映射：
  anger -> 4 (保持)
  surprise -> 2 (保持)
  joy -> 6 (新分配)
  disgust -> 3
  fear -> 5
```

---

## ⚙️ 配置说明

### 主要配置项

**网络架构** (`config/continual_learning.yaml`):

```yaml
network:
  text_dim: 768              # 文本特征维度
  audio_dim: 768             # 音频特征维度
  video_dim: 768             # 视频特征维度
  num_hyperedges: 64         # 超边数量
  num_conv_layers: 2         # 卷积层数
```

**训练参数**:

```yaml
training:
  epochs_per_task: 20        # 每个任务的训练轮数
  warmup_epochs: 3           # Task 0预热轮数
  learning_rate: 1.0e-4      # 网络学习率
  matrix_learning_rate: 1.0e-3  # 矩阵学习率
  gradient_clip: 1.0         # 梯度裁剪
```

**持续学习**:

```yaml
continual_learning:
  use_ewc: true              # 是否使用EWC
  ewc_lambda: 1000.0         # EWC强度
  matrix_reg_lambda: 0.1     # 矩阵正则化强度
  consistency_strategy: majority  # 一致性策略
  min_confidence: 0.8        # 最小置信度
  seen_loss_weight: 1.0      # Seen损失权重
  unseen_loss_weight: 0.3    # Unseen损失权重
```

---

## 📁 任务配置

`config/tasks.json` 定义任务序列：

```json
{
  "tasks": [
    {
      "task_id": 0,
      "task_name": "Task0_MOSEI_Basic_Emotions",
      "dataset_name": "MOSEI",
      "data_dir": "/path/to/mosei/features",
      "seen_emotions": {
        "happy": 0,
        "sad": 1
      },
      "unseen_emotions": {
        "surprise": 5,
        "disgust": 3
      }
    }
  ]
}
```

**字段说明**：

- `dataset_name`: 数据集名称 (MOSEI/MELD)
- `data_dir`: 数据文件目录（绝对路径）
- `seen_emotions`: 有标签的情绪 {名称: 原始标签ID}
- `unseen_emotions`: 无标签的情绪 {名称: 原始标签ID}

---

## 📦 数据格式

### 输入数据

数据文件应放在 `data_dir` 指定的目录中：

**MOSEI格式**:
```
MOSEIhappylabel0.pkl
MOSEIsadlabel1.pkl
MOSEIangerlabel2.pkl
...
```

**MELD格式**:
```
MELD_trainhappylabel0.pkl
MELD_devhappylabel0.pkl
MELD_testhappylabel0.pkl
...
```

### Pickle文件内容

每个`.pkl`文件包含样本列表，每个样本是一个字典：

```python
{
    'text_features': torch.Tensor([768]),
    'audio_features': torch.Tensor([768]),
    'video_features': torch.Tensor([768]),
    'label': int  # 原始标签ID
}
```

---

## 🔍 核心组件

### 1. AU情绪网络 (`core/network.py`)

```
输入: 文本、音频、视频特征
  ↓
单模态编码器
  ↓
超图融合
  ↓
分支1: AU预测器 → AU-EMO矩阵 → 情绪预测（主路径）
分支2: 直接分类器 → 情绪预测（辅助路径）
```

### 2. 可学习AU-EMO矩阵 (`core/learnable_matrix.py`)

- 23×6 可学习参数矩阵
- 初始化为心理学先验
- 通过KL散度正则化保持与先验的联系
- 支持零样本预测

### 3. 多模态一致性检查 (`core/consistency_checker.py`)

为unseen类生成可靠伪标签：

- 比较AU路径和直接路径的预测
- 计算预测置信度
- 过滤低置信度样本

### 4. EWC防遗忘 (`core/ewc.py`)

- 计算每个任务的Fisher信息矩阵
- 惩罚重要参数的变化
- 防止灾难性遗忘

### 5. 类增量标签映射 (`data/dataloader.py`)

- 维护全局标签映射
- 相同情绪在不同任务中保持相同标签
- 新情绪获得递增标签

---

## 🎓 使用示例

### 示例1：修改任务序列

编辑 `config/tasks.json`：

```json
{
  "tasks": [
    {
      "task_id": 0,
      "seen_emotions": {"happy": 0},
      "unseen_emotions": {"sad": 1, "anger": 2}
    },
    {
      "task_id": 1,
      "seen_emotions": {"happy": 0, "sad": 1},
      "unseen_emotions": {"surprise": 5}
    }
  ]
}
```

### 示例2：调整训练参数

编辑 `config/continual_learning.yaml`：

```yaml
training:
  epochs_per_task: 30        # 增加训练轮数
  learning_rate: 5.0e-5      # 降低学习率

continual_learning:
  ewc_lambda: 2000.0         # 增加EWC强度
  unseen_loss_weight: 0.5    # 提高unseen权重
```

### 示例3：使用不同的一致性策略

```yaml
continual_learning:
  consistency_strategy: combined  # 组合策略
  min_confidence: 0.9            # 更高的置信度阈值
```

---

## 📈 监控训练

### 日志输出

训练过程会输出详细日志：

```
[2025-01-01 10:00:00] [INFO] 开始训练 Task 0: Task0_MOSEI_Basic_Emotions
[2025-01-01 10:00:01] [INFO] Epoch 1/20
[2025-01-01 10:00:05] [INFO]   训练损失: 1.2345
[2025-01-01 10:00:05] [INFO]   训练准确率: 0.6543
[2025-01-01 10:00:05] [INFO]   测试准确率: 0.6234
[2025-01-01 10:00:05] [INFO]   矩阵KL散度: 0.1234
[2025-01-01 10:00:05] [INFO]   Seen样本: 1500, Unseen样本: 800, 一致样本: 320
```

### 检查点

每5个epoch保存一次中间检查点：

```
checkpoints/task0_epoch5.pt
checkpoints/task0_epoch10.pt
...
```

任务完成后保存最终检查点：

```
checkpoints/task0_final.pt
checkpoints/task0_matrix.npz
```

---

## ⚠️ 注意事项

### 1. 数据路径

- 必须使用**绝对路径**
- 确保数据文件存在
- 检查文件命名格式

### 2. 标签ID一致性

不同数据集中相同情绪应使用相同的原始标签ID：

```
✅ 正确:
  MOSEI: happy=0, sad=1
  MELD:  happy=0, sad=1 (注意：MELD使用joy代替happy，但ID应一致)

❌ 错误:
  MOSEI: happy=0, sad=1
  MELD:  happy=1, sad=0  (标签ID不一致！)
```

### 3. 内存管理

- 大数据集可能需要调整 `batch_size`
- 使用 `num_workers` 加速数据加载
- 监控GPU内存使用

### 4. 设备选择

根据可用资源选择设备：

```yaml
system:
  device: cuda:0  # 使用GPU 0
  # device: cpu   # 使用CPU（较慢）
```

---

## 🐛 故障排除

### 问题1: FileNotFoundError

**症状**: 找不到数据文件

**解决**: 检查 `tasks.json` 中的 `data_dir` 路径是否正确

### 问题2: CUDA out of memory

**症状**: GPU内存不足

**解决**:
```yaml
dataloader:
  batch_size: 16  # 减小batch size
```

### 问题3: 标签映射不一致

**症状**: 同一情绪在不同任务中标签不同

**解决**: 确保在所有任务中使用同一个 `label_mapper` 实例（代码已自动处理）

### 问题4: 一致性样本数为0

**症状**: Unseen训练时一致样本数为0

**解决**:
```yaml
continual_learning:
  min_confidence: 0.6  # 降低置信度阈值
  consistency_strategy: majority  # 使用多数投票
```

---

## 📚 参考文档

- **codes_v251112/continual_learning/blackbox_learnable/README.md** - Blackbox方法详细说明
- **codes_v251112/continual_learning/blackbox_learnable/DATALOADER_GUIDE.md** - 数据加载器使用指南
- **codes_v251112/continual_learning/blackbox_learnable/CROSS_DATASET_GUIDE.md** - 跨数据集学习指南

---

## 📧 联系方式

如有问题，请查看日志文件或检查配置文件。

---

**最后更新**: 2025-01-19
**版本**: 1.0.0
