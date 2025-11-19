# 跨数据集持续学习指南

## 🌐 概述

支持在不同数据集之间进行持续学习，例如：
- Task 0, 1: MOSEI 数据集
- Task 2: MELD 数据集
- Task 3: 回到 MOSEI

**核心优势**：
- ✅ 统一的标签映射跨所有数据集
- ✅ 灵活配置每个任务的数据源
- ✅ 测试跨域知识迁移能力

---

## 📝 配置文件格式

### 每个任务指定数据集

```json
{
  "num_tasks": 3,
  "default_data_dir": "../../output/mosei_features",
  "default_dataset": "MOSEI",
  "tasks": [
    {
      "task_id": 0,
      "task_name": "Task0_MOSEI",
      "dataset_name": "MOSEI",              // ← 任务级别配置
      "data_dir": "../../output/mosei_features",
      "seen_emotions": {"happy": 0, "sad": 1},
      "unseen_emotions": {"anger": 2}
    },
    {
      "task_id": 1,
      "task_name": "Task1_MELD",
      "dataset_name": "MELD",               // ← 不同的数据集
      "data_dir": "../../output/meld_features",
      "seen_emotions": {"happy": 0, "surprise": 3},
      "unseen_emotions": {"disgust": 4}
    },
    {
      "task_id": 2,
      "task_name": "Task2_MOSEI_Again",
      "dataset_name": "MOSEI",              // ← 回到MOSEI
      "data_dir": "../../output/mosei_features",
      "seen_emotions": {"sad": 1, "anger": 2, "fear": 5},
      "unseen_emotions": {}
    }
  ]
}
```

### 使用默认配置

如果某个任务没有指定 `dataset_name` 或 `data_dir`，会使用全局默认值：

```json
{
  "default_dataset": "MOSEI",
  "default_data_dir": "../../output/mosei_features",
  "tasks": [
    {
      "task_id": 0,
      // 没有指定dataset_name和data_dir
      // 会使用默认的MOSEI
      "seen_emotions": {"happy": 0},
      "unseen_emotions": {"sad": 1}
    }
  ]
}
```

---

## 🚀 使用方法

### 基本使用

```python
from dataloader_continual import create_task_dataloaders, IncrementalLabelMapper

# 创建全局标签映射器
label_mapper = IncrementalLabelMapper()

# Task 0: MOSEI
train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
    task_config_path='task_config_cross_dataset.json',
    task_id=0,
    label_mapper=label_mapper,
    batch_size=32
)

print(f"Task 0 数据集: {task_info['dataset_name']}")  # MOSEI
print(f"Task 0 数据目录: {task_info['data_dir']}")

# Task 1: MELD (使用同一个label_mapper)
train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
    task_config_path='task_config_cross_dataset.json',
    task_id=1,
    label_mapper=label_mapper,  # 关键：传递同一个mapper
    batch_size=32
)

print(f"Task 1 数据集: {task_info['dataset_name']}")  # MELD
print(f"Task 1 数据目录: {task_info['data_dir']}")
```

### 完整训练循环

```python
from dataloader_continual import IncrementalLabelMapper, create_task_dataloaders
import torch
import torch.nn as nn

# 初始化
label_mapper = IncrementalLabelMapper()
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# 持续学习
for task_id in range(3):  # MOSEI, MELD, MOSEI
    # 加载数据
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path='task_config_cross_dataset.json',
        task_id=task_id,
        label_mapper=label_mapper,
        batch_size=32
    )

    dataset_name = task_info['dataset_name']
    print(f"\n训练 Task {task_id} ({dataset_name})")

    # 根据数据集调整策略
    if dataset_name == 'MOSEI':
        lr = 1e-4
        epochs = 10
    elif dataset_name == 'MELD':
        lr = 5e-5  # 可能需要不同的学习率
        epochs = 15

    # 更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 训练
    for epoch in range(epochs):
        for batch in train_loader:
            text = batch['text'].cuda()
            audio = batch['audio'].cuda()
            video = batch['video'].cuda()
            labels = batch['label'].cuda()
            is_seen = batch['is_seen'].cuda()

            # 前向传播
            outputs = model(text, audio, video)

            # 损失计算
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            weights = torch.where(is_seen, 1.0, 0.3)
            loss = (loss * weights).mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存检查点
    torch.save({
        'task_id': task_id,
        'dataset': dataset_name,
        'model': model.state_dict(),
        'label_mapper': label_mapper.original_to_incremental
    }, f'checkpoint_task{task_id}_{dataset_name}.pt')
```

---

## 🔑 关键特性

### 1. 统一的标签映射

即使数据来自不同数据集，同一情绪保持相同的增量标签：

```
Task 0 (MOSEI): happy->0, sad->1, anger->2
Task 1 (MELD):  happy->0 (保持!), surprise->3 (新)
Task 2 (MOSEI): happy->0 (保持!), sad->1 (保持!)
```

### 2. 数据集特定信息

每个任务返回的 `task_info` 包含数据集信息：

```python
task_info = {
    'task_id': 1,
    'task_name': 'Task1_MELD',
    'dataset_name': 'MELD',                    // ← 数据集名称
    'data_dir': '../../output/meld_features',  // ← 数据目录
    'seen_emotions': {...},
    'unseen_emotions': {...},
    'mapping_info': {...},
    'train_stats': {...},
    'test_stats': {...},
    'num_classes_so_far': 4
}
```

### 3. 灵活的数据文件格式

自动适配不同数据集的文件格式：

**MOSEI**:
```
MOSEIhappylabel0.pkl
MOSEIsadlabel1.pkl
```

**MELD**:
```
MELD_trainhappylabel0.pkl
MELD_devhappylabel0.pkl
MELD_testhappylabel0.pkl
```

---

## 📊 示例配置文件

### task_config.json (基础版)

```json
{
  "num_tasks": 3,
  "default_dataset": "MOSEI",
  "default_data_dir": "../../output/mosei_features",
  "tasks": [
    {
      "task_id": 0,
      "task_name": "Task0_MOSEI",
      "dataset_name": "MOSEI",
      "data_dir": "../../output/mosei_features",
      "seen_emotions": {"happy": 0, "sad": 1},
      "unseen_emotions": {"surprise": 3, "disgust": 4}
    },
    {
      "task_id": 1,
      "task_name": "Task1_MOSEI",
      "dataset_name": "MOSEI",
      "data_dir": "../../output/mosei_features",
      "seen_emotions": {"happy": 0, "anger": 2},
      "unseen_emotions": {"fear": 5}
    },
    {
      "task_id": 2,
      "task_name": "Task2_MELD",
      "dataset_name": "MELD",
      "data_dir": "../../output/meld_features",
      "seen_emotions": {
        "happy": 0,
        "sad": 1,
        "anger": 2,
        "surprise": 3,
        "disgust": 4,
        "fear": 5
      },
      "unseen_emotions": {}
    }
  ]
}
```

### task_config_cross_dataset.json (高级版)

提供了更复杂的跨数据集交替场景，参见 `task_config_cross_dataset.json`。

---

## 💡 使用建议

### 1. 数据集特定超参数

不同数据集可能需要不同的训练策略：

```python
dataset_configs = {
    'MOSEI': {
        'lr': 1e-4,
        'epochs': 10,
        'weight_decay': 1e-5
    },
    'MELD': {
        'lr': 5e-5,
        'epochs': 15,
        'weight_decay': 1e-4
    }
}

dataset_name = task_info['dataset_name']
config = dataset_configs[dataset_name]

# 应用配置
for param_group in optimizer.param_groups:
    param_group['lr'] = config['lr']
```

### 2. 跨数据集评估

评估模型在不同数据集间的泛化能力：

```python
# 在所有已训练任务上评估
for eval_task_id in range(current_task_id + 1):
    eval_loader, _, _, eval_info = create_task_dataloaders(
        task_config_path=config_path,
        task_id=eval_task_id,
        label_mapper=label_mapper,
        batch_size=32
    )

    accuracy = evaluate(model, eval_loader)
    print(f"Task {eval_task_id} ({eval_info['dataset_name']}): {accuracy:.4f}")
```

### 3. 域适应策略

从一个数据集切换到另一个时，可以应用域适应技术：

```python
prev_dataset = None

for task_id in range(num_tasks):
    train_loader, _, label_mapper, task_info = create_task_dataloaders(...)

    curr_dataset = task_info['dataset_name']

    # 检测数据集切换
    if prev_dataset and prev_dataset != curr_dataset:
        print(f"数据集切换: {prev_dataset} -> {curr_dataset}")

        # 应用域适应策略
        # 例如: 微调策略、学习率调整、特定层的冻结等
        apply_domain_adaptation(model, prev_dataset, curr_dataset)

    # 训练...

    prev_dataset = curr_dataset
```

---

## 🔍 调试和验证

### 查看数据集分布

```python
import json

with open('task_config_cross_dataset.json', 'r') as f:
    config = json.load(f)

print("数据集使用情况:")
for task in config['tasks']:
    print(f"  Task {task['task_id']}: {task['dataset_name']}")
```

### 验证标签一致性

```python
# 加载所有任务并检查标签映射
label_mapper = IncrementalLabelMapper()

for task_id in range(3):
    _, _, label_mapper, task_info = create_task_dataloaders(
        task_config_path='task_config.json',
        task_id=task_id,
        label_mapper=label_mapper
    )

    print(f"\nTask {task_id} ({task_info['dataset_name']}):")
    print(f"  全局映射: {label_mapper.original_to_incremental}")
```

---

## 🎯 运行示例

```bash
cd codes_v251112/continual_learning/blackbox_learnable

# 运行跨数据集示例
python example_cross_dataset.py
```

示例包括：
1. 配置分析
2. 顺序加载多个数据集
3. 数据集特定处理
4. 完整训练循环（伪代码）

---

## ⚠️ 注意事项

### 1. 数据文件必须存在

确保每个任务的数据目录中有对应的文件：

```bash
# MOSEI
../../output/mosei_features/MOSEIhappylabel0.pkl
../../output/mosei_features/MOSEIsadlabel1.pkl

# MELD
../../output/meld_features/MELD_trainhappylabel0.pkl
../../output/meld_features/MELD_devhappylabel0.pkl
../../output/meld_features/MELD_testhappylabel0.pkl
```

### 2. 标签ID一致性

不同数据集中相同情绪应使用相同的原始标签ID：

```
✅ 正确:
  MOSEI: happy=0, sad=1
  MELD:  happy=0, sad=1  (标签ID一致)

❌ 错误:
  MOSEI: happy=0, sad=1
  MELD:  happy=1, sad=0  (标签ID不一致！)
```

### 3. 特征维度一致性

确保不同数据集的特征维度相同：

```python
# 所有数据集应该有相同的特征维度
text_dim = 768
audio_dim = 768
video_dim = 768
```

---

## 📚 相关文档

- **DATALOADER_GUIDE.md** - 完整数据加载器文档
- **README_DATALOADER.md** - 快速入门
- **task_config_cross_dataset.json** - 跨数据集配置示例
- **example_cross_dataset.py** - 使用示例

---

## 🎓 典型应用场景

### 1. 跨域持续学习

从一个域（MOSEI）迁移到另一个域（MELD）：

```
Task 0: MOSEI (电影评论)
Task 1: MELD (TV剧对话)
Task 2: MOSEI (回到电影评论，测试遗忘)
```

### 2. 渐进式数据集整合

逐步整合多个数据集的数据：

```
Task 0: MOSEI基础情绪
Task 1: 加入MELD复杂情绪
Task 2: 加入第三个数据集
Task 3: 综合评估
```

### 3. 数据集特定微调

在不同数据集上微调特定能力：

```
Task 0: MOSEI (学习基础表示)
Task 1: MELD (微调对话理解)
Task 2: MOSEI (验证知识保持)
```

---

**最后更新**: 2024-11-19
