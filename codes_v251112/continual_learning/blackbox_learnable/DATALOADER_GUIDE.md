# 持续学习数据加载器使用指南

## 📋 概述

`dataloader_continual.py` 实现了**类增量持续学习（Class-Incremental Learning）**的数据加载，支持：

✅ 自动标签重映射（seen类保持标签，新类顺序分配）
✅ Seen/Unseen类区分
✅ 全局标签映射管理
✅ 从JSON配置文件加载任务

---

## 🎯 核心功能

### 1. 标签重映射规则

**问题**：持续学习中，每个任务可能有不同的情绪类别，如何统一标签？

**解决方案**：类增量标签映射

```
Task 0:
  seen: [happy, sad]
  unseen: [surprise, disgust]

  标签分配：
    happy -> 0 (新)
    sad -> 1 (新)
    surprise -> 2 (新)
    disgust -> 3 (新)

Task 1:
  seen: [happy, anger]
  unseen: [fear]

  标签分配：
    happy -> 0 (保持，之前出现过)
    anger -> 4 (新，继续递增)
    fear -> 5 (新)
```

**关键点**：
- ✅ 同一情绪在不同任务中保持相同标签
- ✅ 新出现的情绪获得递增的标签
- ✅ Seen和Unseen都参与全局标签分配

---

## 📁 文件结构

### 1. **task_config.json** - 任务配置文件

```json
{
  "dataset_name": "MOSEI",
  "num_tasks": 3,
  "data_dir": "../../output/mosei_features",
  "tasks": [
    {
      "task_id": 0,
      "task_name": "Task0_Happy_Sad_vs_Surprise_Disgust",
      "seen_emotions": {
        "happy": 0,
        "sad": 1
      },
      "unseen_emotions": {
        "surprise": 3,
        "disgust": 4
      }
    },
    {
      "task_id": 1,
      "task_name": "Task1_Happy_Anger_vs_Fear",
      "seen_emotions": {
        "happy": 0,
        "anger": 2
      },
      "unseen_emotions": {
        "fear": 5
      }
    }
  ]
}
```

### 2. **dataloader_continual.py** - 数据加载器

核心类：
- `IncrementalLabelMapper`: 管理全局标签映射
- `ContinualLearningDataset`: 持续学习数据集
- `create_task_dataloaders()`: 创建单任务dataloader
- `load_all_tasks()`: 加载所有任务

---

## 🚀 快速开始

### 示例1：加载单个任务

```python
from dataloader_continual import create_task_dataloaders

# 加载Task 0
train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
    task_config_path='../../../codes_v251119/config/task_config.json',
    task_id=0,
    batch_size=32,
    num_workers=4,
    train_ratio=0.8
)

# 查看任务信息
print(f"任务名称: {task_info['task_name']}")
print(f"Seen情绪: {task_info['seen_emotions']}")
print(f"Unseen情绪: {task_info['unseen_emotions']}")
print(f"当前总类数: {task_info['num_classes_so_far']}")

# 训练
for batch in train_loader:
    text = batch['text']          # [batch_size, text_dim]
    audio = batch['audio']        # [batch_size, audio_dim]
    video = batch['video']        # [batch_size, video_dim]
    labels = batch['label']       # [batch_size] 增量标签
    is_seen = batch['is_seen']    # [batch_size] bool

    # 训练代码...
```

### 示例2：顺序加载多个任务（持续学习）

```python
from dataloader_continual import create_task_dataloaders, IncrementalLabelMapper

# 创建全局标签映射器
label_mapper = IncrementalLabelMapper()

# Task 0
train_loader_0, test_loader_0, label_mapper, info_0 = create_task_dataloaders(
    task_config_path='task_config.json',
    task_id=0,
    label_mapper=label_mapper  # 传入mapper
)

# 训练Task 0
# ...

# Task 1 (使用同一个label_mapper)
train_loader_1, test_loader_1, label_mapper, info_1 = create_task_dataloaders(
    task_config_path='task_config.json',
    task_id=1,
    label_mapper=label_mapper  # 传入更新后的mapper
)

# 训练Task 1
# ...

print(f"全局标签映射: {label_mapper.original_to_incremental}")
print(f"总类数: {label_mapper.get_num_classes_so_far()}")
```

### 示例3：一次性加载所有任务

```python
from dataloader_continual import load_all_tasks

# 加载所有任务
all_tasks = load_all_tasks(
    task_config_path='task_config.json',
    batch_size=32,
    num_workers=4,
    train_ratio=0.8
)

# 顺序训练
for task_id, (train_loader, test_loader, task_info) in enumerate(all_tasks):
    print(f"\n训练 Task {task_id}: {task_info['task_name']}")

    # 训练
    for batch in train_loader:
        # ...
        pass

    # 评估
    for batch in test_loader:
        # ...
        pass
```

---

## 📦 返回的Batch格式

```python
batch = {
    'text': torch.Tensor,          # [batch_size, text_dim] 例如 [32, 768]
    'audio': torch.Tensor,         # [batch_size, audio_dim] 例如 [32, 768]
    'video': torch.Tensor,         # [batch_size, video_dim] 例如 [32, 768]
    'label': torch.LongTensor,     # [batch_size] 增量标签 (0, 1, 2, ...)
    'original_label': torch.LongTensor,  # [batch_size] 原始MOSEI标签
    'is_seen': torch.BoolTensor    # [batch_size] True=seen, False=unseen
}
```

### 使用示例

```python
for batch in train_loader:
    # 获取数据
    text = batch['text']
    audio = batch['audio']
    video = batch['video']

    # 获取标签
    labels = batch['label']  # 增量标签，用于训练
    is_seen = batch['is_seen']

    # 分离seen和unseen样本
    seen_mask = is_seen
    unseen_mask = ~is_seen

    # Seen样本（有标签，高权重训练）
    if seen_mask.any():
        seen_text = text[seen_mask]
        seen_labels = labels[seen_mask]
        # 训练seen...

    # Unseen样本（无标签，使用伪标签低权重训练）
    if unseen_mask.any():
        unseen_text = text[unseen_mask]
        # 生成伪标签...
        # 训练unseen...
```

---

## 🔧 API文档

### IncrementalLabelMapper

管理全局标签映射。

```python
class IncrementalLabelMapper:
    def add_task(task_id, seen_emotions, unseen_emotions) -> Dict
        """为新任务添加标签映射"""

    def get_incremental_label(original_label: int) -> int
        """获取增量标签"""

    def is_seen(original_label: int) -> bool
        """判断是否是seen类"""

    def get_num_classes_so_far() -> int
        """获取目前为止的总类数"""
```

### create_task_dataloaders()

为单个任务创建数据加载器。

```python
def create_task_dataloaders(
    task_config_path: str,        # 任务配置文件路径
    task_id: int,                 # 任务ID (0, 1, 2, ...)
    label_mapper: IncrementalLabelMapper = None,  # 标签映射器
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,     # 训练集比例
    shuffle_train: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, IncrementalLabelMapper, Dict]:
    """
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        label_mapper: 更新后的标签映射器
        task_info: 任务信息字典
    """
```

### load_all_tasks()

一次性加载所有任务。

```python
def load_all_tasks(
    task_config_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader, Dict]]:
    """
    Returns:
        tasks_data: [(train_loader, test_loader, task_info), ...]
    """
```

---

## 📊 task_info 字典内容

```python
task_info = {
    'task_id': 0,
    'task_name': 'Task0_Happy_Sad_vs_Surprise_Disgust',
    'seen_emotions': {'happy': 0, 'sad': 1},
    'unseen_emotions': {'surprise': 3, 'disgust': 4},
    'mapping_info': {
        'task_id': 0,
        'seen_mapping': {'happy': 0, 'sad': 1},
        'unseen_mapping': {'surprise': 2, 'disgust': 3},
        'global_mapping': {0: 0, 1: 1, 3: 2, 4: 3},
        'next_label': 4
    },
    'train_stats': {
        'total': 5000,
        'seen_count': 3000,
        'unseen_count': 2000,
        'emotion_counts': {'happy': 1500, 'sad': 1500, 'surprise': 1000, 'disgust': 1000},
        'label_counts': {0: 1500, 1: 1500, 2: 1000, 3: 1000}
    },
    'test_stats': { ... },
    'num_classes_so_far': 4  # 当前任务后的总类数
}
```

---

## 💡 完整训练示例

```python
"""
完整的持续学习训练流程
"""
from dataloader_continual import create_task_dataloaders, IncrementalLabelMapper
import torch
import torch.nn as nn
import torch.optim as optim

# 配置
task_config_path = '../../../codes_v251119/config/task_config.json'
num_tasks = 3
batch_size = 32

# 初始化
label_mapper = IncrementalLabelMapper()
model = YourModel()  # 你的模型
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 持续学习循环
for task_id in range(num_tasks):
    print(f"\n{'='*80}")
    print(f"训练 Task {task_id}")
    print(f"{'='*80}")

    # 加载任务数据
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=task_id,
        label_mapper=label_mapper,
        batch_size=batch_size
    )

    # 训练
    model.train()
    for epoch in range(10):
        for batch in train_loader:
            text = batch['text'].cuda()
            audio = batch['audio'].cuda()
            video = batch['video'].cuda()
            labels = batch['label'].cuda()
            is_seen = batch['is_seen'].cuda()

            # 前向传播
            outputs = model(text, audio, video)

            # 计算损失（seen和unseen不同权重）
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)

            # Seen: 高权重, Unseen: 低权重
            weights = torch.where(is_seen,
                                  torch.tensor(1.0).cuda(),
                                  torch.tensor(0.3).cuda())
            loss = (loss * weights).mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].cuda()
            audio = batch['audio'].cuda()
            video = batch['video'].cuda()
            labels = batch['label'].cuda()

            outputs = model(text, audio, video)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"\n  Task {task_id} 测试准确率: {accuracy:.4f}")

    # 保存检查点
    torch.save({
        'task_id': task_id,
        'model_state': model.state_dict(),
        'label_mapper': label_mapper.original_to_incremental,
        'num_classes': label_mapper.get_num_classes_so_far()
    }, f'checkpoint_task{task_id}.pt')

print(f"\n{'='*80}")
print("持续学习训练完成！")
print(f"总类数: {label_mapper.get_num_classes_so_far()}")
print(f"全局标签映射: {label_mapper.original_to_incremental}")
print(f"{'='*80}")
```

---

## 🔍 调试和测试

### 运行测试

```bash
cd codes_v251112/continual_learning/blackbox_learnable

# 运行内置测试
python dataloader_continual.py
```

### 查看数据

```python
from dataloader_continual import create_task_dataloaders

train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
    task_config_path='task_config.json',
    task_id=0,
    batch_size=4,
    num_workers=0  # 设为0便于调试
)

# 查看第一个batch
batch = next(iter(train_loader))
print(f"Text shape: {batch['text'].shape}")
print(f"Labels: {batch['label']}")
print(f"Is seen: {batch['is_seen']}")

# 查看标签映射
print(f"全局映射: {label_mapper.original_to_incremental}")
print(f"总类数: {label_mapper.get_num_classes_so_far()}")
```

---

## ⚠️ 常见问题

### Q1: 文件找不到

**问题**: `FileNotFoundError: MOSEIhappylabel0.pkl`

**解决**:
1. 检查 `data_dir` 路径是否正确
2. 确认文件命名格式: `{DATASET}{emotion}label{id}.pkl`
3. 检查情绪名和标签ID是否匹配

### Q2: 标签不连续

**问题**: 增量标签不是连续的 0, 1, 2, ...

**解决**: 这是正常的！标签按照**出现顺序**分配，如果某个任务跳过了某些情绪，标签会有间隙。

### Q3: Seen和Unseen混淆

**问题**: 不清楚seen和unseen的区别

**说明**:
- **Seen**: 当前任务有真实标签的类，用于监督学习
- **Unseen**: 当前任务没有标签的类，用于零样本学习（需要伪标签）

### Q4: 跨任务标签一致性

**问题**: 同一情绪在不同任务中标签不同

**解决**: 使用 `IncrementalLabelMapper` 确保一致性。必须在所有任务中使用**同一个** `label_mapper` 实例！

```python
# ✅ 正确
label_mapper = IncrementalLabelMapper()
train0, test0, label_mapper, _ = create_task_dataloaders(..., label_mapper=label_mapper)
train1, test1, label_mapper, _ = create_task_dataloaders(..., label_mapper=label_mapper)

# ❌ 错误
train0, test0, mapper0, _ = create_task_dataloaders(..., label_mapper=None)
train1, test1, mapper1, _ = create_task_dataloaders(..., label_mapper=None)
```

---

## 📚 参考

- **dataloader_continual.py**: 完整实现
- **task_config.json**: 配置示例
- **codes_v251112/fusion/dataloader.py**: 原始dataloader参考

---

**最后更新**: 2024-11-19
