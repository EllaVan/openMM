# 持续学习数据加载器 - 快速入门

## 📦 已创建的文件

在 `codes_v251112/continual_learning/blackbox_learnable/` 目录下：

1. **dataloader_continual.py** - 核心实现
   - IncrementalLabelMapper: 标签映射管理器
   - ContinualLearningDataset: 数据集类
   - create_task_dataloaders(): 创建单任务dataloader
   - load_all_tasks(): 加载所有任务

2. **DATALOADER_GUIDE.md** - 完整使用指南
   - API文档
   - 详细示例
   - 常见问题解答

3. **example_dataloader_usage.py** - 使用示例
   - 5个完整示例演示所有功能

4. **task_config.json** (在 codes_v251119/config/)
   - 任务配置文件示例

---

## 🎯 核心功能

### 类增量标签映射

```
Task 0: seen [happy=0, sad=1], unseen [surprise=2, disgust=3]
Task 1: seen [happy=0, anger=4], unseen [fear=5]
        ↑ 保持原标签    ↑ 新分配
```

**关键特点**：
- ✅ 同一情绪在不同任务中保持相同标签
- ✅ 新出现的情绪获得递增标签
- ✅ 自动处理seen/unseen区分

---

## 🚀 快速开始

### 最简单的使用方式

```python
from dataloader_continual import create_task_dataloaders

# 加载Task 0
train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
    task_config_path='../../../codes_v251119/config/task_config.json',
    task_id=0,
    batch_size=32
)

# 训练
for batch in train_loader:
    text = batch['text']          # [batch, 768]
    audio = batch['audio']        # [batch, 768]
    video = batch['video']        # [batch, 768]
    labels = batch['label']       # [batch] 增量标签
    is_seen = batch['is_seen']    # [batch] bool

    # 您的训练代码...
```

### 持续学习（多任务）

```python
from dataloader_continual import IncrementalLabelMapper, create_task_dataloaders

# 创建全局映射器
label_mapper = IncrementalLabelMapper()

# 顺序加载任务
for task_id in range(3):
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path='task_config.json',
        task_id=task_id,
        label_mapper=label_mapper,  # 传递同一个mapper
        batch_size=32
    )

    # 训练Task...
```

---

## 📊 返回的Batch格式

```python
batch = {
    'text': Tensor[batch, 768],        # 文本特征
    'audio': Tensor[batch, 768],       # 音频特征
    'video': Tensor[batch, 768],       # 视频特征
    'label': LongTensor[batch],        # 增量标签 (0,1,2,...)
    'original_label': LongTensor[batch],  # 原始MOSEI标签
    'is_seen': BoolTensor[batch]       # True=seen, False=unseen
}
```

---

## 🔑 关键区别：Seen vs Unseen

| 属性 | Seen样本 | Unseen样本 |
|------|----------|-----------|
| **定义** | 当前任务有真实标签 | 当前任务无标签 |
| **训练方式** | 监督学习（真实标签） | 半监督学习（伪标签） |
| **损失权重** | 高权重 (1.0) | 低权重 (0.3) |
| **is_seen** | True | False |

### 训练中使用示例

```python
for batch in train_loader:
    labels = batch['label'].cuda()
    is_seen = batch['is_seen'].cuda()

    # 前向传播
    outputs = model(batch['text'], batch['audio'], batch['video'])

    # 计算损失（seen和unseen不同权重）
    loss = F.cross_entropy(outputs, labels, reduction='none')

    # Seen: 1.0权重, Unseen: 0.3权重
    weights = torch.where(is_seen, 1.0, 0.3)
    loss = (loss * weights).mean()

    # 反向传播...
```

---

## 📝 task_config.json 格式

```json
{
  "dataset_name": "MOSEI",
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
    }
  ]
}
```

**字段说明**：
- `seen_emotions`: {情绪名: 原始MOSEI标签ID}
- `unseen_emotions`: {情绪名: 原始MOSEI标签ID}

---

## 📚 运行示例

```bash
cd codes_v251112/continual_learning/blackbox_learnable

# 运行所有示例
python example_dataloader_usage.py
```

示例包括：
1. 加载单个任务
2. 顺序加载多个任务
3. 一次性加载所有任务
4. Seen/Unseen样本分离
5. 标签映射细节

---

## ⚠️ 重要提示

### 1. 必须使用同一个label_mapper

```python
# ✅ 正确
mapper = IncrementalLabelMapper()
_, _, mapper, _ = create_task_dataloaders(..., task_id=0, label_mapper=mapper)
_, _, mapper, _ = create_task_dataloaders(..., task_id=1, label_mapper=mapper)

# ❌ 错误（标签会不一致）
_, _, mapper0, _ = create_task_dataloaders(..., task_id=0, label_mapper=None)
_, _, mapper1, _ = create_task_dataloaders(..., task_id=1, label_mapper=None)
```

### 2. 数据文件命名格式

MOSEI数据集文件格式：
```
{DATASET}{emotion}label{id}.pkl

例如:
  MOSEIhappylabel0.pkl
  MOSEIsadlabel1.pkl
  MOSEIangerlabel2.pkl
```

### 3. task_config.json中的标签ID

`seen_emotions`和`unseen_emotions`中的数值是**原始MOSEI标签ID**，不是增量标签！

```json
{
  "seen_emotions": {
    "happy": 0,    // ← 这是原始MOSEI标签ID
    "sad": 1       // ← 不是增量标签
  }
}
```

---

## 🔗 相关文件

- **DATALOADER_GUIDE.md** - 完整文档（API、示例、FAQ）
- **dataloader_continual.py** - 源代码
- **example_dataloader_usage.py** - 使用示例
- **task_config.json** - 配置示例

---

## 📞 下一步

1. **阅读详细文档**: `DATALOADER_GUIDE.md`
2. **运行示例**: `python example_dataloader_usage.py`
3. **修改配置**: 编辑 `task_config.json` 创建您自己的任务
4. **集成到训练**: 将dataloader集成到您的训练循环

---

## 💡 与原dataloader的对比

| 特性 | 原dataloader (fusion/dataloader.py) | 新dataloader (dataloader_continual.py) |
|------|-------------------------------------|----------------------------------------|
| **用途** | 单任务训练 | 多任务持续学习 |
| **标签映射** | seen=[0,1,...], unseen=[len(seen),...] | 全局增量映射（跨任务一致） |
| **配置方式** | Hydra config | JSON task_config |
| **任务管理** | 单个任务 | 多个任务，自动映射 |
| **适用场景** | 单次训练 | 持续学习、类增量学习 |

---

**最后更新**: 2024-11-19
