# 任务配置文件使用指南

## 📋 快速开始

### 方法1：使用预定义序列（最简单）

```bash
# 不需要配置文件，直接使用预定义序列
python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_sequence custom \  # 'demo', 'full', 或 'custom'
    --save_dir ../../checkpoints/blackbox
```

### 方法2：使用自定义配置文件

```bash
# 使用JSON配置文件
python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_config_path ./my_task_config.json \  # 指定配置文件
    --save_dir ../../checkpoints/blackbox
```

---

## 📝 创建配置文件的三种方法

### 方法A：自动策略生成（推荐首次使用）

```python
from continual_learning.domain_splitter import DomainSplitter
from fusion.dataloader import load_mosei_data

# 1. 加载数据集
dataset = load_mosei_data(data_dir='../../output/mosei_features', emotion='all')

# 2. 创建splitter
splitter = DomainSplitter(dataset, exclude_neutral=True)

# 3. 使用策略生成任务
tasks = splitter.create_tasks_by_strategy(
    strategy='small_unseen',     # 策略: 'small_unseen', 'incremental', 'disjoint', 'overlap'
    num_tasks=3,
    seen_classes_base=[0, 1]     # happy=0, sad=1
)

# 4. 保存配置
splitter.save_task_configs(tasks, 'my_task_config.json')
```

**策略说明：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `small_unseen` | 大样本类作seen，小样本类作unseen | 现实场景，数据不平衡 |
| `incremental` | 逐步增加seen类数量 | 渐进式学习 |
| `disjoint` | 每个任务完全不同的类 | 任务分离测试 |
| `overlap` | seen类重叠，unseen类不同 | 共享知识测试 |

### 方法B：使用预定义序列并保存

```python
from continual_learning.domain_splitter import create_predefined_task_sequence
import json

# 1. 创建预定义序列
tasks = create_predefined_task_sequence('custom', 'MOSEI')

# 2. 保存为JSON
output = {
    'num_tasks': len(tasks),
    'exclude_neutral': True,
    'random_seed': 42,
    'tasks': [task.to_dict() for task in tasks]
}

with open('predefined_config.json', 'w') as f:
    json.dump(output, f, indent=2)
```

**预定义序列：**

| 序列名 | 任务数 | 说明 |
|--------|--------|------|
| `demo` | 3 | 快速测试用 |
| `full` | 6 | 完整测试，覆盖所有情绪 |
| `custom` | 3 | 您之前对话中的示例 |

### 方法C：手动编写JSON文件

直接创建 `task_config.json` 文件：

```json
{
  "num_tasks": 3,
  "exclude_neutral": true,
  "random_seed": 42,
  "tasks": [
    {
      "task_id": 0,
      "task_name": "MOSEI_Task0",
      "dataset_name": "MOSEI",
      "seen_classes": [0, 1],
      "unseen_classes": [2],
      "seen_class_names": ["happy", "sad"],
      "unseen_class_names": ["angry"],
      "data_split": null
    },
    {
      "task_id": 1,
      "task_name": "MOSEI_Task1",
      "dataset_name": "MOSEI",
      "seen_classes": [0],
      "unseen_classes": [4, 5],
      "seen_class_names": ["happy"],
      "unseen_class_names": ["disgust", "fear"],
      "data_split": null
    },
    {
      "task_id": 2,
      "task_name": "MOSEI_Task2",
      "dataset_name": "MOSEI",
      "seen_classes": [0, 1],
      "unseen_classes": [3],
      "seen_class_names": ["happy", "sad"],
      "unseen_class_names": ["surprise"],
      "data_split": null
    }
  ]
}
```

---

## 🔧 JSON字段说明

### 必需字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `task_id` | int | 任务ID，从0开始 | `0`, `1`, `2` |
| `task_name` | str | 任务名称（自定义） | `"MOSEI_Task0"` |
| `dataset_name` | str | 数据集名称 | `"MOSEI"` 或 `"MELD"` |
| `seen_classes` | list[int] | 有标签的情绪类别ID | `[0, 1]` |
| `unseen_classes` | list[int] | 无标签的情绪类别ID（零样本） | `[2]` |

### 可选字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `data_split` | float or null | 使用数据的比例 | `null`（全部）, `0.5`（50%） |
| `seen_class_names` | list[str] | 情绪名称（仅用于可读性） | `["happy", "sad"]` |
| `unseen_class_names` | list[str] | 情绪名称（仅用于可读性） | `["angry"]` |

---

## 🎨 情绪类别ID映射

```python
0: 'happy'      # 开心
1: 'sad'        # 悲伤
2: 'angry'      # 愤怒
3: 'surprise'   # 惊讶
4: 'disgust'    # 厌恶
5: 'fear'       # 恐惧
6: 'neutral'    # 中性（通常被排除）
```

---

## 💡 使用示例

### 示例1：运行示例脚本

```bash
cd codes_v251112/continual_learning/blackbox_learnable

# 查看所有创建方法的示例
python example_create_task_config.py
```

这个脚本会：
1. 演示3种创建配置的方法
2. 生成示例配置文件
3. 展示如何加载和使用配置

### 示例2：完整训练流程

```bash
# Step 1: 创建配置（Python脚本）
python -c "
from continual_learning.domain_splitter import DomainSplitter
from fusion.dataloader import load_mosei_data

dataset = load_mosei_data(data_dir='../../output/mosei_features', emotion='all')
splitter = DomainSplitter(dataset, exclude_neutral=True)
tasks = splitter.create_tasks_by_strategy('small_unseen', num_tasks=3, seen_classes_base=[0, 1])
splitter.save_task_configs(tasks, 'my_config.json')
"

# Step 2: 使用配置训练
python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_config_path my_config.json \
    --num_epochs 10 \
    --save_dir ../../checkpoints/blackbox
```

### 示例3：修改已有配置

```bash
# 1. 先生成一个配置
python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_sequence custom

# 这会自动保存配置到 ../../checkpoints/blackbox/task_configs.json

# 2. 修改 task_configs.json 文件（例如改变seen/unseen类）

# 3. 使用修改后的配置重新训练
python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_config_path ../../checkpoints/blackbox/task_configs.json
```

---

## 🔍 加载和使用配置

### Python代码中加载

```python
from continual_learning.domain_splitter import DomainSplitter

# 加载配置文件
tasks = DomainSplitter.load_task_configs('my_task_config.json')

# 输出信息
print(f"加载了 {len(tasks)} 个任务")
for task in tasks:
    print(task)

# 使用配置创建dataloader
from fusion.dataloader import load_mosei_data

dataset = load_mosei_data(data_dir='../../output/mosei_features', emotion='all')
splitter = DomainSplitter(dataset, exclude_neutral=True)

for task_config in tasks:
    seen_loader, unseen_loader = splitter.create_task_dataloaders(
        task_config,
        batch_size=32,
        num_workers=4
    )

    # 训练...
    print(f"Task {task_config.task_id}: "
          f"{len(seen_loader)} seen batches, "
          f"{len(unseen_loader) if unseen_loader else 0} unseen batches")
```

---

## ⚠️ 常见问题

### Q1: 配置文件路径问题

**问题：** `FileNotFoundError: my_task_config.json`

**解决：**
```bash
# 使用绝对路径
--task_config_path /path/to/my_task_config.json

# 或相对于当前目录的路径
--task_config_path ./my_task_config.json
```

### Q2: seen_classes 和 unseen_classes 重复

**问题：** 同一个情绪ID同时出现在seen和unseen中

**解决：** 确保每个情绪ID只出现在一个列表中：
```json
{
  "seen_classes": [0, 1],      // happy, sad
  "unseen_classes": [2, 3, 4]  // angry, surprise, disgust（不能包含0或1）
}
```

### Q3: 没有unseen_classes

**问题：** 某个任务只有seen，没有unseen

**解决：** 这是允许的，unseen_classes可以为空：
```json
{
  "seen_classes": [0, 1, 2, 3, 4, 5],
  "unseen_classes": []  // 空列表，没有零样本学习
}
```

### Q4: 如何查看生成的配置

**解决：** 训练后配置会自动保存：
```bash
# 配置文件保存位置
<save_dir>/task_configs.json

# 例如
checkpoints/blackbox/task_configs.json
```

---

## 📚 参考文档

- **domain_splitter.py**: 完整实现代码
- **blackbox_main.py**: 主训练脚本
- **example_create_task_config.py**: 示例脚本
- **example_task_config.json**: 示例配置文件

---

## 🎯 推荐工作流

1. **首次使用**：使用预定义序列 `--task_sequence custom`
2. **查看生成的配置**：检查 `<save_dir>/task_configs.json`
3. **根据需要修改**：编辑JSON文件调整seen/unseen类
4. **使用修改后的配置**：`--task_config_path <修改后的文件>`
5. **复现实验**：使用同一个配置文件确保一致性

---

**最后更新**: 2024-11-19
