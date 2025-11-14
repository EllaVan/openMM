# 情感数据集DataLoader使用文档

这个模块提供了一个灵活的PyTorch DataLoader系统，用于加载MOSEI和MELD情感数据集。

## 数据加载策略

### MOSEI数据集
- 加载指定emotion和label_id的所有数据
- **自动按7/3比例划分为训练集和测试集**
- 支持自定义划分比例和随机种子

### MELD数据集
- **自动将train和dev合并为训练集**
- test作为测试集

## 数据格式

数据文件应该是pickle格式，文件命名规则如下：

- **MELD数据集**: `MELD_{split}{emotion}label{label_id}.pkl`
  - 例如: `MELD_trainhappylabel0.pkl`, `MELD_devsadlabel1.pkl`, `MELD_testangerlabel2.pkl`

- **MOSEI数据集**: `MOSEI{emotion}label{label_id}.pkl`
  - 例如: `MOSEIhappylabel0.pkl`, `MOSEIsadlabel1.pkl`

## 安装依赖

```bash
pip install torch
```

## 快速开始

### 1. 创建MOSEI DataLoader（7/3自动划分）

```python
from emotion_dataloader import create_dataloaders

# 创建MOSEI数据集的DataLoader
# 自动按7/3划分为训练集和测试集
dataloaders = create_dataloaders(
    data_dir='Data',           # 数据文件目录
    dataset_name='MOSEI',      # 数据集名称
    emotion='happy',           # 情感类型
    label_id=0,                # 标签ID
    batch_size=32,             # 批次大小
    num_workers=4,             # 加载进程数
    train_ratio=0.7,           # 训练集比例（默认0.7）
    seed=42                    # 随机种子（默认42）
)

# 访问训练集和测试集
train_loader = dataloaders['train']
test_loader = dataloaders['test']

# 使用DataLoader
for batch in train_loader:
    # batch是一个字典，包含audio_features, text_features, video_features, label等
    print(batch.keys())
    break

print(f"训练集大小: {len(train_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")
```

### 2. 创建MELD DataLoader（train+dev合并）

```python
from emotion_dataloader import create_dataloaders

# 创建MELD数据集的DataLoader
# 自动将train+dev合并为训练集，test为测试集
dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MELD',
    emotion='sad',
    label_id=1,
    batch_size=32,
    num_workers=4
)

train_loader = dataloaders['train']  # train+dev合并
test_loader = dataloaders['test']     # test

print(f"训练集大小 (train+dev): {len(train_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")
```

### 3. 批量创建多个DataLoader

```python
from emotion_dataloader import create_multiple_dataloaders

# 定义需要加载的情感和标签对
emotion_label_pairs = [
    ('happy', 0),
    ('sad', 1),
    ('anger', 2),
    ('disgust', 3),
    ('surprise', 4),
    ('fear', 5)
]

# 批量创建MOSEI DataLoader
all_dataloaders = create_multiple_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion_label_pairs=emotion_label_pairs,
    batch_size=16,
    num_workers=2,
    train_ratio=0.7
)

# 访问特定的DataLoader
happy_train = all_dataloaders['happy_0']['train']
happy_test = all_dataloaders['happy_0']['test']
sad_train = all_dataloaders['sad_1']['train']
sad_test = all_dataloaders['sad_1']['test']
```

### 4. 使用自定义collate函数

```python
from emotion_dataloader import create_dataloaders, custom_collate_fn

dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=8,
    collate_fn=custom_collate_fn  # 使用自定义collate函数
)

for batch in dataloaders['train']:
    # batch现在有特定的结构
    audio = batch['audio']      # 音频特征
    text = batch['text']        # 文本特征
    video = batch['video']      # 视频特征
    labels = batch['labels']    # 标签
    break
```

### 5. 使用数据转换函数

```python
from emotion_dataloader import create_dataloaders

# 定义transform函数
def my_transform(sample):
    # 对样本进行转换，例如归一化
    if 'audio_features' in sample:
        sample['audio_features'] = sample['audio_features'] / sample['audio_features'].max()
    return sample

dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='surprise',
    label_id=4,
    batch_size=16,
    transform=my_transform  # 应用转换
)
```

### 6. 在训练循环中使用

```python
from emotion_dataloader import create_dataloaders
import torch.nn as nn
import torch.optim as optim

# 创建DataLoader
dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32,
    train_ratio=0.7
)

train_loader = dataloaders['train']
test_loader = dataloaders['test']

# 训练循环
model = YourModel()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch in train_loader:
        audio = batch['audio_features']
        text = batch['text_features']
        video = batch['video_features']
        labels = batch['label']

        # 前向传播
        outputs = model(audio, text, video)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # 测试代码
            pass
```

## API参考

### create_dataloaders

主要函数，创建训练集和测试集的DataLoader。

```python
create_dataloaders(
    data_dir: str,              # 数据文件目录
    dataset_name: str,          # 'MELD' 或 'MOSEI'
    emotion: str,               # 情感类型
    label_id: int,              # 标签ID
    batch_size: int = 32,       # 批次大小
    num_workers: int = 4,       # 加载进程数
    train_ratio: float = 0.7,   # 训练集比例（仅MOSEI）
    seed: int = 42,             # 随机种子（仅MOSEI）
    transform = None,           # 转换函数
    **kwargs                    # 其他DataLoader参数
) -> Dict[str, DataLoader]      # 返回 {'train': loader, 'test': loader}
```

**数据划分策略：**
- **MOSEI**: 按`train_ratio`比例划分，默认7/3
- **MELD**: train+dev合并为训练集，test为测试集

### create_multiple_dataloaders

批量创建多个DataLoader。

```python
create_multiple_dataloaders(
    data_dir: str,
    dataset_name: str,
    emotion_label_pairs: List[Tuple[str, int]],  # [(emotion, label_id), ...]
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,   # 仅MOSEI
    seed: int = 42,             # 仅MOSEI
    transform = None,
    **kwargs
) -> Dict[str, Dict[str, DataLoader]]
```

返回嵌套字典：
```python
{
    'emotion_labelid': {
        'train': train_loader,
        'test': test_loader
    },
    ...
}
```

### EmotionDataset

底层Dataset类。

```python
EmotionDataset(
    data: List,                 # 数据列表
    transform = None            # 数据转换函数
)
```

**方法**:
- `__len__()`: 返回数据集大小
- `__getitem__(idx)`: 获取单个样本
- `get_info()`: 返回数据集信息

### 辅助函数

#### load_mosei_data

```python
load_mosei_data(
    data_dir: str,
    emotion: str,
    label_id: int,
    train_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[List, List]  # (train_data, test_data)
```

#### load_meld_data

```python
load_meld_data(
    data_dir: str,
    emotion: str,
    label_id: int
) -> Tuple[List, List]  # (train_data, test_data)
```

#### load_pkl_file

```python
load_pkl_file(file_path: str) -> List
```

## 支持的情感类型

常见的情感类型包括：
- `happy` (快乐)
- `sad` (悲伤)
- `anger` (愤怒)
- `disgust` (厌恶)
- `surprise` (惊讶)
- `fear` (恐惧)
- `neutral` (中性)

具体支持的情感类型取决于你的数据集。

## 数据格式说明

pkl文件中的数据应该是以下格式之一：

1. **列表格式**: `[sample1, sample2, ...]`
2. **字典格式**: `{key1: sample1, key2: sample2, ...}`

每个sample建议是一个字典，包含：
```python
{
    'audio_features': torch.Tensor,  # 音频特征
    'text_features': torch.Tensor,   # 文本特征
    'video_features': torch.Tensor,  # 视频特征
    'label': int                     # 标签
}
```

## 数据划分对比

### MOSEI数据集
```
原始数据: MOSEIhappylabel0.pkl (1000个样本)
    ↓
自动划分（7/3）
    ↓
训练集: 700个样本 (70%)
测试集: 300个样本 (30%)
```

### MELD数据集
```
原始数据:
- MELD_trainhappylabel0.pkl (600个样本)
- MELD_devhappylabel0.pkl (200个样本)
- MELD_testhappylabel0.pkl (100个样本)
    ↓
自动合并
    ↓
训练集: 800个样本 (train+dev)
测试集: 100个样本 (test)
```

## 示例代码

完整的示例代码请参考 `examples/dataloader_examples.py`

```bash
python examples/dataloader_examples.py
```

示例包括：
1. 创建MOSEI DataLoader（7/3划分）
2. 创建MELD DataLoader（train+dev合并）
3. 批量创建多个MOSEI DataLoader
4. 批量创建多个MELD DataLoader
5. 使用自定义collate函数
6. 在训练循环中使用
7. 使用数据转换函数
8. 比较MOSEI和MELD的数据划分策略

## 常见问题

### 1. FileNotFoundError: 数据文件不存在

确保你的pkl文件命名格式正确，并且放在正确的目录下。

**MOSEI**: `MOSEIhappylabel0.pkl`
**MELD**: `MELD_trainhappylabel0.pkl`, `MELD_devhappylabel0.pkl`, `MELD_testhappylabel0.pkl`

### 2. 如何调整MOSEI的训练集比例？

使用`train_ratio`参数：
```python
dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    train_ratio=0.8  # 80%训练，20%测试
)
```

### 3. 为什么MELD不需要指定split参数了？

新版本自动处理MELD的数据划分：
- 自动加载train、dev、test三个文件
- 自动将train和dev合并为训练集
- test作为测试集

### 4. 如何确保MOSEI每次划分的结果一致？

使用相同的`seed`参数：
```python
dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    seed=42  # 固定随机种子
)
```

### 5. 自定义数据格式

如果你的数据格式与默认格式不同，可以：
- 使用`transform`参数进行数据转换
- 或直接使用底层的`EmotionDataset`类并传入自定义数据

## 完整使用示例

```python
from emotion_dataloader import create_multiple_dataloaders

# 批量创建所有情感的DataLoader
emotion_label_pairs = [
    ('happy', 0), ('sad', 1), ('anger', 2),
    ('disgust', 3), ('surprise', 4), ('fear', 5)
]

# MOSEI数据集（7/3划分）
mosei_loaders = create_multiple_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion_label_pairs=emotion_label_pairs,
    batch_size=32,
    train_ratio=0.7,
    seed=42
)

# MELD数据集（train+dev / test）
meld_loaders = create_multiple_dataloaders(
    data_dir='Data',
    dataset_name='MELD',
    emotion_label_pairs=emotion_label_pairs,
    batch_size=32
)

# 使用
for emotion, label_id in emotion_label_pairs:
    key = f"{emotion}_{label_id}"

    # MOSEI
    mosei_train = mosei_loaders[key]['train']
    mosei_test = mosei_loaders[key]['test']

    # MELD
    meld_train = meld_loaders[key]['train']
    meld_test = meld_loaders[key]['test']

    print(f"{emotion}:")
    print(f"  MOSEI - 训练: {len(mosei_train.dataset)}, 测试: {len(mosei_test.dataset)}")
    print(f"  MELD  - 训练: {len(meld_train.dataset)}, 测试: {len(meld_test.dataset)}")
```

## 许可证

MIT License
