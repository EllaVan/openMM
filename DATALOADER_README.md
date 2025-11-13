# 情感数据集DataLoader使用文档

这个模块提供了一个灵活的PyTorch DataLoader系统，用于加载MOSEI和MELD情感数据集。

## 数据格式

数据文件应该是pickle格式，文件命名规则如下：

- **MELD数据集**: `MELD_{split}{emotion}label{label_id}.pkl`
  - 例如: `MELD_trainhappylabel0.pkl`, `MELD_devsadlabel1.pkl`

- **MOSEI数据集**: `MOSEI{emotion}label{label_id}.pkl`
  - 例如: `MOSEIhappylabel0.pkl`, `MOSEIsadlabel1.pkl`

## 安装依赖

```bash
pip install torch pickle
```

## 快速开始

### 1. 创建单个DataLoader (MOSEI)

```python
from emotion_dataloader import create_emotion_dataloader

# 创建MOSEI数据集的DataLoader
dataloader = create_emotion_dataloader(
    data_dir='Data',           # 数据文件目录
    dataset_name='MOSEI',      # 数据集名称
    emotion='happy',           # 情感类型
    label_id=0,                # 标签ID
    batch_size=32,             # 批次大小
    shuffle=True,              # 是否打乱
    num_workers=4              # 加载进程数
)

# 使用DataLoader
for batch in dataloader:
    # batch是一个字典，包含audio_features, text_features, video_features, label等
    print(batch.keys())
    break
```

### 2. 创建单个DataLoader (MELD)

```python
from emotion_dataloader import create_emotion_dataloader

# MELD数据集需要指定split参数
dataloader = create_emotion_dataloader(
    data_dir='Data',
    dataset_name='MELD',
    emotion='sad',
    label_id=1,
    split='train',             # 必须指定: train/dev/test
    batch_size=32,
    shuffle=True,
    num_workers=4
)
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

# 批量创建
dataloaders = create_multiple_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion_label_pairs=emotion_label_pairs,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

# 访问特定的DataLoader
happy_loader = dataloaders['happy_0']
sad_loader = dataloaders['sad_1']
```

### 4. 创建所有split的DataLoader (仅MELD)

```python
from emotion_dataloader import create_all_splits_dataloaders

# 为MELD创建train/dev/test三个DataLoader
dataloaders = create_all_splits_dataloaders(
    data_dir='Data',
    emotion='anger',
    label_id=2,
    batch_size=32,
    num_workers=4
)

# 访问
train_loader = dataloaders['train']
dev_loader = dataloaders['dev']
test_loader = dataloaders['test']
```

### 5. 使用自定义collate函数

```python
from emotion_dataloader import create_emotion_dataloader, custom_collate_fn

dataloader = create_emotion_dataloader(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=8,
    collate_fn=custom_collate_fn  # 使用自定义collate函数
)

for batch in dataloader:
    # batch现在有特定的结构
    audio = batch['audio']      # 音频特征
    text = batch['text']        # 文本特征
    video = batch['video']      # 视频特征
    labels = batch['labels']    # 标签
    break
```

### 6. 使用数据转换函数

```python
from emotion_dataloader import create_emotion_dataloader

# 定义transform函数
def my_transform(sample):
    # 对样本进行转换，例如归一化
    if 'audio_features' in sample:
        sample['audio_features'] = sample['audio_features'] / sample['audio_features'].max()
    return sample

dataloader = create_emotion_dataloader(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='surprise',
    label_id=4,
    batch_size=16,
    transform=my_transform  # 应用转换
)
```

### 7. 在训练循环中使用

```python
from emotion_dataloader import create_emotion_dataloader
import torch.nn as nn
import torch.optim as optim

# 创建训练和验证集
train_loader = create_emotion_dataloader(
    data_dir='Data',
    dataset_name='MELD',
    emotion='happy',
    label_id=0,
    split='train',
    batch_size=32,
    shuffle=True
)

dev_loader = create_emotion_dataloader(
    data_dir='Data',
    dataset_name='MELD',
    emotion='happy',
    label_id=0,
    split='dev',
    batch_size=32,
    shuffle=False
)

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

    # 验证
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            # 验证代码
            pass
```

## API参考

### EmotionDataset

```python
EmotionDataset(
    data_dir: str,              # 数据文件目录
    dataset_name: str,          # 'MELD' 或 'MOSEI'
    emotion: str,               # 情感类型
    label_id: int,              # 标签ID
    split: Optional[str] = None,# 数据集划分 (MELD需要)
    transform = None            # 数据转换函数
)
```

**方法**:
- `__len__()`: 返回数据集大小
- `__getitem__(idx)`: 获取单个样本
- `get_info()`: 返回数据集信息

### create_emotion_dataloader

```python
create_emotion_dataloader(
    data_dir: str,              # 数据文件目录
    dataset_name: str,          # 数据集名称
    emotion: str,               # 情感类型
    label_id: int,              # 标签ID
    split: Optional[str] = None,# 数据集划分
    batch_size: int = 32,       # 批次大小
    shuffle: bool = True,       # 是否打乱
    num_workers: int = 4,       # 加载进程数
    transform = None,           # 转换函数
    **kwargs                    # 其他DataLoader参数
) -> DataLoader
```

### create_multiple_dataloaders

```python
create_multiple_dataloaders(
    data_dir: str,
    dataset_name: str,
    emotion_label_pairs: List[Tuple[str, int]],  # [(emotion, label_id), ...]
    split: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform = None,
    **kwargs
) -> Dict[str, DataLoader]
```

返回字典，key为`'emotion_labelid'`格式，value为DataLoader。

### create_all_splits_dataloaders

```python
create_all_splits_dataloaders(
    data_dir: str,
    emotion: str,
    label_id: int,
    batch_size: int = 32,
    num_workers: int = 4,
    transform = None,
    **kwargs
) -> Dict[str, DataLoader]
```

返回包含'train', 'dev', 'test'三个DataLoader的字典。

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

## 示例代码

完整的示例代码请参考 `examples/dataloader_examples.py`

```bash
python examples/dataloader_examples.py
```

## 常见问题

### 1. FileNotFoundError: 数据文件不存在

确保你的pkl文件命名格式正确，并且放在正确的目录下。

### 2. MELD数据集缺少split参数

MELD数据集必须指定split参数（train/dev/test）。

### 3. 自定义数据格式

如果你的数据格式与默认格式不同，可以：
- 继承`EmotionDataset`类并重写`__getitem__`方法
- 使用`transform`参数进行数据转换

## 许可证

MIT License
