# 数据集特征提取文档

完整的 MOSEI 和 MELD 数据集特征提取工具，使用统一的多模态特征提取框架。

## 📋 目录

- [功能特性](#功能特性)
- [数据集组织结构](#数据集组织结构)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [配置选项](#配置选项)
- [输出格式](#输出格式)
- [与DataLoader集成](#与dataloader集成)

## 功能特性

✅ **支持多个数据集**
- MOSEI 数据集
- MELD 数据集（train/dev/test）

✅ **灵活的特征提取**
- 多种 backbone 选择（BERT/RoBERTa, Wav2vec2/HuBERT, MediaPipe/ViT）
- 自动时间对齐
- 批量处理

✅ **兼容现有代码**
- 输出格式与 `emotion_dataloader.py` 兼容
- 按情感类型分组保存
- 支持自定义配置

## 数据集组织结构

### MOSEI 数据集

```
MOSEI/
├── audio/
│   └── video_id/
│       └── clip_id.wav
├── video/
│   └── video_id/
│       └── clip_id.mp4
├── text/  (可选)
│   └── ...
└── label/
    └── label.csv
```

**label.csv 必须包含以下列**:
- `video_id`: 视频ID
- `clip_id`: 片段ID
- `text`: 文本内容
- `emotion` 或 `voted_emotion`: 情感标签

### MELD 数据集

```
MELD/
├── train/
│   ├── audio/
│   │   └── video_id/
│   │       └── clip_id.wav
│   ├── video/
│   │   └── video_id/
│   │       └── clip_id.mp4
│   └── label/
│       └── merged_label_new.csv
├── dev/
│   └── ... (同上)
└── test/
    └── ... (同上)
```

**merged_label_new.csv 必须包含以下列**:
- `video_id`: 视频ID
- `clip_id`: 片段ID
- `text`: 文本内容
- `emotion` 或 `voted_emotion`: 情感标签

## 安装依赖

```bash
# 基础依赖
pip install torch transformers librosa numpy opencv-python mediapipe pandas tqdm

# 可选（如果使用 ViT）
pip install Pillow

# 可选（如果使用 OpenFace）
# 需要单独安装 OpenFace
```

## 快速开始

### 方式 1: 交互式运行

```bash
python extract_dataset_features.py
```

按照提示选择：
1. 特征提取器配置
2. 数据集类型
3. 输入路径和输出路径

### 方式 2: 使用配置文件（推荐）

**1. 创建配置文件**

```json
{
  "mosei": {
    "base_dir": "/path/to/MOSEI",
    "label_file": "/path/to/MOSEI/label.csv",
    "output_dir": "./output/mosei",
    "feature_config": "config_roberta_hubert_vit.json"
  },
  "meld": {
    "base_dir": "/path/to/MELD",
    "output_dir": "./output/meld",
    "split": "all",
    "feature_config": "config_roberta_hubert_vit.json"
  }
}
```

**2. 运行批量提取**

```bash
# 提取所有数据集
python batch_extract_datasets.py --config extraction_config.json

# 只提取 MOSEI
python batch_extract_datasets.py --config extraction_config.json --dataset mosei

# 只提取 MELD
python batch_extract_datasets.py --config extraction_config.json --dataset meld
```

### 方式 3: 在代码中使用

```python
from extract_dataset_features import MOSEIFeatureExtractor, MELDFeatureExtractor
import json

# 加载特征提取配置
with open('config_roberta_hubert_vit.json', 'r') as f:
    config = json.load(f)

# 提取 MOSEI
mosei_extractor = MOSEIFeatureExtractor(
    base_dir="/path/to/MOSEI",
    output_dir="./output/mosei",
    label_file="/path/to/MOSEI/label.csv",
    config=config
)
mosei_extractor.process_dataset()

# 提取 MELD
meld_extractor = MELDFeatureExtractor(
    base_dir="/path/to/MELD",
    output_dir="./output/meld",
    config=config
)
meld_extractor.process_dataset(split='all')
```

## 详细使用

### MOSEI 数据集提取

```python
from extract_dataset_features import MOSEIFeatureExtractor

# 创建提取器
extractor = MOSEIFeatureExtractor(
    base_dir="/data/MOSEI",
    output_dir="./output/mosei",
    label_file="/data/MOSEI/label.csv",
    config=None  # 使用默认配置
)

# 处理数据集
results = extractor.process_dataset()

# 查看结果
for emotion, samples in results.items():
    print(f"{emotion}: {len(samples)} 个样本")
```

### MELD 数据集提取

```python
from extract_dataset_features import MELDFeatureExtractor

# 创建提取器
extractor = MELDFeatureExtractor(
    base_dir="/data/MELD",
    output_dir="./output/meld",
    config=None  # 使用默认配置
)

# 处理所有划分
results = extractor.process_dataset(split='all')

# 或只处理特定划分
train_results = extractor.process_dataset(split='train')
```

## 配置选项

### 特征提取器配置

可以使用预定义的配置文件：

| 配置文件 | 文本 | 音频 | 视频 |
|---------|------|------|------|
| `config_aligned.json` | BERT | Wav2vec2 | MediaPipe |
| `config_roberta_hubert_vit.json` | RoBERTa | HuBERT | ViT-16 |
| `config_hubert.json` | BERT | HuBERT | MediaPipe |
| `config_vit.json` | BERT | Wav2vec2 | ViT-16 |

**自定义配置示例**:

```json
{
  "text": {
    "model": "roberta-base",
    "enabled": true
  },
  "audio": {
    "model": "hubert",
    "model_name": "facebook/hubert-base-ls960",
    "sample_rate": 16000,
    "enabled": true
  },
  "video": {
    "model": "vit",
    "model_name": "google/vit-base-patch16-224",
    "feature_mode": "cls",
    "enabled": true
  },
  "alignment": {
    "enabled": true,
    "reference": "audio"
  }
}
```

### 提取配置

**MOSEI 配置**:
```json
{
  "mosei": {
    "base_dir": "/path/to/MOSEI",
    "label_file": "/path/to/MOSEI/label.csv",
    "output_dir": "./output/mosei",
    "feature_config": "config_roberta_hubert_vit.json"
  }
}
```

**MELD 配置**:
```json
{
  "meld": {
    "base_dir": "/path/to/MELD",
    "output_dir": "./output/meld",
    "split": "all",  // "train", "dev", "test", 或 "all"
    "feature_config": "config_roberta_hubert_vit.json"
  }
}
```

## 输出格式

### 文件命名

**MOSEI**:
```
MOSEIhappylabel0.pkl
MOSEIsadlabel1.pkl
MOSEIangerlabel2.pkl
...
```

**MELD**:
```
MELD_trainhappylabel0.pkl
MELD_devhappylabel0.pkl
MELD_testhappylabel0.pkl
MELD_trainsadlabel1.pkl
...
```

### 数据结构

每个 `.pkl` 文件包含一个样本列表，每个样本是一个字典：

```python
{
    'audio_features': torch.Tensor,  # [num_frames, audio_dim]
    'text_features': torch.Tensor,   # [num_frames, 768]
    'video_features': torch.Tensor,  # [num_frames, video_dim]
    'label': int,                    # 情感标签ID
    'emotion': str,                  # 情感类型
    'sample_id': str,                # 样本ID
    'num_frames': int,               # 对齐后的帧数
    'split': str                     # 数据集划分 (仅MELD)
}
```

### 情感标签映射

```python
{
    'happy': 0,
    'sad': 1,
    'anger': 2,
    'disgust': 3,
    'surprise': 4,
    'fear': 5,
    'neutral': 6  # 通常会被跳过
}
```

## 与DataLoader集成

提取的特征可以直接与现有的 `emotion_dataloader.py` 集成使用。

### 示例

```python
from emotion_dataloader import create_dataloaders

# 使用提取的 MOSEI 特征
mosei_loaders = create_dataloaders(
    data_dir='./output/mosei',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32,
    train_ratio=0.7  # MOSEI 自动划分
)

train_loader = mosei_loaders['train']
test_loader = mosei_loaders['test']

# 训练循环
for batch in train_loader:
    audio = batch['audio_features']
    text = batch['text_features']
    video = batch['video_features']
    labels = batch['label']
    # ... 训练模型
```

```python
# 使用提取的 MELD 特征
meld_loaders = create_dataloaders(
    data_dir='./output/meld',
    dataset_name='MELD',
    emotion='sad',
    label_id=1,
    batch_size=32
    # MELD 自动合并 train+dev
)

train_loader = meld_loaders['train']  # train + dev
test_loader = meld_loaders['test']    # test
```

## 高级用法

### 1. 并行处理多个情感

```python
from multiprocessing import Pool
from extract_dataset_features import MOSEIFeatureExtractor

def process_emotion(emotion):
    extractor = MOSEIFeatureExtractor(...)
    # 处理特定情感
    pass

emotions = ['happy', 'sad', 'anger', 'disgust', 'surprise', 'fear']
with Pool(6) as p:
    p.map(process_emotion, emotions)
```

### 2. 自定义特征后处理

```python
class CustomMOSEIExtractor(MOSEIFeatureExtractor):
    def extract_single_sample(self, text, audio_path, video_path, sample_id):
        # 调用父类方法
        features = super().extract_single_sample(
            text, audio_path, video_path, sample_id
        )

        if features:
            # 自定义后处理
            features['audio'] = self.normalize(features['audio'])
            features['video'] = self.augment(features['video'])

        return features

    def normalize(self, tensor):
        # 归一化
        return (tensor - tensor.mean()) / tensor.std()

    def augment(self, tensor):
        # 数据增强
        return tensor
```

### 3. 增量提取

```python
# 检查已处理的样本
processed_samples = set()
output_file = './output/mosei/MOSEIhappylabel0.pkl'

if os.path.exists(output_file):
    with open(output_file, 'rb') as f:
        existing_data = pickle.load(f)
        processed_samples = {s['sample_id'] for s in existing_data}

# 只处理新样本
# ... 在处理循环中跳过已处理的样本
```

## 性能优化

### 1. GPU 加速

确保 PyTorch 检测到 GPU:

```python
import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
```

### 2. 批量处理

对于大型数据集，可以分批处理：

```bash
# 处理 MELD 的不同划分
python batch_extract_datasets.py --config config.json --dataset meld

# 分别处理
python extract_dataset_features.py  # 选择 train
python extract_dataset_features.py  # 选择 dev
python extract_dataset_features.py  # 选择 test
```

### 3. 内存管理

对于内存有限的环境：

```python
# 禁用梯度计算
torch.set_grad_enabled(False)

# 及时释放内存
import gc
gc.collect()
torch.cuda.empty_cache()
```

## 故障排查

### 1. 文件路径错误

**问题**: `FileNotFoundError: 音频/视频文件不存在`

**解决**:
- 检查数据集组织结构是否正确
- 确认 CSV 文件中的 `video_id` 和 `clip_id` 与实际文件名匹配
- 使用绝对路径

### 2. 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决**:
- 减少视频帧率
- 使用 CPU 模式
- 分批处理数据集

### 3. CSV 列名不匹配

**问题**: `KeyError: 'emotion'`

**解决**:
- 确保 CSV 包含 `emotion` 或 `voted_emotion` 列
- 检查列名大小写
- 查看 CSV 文件的前几行确认格式

### 4. 模型下载失败

**问题**: HuggingFace 模型下载失败

**解决**:
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或预先下载模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('roberta-base')"
```

## 完整示例

### 示例 1: 提取 MOSEI 数据集

```bash
# 1. 创建配置文件
cat > mosei_config.json << EOF
{
  "mosei": {
    "base_dir": "/data/MOSEI",
    "label_file": "/data/MOSEI/label.csv",
    "output_dir": "./output/mosei",
    "feature_config": "config_roberta_hubert_vit.json"
  }
}
EOF

# 2. 运行提取
python batch_extract_datasets.py --config mosei_config.json --dataset mosei

# 3. 查看输出
ls -lh ./output/mosei/
```

### 示例 2: 提取 MELD 数据集

```bash
# 1. 创建配置文件
cat > meld_config.json << EOF
{
  "meld": {
    "base_dir": "/data/MELD",
    "output_dir": "./output/meld",
    "split": "all",
    "feature_config": "config_vit.json"
  }
}
EOF

# 2. 运行提取
python batch_extract_datasets.py --config meld_config.json --dataset meld

# 3. 查看输出
ls -lh ./output/meld/
```

### 示例 3: 与 DataLoader 集成

```python
from emotion_dataloader import create_dataloaders
import torch.nn as nn
import torch.optim as optim

# 加载数据
dataloaders = create_dataloaders(
    data_dir='./output/mosei',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

# 定义模型
class MultimodalModel(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, hidden_dim, num_classes):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, audio, text, video):
        audio_h = self.audio_proj(audio.mean(dim=1))
        text_h = self.text_proj(text.mean(dim=1))
        video_h = self.video_proj(video.mean(dim=1))
        combined = torch.cat([audio_h, text_h, video_h], dim=1)
        return self.classifier(combined)

# 训练
model = MultimodalModel(768, 768, 768, 256, 2)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in dataloaders['train']:
        audio = batch['audio_features']
        text = batch['text_features']
        video = batch['video_features']
        labels = batch['label']

        outputs = model(audio, text, video)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 参考资料

- **特征提取框架**: `feature_extraction_demo.py`
- **对齐机制**: `ALIGNMENT_MECHANISM.md`
- **Backbone 对比**: `FAQ_BACKBONE_ALIGNMENT.md`
- **DataLoader**: `DATALOADER_README.md`
- **配置示例**: `extraction_config_example.json`

## 许可证

MIT License
