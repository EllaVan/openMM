# 多模态特征提取工具

使用本地 RoBERTa、HuBERT 和 ViT 模型提取 MOSEI 和 MELD 数据集的对齐特征。

## 📁 目录结构

```
unimodal_features/
├── feature_extractor.py       # 多模态特征提取器
├── dataset_extractor.py       # 数据集提取器
├── batch_extract.py           # 批量提取脚本
├── config.json                # 模型路径配置
├── extraction_config.json     # 数据集路径配置
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 配置模型路径

编辑 `config.json`，填入本地模型路径：

```json
{
  "text": {
    "model_path": "/path/to/roberta-base"
  },
  "audio": {
    "model_path": "/path/to/hubert-base-ls960"
  },
  "video": {
    "model_path": "/path/to/vit-base-patch16-224-in21k"
  }
}
```

### 2. 配置数据集路径

编辑 `extraction_config.json`：

```json
{
  "mosei": {
    "base_dir": "/path/to/MOSEI",
    "label_file": "/path/to/MOSEI/label/label.csv",
    "output_dir": "./output/mosei_features"
  },
  "meld": {
    "base_dir": "/path/to/MELD",
    "output_dir": "./output/meld_features",
    "split": "all"
  }
}
```

### 3. 运行提取

```bash
# 提取 MOSEI
python unimodal_features/batch_extract.py --dataset mosei

# 提取 MELD
python unimodal_features/batch_extract.py --dataset meld

# 提取所有
python unimodal_features/batch_extract.py --dataset all
```

## 📊 数据集格式

### MOSEI 数据集

```
MOSEI/
├── audio/
│   └── video_id/
│       ├── 0.wav
│       ├── 1.wav
│       └── ...
├── video/
│   └── video_id/
│       ├── 0.mp4
│       ├── 1.mp4
│       └── ...
└── label/
    └── label.csv  # 列: video_id, clip_id, text, emotion
```

### MELD 数据集

```
MELD/
├── train/
│   ├── audio/
│   │   ├── file_id.wav
│   │   └── ...
│   ├── video/
│   │   ├── file_id.mp4
│   │   └── ...
│   └── label.csv  # 列: file_id, text, emotion
├── dev/
│   └── ... (同上)
└── test/
    └── ... (同上)
```

## 🔧 核心功能

### 特征提取器 (`feature_extractor.py`)

- **文本**: RoBERTa → `[num_frames, 768]`
- **音频**: HuBERT → `[num_frames, 768]`
- **视频**: ViT → `[num_frames, 768]`

自动时间对齐：
1. 音频作为对齐基准，提取时间戳
2. 文本通过线性插值对齐
3. 视频根据时间戳采样帧

### 数据集提取器 (`dataset_extractor.py`)

- `MOSEIFeatureExtractor`: 处理 MOSEI 数据集
- `MELDFeatureExtractor`: 处理 MELD 数据集（修正后结构）

输出格式：`{dataset}{split}{emotion}label{id}.pkl`

例如：
- `MOSEIhappylabel0.pkl`
- `MELD_trainhappylabel0.pkl`

### 批量提取 (`batch_extract.py`)

命令行工具，支持：
- 单个或多个数据集提取
- 自定义配置文件路径
- 日志记录

## 📦 输出格式

每个 `.pkl` 文件包含一个列表，每个元素为：

```python
{
    'audio_features': torch.Tensor,  # [num_frames, 768]
    'text_features': torch.Tensor,   # [num_frames, 768]
    'video_features': torch.Tensor,  # [num_frames, 768]
    'label': int,                    # 情感标签 ID
    'emotion': str,                  # 情感名称
    'sample_id': str,                # 样本 ID
    'num_frames': int                # 总帧数
}
```

## 💡 使用示例

### Python API

```python
import json
from unimodal_features.feature_extractor import MultimodalFeatureExtractor

# 加载配置
with open('unimodal_features/config.json', 'r') as f:
    config = json.load(f)

# 创建提取器
extractor = MultimodalFeatureExtractor(config)

# 提取单个样本
features = extractor.extract_multimodal_features(
    text="This is a test.",
    audio_path="sample.wav",
    video_path="sample.mp4"
)

print(features['audio_features'].shape)  # [T, 768]
print(features['text_features'].shape)   # [T, 768]
print(features['video_features'].shape)  # [T, 768]
```

### 数据集提取

```python
from unimodal_features.dataset_extractor import MOSEIFeatureExtractor

extractor = MOSEIFeatureExtractor(
    base_dir='/path/to/MOSEI',
    output_dir='./output/mosei_features',
    label_file='/path/to/MOSEI/label/label.csv',
    config=config
)

stats = extractor.process_dataset()
print(stats)  # {'happy': 123, 'sad': 456, ...}
```

## ⚙️ 依赖

```bash
pip install torch transformers librosa soundfile opencv-python pandas tqdm
```

## 🎯 特点

- ✅ **仅支持本地模型**：无需 HuggingFace 下载
- ✅ **自动时间对齐**：三模态特征帧数一致
- ✅ **批量处理**：支持整个数据集提取
- ✅ **修正 MELD 结构**：适配新的目录组织
- ✅ **日志记录**：完整的提取过程记录

## 📝 注意事项

1. **模型路径**：确保本地模型路径包含 `config.json` 和模型权重文件
2. **GPU 加速**：自动检测 CUDA，建议使用 GPU 加速提取
3. **内存管理**：大数据集建议分批次提取
4. **文件检查**：自动跳过缺失的音频/视频文件

## 🔍 故障排查

### 模型加载失败

```bash
# 检查模型路径
ls /path/to/roberta-base/config.json
ls /path/to/roberta-base/pytorch_model.bin
```

### CUDA 错误

```python
import torch
print(torch.cuda.is_available())  # 应为 True
```

### 特征维度不匹配

确保使用的模型为：
- RoBERTa-base: 768 维
- HuBERT-base: 768 维
- ViT-base: 768 维

---

**完成时间对齐的多模态特征提取！** 🎉
