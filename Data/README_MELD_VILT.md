# MELD数据集特征提取 (使用ViLT)

## 概述

这个脚本用于从MELD数据集中提取多模态特征:
- **文本特征**: BERT (bert-base-uncased)
- **视频特征**: ViLT (Vision-and-Language Transformer)
- **音频特征**: HuBERT (hubert-base-ls960)

## 与原版的区别

相比 `prepare_data_meld.py`:
- 视频特征提取从 **ViT** 改为 **ViLT**
- ViLT支持视觉-语言联合编码,可以利用文本信息增强视频特征
- 其他部分保持一致

## 使用前准备

### 1. 修改路径配置

在 `prepare_data_meld_vilt.py` 中找到以下需要修改的路径:

```python
# 数据集路径
base_dir = '/path/to/Datasets/MELD/organized/dev'  # TODO: 修改

# 预训练模型路径
BERT_MODEL_PATH = "/path/to/bert-base-uncased"      # TODO: 修改
HUBERT_MODEL_PATH = "/path/to/hubert-base-ls960"    # TODO: 修改
VILT_MODEL_PATH = "/path/to/vilt-b32-mlm"           # TODO: 修改

# 输出文件
save_path = 'meld_multimodal_features_vilt.npy'     # TODO: 可选修改
```

### 2. 数据集目录结构

确保你的MELD数据集组织如下:

```
Datasets/MELD/organized/dev/
├── audio/
│   ├── video_id_1/
│   │   ├── clip_1.wav
│   │   ├── clip_2.wav
│   │   └── ...
│   └── video_id_2/
│       └── ...
├── video/
│   ├── video_id_1/
│   │   ├── clip_1.mp4
│   │   ├── clip_2.mp4
│   │   └── ...
│   └── video_id_2/
│       └── ...
├── text/
│   └── ...
└── label/
    └── merged_label_new.csv
```

### 3. 标签文件格式

CSV文件应包含以下列:
- `video_id`: 视频ID
- `clip_id`: 片段ID
- `text`: 对话文本
- `voted_emotion`: 情感标签 (happy/sad/anger/disgust/surprise/fear/neutral)

示例:
```csv
video_id,clip_id,text,voted_emotion
dia0_utt0,clip_001,"Hello, how are you?",happy
dia0_utt1,clip_002,"I'm fine, thanks.",neutral
```

### 4. 安装依赖

```bash
pip install torch transformers torchaudio opencv-python pandas numpy tqdm pillow
```

## 运行脚本

```bash
python Data/prepare_data_meld_vilt.py
```

## 输出格式

输出的 `.npy` 文件包含多个样本,每个样本是一个字典:

```python
{
    'audio_features': torch.Tensor,  # shape: (1, 512, 768)
    'text_features': torch.Tensor,   # shape: (1, 512, 768)
    'video_features': torch.Tensor,  # shape: (1, 512, 768)
    'label': int,                    # 情感标签 0-6
    'video_id': str,                 # 视频ID
    'clip_id': str                   # 片段ID
}
```

## 读取保存的数据

```python
import numpy as np

data_list = []
with open('meld_multimodal_features_vilt.npy', 'rb') as f:
    while True:
        try:
            sample = np.load(f, allow_pickle=True).item()
            data_list.append(sample)
        except:
            break

print(f"共加载 {len(data_list)} 个样本")
```

## 关键参数调整

### 视频采样率
在 `extract_video_features()` 函数中:
```python
frames_per_second = 5  # 每秒采样5帧,可根据需要调整
```

### 特征对齐长度
在 `align_features()` 函数中:
```python
target_length = 512  # 所有模态对齐到512长度,可根据需要调整
```

### 跳过neutral类别
如果不需要neutral类别,在main()函数中取消注释:
```python
if voted_emotion == 'neutral':
    continue
```

## ViLT模型说明

ViLT (Vision-and-Language Transformer) 的优势:
1. **联合编码**: 同时处理图像和文本,捕获跨模态关系
2. **轻量级**: 相比其他视觉-语言模型更高效
3. **适用场景**: 特别适合有文本上下文的视频理解任务

在本脚本中,ViLT会同时接收:
- **视觉输入**: 视频帧
- **文本输入**: 对话文本

这样可以更好地理解视频内容与对话的关系。

## 预训练模型下载

### BERT
```bash
# Hugging Face
bert-base-uncased
```

### HuBERT
```bash
# Hugging Face
facebook/hubert-base-ls960
```

### ViLT
```bash
# Hugging Face
dandelin/vilt-b32-mlm
# 或
dandelin/vilt-b32-finetuned-coco
```

## 常见问题

### Q: 内存不足怎么办?
A: 可以减小batch处理的帧数,或者降低target_length

### Q: 处理速度慢怎么办?
A:
- 确保使用GPU: `device = torch.device("cuda")`
- 减少视频采样率: `frames_per_second = 3`
- 使用更小的ViLT模型

### Q: 某些视频/音频文件损坏?
A: 脚本会自动跳过无法处理的文件,并在控制台输出错误信息

## 许可

本代码基于 `prepare_data_meld.py` 修改而来。
