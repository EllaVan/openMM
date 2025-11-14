# 多模态特征提取 Demo

基于 [MMSA-FET](https://github.com/thuiar/MMSA-FET) 的多模态特征提取工具，支持从原始数据（.txt、.wav、.mp4）提取特征并进行时间步对齐。

## 功能特性

### 支持的模态

1. **文本 (Text)**
   - BERT 预训练模型提取文本特征
   - 支持词级别的特征提取
   - 输出维度: [seq_len, 768]

2. **音频 (Audio)**
   - **Wav2vec2**: 深度学习音频特征 [time_steps, 768]
   - **Librosa**: 传统声学特征 (MFCC, RMS, ZCR, Spectral Centroid)
   - 自动生成时间戳

3. **视频 (Video)**
   - **MediaPipe**: 面部网格特征 (468个关键点)
   - **OpenFace**: 面部动作单元和关键点 (需要外部安装)
   - 支持多帧提取

### 时间对齐

- **基于时间戳的对齐**: 使用音频时间戳作为基准
- **线性插值**: 自动调整不同模态的时间步
- **帧级对齐**: 确保所有模态在同一时间点的特征对齐

## 安装依赖

### 基础依赖

```bash
pip install torch transformers librosa numpy opencv-python mediapipe
```

### 详细说明

```bash
# PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio

# Transformers (BERT, Wav2vec2)
pip install transformers

# 音频处理
pip install librosa soundfile

# 视频处理
pip install opencv-python mediapipe

# 其他
pip install numpy pickle-mixin
```

### 可选依赖

```bash
# OpenFace (需要单独安装，参考官方文档)
# https://github.com/TadasBaltrusaitis/OpenFace

# openSMILE (高级音频特征)
# https://github.com/audeering/opensmile-python
pip install opensmile
```

## 快速开始

### 1. 准备数据

创建示例数据目录并放置文件:

```bash
mkdir -p example_data
# 放置以下文件:
# - sample.txt (文本文件)
# - sample.wav (音频文件，16kHz推荐)
# - sample.mp4 (视频文件)
```

### 2. 基本使用

```python
from feature_extraction_demo import MultimodalFeatureExtractor

# 初始化提取器 (使用默认 aligned 配置)
extractor = MultimodalFeatureExtractor()

# 从原始文件提取对齐特征
features = extractor.extract_from_files(
    text_file="example_data/sample.txt",
    audio_file="example_data/sample.wav",
    video_file="example_data/sample.mp4",
    output_file="output/sample_features.pkl"
)

# 查看对齐后的特征
print("对齐后的特征:")
print(f"  时间步数: {features['num_frames']}")
print(f"  音频特征: {features['audio'].shape}")
print(f"  文本特征: {features['text'].shape}")
print(f"  视频特征: {features['video'].shape}")
```

### 3. 自定义配置

```python
# 自定义配置
custom_config = {
    'text': {
        'model': 'bert-base-uncased',
        'enabled': True
    },
    'audio': {
        'model': 'librosa',  # 或 'wav2vec2'
        'sample_rate': 16000,
        'enabled': True
    },
    'video': {
        'model': 'mediapipe',  # 或 'openface'
        'fps': 25,
        'enabled': True
    },
    'alignment': {
        'method': 'wav2vec_ctc',
        'enabled': True
    }
}

extractor = MultimodalFeatureExtractor(config=custom_config)
```

### 4. 批量处理

```python
from feature_extraction_demo import MultimodalFeatureExtractor
import os

extractor = MultimodalFeatureExtractor()

# 定义文件列表
file_list = [
    {'id': 'video1', 'text': 'data/video1.txt', 'audio': 'data/video1.wav', 'video': 'data/video1.mp4'},
    {'id': 'video2', 'text': 'data/video2.txt', 'audio': 'data/video2.wav', 'video': 'data/video2.mp4'},
]

# 批量提取
for item in file_list:
    features = extractor.extract_from_files(
        text_file=item['text'],
        audio_file=item['audio'],
        video_file=item['video'],
        output_file=f"output/{item['id']}_features.pkl"
    )
    print(f"✓ {item['id']} 处理完成")
```

## 配置详解

### 默认配置 (Aligned Config)

```python
{
    'text': {
        'model': 'bert-base-uncased',  # BERT模型
        'enabled': True                # 启用文本提取
    },
    'audio': {
        'model': 'wav2vec2',           # 音频模型 (wav2vec2 或 librosa)
        'sample_rate': 16000,          # 采样率
        'enabled': True                # 启用音频提取
    },
    'video': {
        'model': 'openface',           # 视频模型 (openface 或 mediapipe)
        'fps': 25,                     # 帧率
        'enabled': True                # 启用视频提取
    },
    'alignment': {
        'method': 'wav2vec_ctc',       # 对齐方法
        'enabled': True                # 启用时间对齐
    }
}
```

### 文本配置选项

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| model | str | BERT模型名称 | bert-base-uncased, bert-large-uncased, roberta-base 等 |
| enabled | bool | 是否启用文本提取 | True, False |

### 音频配置选项

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| model | str | 音频提取模型 | wav2vec2, librosa |
| sample_rate | int | 音频采样率 | 16000, 22050, 44100 |
| enabled | bool | 是否启用音频提取 | True, False |

**Wav2vec2 vs Librosa:**

- **Wav2vec2**:
  - 优点: 深度学习特征，语义信息丰富
  - 缺点: 需要GPU，速度较慢
  - 特征维度: [time_steps, 768]

- **Librosa**:
  - 优点: 速度快，传统声学特征
  - 缺点: 语义信息较少
  - 特征: MFCC (13), RMS (1), ZCR (1), Spectral Centroid (1)

### 视频配置选项

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| model | str | 视频提取模型 | mediapipe, openface |
| fps | int | 视频帧率 | 25, 30 |
| enabled | bool | 是否启用视频提取 | True, False |

**MediaPipe vs OpenFace:**

- **MediaPipe**:
  - 优点: 易安装，速度快
  - 特征: 468个面部关键点 (x, y, z)
  - 输出维度: [num_frames, 1404]

- **OpenFace**:
  - 优点: 功能全面 (AU, 头部姿态等)
  - 缺点: 需要外部安装
  - 特征: 68个关键点 + AU + 头部姿态

## 时间对齐机制

### 对齐流程

1. **选择基准**: 使用音频时间戳作为对齐基准
2. **文本对齐**: 使用线性插值将文本特征对齐到音频帧数
3. **视频对齐**: 根据时间戳找到最接近的视频帧

### 对齐示例

```
音频: [0.00s, 0.02s, 0.04s, ..., 10.00s] -> 500帧
文本: [token1, token2, ..., token50]      -> 50个token
视频: [frame1, frame2, ..., frame250]     -> 250帧 (25fps)

对齐后:
- 音频: [500, 768]  # 500帧，每帧768维
- 文本: [500, 768]  # 插值到500帧
- 视频: [500, 1404] # 根据时间戳对齐到500帧
```

### 手动对齐

```python
# 如果需要自定义对齐方式
aligned_features = extractor.align_features(
    text_features=text_features,
    audio_features=audio_features,
    video_features=video_features
)
```

## 输出格式

### 对齐后的特征字典

```python
{
    'timestamps': np.array,     # 时间戳数组 [num_frames]
    'audio': torch.Tensor,      # 音频特征 [num_frames, audio_dim]
    'text': torch.Tensor,       # 文本特征 [num_frames, 768]
    'video': torch.Tensor,      # 视频特征 [num_frames, video_dim]
    'num_frames': int           # 总帧数
}
```

### 保存和加载

```python
# 保存特征
extractor.save_features(features, "output/features.pkl")

# 加载特征
loaded_features = extractor.load_features("output/features.pkl")
```

## 完整示例

### 示例 1: 单个文件提取

```python
from feature_extraction_demo import MultimodalFeatureExtractor

# 初始化
extractor = MultimodalFeatureExtractor()

# 提取特征
features = extractor.extract_from_files(
    text_file="data/interview.txt",
    audio_file="data/interview.wav",
    video_file="data/interview.mp4",
    output_file="output/interview_features.pkl"
)

# 使用特征进行下游任务
audio_features = features['audio']  # [T, 768]
text_features = features['text']    # [T, 768]
video_features = features['video']  # [T, 1404]

# 可以输入到多模态模型
# output = model(audio_features, text_features, video_features)
```

### 示例 2: 仅提取部分模态

```python
# 只提取音频和文本
features = extractor.extract_from_files(
    text_file="data/interview.txt",
    audio_file="data/interview.wav",
    # video_file=None,  # 不提供视频文件
    output_file="output/audio_text_features.pkl"
)
```

### 示例 3: 使用 Librosa 提取音频特征

```python
config = {
    'text': {'model': 'bert-base-uncased', 'enabled': True},
    'audio': {'model': 'librosa', 'sample_rate': 16000, 'enabled': True},
    'video': {'model': 'mediapipe', 'fps': 25, 'enabled': True},
    'alignment': {'method': 'interpolation', 'enabled': True}
}

extractor = MultimodalFeatureExtractor(config=config)
features = extractor.extract_from_files(
    text_file="data/sample.txt",
    audio_file="data/sample.wav",
    video_file="data/sample.mp4"
)
```

## 与 MMSA 框架集成

提取的特征可以直接用于 [MMSA](https://github.com/thuiar/MMSA) 框架进行多模态情感分析:

```python
from feature_extraction_demo import MultimodalFeatureExtractor
import pickle

# 1. 提取特征
extractor = MultimodalFeatureExtractor()
features = extractor.extract_from_files(
    text_file="data/sample.txt",
    audio_file="data/sample.wav",
    video_file="data/sample.mp4"
)

# 2. 转换为 MMSA 格式
mmsa_data = {
    'audio': features['audio'].numpy(),
    'vision': features['video'].numpy(),
    'text': features['text'].numpy(),
    'audio_lengths': features['num_frames'],
    'vision_lengths': features['num_frames'],
    'text_lengths': features['num_frames']
}

# 3. 保存为 MMSA 格式
with open('output/mmsa_format.pkl', 'wb') as f:
    pickle.dump(mmsa_data, f)
```

## 常见问题

### 1. 内存不足

**问题**: 处理长视频时内存溢出

**解决方案**:
- 分段处理视频
- 使用较低的视频帧率
- 禁用不需要的模态

```python
config = {
    'text': {'enabled': True},
    'audio': {'enabled': True},
    'video': {'enabled': False}  # 禁用视频
}
```

### 2. OpenFace 安装问题

**问题**: OpenFace 难以安装

**解决方案**: 使用 MediaPipe 作为替代

```python
config = {
    'video': {'model': 'mediapipe', 'enabled': True}
}
```

### 3. 时间对齐不准确

**问题**: 不同模态的时间对齐有偏差

**解决方案**:
- 确保音频和视频来自同一源
- 使用更精确的对齐方法 (Wav2vec CTC)
- 手动调整时间戳

### 4. GPU 内存不足

**问题**: 使用 Wav2vec2 时 GPU 内存不足

**解决方案**:
- 切换到 Librosa
- 分段处理音频
- 使用 CPU 模式

```python
config = {
    'audio': {'model': 'librosa', 'enabled': True}
}
```

### 5. 视频无人脸检测

**问题**: MediaPipe 无法检测到人脸

**解决方案**:
- 检查视频质量
- 调整检测置信度阈值
- 使用 OpenFace

## 性能优化

### 1. 使用批处理

```python
# 批量处理多个文件
file_list = [...]  # 文件列表
for item in file_list:
    features = extractor.extract_from_files(...)
```

### 2. 多进程处理

```python
from multiprocessing import Pool

def process_file(item):
    extractor = MultimodalFeatureExtractor()
    return extractor.extract_from_files(...)

with Pool(4) as p:
    results = p.map(process_file, file_list)
```

### 3. 缓存提取结果

```python
import os

output_file = f"output/{item['id']}_features.pkl"
if not os.path.exists(output_file):
    features = extractor.extract_from_files(...)
else:
    features = extractor.load_features(output_file)
```

## 参考资料

- [MMSA-FET GitHub](https://github.com/thuiar/MMSA-FET)
- [MMSA Framework](https://github.com/thuiar/MMSA)
- [Wav2vec2 Paper](https://arxiv.org/abs/2006.11477)
- [MediaPipe Docs](https://google.github.io/mediapipe/)
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

## 贡献

欢迎提交 Issue 和 Pull Request!

## 许可证

MIT License
