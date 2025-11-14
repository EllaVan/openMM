# 快速开始指南

完整的多模态特征提取和数据集处理工具包。

## 🎯 核心功能

### 1. 多模态特征提取
从原始数据（.txt, .wav, .mp4）提取对齐的特征

**支持的模型**:
- **文本**: BERT, RoBERTa, DistilBERT, ALBERT
- **音频**: Wav2vec2, HuBERT, Librosa
- **视频**: MediaPipe, ViT-16, OpenFace

**核心特性**:
- ✅ 自动时间步对齐
- ✅ 多种 backbone 选择
- ✅ 灵活配置

### 2. 数据集批量处理
批量提取 MOSEI 和 MELD 数据集特征

**支持的数据集**:
- MOSEI
- MELD (train/dev/test)

**核心特性**:
- ✅ 按情感类型分组
- ✅ 与 DataLoader 兼容
- ✅ 批量处理

---

## 📦 安装

```bash
pip install torch transformers librosa numpy opencv-python mediapipe pandas tqdm
```

---

## 🚀 使用方法

### 场景 1: 单个文件特征提取

**适用于**: 提取单个样本的特征

```python
from feature_extraction_demo import MultimodalFeatureExtractor

# 初始化
extractor = MultimodalFeatureExtractor()

# 提取特征
features = extractor.extract_from_files(
    text_file="data/sample.txt",
    audio_file="data/sample.wav",
    video_file="data/sample.mp4",
    output_file="output/features.pkl"
)

# 查看结果
print(f"总帧数: {features['num_frames']}")
print(f"音频: {features['audio'].shape}")
print(f"文本: {features['text'].shape}")
print(f"视频: {features['video'].shape}")
```

**输出**:
```
总帧数: 500
音频: torch.Size([500, 768])
文本: torch.Size([500, 768])
视频: torch.Size([500, 768])
```

---

### 场景 2: 使用不同的 Backbone

**适用于**: 需要特定模型的特征

```python
from feature_extraction_demo import MultimodalFeatureExtractor

# 配置 RoBERTa + HuBERT + ViT
config = {
    'text': {'model': 'roberta-base', 'enabled': True},
    'audio': {'model': 'hubert', 'model_name': 'facebook/hubert-base-ls960',
              'sample_rate': 16000, 'enabled': True},
    'video': {'model': 'vit', 'model_name': 'google/vit-base-patch16-224',
              'feature_mode': 'cls', 'enabled': True},
    'alignment': {'enabled': True, 'reference': 'audio'}
}

extractor = MultimodalFeatureExtractor(config=config)
features = extractor.extract_from_files(...)
```

**或使用配置文件**:

```python
extractor = MultimodalFeatureExtractor(config='config_roberta_hubert_vit.json')
```

---

### 场景 3: 批量处理 MOSEI 数据集

**适用于**: 提取整个 MOSEI 数据集的特征

**方式 1: 交互式**

```bash
python extract_dataset_features.py
```

选择:
1. 配置: RoBERTa + HuBERT + ViT-16
2. 数据集: MOSEI
3. 输入路径和输出路径

**方式 2: 配置文件（推荐）**

创建 `mosei_config.json`:
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

运行:
```bash
python batch_extract_datasets.py --config mosei_config.json --dataset mosei
```

**输出**:
```
output/mosei/
├── MOSEIhappylabel0.pkl
├── MOSEIsadlabel1.pkl
├── MOSEIangerlabel2.pkl
├── MOSEIdisgustlabel3.pkl
├── MOSEIsurpriselabel4.pkl
└── MOSEIfearlabel5.pkl
```

---

### 场景 4: 批量处理 MELD 数据集

**适用于**: 提取整个 MELD 数据集的特征

创建 `meld_config.json`:
```json
{
  "meld": {
    "base_dir": "/path/to/MELD",
    "output_dir": "./output/meld",
    "split": "all",
    "feature_config": "config_vit.json"
  }
}
```

运行:
```bash
python batch_extract_datasets.py --config meld_config.json --dataset meld
```

**输出**:
```
output/meld/
├── MELD_trainhappylabel0.pkl
├── MELD_devhappylabel0.pkl
├── MELD_testhappylabel0.pkl
├── MELD_trainsadlabel1.pkl
├── MELD_devsadlabel1.pkl
├── MELD_testsadlabel1.pkl
└── ...
```

---

### 场景 5: 与 DataLoader 集成训练

**适用于**: 使用提取的特征训练模型

```python
from emotion_dataloader import create_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 加载数据
dataloaders = create_dataloaders(
    data_dir='./output/mosei',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32,
    train_ratio=0.7
)

train_loader = dataloaders['train']
test_loader = dataloaders['test']

# 2. 定义模型
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Linear(768, 256)
        self.text_encoder = nn.Linear(768, 256)
        self.video_encoder = nn.Linear(768, 256)
        self.classifier = nn.Linear(256 * 3, 2)

    def forward(self, audio, text, video):
        # 平均池化时间维度
        audio_h = self.audio_encoder(audio.mean(dim=1))
        text_h = self.text_encoder(text.mean(dim=1))
        video_h = self.video_encoder(video.mean(dim=1))

        # 拼接
        combined = torch.cat([audio_h, text_h, video_h], dim=1)
        return self.classifier(combined)

# 3. 训练
model = MultimodalModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch in train_loader:
        audio = batch['audio_features']
        text = batch['text_features']
        video = batch['video_features']
        labels = batch['label']

        outputs = model(audio, text, video)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            audio = batch['audio_features']
            text = batch['text_features']
            video = batch['video_features']
            labels = batch['label']

            outputs = model(audio, text, video)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')
```

---

### 场景 6: Backbone 对比实验

**适用于**: 对比不同特征提取器的效果

```bash
cd examples
python backbone_comparison.py
```

选择对比项目:
1. BERT vs RoBERTa (文本)
2. Wav2vec2 vs HuBERT (音频)
3. MediaPipe vs ViT-16 (视频)
4. 完整对齐测试

**示例输出**:
```
对比结果:
Wav2vec2 特征:
  - 形状: (1, 500, 768)
  - 时间步数: 500
  - 时间分辨率: 20.0ms/帧

HuBERT 特征:
  - 形状: (1, 500, 768)
  - 时间步数: 500
  - 时间分辨率: 20.0ms/帧

分析:
  ✓ 时间步数: 相同 (Wav2vec2: 500, HuBERT: 500)
  ✓ 特征维度: 相同

注意: Wav2vec2 和 HuBERT 使用相同的卷积架构
     下采样率相同 (320x),因此时间步数应该一致
```

---

## 📚 文档索引

### 核心文档
- **QUICKSTART.md** (本文档): 快速开始指南
- **FEATURE_EXTRACTION_README.md**: 特征提取详细文档
- **DATASET_EXTRACTION_README.md**: 数据集提取详细文档
- **DATALOADER_README.md**: DataLoader 使用文档

### 技术文档
- **ALIGNMENT_MECHANISM.md**: 时间对齐机制详解
- **FAQ_BACKBONE_ALIGNMENT.md**: Backbone 更换常见问题

### 示例代码
- **examples/feature_extraction_example.py**: 特征提取示例
- **examples/backbone_comparison.py**: Backbone 对比示例
- **examples/dataloader_examples.py**: DataLoader 示例

### 配置文件
- **config_aligned.json**: BERT + Wav2vec2 + MediaPipe
- **config_roberta_hubert_vit.json**: RoBERTa + HuBERT + ViT
- **config_hubert.json**: HuBERT 音频配置
- **config_vit.json**: ViT 视频配置
- **extraction_config_example.json**: 数据集提取配置示例

---

## 🔧 配置选项速查

### 特征提取器配置

| 配置文件 | 文本 | 音频 | 视频 | 用途 |
|---------|------|------|------|------|
| `config_aligned.json` | BERT | Wav2vec2 | MediaPipe | 默认配置 |
| `config_roberta_hubert_vit.json` | RoBERTa | HuBERT | ViT-16 | 高质量特征 |
| `config_hubert.json` | BERT | HuBERT | MediaPipe | HuBERT 音频 |
| `config_vit.json` | BERT | Wav2vec2 | ViT-16 | ViT 视频 |

### 时间对齐

- **对齐基准**: 音频时间步
- **文本对齐**: 线性插值
- **视频对齐**: 时间戳最近邻匹配
- **总帧数**: 由音频模型决定

### 情感标签

```python
{
    'happy': 0,
    'sad': 1,
    'anger': 2,
    'disgust': 3,
    'surprise': 4,
    'fear': 5,
    'neutral': 6
}
```

---

## 💡 常见问题

### Q1: 如何更换特征提取 backbone?

**A**: 使用配置文件或直接在代码中指定:

```python
config = {
    'text': {'model': 'roberta-base'},
    'audio': {'model': 'hubert'},
    'video': {'model': 'vit', 'feature_mode': 'cls'}
}
extractor = MultimodalFeatureExtractor(config=config)
```

### Q2: 更换 backbone 会影响对齐吗？

**A**: 基本不会。只要音频模型的下采样率相同，总帧数就一致。
- Wav2vec2 和 HuBERT: 下采样率相同 (320x)
- 文本和视频会自动对齐到音频的时间步

详见: `FAQ_BACKBONE_ALIGNMENT.md`

### Q3: 对齐后的总帧数由谁决定？

**A**: 由音频特征的时间步数决定。

计算公式:
```
总帧数 = (音频采样点数) / 下采样率
       = (时长 × 采样率) / 320
```

示例: 10秒音频 = (10 × 16000) / 320 = 500 帧

### Q4: 如何处理 GPU 内存不足？

**A**:
1. 减少视频帧率
2. 使用 CPU 模式
3. 分批处理数据集
4. 使用 Librosa 替代 Wav2vec2

### Q5: 输出格式是什么？

**A**: 与 `emotion_dataloader.py` 兼容的 pickle 格式:

```python
{
    'audio_features': torch.Tensor,  # [num_frames, audio_dim]
    'text_features': torch.Tensor,   # [num_frames, 768]
    'video_features': torch.Tensor,  # [num_frames, video_dim]
    'label': int,                    # 情感标签ID
    'emotion': str,                  # 情感类型
    'sample_id': str,                # 样本ID
    'num_frames': int                # 对齐后的帧数
}
```

---

## 🎓 学习路径

### 初学者
1. 阅读本文档 (QUICKSTART.md)
2. 运行场景 1: 单个文件提取
3. 尝试场景 2: 更换 backbone
4. 阅读 FAQ_BACKBONE_ALIGNMENT.md

### 进阶用户
1. 运行场景 3/4: 批量处理数据集
2. 阅读 DATASET_EXTRACTION_README.md
3. 运行场景 5: 训练模型
4. 阅读 ALIGNMENT_MECHANISM.md

### 高级用户
1. 阅读所有技术文档
2. 运行场景 6: Backbone 对比
3. 自定义配置和模型
4. 贡献代码

---

## 📞 获取帮助

- **文档**: 查看 `docs/` 目录下的详细文档
- **示例**: 运行 `examples/` 目录下的示例代码
- **问题**: 参考 FAQ_BACKBONE_ALIGNMENT.md
- **调试**: 查看 DATASET_EXTRACTION_README.md 的故障排查部分

---

## 🚀 下一步

选择你的使用场景:

- **单个文件**: → `feature_extraction_demo.py`
- **MOSEI 数据集**: → `extract_dataset_features.py` 或 `batch_extract_datasets.py`
- **MELD 数据集**: → `extract_dataset_features.py` 或 `batch_extract_datasets.py`
- **训练模型**: → `emotion_dataloader.py` + 你的训练代码
- **对比实验**: → `examples/backbone_comparison.py`

祝你使用愉快！🎉
