# FAQ: Backbone 更换与时间对齐

## 你的问题及答案

### Q1: 能否使用 RoBERTa/HuBERT 和 ViT-16 代替原本代码中的特征提取 backbone?

**答：✅ 可以！现在已经全面支持。**

#### 使用方法

**方法 1: 使用配置文件**

```bash
# 使用 RoBERTa + HuBERT + ViT-16 组合
python feature_extraction_demo.py --config config_roberta_hubert_vit.json
```

**方法 2: 在代码中指定**

```python
from feature_extraction_demo import MultimodalFeatureExtractor

config = {
    'text': {
        'model': 'roberta-base',  # 或 'roberta-large'
        'enabled': True
    },
    'audio': {
        'model': 'hubert',
        'model_name': 'facebook/hubert-base-ls960',  # 或 hubert-large-ll60k
        'sample_rate': 16000,
        'enabled': True
    },
    'video': {
        'model': 'vit',
        'model_name': 'google/vit-base-patch16-224',  # 或其他 ViT 变体
        'feature_mode': 'cls',  # 'cls', 'pooled', 或 'mean'
        'enabled': True
    }
}

extractor = MultimodalFeatureExtractor(config=config)
```

#### 支持的模型

| 模态 | 原始 | 新增支持 | 配置参数 |
|------|------|----------|----------|
| 文本 | BERT | RoBERTa, DistilBERT, ALBERT, 等 | `text.model` |
| 音频 | Wav2vec2 | HuBERT | `audio.model` |
| 视频 | MediaPipe | ViT-16, ViT-32 等 | `video.model` |

---

### Q2: 现在的代码是如何对齐时间步的？

**答：使用音频时间戳作为基准，通过插值和匹配对齐其他模态。**

#### 对齐流程

```
┌─────────────────────────────────────────┐
│  1. 音频特征提取 (基准)                  │
│     - Wav2vec2/HuBERT 提取特征           │
│     - 计算时间戳: [0.00s, 0.02s, ...]   │
│     - 确定总帧数: N_audio               │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  2. 文本特征对齐 (线性插值)              │
│     输入: [N_text, 768]                 │
│     输出: [N_audio, 768]                │
│     方法: torch.nn.functional.interpolate│
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  3. 视频特征对齐 (时间戳匹配)            │
│     输入: [N_video, D_video]            │
│     输出: [N_audio, D_video]            │
│     方法: 最近邻时间戳匹配               │
└─────────────────────────────────────────┘
```

#### 代码位置

详见 `feature_extraction_demo.py`:
- **对齐函数**: `align_features()` (line 313-367)
- **文本插值**: `_interpolate_features()` (line 370-388)
- **视频匹配**: `_align_by_timestamps()` (line 390-415)

#### 具体算法

**文本对齐 (线性插值)**:
```python
# 输入: [10, 768] (10个token)
# 目标: [500, 768] (500个音频帧)

features_t = features.T.unsqueeze(0)  # [1, 768, 10]
interpolated = F.interpolate(
    features_t,
    size=500,  # 目标长度 = 音频帧数
    mode='linear'
)
result = interpolated.squeeze(0).T  # [500, 768]
```

**视频对齐 (时间戳匹配)**:
```python
# 音频时间戳: [0.00, 0.02, 0.04, ...]
# 视频时间戳: [0.00, 0.04, 0.08, ...]

for audio_t in audio_timestamps:
    # 找最接近的视频帧
    idx = argmin(|video_timestamps - audio_t|)
    aligned_video.append(video_features[idx])
```

---

### Q3: 更改提取 backbone 是否会影响对齐？

**答：基本不会影响，只要满足以下条件。**

#### 影响分析

| Backbone 变化 | 是否影响对齐 | 原因 | 注意事项 |
|--------------|-------------|------|---------|
| BERT → RoBERTa | ❌ 不影响 | 插值只关心序列长度 | token 数量可能不同，但会自动处理 |
| Wav2vec2 → HuBERT | ❌ 不影响 | 两者下采样率相同 (320x) | 总帧数保持一致 |
| MediaPipe → ViT | ❌ 不影响 | 时间戳匹配与特征维度无关 | 使用 `feature_mode='cls'` |

#### 详细分析

**1. 文本 Backbone 变化 (BERT → RoBERTa)**

```python
# BERT
输入文本: "I am happy"
Token 数: 10
特征: [10, 768]
↓ 插值到 500 帧
对齐后: [500, 768]

# RoBERTa (使用不同的 tokenizer)
输入文本: "I am happy"
Token 数: 12  # 可能不同！
特征: [12, 768]
↓ 插值到 500 帧
对齐后: [500, 768]  # 最终维度相同
```

✅ **结论**: 插值会自动处理不同的 token 数量

**2. 音频 Backbone 变化 (Wav2vec2 → HuBERT)**

```python
# Wav2vec2
音频: 10s @ 16kHz = 160,000 采样点
下采样: 320x
时间步数: 160,000 / 320 = 500 帧
特征: [500, 768]

# HuBERT (相同架构)
音频: 10s @ 16kHz = 160,000 采样点
下采样: 320x  # 与 Wav2vec2 相同
时间步数: 160,000 / 320 = 500 帧  # 相同！
特征: [500, 768]
```

✅ **结论**: 时间步数完全相同，对齐一致

**3. 视频 Backbone 变化 (MediaPipe → ViT)**

```python
# MediaPipe
视频: 10s @ 25fps = 250 帧
特征: [250, 1404]  # 468个关键点 × 3
时间戳: [0.00, 0.04, 0.08, ...]
↓ 时间戳匹配到音频 (500帧)
对齐后: [500, 1404]

# ViT-16
视频: 10s @ 25fps = 250 帧
特征: [250, 768]  # [CLS] token
时间戳: [0.00, 0.04, 0.08, ...]  # 相同！
↓ 时间戳匹配到音频 (500帧)
对齐后: [500, 768]
```

✅ **结论**: 时间戳匹配与特征维度无关

---

### Q4: 对齐后的总帧数是由谁决定的？

**答：由音频特征的时间步数决定。**

#### 决定因素

```python
# 从 feature_extraction_demo.py line 330-336
def align_features(self, text_features, audio_features, video_features):
    # 使用音频时间戳作为基准
    reference_timestamps = audio_features['timestamps']
    num_frames = len(reference_timestamps)  # ← 这里决定总帧数！

    # 其他模态都对齐到这个帧数
    aligned_text = interpolate(text, target_length=num_frames)
    aligned_video = match_timestamps(video, reference_timestamps)
```

#### 为什么选择音频？

1. **时间精度最高**: 音频采样率固定 (16kHz)，时间对应关系明确
2. **物理意义清晰**: 每帧对应固定时长 (约 20ms)
3. **连续性好**: 音频信号连续，不像视频可能有帧丢失

#### 计算公式

```python
# 音频采样点数
N_samples = duration × sample_rate
# 例: 10s × 16000 = 160,000

# Wav2vec2/HuBERT 下采样率
downsample_rate = 320

# 总帧数 (对齐后的时间步数)
num_frames = N_samples / downsample_rate
# 例: 160,000 / 320 = 500 帧

# 每帧对应时长
frame_duration = duration / num_frames
# 例: 10s / 500 = 0.02s = 20ms
```

#### 实际示例

| 音频时长 | 采样率 | 下采样率 | 总帧数 | 帧时长 |
|---------|--------|---------|--------|--------|
| 5s | 16000 Hz | 320x | 250 | 20ms |
| 10s | 16000 Hz | 320x | 500 | 20ms |
| 30s | 16000 Hz | 320x | 1500 | 20ms |

---

## 实战示例

### 示例 1: 使用 RoBERTa + HuBERT + ViT

```python
from feature_extraction_demo import MultimodalFeatureExtractor

# 加载配置
config = {
    'text': {'model': 'roberta-base', 'enabled': True},
    'audio': {'model': 'hubert', 'model_name': 'facebook/hubert-base-ls960',
              'sample_rate': 16000, 'enabled': True},
    'video': {'model': 'vit', 'model_name': 'google/vit-base-patch16-224',
              'feature_mode': 'cls', 'enabled': True},
    'alignment': {'enabled': True, 'reference': 'audio'}
}

extractor = MultimodalFeatureExtractor(config=config)

# 提取特征
features = extractor.extract_from_files(
    text_file="data/sample.txt",
    audio_file="data/sample.wav",
    video_file="data/sample.mp4"
)

# 查看对齐结果
print(f"总帧数: {features['num_frames']}")  # 由 HuBERT 决定
print(f"音频: {features['audio'].shape}")   # [num_frames, 768]
print(f"文本: {features['text'].shape}")    # [num_frames, 768]
print(f"视频: {features['video'].shape}")   # [num_frames, 768]
```

### 示例 2: 对比不同 Backbone

```bash
# 运行对比脚本
cd examples
python backbone_comparison.py

# 选择对比项目:
# 1. BERT vs RoBERTa
# 2. Wav2vec2 vs HuBERT
# 3. MediaPipe vs ViT-16
# 4. 完整对齐测试
```

---

## 关键总结

### ✅ 可以做的

1. **自由更换 Backbone**
   - 文本: BERT, RoBERTa, DistilBERT 等
   - 音频: Wav2vec2, HuBERT
   - 视频: MediaPipe, ViT-16

2. **对齐自动处理**
   - Token 数量不同 → 自动插值
   - 特征维度不同 → 不影响对齐
   - 视频帧率不同 → 时间戳匹配

3. **灵活配置**
   - JSON 配置文件
   - Python 代码配置
   - 命令行参数

### ⚠️ 注意事项

1. **音频模型选择**
   - 必须使用相同下采样率的模型
   - Wav2vec2 和 HuBERT: ✅ 下采样率相同 (320x)
   - 其他音频模型: 需验证下采样率

2. **视频 ViT 配置**
   - 推荐使用 `feature_mode='cls'`
   - 避免使用 patch-level 特征（需额外处理）

3. **计算资源**
   - ViT 提取速度比 MediaPipe 慢
   - HuBERT 和 Wav2vec2 资源消耗相近
   - RoBERTa 比 BERT 略慢

---

## 参考文档

- **对齐机制详解**: `ALIGNMENT_MECHANISM.md`
- **配置示例**: `config_roberta_hubert_vit.json`
- **对比脚本**: `examples/backbone_comparison.py`
- **主要代码**: `feature_extraction_demo.py`

## 相关论文

- **HuBERT**: [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
- **ViT**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **RoBERTa**: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
