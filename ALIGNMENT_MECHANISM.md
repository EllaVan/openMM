# 时间步对齐机制详解

## 当前对齐机制分析

### 1. 对齐流程概述

```
┌─────────────────────────────────────────────────────────┐
│                    对齐流程                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  文本特征 [N_text, 768]                                  │
│       │                                                  │
│       │  线性插值                                        │
│       ↓                                                  │
│  对齐文本 [N_audio, 768] ←─────────────┐                │
│                                        │                │
│  音频特征 [N_audio, D_audio] ─────────┼→ 基准时间步     │
│       │                                │                │
│       ├→ 时间戳: [t0, t1, ..., tN]    │                │
│                                        │                │
│  视频特征 [N_video, D_video]           │                │
│       │                                │                │
│       │  时间戳匹配                     │                │
│       ↓                                │                │
│  对齐视频 [N_audio, D_video] ──────────┘                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2. 对齐后的总帧数由谁决定？

**答案：音频特征的时间步数**

```python
# 从 feature_extraction_demo.py line 330-336
reference_timestamps = audio_features['timestamps']
num_frames = len(reference_timestamps)  # ← 这里决定总帧数
```

**原因**：
- 音频信号是连续的，时间戳最精确
- 音频采样率固定（如 16kHz），时间对应关系明确
- Wav2vec2 输出的时间步有明确的物理意义

**计算示例**：
```python
# Wav2vec2 示例
音频时长: 10 秒
采样率: 16000 Hz
总采样点: 10 * 16000 = 160,000

Wav2vec2 输出:
- 卷积下采样率约为 320 (facebook/wav2vec2-base-960h)
- 时间步数 = 160,000 / 320 ≈ 500 帧
- 每帧对应时间 = 10s / 500 = 0.02s (20ms)

最终对齐后总帧数 = 500
```

### 3. 当前对齐方法详解

#### 3.1 文本特征对齐 (线性插值)

```python
# 从 feature_extraction_demo.py line 370-388
def _interpolate_features(self, features: torch.Tensor, target_length: int):
    """
    输入: features [N_text, 768]
    输出: aligned_features [N_audio, 768]

    方法: PyTorch 的 F.interpolate
    """
    # 转置: [768, N_text]
    features_t = features.T.unsqueeze(0)  # [1, 768, N_text]

    # 线性插值到目标长度
    interpolated = torch.nn.functional.interpolate(
        features_t,
        size=target_length,  # N_audio
        mode='linear',
        align_corners=False
    )

    # 转回: [N_audio, 768]
    return interpolated.squeeze(0).T
```

**示例**：
```
文本: "I am happy" → BERT → [10, 768]  (10个token)
音频: 500 帧

插值过程:
- 将 [10, 768] 插值到 [500, 768]
- 每个原始token平均对应 50 个音频帧
- token 0 的特征被平滑地分配到帧 0-50
- token 1 的特征被平滑地分配到帧 50-100
- ...
```

#### 3.2 视频特征对齐 (时间戳匹配)

```python
# 从 feature_extraction_demo.py line 390-415
def _align_by_timestamps(self, features, source_timestamps, target_timestamps):
    """
    输入:
      - features: [N_video, D_video]
      - source_timestamps: 视频帧时间戳 [N_video]
      - target_timestamps: 音频帧时间戳 [N_audio]

    输出: aligned_features [N_audio, D_video]

    方法: 最近邻匹配
    """
    aligned_features = []

    for t in target_timestamps:
        # 找到最接近的视频帧
        idx = np.argmin(np.abs(source_timestamps - t))
        aligned_features.append(features[idx])

    return torch.stack(aligned_features)
```

**示例**：
```
视频: 25 fps, 10秒 → 250 帧
视频时间戳: [0.00, 0.04, 0.08, ..., 9.96]s

音频: 500 帧
音频时间戳: [0.00, 0.02, 0.04, ..., 9.98]s

对齐过程:
- 音频帧 t=0.00s → 找最近的视频帧 t=0.00s (idx=0)
- 音频帧 t=0.02s → 找最近的视频帧 t=0.04s (idx=1)
- 音频帧 t=0.04s → 找最近的视频帧 t=0.04s (idx=1)
- ...

结果: [500, D_video]，部分视频帧会被重复使用
```

### 4. 更换 Backbone 的影响

#### 4.1 文本 Backbone (BERT → RoBERTa)

**特征维度变化**：
| Backbone | 输出维度 | 时间步数 | 影响 |
|----------|----------|----------|------|
| BERT-base | [N_tokens, 768] | N_tokens | 基准 |
| RoBERTa-base | [N_tokens, 768] | N_tokens | ✅ 无影响 |
| RoBERTa-large | [N_tokens, 1024] | N_tokens | ⚠️ 特征维度变化 |

**对齐影响**：
- ✅ **无影响**：插值只关心序列长度，不关心特征维度
- ✅ **token 数量可能不同**：RoBERTa 的 tokenizer 可能产生不同数量的 token，但插值会自动处理

**代码无需修改**，插值函数自动适应特征维度。

#### 4.2 音频 Backbone (Wav2vec2 → HuBERT)

**特征维度变化**：
| Backbone | 输出维度 | 时间步数 | 下采样率 | 影响 |
|----------|----------|----------|----------|------|
| Wav2vec2-base | [T, 768] | T ≈ L/320 | 320x | 基准 |
| HuBERT-base | [T, 768] | T ≈ L/320 | 320x | ✅ 无影响 |
| HuBERT-large | [T, 1024] | T ≈ L/320 | 320x | ⚠️ 特征维度变化 |

**对齐影响**：
- ✅ **基本无影响**：HuBERT 和 Wav2vec2 架构相似，下采样率相同
- ⚠️ **时间步数一致性**：只要下采样率相同，总帧数就一致
- ⚠️ **不同模型可能有微小差异**：需要验证具体模型的卷积配置

**关键**：总帧数由音频时间步数决定，HuBERT 的时间步数计算方式与 Wav2vec2 相同。

#### 4.3 视频 Backbone (MediaPipe → ViT-16)

**特征维度变化**：
| Backbone | 输出维度 | 时间步数 | 特征类型 | 影响 |
|----------|----------|----------|----------|------|
| MediaPipe | [N_frames, 1404] | N_frames = fps × duration | 关键点坐标 | 基准 |
| ViT-16 | [N_frames, 768] | N_frames = fps × duration | 图像特征 | ⚠️ 特征类型变化 |
| ViT-16 (patch) | [N_frames, N_patches, 768] | N_patches = (H/16)×(W/16) | patch 特征 | ⚠️⚠️ 结构变化 |

**对齐影响**：
- ✅ **无影响（frame-level）**：如果每帧提取一个全局特征 [CLS] token
- ⚠️⚠️ **需要修改（patch-level）**：如果保留所有 patch 特征，需要额外的空间维度处理

**推荐方案**：
1. **使用 [CLS] token**：提取每帧的全局特征 `[N_frames, 768]`
2. **平均池化**：对所有 patch 进行平均 `[N_frames, 768]`
3. **保留 patch**：修改对齐逻辑，支持 `[N_frames, N_patches, 768]` 结构

### 5. 对齐质量评估

#### 5.1 文本-音频对齐质量

**问题**：线性插值假设文本特征均匀分布在时间上，但实际说话速度不均匀。

**改进方案**：
- 使用强制对齐工具（如 Montreal Forced Aligner）
- 使用 CTC 对齐（Wav2vec2 CTC）
- 保留原始 token 时间戳信息

**当前方案的局限性**：
```
文本: "Hello    [long pause]    world"
当前: token均匀分布 → 不准确
理想: 根据实际说话时间对齐
```

#### 5.2 视频-音频对齐质量

**问题**：最近邻匹配可能导致视频帧重复或跳过。

**示例**：
```
视频 25fps：帧间隔 0.04s
音频 500帧/10s：帧间隔 0.02s

→ 音频帧数是视频帧数的 2倍
→ 每个视频帧平均被使用 2次
```

**改进方案**：
- 线性插值视频特征（对于连续特征如 ViT）
- 保持原始帧率，调整音频对齐基准
- 使用更高的视频帧率

### 6. 总结

| 问题 | 答案 |
|------|------|
| **总帧数由谁决定？** | 音频特征的时间步数 |
| **能否替换 backbone？** | ✅ 可以，但需要注意特征维度和结构 |
| **替换是否影响对齐？** | 文本/音频：基本无影响<br>视频：需要处理 patch 维度 |
| **当前对齐方法** | 文本：线性插值<br>视频：时间戳最近邻匹配 |
| **对齐质量** | 中等，适合粗粒度任务<br>精细任务需要强制对齐 |

### 7. 推荐配置

#### 7.1 高质量对齐配置

```python
{
    "text": {
        "model": "roberta-base",  # 或 bert-base
        "enabled": true
    },
    "audio": {
        "model": "hubert",  # 或 wav2vec2
        "sample_rate": 16000,
        "enabled": true,
        "alignment_method": "ctc"  # 使用 CTC 对齐
    },
    "video": {
        "model": "vit",  # 使用 ViT-16
        "fps": 25,
        "enabled": true,
        "feature_type": "cls"  # 使用 [CLS] token
    },
    "alignment": {
        "reference": "audio",  # 音频作为基准
        "method": "timestamp_matching",
        "interpolation_mode": "linear"
    }
}
```

#### 7.2 计算高效配置

```python
{
    "text": {
        "model": "bert-base-uncased",
        "enabled": true
    },
    "audio": {
        "model": "librosa",  # 更快
        "enabled": true
    },
    "video": {
        "model": "mediapipe",  # 更快
        "enabled": true
    },
    "alignment": {
        "reference": "audio",
        "method": "interpolation"
    }
}
```

### 8. 下一步改进

1. **添加 CTC 对齐**：使用 Wav2vec2 的 CTC 输出进行词级对齐
2. **支持多参考基准**：允许使用视频或文本作为对齐基准
3. **Patch-level 特征支持**：支持 ViT 的 patch 特征
4. **对齐质量评估**：提供对齐质量的度量指标
5. **灵活的帧率配置**：支持自定义目标帧率

---

## 附录：各模型的时间步计算

### Wav2vec2
```python
# facebook/wav2vec2-base-960h
卷积层配置:
- Conv1: kernel=10, stride=5  → 下采样 5x
- Conv2: kernel=3,  stride=2  → 下采样 2x
- Conv3: kernel=3,  stride=2  → 下采样 2x
- Conv4: kernel=3,  stride=2  → 下采样 2x
- Conv5: kernel=3,  stride=2  → 下采样 2x
- Conv6: kernel=2,  stride=2  → 下采样 2x
- Conv7: kernel=2,  stride=2  → 下采样 2x

总下采样: 5 × 2^6 = 320

时间步数 = (采样点数) / 320
```

### HuBERT
```python
# facebook/hubert-base-ls960
与 Wav2vec2 相同的架构
总下采样: 320

时间步数 = (采样点数) / 320
```

### ViT-16
```python
# google/vit-base-patch16-224
输入图像: 224×224
Patch 大小: 16×16

Patch 数量: (224/16) × (224/16) = 14 × 14 = 196 patches
输出: [CLS] + 196 patches = 197 tokens

每帧特征:
- [CLS]: [768]  ← 全局特征
- Patches: [196, 768]  ← 空间特征
```
