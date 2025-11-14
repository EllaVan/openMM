# 帧数处理机制分析

## 📊 当前实现的帧数处理方式

### 1. 特征提取阶段（`unimodal_features/feature_extractor.py`）

**每个样本的帧数由音频长度决定，不同样本帧数不同。**

```python
def extract_multimodal_features(self, text, audio_path, video_path):
    # 1. 提取音频特征（作为对齐基准）
    audio_features, timestamps = self.extract_audio_features(audio_path)
    num_frames = len(timestamps)  # 每个样本的帧数由音频决定

    # 2. 文本特征对齐到音频帧数
    text_features = self.extract_text_features(text, target_frames=num_frames)

    # 3. 视频特征根据音频时间戳采样
    video_features = self.extract_video_features(video_path, timestamps)

    return {
        'audio_features': audio_features,  # [num_frames, 768]
        'text_features': text_features,    # [num_frames, 768]
        'video_features': video_features,  # [num_frames, 768]
        'num_frames': num_frames           # 帧数（不同样本不同）
    }
```

**结果：**
- ✅ 样本内三模态对齐（同一样本的文本、音频、视频帧数相同）
- ❌ 样本间帧数不同（不同样本的 num_frames 不同）

**示例：**
```
sample_1: num_frames = 150  (3秒音频)
sample_2: num_frames = 200  (4秒音频)
sample_3: num_frames = 100  (2秒音频)
```

### 2. 模型期望的输入形状（`hypergraph_network.py`）

**模型要求批次内所有样本的帧数相同。**

```python
def forward(
    self,
    text_features: torch.Tensor,    # [batch_size, T, 768]
    audio_features: torch.Tensor,   # [batch_size, T, 768]
    video_features: torch.Tensor,   # [batch_size, T, 768]
    labels: Optional[torch.Tensor] = None
):
    # LSTM 处理固定长度序列
    text_encoded = self.text_encoder(text_features)    # 期望 T 固定
    audio_encoded = self.audio_encoder(audio_features)  # 期望 T 固定
    video_encoded = self.video_encoder(video_features)  # 期望 T 固定

    # 超图构建（期望所有样本节点数相同）
    nodes = torch.cat([text_encoded, audio_encoded, video_encoded], dim=1)
    # [batch_size, 3T, hidden_dim]
```

**要求：**
- ✅ 批次内所有样本必须有相同的 T
- ❌ 使用 `torch.stack()` 组成批次，无法处理变长序列

### 3. DataLoader 的批处理（`emotion_dataloader.py`）

**默认 collate_fn 使用 torch.stack，无法处理变长序列。**

```python
# 默认行为（会报错）
def default_collate(batch):
    audio_features = torch.stack([item['audio_features'] for item in batch])
    # 如果 item1: [150, 768], item2: [200, 768]
    # torch.stack() 会报错：张量形状不匹配
```

## ⚠️ 当前实现的问题

**存在矛盾：**

1. **特征提取** → 每个样本帧数不同 `[num_frames, 768]`
2. **模型输入** → 要求批次内帧数相同 `[batch_size, T, 768]`
3. **DataLoader** → 使用 `torch.stack()` 无法处理变长序列

**如果直接使用当前代码训练，会出现以下错误：**

```
RuntimeError: stack expects each tensor to be equal size, but got [150, 768] at entry 0 and [200, 768] at entry 1
```

## ✅ 解决方案

### 方案 1: Padding + Masking（推荐）

**为批次内的样本填充到相同长度，并使用 mask 标记有效帧。**

#### 1.1 修改 collate_fn

```python
def padded_collate_fn(batch):
    """支持变长序列的 collate 函数"""
    # 找到最大帧数
    max_frames = max(item['num_frames'] for item in batch)

    batch_size = len(batch)
    audio_dim = batch[0]['audio_features'].shape[1]
    text_dim = batch[0]['text_features'].shape[1]
    video_dim = batch[0]['video_features'].shape[1]

    # 初始化填充后的张量
    audio_padded = torch.zeros(batch_size, max_frames, audio_dim)
    text_padded = torch.zeros(batch_size, max_frames, text_dim)
    video_padded = torch.zeros(batch_size, max_frames, video_dim)
    masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    labels = torch.tensor([item['label'] for item in batch])

    # 填充数据
    for i, item in enumerate(batch):
        num_frames = item['num_frames']
        audio_padded[i, :num_frames] = item['audio_features']
        text_padded[i, :num_frames] = item['text_features']
        video_padded[i, :num_frames] = item['video_features']
        masks[i, :num_frames] = True  # 标记有效帧

    return {
        'audio_features': audio_padded,   # [batch, max_frames, 768]
        'text_features': text_padded,     # [batch, max_frames, 768]
        'video_features': video_padded,   # [batch, max_frames, 768]
        'masks': masks,                   # [batch, max_frames]
        'label': labels
    }
```

#### 1.2 修改模型使用 mask

```python
class UnimodalEncoder(nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] - True 表示有效位置
        """
        # LSTM 支持 pack_padded_sequence
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()  # 每个样本的有效长度
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        out = self.fc(lstm_out)
        return out
```

#### 1.3 超图构建时应用 mask

```python
class MultimodalHypergraphLayer(nn.Module):
    def forward(self, text_features, audio_features, video_features, mask=None):
        # 投影到统一维度
        text_proj = self.text_proj(text_features)
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)

        # 拼接节点
        nodes = torch.cat([text_proj, audio_proj, video_proj], dim=1)
        # [batch, 3T, hidden_dim]

        # 如果有 mask，扩展到三个模态
        if mask is not None:
            mask_3x = torch.cat([mask, mask, mask], dim=1)  # [batch, 3T]
            # 在超图构建时忽略填充的节点
            nodes = nodes * mask_3x.unsqueeze(-1)

        # 构建超图
        H = self.hypergraph_init(nodes)
        # 超图卷积...
```

**优点：**
- ✅ 支持变长序列
- ✅ 不损失信息
- ✅ GPU 友好（批处理）

**缺点：**
- ❌ 增加计算和内存开销（填充部分）
- ❌ 需要修改模型代码

---

### 方案 2: 固定帧数（最简单）

**在特征提取时，将所有样本统一调整到固定帧数。**

#### 2.1 修改特征提取器

```python
class MultimodalFeatureExtractor:
    def __init__(self, config: Dict, fixed_frames: int = 200):
        """
        Args:
            fixed_frames: 固定的帧数（默认 200）
        """
        self.fixed_frames = fixed_frames
        # ...

    def extract_multimodal_features(self, text, audio_path, video_path):
        # 1. 提取音频特征
        audio_features, timestamps = self.extract_audio_features(audio_path)

        # 2. 调整到固定帧数
        audio_features = self._resize_to_fixed_frames(audio_features)

        # 3. 文本和视频也调整到固定帧数
        text_features = self.extract_text_features(text, target_frames=self.fixed_frames)

        # 重新生成固定帧数的时间戳
        duration = len(timestamps) / len(audio_features) * self.fixed_frames
        fixed_timestamps = np.linspace(0, timestamps[-1], self.fixed_frames)
        video_features = self.extract_video_features(video_path, fixed_timestamps)

        return {
            'audio_features': audio_features,   # [fixed_frames, 768]
            'text_features': text_features,     # [fixed_frames, 768]
            'video_features': video_features,   # [fixed_frames, 768]
            'num_frames': self.fixed_frames     # 固定值
        }

    def _resize_to_fixed_frames(self, features: torch.Tensor) -> torch.Tensor:
        """
        将特征调整到固定帧数（插值或截断）

        Args:
            features: [current_frames, dim]

        Returns:
            resized: [fixed_frames, dim]
        """
        current_frames = features.shape[0]

        if current_frames == self.fixed_frames:
            return features

        # 使用线性插值
        features = features.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            features.permute(0, 3, 1, 2),
            size=(1, self.fixed_frames),
            mode='bilinear',
            align_corners=False
        )
        resized = resized.permute(0, 2, 3, 1).squeeze(0).squeeze(0)

        return resized
```

**优点：**
- ✅ 最简单，无需修改模型
- ✅ 内存效率高（无填充）
- ✅ 可以直接使用现有代码

**缺点：**
- ❌ 可能损失信息（短序列拉长，长序列压缩）
- ❌ 需要手动调整 fixed_frames 参数

**如何选择 fixed_frames：**
```python
# 统计数据集帧数分布
import pickle
import numpy as np

frames_list = []
for file in pkl_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        frames_list.extend([item['num_frames'] for item in data])

print(f"平均帧数: {np.mean(frames_list):.1f}")
print(f"中位数: {np.median(frames_list):.1f}")
print(f"最大值: {np.max(frames_list)}")
print(f"最小值: {np.min(frames_list)}")
print(f"95分位: {np.percentile(frames_list, 95):.1f}")

# 建议使用 95 分位或平均值作为 fixed_frames
```

---

### 方案 3: 动态批处理（高级）

**将相似长度的样本组成一个批次。**

#### 3.1 按长度排序并分组

```python
from torch.utils.data import Sampler

class LengthGroupedSampler(Sampler):
    """按序列长度分组的采样器"""

    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 按长度排序索引
        lengths = [item['num_frames'] for item in dataset.data]
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self):
        # 分组成批次
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i:i+self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # 打乱批次顺序
        import random
        random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size * self.batch_size
        return len(self.dataset)

# 使用
from emotion_dataloader import EmotionDataset
from torch.utils.data import DataLoader

dataset = EmotionDataset(data)
sampler = LengthGroupedSampler(dataset, batch_size=32)

dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=padded_collate_fn  # 仍需要 padding
)
```

**优点：**
- ✅ 减少填充开销（同批次长度相近）
- ✅ 训练效率高

**缺点：**
- ❌ 实现复杂
- ❌ 仍需要 padding 和 mask

---

## 🎯 推荐方案

### 对于快速实验：**方案 2（固定帧数）**

- 最简单，立即可用
- 在 `config.json` 中添加 `"fixed_frames": 200`
- 修改 `feature_extractor.py` 中的 `extract_multimodal_features`

### 对于正式研究：**方案 1（Padding + Masking）**

- 不损失信息
- 更符合学术规范
- 需要修改 `collate_fn` 和模型代码

## 📝 总结

| 方面 | 当前实现 | 方案 1 (Padding) | 方案 2 (Fixed) | 方案 3 (Grouped) |
|------|---------|-----------------|----------------|------------------|
| 支持变长 | ❌ | ✅ | ❌ | ✅ |
| 实现难度 | - | 中 | 低 | 高 |
| 信息损失 | - | 无 | 有 | 无 |
| 内存效率 | - | 中 | 高 | 高 |
| 训练速度 | - | 中 | 快 | 快 |
| 推荐度 | - | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

**下一步操作：**

1. **确认需求**：是否需要保留原始时序信息？
2. **选择方案**：快速实验用方案 2，正式研究用方案 1
3. **实现修改**：我可以帮您实现任何一个方案
