# Hypergraph Fusion Network - 支持 Padding + Masking

多模态超图融合网络，用于情感识别任务。支持变长序列的 Padding + Masking 策略。

## 📁 目录结构

```
hyper_fusion/
├── __init__.py          # 模块初始化
├── dataloader.py        # 支持 Padding + Masking 的 DataLoader
├── modules.py           # 超图模块（支持 mask）
├── network.py           # 完整网络（支持 mask）
├── train.py             # 训练脚本
├── config.json          # 配置示例
└── README.md            # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio
```

### 2. 准备数据

确保已经使用 `unimodal_features` 提取了特征：

```bash
# 特征提取
python unimodal_features/batch_extract.py --dataset mosei

# 输出目录: ./output/mosei_features/
```

### 3. 训练模型

```bash
python hyper_fusion/train.py \
  --data_dir ./output/mosei_features \
  --dataset MOSEI \
  --emotion happy \
  --label_id 0 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --num_hyperedges 64 \
  --num_conv_layers 2 \
  --use_contrastive \
  --use_bottleneck \
  --save_dir ./checkpoints
```

### 4. 查看结果

```
训练日志示例:
================================================================
Epoch 1/50
Train - Loss: 0.6234, Cls Loss: 0.6123, Contrastive Loss: 0.0111, Acc: 65.23%
Test  - Loss: 0.5987, Cls Loss: 0.5876, Acc: 68.45%
✓ 保存最佳模型 (Acc: 68.45%)
================================================================
```

## 📊 核心功能

### 1. Padding + Masking DataLoader

**支持变长序列的批处理**

```python
from hyper_fusion import create_dataloaders

dataloaders = create_dataloaders(
    data_dir='./output/mosei_features',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

# 获取一个批次
batch = next(iter(dataloaders['train']))
print(batch.keys())
# dict_keys(['audio_features', 'text_features', 'video_features',
#            'masks', 'labels', 'num_frames'])

print(batch['audio_features'].shape)  # [32, max_frames, 768]
print(batch['masks'].shape)           # [32, max_frames]
print(batch['num_frames'])            # [30, 40, 50, 25, ...]
```

**Mask 说明：**
- `True`: 有效帧
- `False`: 填充帧

批次内自动填充到最大帧数，mask 标记有效位置。

### 2. 支持 Mask 的超图模块

**所有模块都考虑 mask，忽略填充帧：**

```python
from hyper_fusion import MultimodalHypergraphLayer

hypergraph = MultimodalHypergraphLayer(
    text_dim=768,
    audio_dim=768,
    video_dim=768,
    hidden_dim=256,
    num_hyperedges=64,
    num_conv_layers=2
)

# 前向传播（自动处理 mask）
output = hypergraph(
    text_features,
    audio_features,
    video_features,
    mask=masks  # 传入 mask
)
```

**关键特性：**
- 超图初始化时，填充节点不参与连接矩阵计算
- 超图卷积时，填充节点特征始终为 0
- 池化时，只对有效帧进行平均（Masked Pooling）

### 3. 完整网络

```python
from hyper_fusion import HypergraphEmotionClassifier

model = HypergraphEmotionClassifier(
    feature_dims={'text': 768, 'audio': 768, 'video': 768},
    num_classes=2,
    config={
        'num_hyperedges': 64,
        'num_conv_layers': 2,
        'use_contrastive': True,
        'use_bottleneck': True
    }
)

# 训练
output = model(batch)
loss = output['loss']
loss.backward()

# 预测
predictions = model.predict(batch)
```

## 🏗️ 网络架构

```
输入:
  - text_features:  [batch, T, 768]
  - audio_features: [batch, T, 768]
  - video_features: [batch, T, 768]
  - masks:          [batch, T]

↓ 单模态编码器 (Bi-LSTM + Mask)
  - text_encoded:   [batch, T, 256]
  - audio_encoded:  [batch, T, 256]
  - video_encoded:  [batch, T, 256]

↓ 多模态超图层 (Hypergraph Fusion + Mask)
  1. 拼接节点: [batch, 3T, 256]
  2. 超图初始化 (基于相关性)
  3. 超图增强 (随机删除超边)
  4. 超图卷积 x2 (两阶段传播)

↓ Bottleneck 层 (可选)
  - 降维 → 升维

↓ Masked Pooling (只对有效帧平均)
  - text_pooled:  [batch, 256]
  - audio_pooled: [batch, 256]
  - video_pooled: [batch, 256]

↓ 拼接 + 分类器
  - multimodal:   [batch, 768]
  - logits:       [batch, 2]

损失函数:
  loss = cls_loss + λ × contrastive_loss + μ × l2_reg
```

## 📈 训练参数

### 数据参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据目录 | - |
| `--dataset` | MOSEI 或 MELD | - |
| `--emotion` | 情感类型 | - |
| `--label_id` | 标签 ID | - |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--encoder_hidden_dim` | 编码器隐藏层维度 | 256 |
| `--encoder_output_dim` | 编码器输出维度 | 256 |
| `--hypergraph_hidden_dim` | 超图隐藏层维度 | 256 |
| `--num_hyperedges` | 超边数量 | 64 |
| `--num_conv_layers` | 超图卷积层数 | 2 |
| `--bottleneck_dim` | Bottleneck 维度 | 128 |
| `--dropout` | Dropout 率 | 0.1 |
| `--hyperedge_drop_rate` | 超边删除率 | 0.2 |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 批次大小 | 32 |
| `--epochs` | 训练轮数 | 50 |
| `--lr` | 学习率 | 1e-4 |
| `--weight_decay` | 权重衰减 | 1e-4 |
| `--use_contrastive` | 使用对比学习 | True |
| `--contrastive_weight` | 对比学习权重 | 0.1 |
| `--use_bottleneck` | 使用 Bottleneck | True |

## 🔍 Padding + Masking 工作原理

### 问题

提取的特征具有不同的帧数：
- sample_1: 150 帧
- sample_2: 200 帧
- sample_3: 100 帧

无法直接组成批次（`torch.stack` 要求相同形状）。

### 解决方案

**1. Padding: 填充到批次内最大长度**

```python
max_frames = 200  # 批次内最大值

# 填充到 max_frames
audio_padded[0, :150] = sample_1_audio  # 150 帧
audio_padded[0, 150:] = 0               # 填充 50 帧

audio_padded[1, :200] = sample_2_audio  # 200 帧
audio_padded[2, :100] = sample_3_audio  # 100 帧
audio_padded[2, 100:] = 0               # 填充 100 帧
```

**2. Masking: 标记有效帧**

```python
masks[0, :150] = True   # sample_1 有效帧
masks[0, 150:] = False  # 填充位置

masks[1, :200] = True   # sample_2 有效帧
masks[2, :100] = True   # sample_3 有效帧
masks[2, 100:] = False  # 填充位置
```

**3. 模型处理**

```python
# 超图初始化: 填充节点不参与计算
H_hat = H_hat.masked_fill(~mask_expanded, float('-inf'))

# 超图卷积: 填充节点特征置零
nodes = nodes * mask.unsqueeze(-1).float()

# Masked Pooling: 只对有效帧求平均
valid_counts = masks.sum(dim=1)
pooled = (nodes * mask_expanded).sum(dim=1) / valid_counts
```

## 💡 使用示例

### 示例 1: 基础训练

```bash
python hyper_fusion/train.py \
  --data_dir ./output/mosei_features \
  --dataset MOSEI \
  --emotion happy \
  --label_id 0 \
  --batch_size 32 \
  --epochs 50
```

### 示例 2: 自定义配置

```bash
python hyper_fusion/train.py \
  --data_dir ./output/meld_features \
  --dataset MELD \
  --emotion sad \
  --label_id 1 \
  --batch_size 16 \
  --num_hyperedges 128 \
  --num_conv_layers 3 \
  --dropout 0.2 \
  --lr 5e-5
```

### 示例 3: Python API

```python
from hyper_fusion import create_dataloaders, HypergraphEmotionClassifier
import torch

# 创建 DataLoader
dataloaders = create_dataloaders(
    data_dir='./output/mosei_features',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

# 创建模型
feature_dims = {'text': 768, 'audio': 768, 'video': 768}
model = HypergraphEmotionClassifier(feature_dims, num_classes=2)

# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloaders['train']:
    output = model(batch)
    loss = output['loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
```

## 📝 输出说明

### 训练输出

```python
output = model(batch)

# 包含:
{
    'logits': torch.Tensor,           # [batch, num_classes]
    'loss': torch.Tensor,             # 总损失
    'cls_loss': torch.Tensor,         # 分类损失
    'contrastive_loss': torch.Tensor, # 对比学习损失
    'l2_reg': torch.Tensor,           # L2 正则化
    'H': torch.Tensor,                # 超图连接矩阵 [batch, 3T, K]
    'text_pooled': torch.Tensor,      # 池化后文本特征 [batch, hidden_dim]
    'audio_pooled': torch.Tensor,     # 池化后音频特征
    'video_pooled': torch.Tensor      # 池化后视频特征
}
```

### 保存的模型

```python
checkpoint = torch.load('checkpoints/best_model_MOSEI_happy.pth')

# 包含:
{
    'epoch': int,                     # 最佳 epoch
    'model_state_dict': dict,         # 模型参数
    'optimizer_state_dict': dict,     # 优化器状态
    'accuracy': float,                # 最佳准确率
    'config': dict,                   # 模型配置
    'feature_dims': dict,             # 特征维度
    'args': dict                      # 训练参数
}
```

## 🎯 核心优势

### 1. 支持变长序列
- ✅ 自动 Padding + Masking
- ✅ 批处理高效
- ✅ 不损失信息

### 2. 超图融合
- ✅ 基于相关性的超图初始化
- ✅ 两阶段超图卷积
- ✅ 捕获高阶关系

### 3. 多任务学习
- ✅ 分类损失
- ✅ 对比学习损失
- ✅ L2 正则化

### 4. 易于使用
- ✅ 简洁的 API
- ✅ 命令行训练
- ✅ 完整文档

## 🔧 高级用法

### 加载预训练模型

```python
checkpoint = torch.load('checkpoints/best_model_MOSEI_happy.pth')

model = HypergraphEmotionClassifier(
    feature_dims=checkpoint['feature_dims'],
    num_classes=2,
    config=checkpoint['config']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测
predictions = model.predict(batch)
```

### 自定义 collate_fn

```python
from hyper_fusion.dataloader import padded_collate_fn

# 已经是最优实现，直接使用即可
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=padded_collate_fn
)
```

## 📚 参考

- [hyper_graph_fusion_instruct.md](../hyper_graph_fusion_instruct.md) - 原始技术文档
- [FRAME_HANDLING_ANALYSIS.md](../FRAME_HANDLING_ANALYSIS.md) - 帧数处理分析

---

**完整的 Padding + Masking 超图融合网络！** 🎉
