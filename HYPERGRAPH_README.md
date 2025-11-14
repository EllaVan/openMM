```markdown
# 多模态超图融合网络

完整的基于超图的多模态情感分析网络实现，基于论文方法构建。

## 📋 目录

- [核心概念](#核心概念)
- [网络架构](#网络架构)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [训练模型](#训练模型)
- [超图可视化](#超图可视化)
- [API参考](#api参考)

## 核心概念

### 什么是超图？

**传统图 vs 超图**:

```
传统图:
节点A ──边── 节点B  (一条边只连接2个节点)

超图:
        ┌─ 节点A
超边1 ──┼─ 节点B
        ├─ 节点C
        └─ 节点D  (一条超边可以连接多个节点)
```

### 为什么使用超图？

**优势**:
1. ✅ 可以建模**高阶关系** (多于2个节点的关系)
2. ✅ 适合**多模态融合** (同时连接文本、音频、视频节点)
3. ✅ 捕捉**复杂的情感模式** (单模态、双模态、三模态的组合)

### 本实现的创新点

1. **多模态统一超图**
   - 将三个模态的所有时间步作为节点
   - 一个超边可以同时连接不同模态的节点
   - 自动学习跨模态的高阶关系

2. **基于相关性的初始化**
   - 不需要人工定义连接规则
   - 通过学习自动发现节点间的关系

3. **超图增强**
   - 随机删除超边，减少冗余
   - 提高模型鲁棒性

4. **图对比学习**
   - 增强同一样本不同模态的一致性
   - 提高特征判别能力

## 网络架构

### 整体流程

```
输入特征
   ↓
┌──────────────────────────────────────┐
│  1. 单模态编码器 (Bi-LSTM)            │
│     文本/音频/视频 → 统一维度         │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  2. 超图初始化                        │
│     H = softmax((W_N·N)(W_E·N)^T/√d)│
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  3. 超图增强                          │
│     随机删除部分超边                  │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  4. 超图卷积 (多层)                   │
│     两阶段传播:                       │
│     - 节点 → 超边                    │
│     - 超边 → 节点                    │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  5. Bottleneck (可选)                │
│     压缩特征维度                      │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  6. 特征聚合                          │
│     分离三个模态 → 平均池化           │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  7. 分类器                            │
│     多模态特征 → 情感类别             │
└──────────────────────────────────────┘
   ↓
输出 + 损失
- 分类损失
- 对比学习损失 (可选)
- 正则化损失
```

### 核心组件

#### 1. 超图初始化

**功能**: 构建节点与超边的连接矩阵

**输入**: 节点特征 `N ∈ ℝ^(3T×d)`
- 3T 个节点 (文本T步 + 音频T步 + 视频T步)
- d 维特征

**输出**: 连接矩阵 `H ∈ ℝ^(3T×K)`
- K 个超边
- H[i,j] 表示节点 i 属于超边 j 的程度 (0-1)

**公式**:
```
Ĥ = (W_N · N)(W_E · N)^T
H = softmax(Ĥ / √d)
```

#### 2. 超图卷积

**功能**: 两阶段传播聚合高阶特征

**阶段1**: 节点 → 超边
```
E = D_e^(-1/2) H^T D_n^(-1/2) N
```
- 聚合同一超边的节点特征

**阶段2**: 超边 → 节点
```
N' = D_n^(-1/2) H W E
N'' = σ(N' θ)
```
- 将超边特征传播回节点

#### 3. 图对比学习

**功能**: 增强模态间一致性

**损失函数**:
```
L_contrastive = -1/|P(i)| Σ log(exp(z_i·z_p/τ) / Σ exp(z_i·z_a/τ))
```
- P(i): 同一样本的其他模态 (正例)
- A(i): 不同样本的模态 (负例)

## 安装依赖

```bash
pip install torch numpy tqdm
```

## 快速开始

### 示例 1: 基础使用

```python
import torch
from hypergraph_network import HypergraphEmotionClassifier

# 准备数据
batch = {
    'text_features': torch.randn(8, 50, 768),   # [batch, T, dim]
    'audio_features': torch.randn(8, 50, 768),
    'video_features': torch.randn(8, 50, 768),
    'label': torch.randint(0, 7, (8,))
}

# 创建模型
model = HypergraphEmotionClassifier(
    feature_dims={'text': 768, 'audio': 768, 'video': 768},
    num_classes=7
)

# 前向传播
output = model(batch)
print(f"Loss: {output['loss'].item():.4f}")
print(f"Predictions: {torch.argmax(output['logits'], dim=1)}")
```

### 示例 2: 使用已提取的特征

```python
from emotion_dataloader import create_dataloaders
from hypergraph_network import HypergraphEmotionClassifier

# 加载数据
dataloaders = create_dataloaders(
    data_dir='./output/mosei',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=16
)

# 获取特征维度
batch = next(iter(dataloaders['train']))
feature_dims = {
    'text': batch['text_features'].shape[-1],
    'audio': batch['audio_features'].shape[-1],
    'video': batch['video_features'].shape[-1]
}

# 创建模型
model = HypergraphEmotionClassifier(
    feature_dims=feature_dims,
    num_classes=2  # 二分类
)

# 训练
for batch in dataloaders['train']:
    output = model(batch)
    loss = output['loss']
    # ... 训练代码
```

## 详细使用

### 自定义配置

```python
config = {
    # 编码器参数
    'encoder_hidden_dim': 256,      # LSTM 隐藏层维度
    'encoder_output_dim': 512,      # 编码器输出维度

    # 超图参数
    'hypergraph_hidden_dim': 512,   # 超图隐藏层维度
    'num_hyperedges': 128,          # 超边数量
    'num_conv_layers': 3,           # 超图卷积层数

    # Bottleneck
    'bottleneck_dim': 256,          # Bottleneck 维度
    'use_bottleneck': True,         # 是否使用

    # Dropout
    'dropout': 0.2,                 # Dropout 率
    'hyperedge_drop_rate': 0.3,     # 超边删除率

    # 对比学习
    'use_contrastive': True,        # 是否使用
    'contrastive_weight': 0.2       # 对比学习权重
}

model = HypergraphEmotionClassifier(
    feature_dims=feature_dims,
    num_classes=7,
    config=config
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `encoder_hidden_dim` | 128 | LSTM 隐藏层维度 |
| `encoder_output_dim` | 256 | 编码器输出维度，会投影到统一空间 |
| `hypergraph_hidden_dim` | 256 | 超图节点特征维度 |
| `num_hyperedges` | 64 | 超边数量，影响模型容量 |
| `num_conv_layers` | 2 | 超图卷积层数，建议 2-4 层 |
| `bottleneck_dim` | 128 | Bottleneck 压缩维度 |
| `dropout` | 0.1 | Dropout 概率 |
| `hyperedge_drop_rate` | 0.2 | 超边删除率，用于增强 |
| `use_contrastive` | True | 是否使用对比学习 |
| `contrastive_weight` | 0.1 | 对比学习损失权重 |

## 训练模型

### 方式 1: 使用训练脚本

```bash
python train_hypergraph.py \
  --data_dir ./output/mosei \
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

### 方式 2: 自定义训练循环

```python
import torch
import torch.optim as optim
from hypergraph_network import HypergraphEmotionClassifier

# 创建模型
model = HypergraphEmotionClassifier(...)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向传播
        output = model(batch)
        loss = output['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        print(f"Loss: {loss.item():.4f}")
        if 'contrastive_loss' in output:
            print(f"Contrastive: {output['contrastive_loss'].item():.4f}")
```

### 训练技巧

1. **学习率调度**:
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5
)

# 每个 epoch 后
scheduler.step(val_accuracy)
```

2. **早停**:
```python
best_acc = 0
patience = 10
counter = 0

for epoch in range(epochs):
    val_acc = evaluate(model, val_loader)

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        # 保存模型
    else:
        counter += 1
        if counter >= patience:
            break  # 早停
```

3. **梯度裁剪**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 超图可视化

### 分析超图连接矩阵

```python
model.eval()
with torch.no_grad():
    output = model(batch)
    H = output['H']  # [batch, 3T, num_hyperedges]

# 分析第一个样本
H_0 = H[0]  # [3T, K]

# 节点度分析
node_degrees = (H_0 > 0.01).sum(dim=1)
print(f"平均节点度: {node_degrees.float().mean():.2f}")

# 超边大小分析
edge_sizes = (H_0 > 0.01).sum(dim=0)
print(f"平均超边大小: {edge_sizes.float().mean():.2f}")

# 跨模态连接
T = H_0.shape[0] // 3
text_nodes = H_0[:T, :]
audio_nodes = H_0[T:2*T, :]
video_nodes = H_0[2*T:, :]

# 三模态共享的超边数量
shared_edges = ((text_nodes > 0.01) &
                (audio_nodes > 0.01) &
                (video_nodes > 0.01)).sum()
print(f"三模态共享超边: {shared_edges}")
```

### 可视化超图结构

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 热力图
plt.figure(figsize=(12, 8))
sns.heatmap(
    H_0.cpu().numpy(),
    cmap='viridis',
    cbar_kws={'label': 'Connection Weight'}
)
plt.xlabel('Hyperedges')
plt.ylabel('Nodes (Text→Audio→Video)')
plt.title('Hypergraph Incidence Matrix')
plt.tight_layout()
plt.savefig('hypergraph_heatmap.png')
```

## API 参考

### HypergraphEmotionClassifier

主要的模型类，封装了完整的超图网络。

```python
model = HypergraphEmotionClassifier(
    feature_dims: Dict[str, int],  # 特征维度
    num_classes: int = 7,          # 类别数
    config: Optional[Dict] = None  # 配置字典
)
```

**方法**:

#### `forward(batch)`

前向传播。

**参数**:
- `batch`: 包含 'text_features', 'audio_features', 'video_features', 'label' 的字典

**返回**:
```python
{
    'logits': 分类 logits [batch_size, num_classes],
    'loss': 总损失,
    'cls_loss': 分类损失,
    'contrastive_loss': 对比学习损失 (可选),
    'reg_loss': 正则化损失,
    'H': 超图连接矩阵 [batch, 3T, K],
    'multimodal_feature': 融合特征 [batch, hidden_dim*3]
}
```

#### `predict(batch)`

预测类别。

**返回**: 预测的类别 `[batch_size]`

### MultimodalHypergraphLayer

多模态超图层，核心组件。

```python
layer = MultimodalHypergraphLayer(
    text_dim: int,
    audio_dim: int,
    video_dim: int,
    hidden_dim: int,
    num_hyperedges: int,
    num_conv_layers: int = 2
)
```

### HypergraphConvolution

超图卷积层。

```python
conv = HypergraphConvolution(
    in_dim: int,
    out_dim: int,
    use_bn: bool = True,
    dropout: float = 0.1
)
```

### GraphContrastiveLearning

图对比学习模块。

```python
contrastive = GraphContrastiveLearning(
    feature_dim: int,
    projection_dim: int = 128,
    temperature: float = 0.07
)
```

## 完整示例

### 端到端训练

```python
# 1. 准备数据
from emotion_dataloader import create_dataloaders

dataloaders = create_dataloaders(
    data_dir='./output/mosei',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

# 2. 创建模型
from hypergraph_network import HypergraphEmotionClassifier

batch = next(iter(dataloaders['train']))
feature_dims = {
    'text': batch['text_features'].shape[-1],
    'audio': batch['audio_features'].shape[-1],
    'video': batch['video_features'].shape[-1]
}

model = HypergraphEmotionClassifier(
    feature_dims=feature_dims,
    num_classes=2,
    config={
        'num_hyperedges': 64,
        'num_conv_layers': 2,
        'use_contrastive': True,
        'use_bottleneck': True
    }
)

# 3. 训练
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    for batch in dataloaders['train']:
        output = model(batch)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloaders['test']:
            predictions = model.predict(batch)
            correct += (predictions == batch['label']).sum().item()
            total += batch['label'].size(0)

    accuracy = correct / total
    print(f'Epoch {epoch+1}: Accuracy = {accuracy:.4f}')
```

## 性能优化

### 1. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(batch)
        loss = output['loss']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = output['loss'] / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 模型并行

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 常见问题

### Q1: 如何选择超边数量？

**A**: 一般建议:
- 小数据集: 32-64
- 中等数据集: 64-128
- 大数据集: 128-256

超边数量越多，模型容量越大，但计算成本也越高。

### Q2: 超图卷积层数选择？

**A**: 建议 2-4 层
- 2层: 基础，捕捉局部关系
- 3-4层: 捕捉更长程的依赖
- 过多层: 可能导致过平滑

### Q3: 对比学习权重如何设置？

**A**: 通常 0.05-0.2 之间
- 权重太小: 对比学习作用不明显
- 权重太大: 可能影响分类性能

### Q4: GPU 内存不足怎么办？

**A**:
1. 减少 batch_size
2. 减少超边数量
3. 减少超图卷积层数
4. 使用梯度累积
5. 使用混合精度训练

## 参考论文

基于论文的超图融合方法实现，包含以下关键技术:
1. 基于相关性的超图初始化
2. 超图增强 (超边删除)
3. 两阶段超图卷积
4. 监督对比学习
5. Bottleneck 压缩

## 许可证

MIT License
```
