# 基于超图的多模态融合预测网络

本项目实现了一个基于超图结构的多模态融合神经网络，结合超图卷积和图对比学习，用于多模态情感分类任务。

## 目录

- [架构概述](#架构概述)
- [超图结构](#超图结构)
- [核心组件](#核心组件)
- [快速开始](#快速开始)
- [训练模型](#训练模型)
- [API文档](#api文档)
- [示例代码](#示例代码)

## 架构概述

### 整体流程

```
多模态输入 (Text/Video/Audio)
    ↓
特征投影到统一空间
    ↓
构建超图结构
    ↓
超图卷积 (多层)
    ↓
节点特征聚合
    ↓
分类 + 图对比学习
    ↓
输出预测结果
```

### 关键创新

1. **超图结构建模**: 同时捕获样本内多模态交互和跨样本模态关联
2. **图对比学习**: 利用标签监督，增强特征判别性
3. **端到端训练**: 超图构建、卷积和分类联合优化

## 超图结构

### 节点定义

- **节点数量**: M × N
  - N: 样本数量
  - M: 模态数量 (M=3: text/video/audio)
- **每个节点**: 表示一个样本的某个模态特征

### 超边定义

- **超边数量**: M + N

#### 1. 样本内超边 (N条)

每个样本的所有模态特征之间有一条超边连接，用于**模态融合**。

```
样本1: [text₁, video₁, audio₁] ←→ 超边1
样本2: [text₂, video₂, audio₂] ←→ 超边2
...
```

#### 2. 模态间超边 (M条)

不同样本的同一模态之间，满足**K最近邻**时有超边连接，用于**跨样本关联**。

```
Text模态: [text₁, text₂, ..., textₙ] (K-NN连接) ←→ 超边(N+1)
Video模态: [video₁, video₂, ..., videoₙ] (K-NN连接) ←→ 超边(N+2)
Audio模态: [audio₁, audio₂, ..., audioₙ] (K-NN连接) ←→ 超边(N+3)
```

### 关联矩阵

关联矩阵 **H**: (M×N) × (M+N)

- H[i, j] = 1: 节点i与超边j相连
- H[i, j] = 0: 节点i与超边j不相连

**示例** (N=2, M=3, K=1):

```
节点索引:
  0: text₁, 1: video₁, 2: audio₁
  3: text₂, 4: video₂, 5: audio₂

超边索引:
  0: 样本1内部连接
  1: 样本2内部连接
  2: Text模态K-NN
  3: Video模态K-NN
  4: Audio模态K-NN

关联矩阵 H (6×5):
       超边0  超边1  超边2  超边3  超边4
节点0    1     0     1     0     0    (text₁)
节点1    1     0     0     1     0    (video₁)
节点2    1     0     0     0     1    (audio₁)
节点3    0     1     1     0     0    (text₂)
节点4    0     1     0     1     0    (video₂)
节点5    0     1     0     0     1    (audio₂)
```

## 核心组件

### 1. HypergraphConstructor

构建超图关联矩阵。

```python
constructor = HypergraphConstructor(
    num_samples=32,        # 批次大小
    num_modalities=3,      # 模态数量
    k_neighbors=5          # K最近邻
)

H = constructor.construct_incidence_matrix(features_list)
```

### 2. HypergraphConvolution

超图卷积层，实现信息传播。

**公式**:
```
X' = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Θ
```

其中:
- D_v: 节点度矩阵
- D_e: 超边度矩阵
- W: 超边权重矩阵
- Θ: 可学习参数

```python
hgcn = HypergraphConvolution(
    in_features=256,
    out_features=128,
    dropout=0.5
)

X_out = hgcn(X, H)
```

### 3. ContrastiveLearning

监督对比学习模块。

**目标**:
- 同类样本特征拉近
- 异类样本特征推远

```python
contrastive = ContrastiveLearning(temperature=0.07)
loss = contrastive(embeddings, labels)
```

### 4. MultimodalHypergraphNetwork

完整的端到端网络。

```python
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],  # text/video/audio维度
    hidden_dim=256,
    output_dim=128,
    num_classes=6,
    num_hgcn_layers=2,
    k_neighbors=5,
    dropout=0.5,
    temperature=0.07
)
```

## 快速开始

### 安装依赖

```bash
pip install torch numpy scikit-learn tqdm
```

### 基本使用

```python
import torch
from hypergraph_model import MultimodalHypergraphNetwork

# 创建模拟数据
batch_size = 32
text_features = torch.randn(batch_size, 768)
video_features = torch.randn(batch_size, 512)
audio_features = torch.randn(batch_size, 256)
labels = torch.randint(0, 6, (batch_size,))

features_list = [text_features, video_features, audio_features]

# 创建模型
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],
    num_classes=6
)

# 训练
outputs = model(features_list, labels)
loss = outputs['loss']
loss.backward()

# 推理
predictions = model.predict(features_list)
```

## 训练模型

### 使用训练脚本

```bash
# MOSEI数据集
python train_hypergraph.py \
    --dataset MOSEI \
    --emotion happy \
    --label_id 0 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --k_neighbors 5 \
    --save_model

# MELD数据集
python train_hypergraph.py \
    --dataset MELD \
    --emotion sad \
    --label_id 1 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --k_neighbors 5 \
    --save_model
```

### 训练参数说明

**数据参数**:
- `--dataset`: 数据集名称 (MOSEI/MELD)
- `--emotion`: 情感类型 (happy/sad/anger/etc.)
- `--label_id`: 标签ID
- `--data_dir`: 数据目录 (默认'Data')
- `--train_ratio`: 训练集比例 (MOSEI, 默认0.7)

**模型参数**:
- `--hidden_dim`: 隐藏层维度 (默认256)
- `--output_dim`: 输出嵌入维度 (默认128)
- `--num_classes`: 分类类别数 (默认6)
- `--num_hgcn_layers`: 超图卷积层数 (默认2)
- `--k_neighbors`: K最近邻的K值 (默认5)
- `--dropout`: Dropout概率 (默认0.5)
- `--temperature`: 对比学习温度 (默认0.07)

**训练参数**:
- `--batch_size`: 批次大小 (默认32)
- `--epochs`: 训练轮数 (默认50)
- `--lr`: 学习率 (默认0.001)
- `--weight_decay`: 权重衰减 (默认1e-4)
- `--lr_step`: 学习率衰减步长 (默认20)
- `--lr_gamma`: 学习率衰减系数 (默认0.5)

## API文档

### MultimodalHypergraphNetwork

#### 初始化

```python
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],  # 每个模态的特征维度
    hidden_dim=256,                # 隐藏层维度
    output_dim=128,                # 输出嵌入维度
    num_classes=6,                 # 分类类别数
    num_hgcn_layers=2,             # 超图卷积层数量
    k_neighbors=5,                 # K最近邻的K值
    dropout=0.5,                   # Dropout概率
    temperature=0.07               # 对比学习温度
)
```

#### forward()

```python
outputs = model(
    features_list,        # List[Tensor]: [text, video, audio]
    labels=None,          # Tensor: 标签 (训练时需要)
    return_embeddings=False  # bool: 是否返回嵌入
)

# 返回字典:
# {
#     'logits': Tensor,              # 分类logits (N, num_classes)
#     'loss': Tensor,                # 总损失 (需要labels)
#     'classification_loss': Tensor, # 分类损失
#     'contrastive_loss': Tensor,    # 对比学习损失
#     'embeddings': Tensor,          # 样本嵌入 (可选)
#     'node_features': Tensor        # 节点特征 (可选)
# }
```

#### predict()

```python
predictions = model.predict(features_list)
# 返回: Tensor (N,) 预测类别
```

### HypergraphConstructor

```python
constructor = HypergraphConstructor(
    num_samples=32,       # 样本数量
    num_modalities=3,     # 模态数量
    k_neighbors=5         # K最近邻
)

H = constructor.construct_incidence_matrix(features_list)
# 返回: Tensor (M*N, M+N) 关联矩阵
```

### HypergraphConvolution

```python
hgcn = HypergraphConvolution(
    in_features=256,      # 输入特征维度
    out_features=128,     # 输出特征维度
    dropout=0.5           # Dropout概率
)

X_out = hgcn(X, H)
# X: Tensor (num_nodes, in_features)
# H: Tensor (num_nodes, num_hyperedges)
# 返回: Tensor (num_nodes, out_features)
```

### ContrastiveLearning

```python
contrastive = ContrastiveLearning(temperature=0.07)

loss = contrastive(features, labels)
# features: Tensor (batch_size, feature_dim)
# labels: Tensor (batch_size,)
# 返回: Tensor (标量) 对比学习损失
```

## 示例代码

### 示例1: 完整训练流程

```python
import torch
import torch.optim as optim
from hypergraph_model import MultimodalHypergraphNetwork
from emotion_dataloader import create_dataloaders

# 创建数据加载器
dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

train_loader = dataloaders['train']
test_loader = dataloaders['test']

# 创建模型
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],
    num_classes=6
).cuda()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(50):
    model.train()
    for batch in train_loader:
        # 准备数据
        text = batch['text'].cuda()
        video = batch['video'].cuda()
        audio = batch['audio'].cuda()
        labels = batch['labels'].cuda()

        features_list = [text, video, audio]

        # 前向传播
        outputs = model(features_list, labels)
        loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].cuda()
            video = batch['video'].cuda()
            audio = batch['audio'].cuda()
            labels = batch['labels'].cuda()

            predictions = model.predict([text, video, audio])
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Test Acc: {accuracy:.2f}%')
```

### 示例2: 可视化超图结构

```python
from hypergraph_model import HypergraphConstructor
import matplotlib.pyplot as plt

# 创建超图
constructor = HypergraphConstructor(
    num_samples=10,
    num_modalities=3,
    k_neighbors=3
)

# 构建关联矩阵
H = constructor.construct_incidence_matrix(features_list)

# 可视化
plt.figure(figsize=(12, 8))
plt.imshow(H.cpu().numpy(), cmap='Blues', aspect='auto')
plt.colorbar()
plt.xlabel('Hyperedges')
plt.ylabel('Nodes')
plt.title('Hypergraph Incidence Matrix')
plt.savefig('hypergraph_structure.png')
```

### 示例3: 特征嵌入提取

```python
# 提取样本嵌入用于下游任务
model.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        text = batch['text'].cuda()
        video = batch['video'].cuda()
        audio = batch['audio'].cuda()
        labels = batch['labels']

        outputs = model([text, video, audio], return_embeddings=True)
        embeddings = outputs['embeddings'].cpu()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, output_dim)
all_labels = torch.cat(all_labels, dim=0)           # (N,)

# 可以用于t-SNE可视化、聚类等
```

### 示例4: 自定义超图构建

```python
class CustomHypergraphConstructor(HypergraphConstructor):
    """自定义超图构建策略"""

    def construct_incidence_matrix(self, features_list):
        # 调用父类方法获取基础超图
        H = super().construct_incidence_matrix(features_list)

        # 添加自定义超边
        # 例如: 基于语义相似度的超边

        return H

# 使用自定义构建器
model.hypergraph_constructor = CustomHypergraphConstructor(
    num_samples=batch_size,
    num_modalities=3,
    k_neighbors=5
)
```

## 运行示例

```bash
# 运行所有示例
python examples/hypergraph_example.py

# 运行训练脚本
python train_hypergraph.py --dataset MOSEI --emotion happy --label_id 0 --save_model
```

## 性能优化建议

1. **批次大小**: 根据GPU内存调整，推荐16-64
2. **K值选择**: 较小的K(3-7)通常效果更好
3. **学习率**: 从0.001开始，使用学习率衰减
4. **层数**: 2-3层超图卷积层通常足够
5. **温度参数**: 0.05-0.1之间调整对比学习强度

## 常见问题

### 1. 内存不足

- 减小batch_size
- 减小hidden_dim和output_dim
- 减少超图卷积层数

### 2. 训练不稳定

- 降低学习率
- 增加dropout
- 检查数据预处理和归一化

### 3. 精度不高

- 增加训练轮数
- 调整K值
- 尝试不同的temperature参数
- 检查数据质量

## 引用

如果你使用了这个代码，请引用相关论文：

```bibtex
@article{hypergraph_gcn,
  title={Hypergraph Convolution and Hypergraph Attention},
  author={...},
  journal={...},
  year={2020}
}

@article{supervised_contrastive,
  title={Supervised Contrastive Learning},
  author={Khosla et al.},
  journal={NeurIPS},
  year={2020}
}
```

## 许可证

MIT License
