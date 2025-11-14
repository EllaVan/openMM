# 快速开始指南

本指南帮助你快速上手使用超图多模态融合网络进行情感分类。

## 目录

1. [安装依赖](#1-安装依赖)
2. [准备数据](#2-准备数据)
3. [运行示例](#3-运行示例)
4. [训练模型](#4-训练模型)
5. [使用模型](#5-使用模型)

## 1. 安装依赖

```bash
# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio

# 安装其他依赖
pip install numpy scikit-learn tqdm
```

## 2. 准备数据

确保你的数据文件按以下格式组织：

### MOSEI数据集

```
Data/
├── MOSEIhappylabel0.pkl
├── MOSEIsadlabel1.pkl
├── MOSEIangerlabel2.pkl
└── ...
```

### MELD数据集

```
Data/
├── MELD_trainhappylabel0.pkl
├── MELD_devhappylabel0.pkl
├── MELD_testhappylabel0.pkl
├── MELD_trainsadlabel1.pkl
└── ...
```

**数据格式**: 每个pkl文件包含一个列表或字典，每个样本是一个字典:

```python
{
    'audio_features': numpy.ndarray or torch.Tensor,  # shape: (seq_len, audio_dim)
    'text_features': numpy.ndarray or torch.Tensor,   # shape: (seq_len, text_dim)
    'video_features': numpy.ndarray or torch.Tensor,  # shape: (seq_len, video_dim)
    'label': int  # 类别标签
}
```

## 3. 运行示例

### 查看超图网络示例

```bash
python examples/hypergraph_example.py
```

这将运行6个示例，展示：
1. 超图构建
2. 超图卷积
3. 图对比学习
4. 完整网络
5. 训练步骤
6. 超图结构分析

### 查看数据加载器示例

```bash
python examples/dataloader_examples.py
```

## 4. 训练模型

### 最简单的训练命令

```bash
# 使用MOSEI数据集训练
python train_hypergraph.py \
    --dataset MOSEI \
    --emotion happy \
    --label_id 0 \
    --save_model
```

### 完整的训练命令（所有参数）

```bash
python train_hypergraph.py \
    --dataset MOSEI \
    --emotion happy \
    --label_id 0 \
    --data_dir Data \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --hidden_dim 256 \
    --output_dim 128 \
    --num_classes 6 \
    --num_hgcn_layers 2 \
    --k_neighbors 5 \
    --dropout 0.5 \
    --temperature 0.07 \
    --save_model
```

### 训练MELD数据集

```bash
python train_hypergraph.py \
    --dataset MELD \
    --emotion sad \
    --label_id 1 \
    --batch_size 32 \
    --epochs 50 \
    --save_model
```

### 训练输出

训练过程中你会看到：

```
Using device: cuda

Loading MOSEI dataset...
Train samples: 700
Test samples: 300

Feature dimensions:
  Text: 768
  Video: 512
  Audio: 256

Model created with 1234567 parameters

Starting training for 50 epochs...
================================================================================

Epoch [1/50]
Training: 100%|████████████| 22/22 [00:15<00:00,  1.41it/s, loss=1.2345, acc=45.67%]
Evaluating: 100%|█████████| 10/10 [00:03<00:00,  3.21it/s, loss=1.1234, acc=52.34%]

Train Loss: 1.2345 (Cls: 1.1234, Con: 0.1111)
Train Acc: 45.67%
Test Loss: 1.1234
Test Acc: 52.34%
LR: 0.001000
Saved best model with acc: 52.34%

...
```

## 5. 使用模型

### 5.1 基本使用

```python
import torch
from hypergraph_model import MultimodalHypergraphNetwork

# 创建模型
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],  # text/video/audio维度
    hidden_dim=256,
    output_dim=128,
    num_classes=6,
    num_hgcn_layers=2,
    k_neighbors=5
)

# 准备数据
text_features = torch.randn(32, 768)
video_features = torch.randn(32, 512)
audio_features = torch.randn(32, 256)
labels = torch.randint(0, 6, (32,))

features_list = [text_features, video_features, audio_features]

# 训练
model.train()
outputs = model(features_list, labels)
loss = outputs['loss']
loss.backward()

# 推理
model.eval()
predictions = model.predict(features_list)
print(f"Predictions: {predictions}")
```

### 5.2 加载已训练模型

```python
import torch
from hypergraph_model import MultimodalHypergraphNetwork

# 创建模型
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],
    num_classes=6
)

# 加载权重
checkpoint = torch.load('checkpoints/best_model_MOSEI_happy.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model with test accuracy: {checkpoint['test_acc']:.2f}%")

# 进行预测
with torch.no_grad():
    predictions = model.predict(features_list)
```

### 5.3 提取特征嵌入

```python
model.eval()
with torch.no_grad():
    outputs = model(features_list, return_embeddings=True)

    # 获取样本嵌入
    embeddings = outputs['embeddings']  # shape: (batch_size, output_dim)

    # 可用于可视化、聚类等下游任务
    print(f"Embeddings shape: {embeddings.shape}")
```

### 5.4 完整的训练循环

```python
import torch
import torch.optim as optim
from hypergraph_model import MultimodalHypergraphNetwork
from emotion_dataloader import create_dataloaders

# 1. 准备数据
dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

train_loader = dataloaders['train']
test_loader = dataloaders['test']

# 2. 创建模型
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],
    num_classes=6
).cuda()

# 3. 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 4. 训练
best_acc = 0
for epoch in range(50):
    # 训练阶段
    model.train()
    for batch in train_loader:
        text = batch['text'].cuda()
        video = batch['video'].cuda()
        audio = batch['audio'].cuda()
        labels = batch['labels'].cuda()

        # 前向传播
        outputs = model([text, video, audio], labels)
        loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估阶段
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

    # 保存最佳模型
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), 'best_model.pt')

    scheduler.step()

print(f'Best Test Accuracy: {best_acc:.2f}%')
```

## 参数调优建议

### 开始调优

1. **先使用默认参数训练**:
   ```bash
   python train_hypergraph.py --dataset MOSEI --emotion happy --label_id 0
   ```

2. **观察训练曲线**，然后调整：

### 如果训练损失下降很慢

- 提高学习率: `--lr 0.005`
- 减少dropout: `--dropout 0.3`
- 增加模型容量: `--hidden_dim 512 --output_dim 256`

### 如果出现过拟合（训练acc高但测试acc低）

- 降低学习率: `--lr 0.0005`
- 增加dropout: `--dropout 0.6`
- 减少模型容量: `--hidden_dim 128 --output_dim 64`
- 增强对比学习: `--temperature 0.05`

### 如果内存不足

- 减小batch size: `--batch_size 16`
- 减少模型维度: `--hidden_dim 128 --output_dim 64`
- 减少层数: `--num_hgcn_layers 1`

### K值调优

K值控制跨样本模态关联的强度：

- **小K值 (3-5)**: 只连接最相似的样本，适合数据分布清晰的情况
- **大K值 (7-10)**: 连接更多样本，适合数据分布复杂的情况

```bash
# 尝试不同的K值
python train_hypergraph.py --dataset MOSEI --emotion happy --label_id 0 --k_neighbors 3
python train_hypergraph.py --dataset MOSEI --emotion happy --label_id 0 --k_neighbors 7
```

### 温度参数调优

温度参数控制对比学习的强度：

- **低温度 (0.05)**: 对比学习更强，特征更判别
- **高温度 (0.1)**: 对比学习更弱，特征更平滑

```bash
python train_hypergraph.py --dataset MOSEI --emotion happy --label_id 0 --temperature 0.05
```

## 常见问题

### Q1: 训练速度很慢

**A**: 尝试以下方法：
- 增加 `--num_workers` (如 `--num_workers 8`)
- 使用更小的batch size但更多的epoch
- 确保使用GPU训练

### Q2: 准确率一直很低

**A**: 检查以下几点：
- 数据是否正确加载（运行 `examples/dataloader_examples.py`）
- 标签是否正确
- 特征维度是否匹配
- 尝试从简单任务开始（如二分类）

### Q3: 内存溢出

**A**:
- 减小 `--batch_size`
- 减小 `--hidden_dim` 和 `--output_dim`
- 减少 `--num_hgcn_layers`

### Q4: 如何可视化超图结构

**A**:
```python
import matplotlib.pyplot as plt
from hypergraph_model import HypergraphConstructor

constructor = HypergraphConstructor(num_samples=10, num_modalities=3, k_neighbors=3)
H = constructor.construct_incidence_matrix(features_list)

plt.figure(figsize=(12, 8))
plt.imshow(H.cpu().numpy(), cmap='Blues', aspect='auto')
plt.colorbar()
plt.xlabel('Hyperedges')
plt.ylabel('Nodes')
plt.savefig('hypergraph.png')
```

## 下一步

- 阅读 [HYPERGRAPH_README.md](HYPERGRAPH_README.md) 了解详细的架构说明
- 查看 [examples/hypergraph_example.py](examples/hypergraph_example.py) 学习更多用法
- 尝试调整超参数以获得更好的性能
- 在自己的数据集上训练模型

## 获取帮助

如果遇到问题：

1. 检查数据格式是否正确
2. 运行示例代码确认环境配置
3. 查看详细文档
4. 提交issue描述问题

祝你使用愉快！
