# 样本级别超图融合网络

## 核心设计

### 数据格式
- 输入：每个样本的每个模态特征为 **[768]** 维度（无时序）
- 特征类型：text_features, audio_features, video_features

### 超图结构

**节点** (3N个)：N个样本 × 3个模态
- 文本节点：N个样本的文本特征
- 音频节点：N个样本的音频特征  
- 视频节点：N个样本的视频特征

**超边** (N+3条)：
1. **N条样本内超边**：连接同一样本的3个模态
2. **3条模态内超边**：连接所有样本的同一模态，权重为余弦相似度

### 标签映射

**重要**：seen 和 unseen emotions 都会重新映射标签！

- **Seen emotions**: 0, 1, 2, ...
- **Unseen emotions**: len(seen), len(seen)+1, ...

**示例**：
```yaml
seen_emotions:
  happy: 0  # 原始标签0 → 新标签0
  sad: 1    # 原始标签1 → 新标签1
unseen_emotions:
  fear: 5   # 原始标签5 → 新标签2
```

## 使用方法

### 1. 基本运行

```bash
python codes_v251112/fusion/train.py
```

### 2. 修改配置

编辑 `codes_v251112/fusion/config/config.yaml`:

```yaml
dataset:
  seen_emotions:
    happy: 0
    sad: 1
  unseen_emotions:
    fear: 5
```

### 3. 命令行覆盖

```bash
# 修改学习率
python codes_v251112/fusion/train.py training.learning_rate=0.001

# 修改batch size
python codes_v251112/fusion/train.py dataloader.batch_size=64

# 禁用unseen emotions
python codes_v251112/fusion/train.py dataset.unseen_emotions=null

# 修改超图卷积层数
python codes_v251112/fusion/train.py model.hypergraph.num_conv_layers=3
```

### 4. 多运行实验

```bash
# 测试不同学习率
python codes_v251112/fusion/train.py --multirun \
    training.learning_rate=0.0001,0.001,0.01

# 测试不同超图配置
python codes_v251112/fusion/train.py --multirun \
    model.hypergraph.num_conv_layers=1,2,3 \
    model.sample_hypergraph.similarity_temperature=0.5,1.0,2.0
```

## 文件结构

```
fusion/
├── config/
│   └── config.yaml              # 配置文件
├── sample_hypergraph.py         # 超图模块
├── sample_network.py            # 网络模型
├── dataloader.py                # 数据加载器
├── train.py                     # 训练脚本
└── README.md                    # 本文档
```

## 关键参数

### 模型参数

```yaml
model:
  hypergraph:
    hidden_dim: 256              # 超图隐藏层维度
    num_conv_layers: 2           # 超图卷积层数
  sample_hypergraph:
    use_edge_weights: true       # 是否使用余弦相似度权重
    similarity_temperature: 1.0   # 温度参数（越高越均匀）
```

### 训练参数

```yaml
training:
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler:
    mode: max                    # 监控准确率
    patience: 5                  # 5个epoch不提升则降低学习率
  early_stopping:
    patience: 10                 # 10个epoch不提升则早停
```

## 工作原理

1. **数据加载**：加载 seen/unseen emotions，重新映射标签
2. **特征提取**：直接使用 [768] 维度的样本特征
3. **超图构建**：
   - 样本内超边：连接同一样本的3个模态
   - 模态内超边：连接所有样本的同一模态，权重为余弦相似度
4. **超图卷积**：多层超图卷积传播信息
5. **样本聚合**：融合3个模态特征得到样本表示
6. **分类**：全连接层进行分类

## 注意事项

1. **Hydra工作目录**：代码中已处理 `os.chdir(exc_dir)`
2. **标签映射**：unseen emotions 也会分配标签
3. **Batch Size**：建议 ≥32，以便学习样本间关系
4. **数据路径**：确保 `data_dir` 指向正确的特征文件目录

## 依赖

- Python 3.7+
- PyTorch 1.9+
- Hydra 1.2+
- OmegaConf
