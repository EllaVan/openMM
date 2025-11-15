# 样本级别超图融合网络

## 📋 概述

这是一个基于**样本级别超图**的多模态情感识别网络。与之前基于时序帧的超图不同，这个版本在批次级别构建超图，将每个样本的每个模态视为一个节点。

## 🎯 核心设计

### 超图结构

假设模态数量为 **3** (文本、音频、视频)，批次大小为 **N**：

#### 节点 (3N个)
- **文本节点**: N个样本的文本特征
- **音频节点**: N个样本的音频特征
- **视频节点**: N个样本的视频特征

#### 超边 (N+3条)

**1. 样本内超边 (N条)**
- 每条超边连接**同一样本**的3个不同模态
- 建模单个样本内的多模态交互
- 权重: 均匀权重 (1.0)

**2. 模态内超边 (3条)**
- 每条超边连接**所有样本**的同一模态
- 建模不同样本间同一模态的关系
- 权重: 样本间的**余弦相似度**

#### 样本表示
- 聚合每个样本的3个模态节点特征
- 通过全连接层融合得到最终的样本表示

### 超图关联矩阵示例

对于批次大小 N=4，关联矩阵 H 的形状为 [12, 7]:

```
节点索引:
- 0-3:   文本节点 (样本0-3)
- 4-7:   音频节点 (样本0-3)
- 8-11:  视频节点 (样本0-3)

超边索引:
- 0-3:  样本内超边 (样本0-3)
- 4:    文本模态内超边
- 5:    音频模态内超边
- 6:    视频模态内超边

关联矩阵 H[12, 7]:
         超边0 超边1 超边2 超边3 | 超边4(文本) 超边5(音频) 超边6(视频)
节点0(文本0)  1    0    0    0  |   w0          0          0
节点1(文本1)  0    1    0    0  |   w1          0          0
节点2(文本2)  0    0    1    0  |   w2          0          0
节点3(文本3)  0    0    0    1  |   w3          0          0
节点4(音频0)  1    0    0    0  |   0          w0          0
节点5(音频1)  0    1    0    0  |   0          w1          0
节点6(音频2)  0    0    1    0  |   0          w2          0
节点7(音频3)  0    0    0    1  |   0          w3          0
节点8(视频0)  1    0    0    0  |   0           0         w0
节点9(视频1)  0    1    0    0  |   0           0         w1
节点10(视频2) 0    0    1    0  |   0           0         w2
节点11(视频3) 0    0    0    1  |   0           0         w3

其中 w0, w1, w2, w3 是基于余弦相似度计算的权重
```

## 🏗️ 网络架构

```
输入: [batch_size, T, feature_dim] (时序特征)
  │
  ├─> 时序编码器 (Bi-LSTM)
  │   └─> [batch_size, T, encoder_output_dim]
  │
  ├─> 时序池化 (Masked Average/Max/Last)
  │   └─> [batch_size, encoder_output_dim]
  │
  ├─> 样本级别超图融合
  │   ├─> 投影到统一维度
  │   │   └─> [3N, hidden_dim]
  │   │
  │   ├─> 构建超图 (H, W)
  │   │   ├─> 关联矩阵 H: [3N, N+3]
  │   │   └─> 超边权重 W: [N+3]
  │   │
  │   ├─> 超图卷积 (多层)
  │   │   └─> [3N, hidden_dim]
  │   │
  │   └─> 样本特征聚合
  │       └─> [batch_size, hidden_dim]
  │
  └─> 分类器
      └─> [batch_size, num_classes]
```

## 📁 文件结构

```
fusion/
├── sample_hypergraph.py              # 样本级别超图模块 ⭐
│   ├── SampleLevelHypergraph        # 超图构建和卷积
│   └── SampleHypergraphConv         # 超图卷积层
│
├── sample_network.py                 # 完整网络架构 ⭐
│   ├── TemporalEncoder              # 时序编码器
│   ├── TemporalPooling              # 时序池化
│   ├── SampleHypergraphNetwork      # 主网络
│   └── SampleHypergraphClassifier   # 分类器封装
│
├── config_sample_hypergraph.yaml    # 配置文件 ⭐
├── train_sample_hypergraph.py       # 训练脚本 ⭐
├── config_utils.py                  # 配置读取工具
├── dataloader.py                    # 数据加载器
└── README_SAMPLE_HYPERGRAPH.md      # 本文档
```

## ⚙️ 配置说明

### 模型配置

```yaml
model:
  # 编码器配置
  encoder:
    hidden_dim: 256         # LSTM隐藏层维度
    output_dim: 256         # 编码器输出维度
    dropout: 0.1

  # 超图配置
  hypergraph:
    hidden_dim: 256         # 超图隐藏层维度
    num_conv_layers: 2      # 超图卷积层数

  # 池化配置
  pooling:
    pooling_type: "masked_mean"  # 池化类型: masked_mean, max, last

  # 样本级别超图特有配置
  sample_hypergraph:
    use_edge_weights: true           # 是否使用余弦相似度作为权重
    similarity_temperature: 1.0       # 相似度温度参数
```

### 池化类型说明

- **masked_mean**: 带mask的平均池化 (推荐)
  - 只对有效帧计算平均
  - 适合变长序列

- **max**: 最大池化
  - 取每个维度的最大值
  - 保留最显著的特征

- **last**: 使用最后一个有效帧
  - 适合序列信息累积的场景
  - 需要准确的mask

## 🚀 使用方法

### 1. 准备数据

确保数据集格式符合 `instruct.md` 中的要求。

### 2. 修改配置

编辑 `config_sample_hypergraph.yaml`:

```yaml
# 示例: Happy vs Sad 二分类
dataset:
  name: "MELD"
  data_dir: "./output/meld_utterance_features"
  seen_emotions:
    happy: 0
    sad: 1
  unseen_emotions: {}

# 调整超图参数
model:
  hypergraph:
    num_conv_layers: 2  # 增加卷积层可能提高性能
  sample_hypergraph:
    use_edge_weights: true  # 使用余弦相似度
    similarity_temperature: 1.0  # 温度越高，权重越均匀
```

### 3. 运行训练

```bash
# 使用默认配置
python codes_v251112/fusion/train_sample_hypergraph.py

# 使用自定义配置
python codes_v251112/fusion/train_sample_hypergraph.py \
    --config path/to/your/config.yaml
```

### 4. 测试模块

```bash
# 测试超图模块
python codes_v251112/fusion/sample_hypergraph.py

# 测试完整网络
python codes_v251112/fusion/sample_network.py
```

## 🔍 关键特性

### 1. 样本内多模态交互

每个样本的3个模态通过**样本内超边**连接，学习单个样本的多模态融合。

```python
# 样本0的超边连接
H[text_0, hyperedge_0] = 1
H[audio_0, hyperedge_0] = 1
H[video_0, hyperedge_0] = 1
```

### 2. 样本间同模态关系

同一模态的不同样本通过**模态内超边**连接，权重为余弦相似度。

```python
# 文本模态内超边
for sample_i in range(N):
    # 计算样本i与所有样本的文本相似度
    similarity = cosine_similarity(text_i, all_text_samples)
    weight_i = softmax(similarity)
    H[text_i, hyperedge_text] = weight_i
```

### 3. 超图卷积

两阶段信息传播：
1. **节点 → 超边**: 聚合连接到同一超边的节点特征
2. **超边 → 节点**: 将超边特征传播回节点

```python
# 归一化的超图拉普拉斯
H_norm = D_n^{-1/2} H W D_e^{-1/2}
N' = H_norm H_norm^T N θ
```

## 📊 与之前版本的对比

| 特性 | 时序帧超图 | 样本级别超图 (新) |
|------|-----------|------------------|
| 节点粒度 | 每个帧 | 每个样本的每个模态 |
| 节点数 | 3T (T为帧数) | 3N (N为批次大小) |
| 超边数 | K个可学习 | N+3个固定结构 |
| 超边语义 | 学习的模式 | 样本内/模态内关系 |
| 权重 | 学习的 | 余弦相似度 |
| 适用场景 | 细粒度时序建模 | 样本间关系建模 |

## 💡 优势

1. **明确的语义**: 超边有清晰的含义（样本内/模态内）
2. **自适应权重**: 基于余弦相似度的动态权重
3. **样本交互**: 显式建模batch内样本间的关系
4. **计算高效**: 节点数与batch size线性相关
5. **可解释性**: 超图结构直观，易于理解和分析

## 🎓 理论基础

### 超图卷积公式

$$
\mathbf{N}' = \sigma(\mathbf{D}_n^{-\frac{1}{2}} \mathbf{H} \mathbf{W} \mathbf{D}_e^{-\frac{1}{2}} \mathbf{H}^T \mathbf{D}_n^{-\frac{1}{2}} \mathbf{N} \mathbf{\Theta})
$$

其中:
- $\mathbf{N}$: 节点特征矩阵 [3N, d]
- $\mathbf{H}$: 关联矩阵 [3N, N+3]
- $\mathbf{W}$: 超边权重对角矩阵 [N+3, N+3]
- $\mathbf{D}_n$: 节点度矩阵 (对角矩阵)
- $\mathbf{D}_e$: 超边度矩阵 (对角矩阵)
- $\mathbf{\Theta}$: 可学习参数
- $\sigma$: 激活函数

### 余弦相似度权重

$$
w_i = \frac{\exp(\text{sim}(f_i, \bar{f}) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(f_j, \bar{f}) / \tau)}
$$

其中:
- $f_i$: 样本i的特征
- $\bar{f}$: 所有样本特征的均值
- $\tau$: 温度参数
- $\text{sim}(\cdot, \cdot)$: 余弦相似度

## 🔧 调优建议

### 1. 超图卷积层数

```yaml
# 少层: 快速但可能欠拟合
num_conv_layers: 1

# 中等: 平衡性能和计算 (推荐)
num_conv_layers: 2

# 多层: 更深的特征但可能过拟合
num_conv_layers: 3
```

### 2. 相似度温度

```yaml
# 低温度: 权重更集中在相似样本
similarity_temperature: 0.5

# 中等温度: 平衡 (推荐)
similarity_temperature: 1.0

# 高温度: 权重更均匀
similarity_temperature: 2.0
```

### 3. 池化类型

```yaml
# 序列信息均匀分布: masked_mean (推荐)
pooling_type: "masked_mean"

# 关键信息在峰值: max
pooling_type: "max"

# 信息累积: last
pooling_type: "last"
```

## 📈 预期效果

- **训练速度**: 比时序帧超图快 (节点数更少)
- **内存占用**: 更低 (不需要保存大量时序节点)
- **性能**: 在样本关系重要的任务上表现更好
- **可扩展性**: 支持大batch训练

## 🐛 注意事项

1. **Batch Size**: 建议使用较大的batch size (≥32)，以便学习更好的样本间关系
2. **数据平衡**: 确保batch内各类别样本均衡，避免相似度偏差
3. **梯度检查**: 如遇梯度爆炸，降低学习率或启用梯度裁剪

## 📚 参考

这个设计结合了以下思想:
- 超图神经网络 (Hypergraph Neural Networks)
- 自注意力机制 (Self-Attention)
- 多模态融合 (Multimodal Fusion)
- 度量学习 (Metric Learning)
