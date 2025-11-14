# openMM - 多模态情感分类系统

基于超图结构的多模态融合神经网络，用于情感分类任务。

## 项目简介

本项目实现了一个完整的多模态情感分类系统，包括：

1. **灵活的数据加载器** - 支持MOSEI和MELD数据集
2. **超图神经网络** - 结合超图卷积和图对比学习
3. **端到端训练** - 完整的训练和评估流程

### 主要特性

- ✨ **超图结构建模**: 同时捕获样本内多模态交互和跨样本模态关联
- 🎯 **图对比学习**: 利用标签监督增强特征判别性
- 📊 **多数据集支持**: MOSEI和MELD数据集
- 🔧 **易于使用**: 清晰的API和丰富的文档
- 🚀 **高效训练**: 优化的训练流程和参数

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/EllaVan/openMM.git
cd openMM

# 安装依赖
pip install torch numpy scikit-learn tqdm
```

### 基本使用

```python
from hypergraph_model import MultimodalHypergraphNetwork

# 创建模型
model = MultimodalHypergraphNetwork(
    feature_dims=[768, 512, 256],  # text/video/audio
    num_classes=6
)

# 训练
outputs = model(features_list, labels)
loss = outputs['loss']
loss.backward()

# 推理
predictions = model.predict(features_list)
```

### 训练模型

```bash
# MOSEI数据集
python train_hypergraph.py \
    --dataset MOSEI \
    --emotion happy \
    --label_id 0 \
    --batch_size 32 \
    --epochs 50 \
    --save_model

# MELD数据集
python train_hypergraph.py \
    --dataset MELD \
    --emotion sad \
    --label_id 1 \
    --batch_size 32 \
    --epochs 50 \
    --save_model
```

详细的快速开始指南请查看 [QUICKSTART.md](QUICKSTART.md)

## 项目结构

```
openMM/
├── hypergraph_model.py          # 超图网络模型
├── emotion_dataloader.py        # 数据加载器
├── train_hypergraph.py          # 训练脚本
├── examples/
│   ├── dataloader_examples.py   # 数据加载器示例
│   └── hypergraph_example.py    # 超图网络示例
├── Data/                        # 数据目录
├── checkpoints/                 # 模型检查点
├── README.md                    # 本文件
├── QUICKSTART.md                # 快速开始指南
├── DATALOADER_README.md         # 数据加载器文档
└── HYPERGRAPH_README.md         # 超图网络详细文档
```

## 核心组件

### 1. 数据加载器

支持MOSEI和MELD两种数据集，自动处理数据划分：

- **MOSEI**: 按7/3比例自动划分训练集和测试集
- **MELD**: 自动合并train+dev为训练集，test为测试集

```python
from emotion_dataloader import create_dataloaders

dataloaders = create_dataloaders(
    data_dir='Data',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

train_loader = dataloaders['train']
test_loader = dataloaders['test']
```

详细文档: [DATALOADER_README.md](DATALOADER_README.md)

### 2. 超图神经网络

基于超图结构的多模态融合网络：

#### 超图结构

- **节点**: M×N 个 (N个样本，M=3个模态)
- **超边**: M+N 条
  - N条样本内超边: 连接每个样本的所有模态
  - M条模态间超边: 通过K-NN连接跨样本的同一模态

#### 网络架构

```
输入特征 → 特征投影 → 超图构建 → 超图卷积 → 节点聚合 → 分类+对比学习 → 输出
```

#### 核心模块

- **HypergraphConstructor**: 构建超图关联矩阵
- **HypergraphConvolution**: 超图卷积层
- **ContrastiveLearning**: 图对比学习
- **MultimodalHypergraphNetwork**: 端到端网络

详细文档: [HYPERGRAPH_README.md](HYPERGRAPH_README.md)

## 示例

### 数据加载示例

```bash
# 运行数据加载器示例
python examples/dataloader_examples.py
```

包含8个示例：
1. MOSEI单个DataLoader (7/3划分)
2. MELD单个DataLoader (train+dev合并)
3. 批量创建多个MOSEI DataLoader
4. 批量创建多个MELD DataLoader
5. 使用自定义collate函数
6. 在训练循环中使用
7. 使用数据转换函数
8. 比较MOSEI和MELD的数据划分策略

### 超图网络示例

```bash
# 运行超图网络示例
python examples/hypergraph_example.py
```

包含6个示例：
1. 超图构建
2. 超图卷积
3. 图对比学习
4. 完整网络
5. 训练步骤
6. 超图结构分析

## 训练参数

### 数据参数

- `--dataset`: 数据集名称 (MOSEI/MELD)
- `--emotion`: 情感类型 (happy/sad/anger/etc.)
- `--label_id`: 标签ID
- `--train_ratio`: 训练集比例 (MOSEI, 默认0.7)

### 模型参数

- `--hidden_dim`: 隐藏层维度 (默认256)
- `--output_dim`: 输出嵌入维度 (默认128)
- `--num_classes`: 分类类别数 (默认6)
- `--num_hgcn_layers`: 超图卷积层数 (默认2)
- `--k_neighbors`: K最近邻的K值 (默认5)
- `--dropout`: Dropout概率 (默认0.5)
- `--temperature`: 对比学习温度 (默认0.07)

### 训练参数

- `--batch_size`: 批次大小 (默认32)
- `--epochs`: 训练轮数 (默认50)
- `--lr`: 学习率 (默认0.001)
- `--weight_decay`: 权重衰减 (默认1e-4)

完整参数列表请运行: `python train_hypergraph.py --help`

## 数据格式

### MOSEI数据集

```
Data/
├── MOSEIhappylabel0.pkl
├── MOSEIsadlabel1.pkl
└── ...
```

### MELD数据集

```
Data/
├── MELD_trainhappylabel0.pkl
├── MELD_devhappylabel0.pkl
├── MELD_testhappylabel0.pkl
└── ...
```

### 样本格式

每个pkl文件包含列表或字典，每个样本是字典：

```python
{
    'audio_features': torch.Tensor,  # (seq_len, audio_dim)
    'text_features': torch.Tensor,   # (seq_len, text_dim)
    'video_features': torch.Tensor,  # (seq_len, video_dim)
    'label': int                     # 类别标签
}
```

## 性能

### 推荐配置

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Batch Size**: 16-64
- **Learning Rate**: 0.001 (with decay)
- **K Neighbors**: 3-7
- **Temperature**: 0.05-0.1

### 优化建议

1. **提高准确率**:
   - 增加训练轮数
   - 调整K值和温度参数
   - 增加模型容量

2. **减少过拟合**:
   - 增加dropout
   - 使用更强的对比学习 (降低temperature)
   - 数据增强

3. **加速训练**:
   - 增加num_workers
   - 使用混合精度训练
   - 优化batch size

## 文档

- [快速开始指南](QUICKSTART.md) - 快速上手教程
- [数据加载器文档](DATALOADER_README.md) - 数据加载详细说明
- [超图网络文档](HYPERGRAPH_README.md) - 网络架构详细说明

## 开发计划

- [ ] 支持更多数据集
- [ ] 添加更多评估指标
- [ ] 可视化工具
- [ ] 预训练模型
- [ ] TensorBoard支持
- [ ] 模型导出 (ONNX)

## 引用

如果你使用了这个项目，请引用相关论文：

```bibtex
@article{hypergraph_multimodal,
  title={Hypergraph-based Multimodal Fusion with Contrastive Learning},
  author={...},
  year={2024}
}
```

## 许可证

MIT License

## 致谢

- PyTorch团队
- 超图神经网络相关研究
- 对比学习相关研究

---

**Happy Coding! 🚀**
