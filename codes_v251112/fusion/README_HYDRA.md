# 使用 Hydra 配置框架训练样本级别超图网络

## 📋 概述

这个版本使用 **Hydra** 配置框架，提供更强大的配置管理功能，包括：
- 命令行覆盖配置参数
- 配置组合
- 多运行实验
- 更灵活的参数管理

## 🔧 Hydra vs PyYAML

| 特性 | PyYAML 版本 | Hydra 版本 |
|------|------------|-----------|
| 配置文件 | `config.yaml` | `config/config_sample_hypergraph.yaml` |
| 训练脚本 | `train_sample_hypergraph.py` | `train_sample_hypergraph_hydra.py` |
| 配置访问 | `config.get('key', default)` | `cfg.key` |
| 命令行覆盖 | ❌ 不支持 | ✅ 支持 |
| 配置组合 | ❌ 不支持 | ✅ 支持 |
| 工作目录 | 不变 | **会改变** (需要 `os.chdir(exc_dir)`) |

## 📁 文件结构

```
fusion/
├── config/                                  # Hydra 配置目录 ⭐
│   └── config_sample_hypergraph.yaml       # 主配置文件
│
├── train_sample_hypergraph_hydra.py        # Hydra 训练脚本 ⭐
├── dataloader.py                           # 数据加载器 (已添加 Hydra 支持)
└── sample_network.py                       # 网络模型
```

## 🚀 基本使用

### 1. 使用默认配置

```bash
python codes_v251112/fusion/train_sample_hypergraph_hydra.py
```

### 2. 命令行覆盖配置

这是 Hydra 的核心优势！你可以在命令行直接修改任何配置参数：

```bash
# 修改学习率
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    training.learning_rate=0.001

# 修改 batch size
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    dataloader.batch_size=64

# 修改多个参数
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    training.learning_rate=0.001 \
    dataloader.batch_size=64 \
    training.epochs=100

# 修改实验名称
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    experiment.name=experiment_lr001_bs64
```

### 3. 修改 seen/unseen emotions

```bash
# 训练 happy vs sad vs angry (三分类)
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    dataset.seen_emotions.angry=2

# 移除 unseen emotions
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    dataset.unseen_emotions=null

# 修改 unseen emotion
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    dataset.unseen_emotions.disgust=4
```

### 4. 修改模型参数

```bash
# 修改超图卷积层数
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    model.hypergraph.num_conv_layers=3

# 修改池化类型
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    model.pooling.pooling_type=max

# 禁用边权重
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    model.sample_hypergraph.use_edge_weights=false

# 修改相似度温度
python codes_v251112/fusion/train_sample_hypergraph_hydra.py \
    model.sample_hypergraph.similarity_temperature=2.0
```

## 📝 配置文件格式

### Hydra 配置文件示例

```yaml
# config/config_sample_hypergraph.yaml

dataset:
  name: MELD
  data_dir: ./output/meld_utterance_features
  seen_emotions:
    happy: 0
    sad: 1
  unseen_emotions:
    fear: 5

model:
  encoder:
    hidden_dim: 256
    output_dim: 256
  pooling:
    pooling_type: masked_mean

training:
  epochs: 50
  learning_rate: 0.0001
```

### 在代码中访问配置

```python
@hydra.main(config_path="config", config_name="config_sample_hypergraph", version_base=None)
def run_main(cfg: DictConfig):
    # Hydra 会改变工作目录，需要切换回来
    os.chdir(exc_dir)

    # 访问配置 (使用点号)
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.dataloader.batch_size

    # 访问嵌套配置
    hidden_dim = cfg.model.encoder.hidden_dim

    # 访问字典型配置
    seen_emotions = dict(cfg.dataset.seen_emotions)
    # {'happy': 0, 'sad': 1}

    # 打印完整配置
    print(OmegaConf.to_yaml(cfg))
```

## 🔍 关键代码说明

### 1. 装饰器

```python
@hydra.main(config_path="config", config_name="config_sample_hypergraph", version_base=None)
def run_main(cfg: DictConfig):
    pass
```

- `config_path`: 配置文件所在目录
- `config_name`: 配置文件名（不含 `.yaml` 后缀）
- `version_base=None`: 使用最新版本的 Hydra

### 2. 工作目录问题 ⚠️

**重要**：Hydra 会自动改变工作目录到输出目录！

```python
# 保存原始执行目录
exc_dir = os.getcwd()

@hydra.main(...)
def run_main(cfg: DictConfig):
    # 切换回原始目录
    os.chdir(exc_dir)
```

如果不切换回来，数据路径会找不到！

### 3. 配置访问

```python
# ✅ Hydra 方式 (推荐)
learning_rate = cfg.training.learning_rate

# ✅ 带默认值
learning_rate = cfg.training.get('learning_rate', 0.0001)

# ❌ PyYAML 方式 (不兼容)
learning_rate = config.training.get('learning_rate', 0.0001)
```

### 4. 动态修改配置

```python
@hydra.main(...)
def run_main(cfg: DictConfig):
    # 在代码中动态修改配置
    cfg.dataset.missing_type = 'text'

    # 使用 OmegaConf 修改
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)  # 允许添加新键
    cfg.new_param = 'value'
    OmegaConf.set_struct(cfg, True)   # 恢复结构限制
```

## 💡 高级用法

### 1. 配置组 (Config Groups)

创建多个配置变体：

```
config/
├── config_sample_hypergraph.yaml    # 主配置
├── dataset/
│   ├── meld.yaml                    # MELD 数据集配置
│   └── mosei.yaml                   # MOSEI 数据集配置
└── model/
    ├── small.yaml                   # 小模型
    └── large.yaml                   # 大模型
```

使用：
```bash
python train_sample_hypergraph_hydra.py \
    dataset=mosei \
    model=large
```

### 2. 多运行实验 (Multirun)

自动运行多组实验：

```bash
# 测试不同学习率
python train_sample_hypergraph_hydra.py \
    --multirun \
    training.learning_rate=0.0001,0.001,0.01

# 测试不同 batch size 和学习率的组合
python train_sample_hypergraph_hydra.py \
    --multirun \
    dataloader.batch_size=16,32,64 \
    training.learning_rate=0.0001,0.001
```

这会自动运行 3×2=6 个实验！

### 3. 配置继承

```yaml
# config/base_config.yaml
defaults:
  - override hydra/launcher: basic

dataset:
  name: MELD
  train_ratio: 0.7

# config/config_happy_sad.yaml
defaults:
  - base_config

dataset:
  seen_emotions:
    happy: 0
    sad: 1
```

## 📊 常见使用场景

### 场景 1: 快速调试

```bash
# 使用小 batch size 和少量 epochs 快速测试
python train_sample_hypergraph_hydra.py \
    dataloader.batch_size=4 \
    training.epochs=2 \
    experiment.name=debug
```

### 场景 2: 超参数调优

```bash
# 测试不同的超图配置
python train_sample_hypergraph_hydra.py \
    --multirun \
    model.hypergraph.num_conv_layers=1,2,3 \
    model.sample_hypergraph.similarity_temperature=0.5,1.0,2.0
```

### 场景 3: 不同情感组合

```bash
# Happy vs Sad
python train_sample_hypergraph_hydra.py \
    dataset.seen_emotions='{happy: 0, sad: 1}' \
    experiment.name=happy_vs_sad

# Happy vs Sad vs Angry
python train_sample_hypergraph_hydra.py \
    dataset.seen_emotions='{happy: 0, sad: 1, angry: 2}' \
    experiment.name=three_class
```

### 场景 4: 修改数据路径

```bash
# 使用 MOSEI 数据集
python train_sample_hypergraph_hydra.py \
    dataset.name=MOSEI \
    dataset.data_dir=./output/mosei_utterance_features \
    experiment.name=mosei_experiment
```

## 🔧 与 PyYAML 版本的差异

### 配置读取

**PyYAML 版本**:
```python
from config_utils import load_config

config = load_config('config.yaml')
learning_rate = config.training.get('learning_rate', 1e-4)
```

**Hydra 版本**:
```python
@hydra.main(config_path="config", config_name="config_sample_hypergraph")
def run_main(cfg: DictConfig):
    learning_rate = cfg.training.learning_rate  # 直接访问
```

### 字典访问

**PyYAML 版本**:
```python
seen_emotions = config.get_seen_emotions()  # 自定义方法
```

**Hydra 版本**:
```python
seen_emotions = dict(cfg.dataset.seen_emotions)  # 转为字典
```

### 工作目录

**PyYAML 版本**: 工作目录不变

**Hydra 版本**:
```python
exc_dir = os.getcwd()  # 保存原始目录

@hydra.main(...)
def run_main(cfg: DictConfig):
    os.chdir(exc_dir)  # 切换回来
```

## 📚 Hydra 资源

- 官方文档: https://hydra.cc/
- 快速入门: https://hydra.cc/docs/intro/
- 配置组: https://hydra.cc/docs/tutorials/structured_config/config_groups/
- 多运行: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

## ⚠️ 注意事项

1. **工作目录**：Hydra 会改变工作目录，记得 `os.chdir(exc_dir)`

2. **配置路径**：配置文件必须在 `config/` 目录下

3. **版本兼容**：使用 `version_base=None` 以兼容最新版本

4. **字典转换**：`cfg.dataset.seen_emotions` 返回的是 DictConfig，需要 `dict()` 转换

5. **命令行覆盖**：使用 `key=value` 格式，嵌套用点号分隔 `parent.child=value`

## 🎯 推荐使用

- **开发阶段**: 使用 Hydra，方便快速调参
- **生产部署**: 可以用 PyYAML 版本，更简单稳定
- **实验研究**: 强烈推荐 Hydra 的 multirun 功能
