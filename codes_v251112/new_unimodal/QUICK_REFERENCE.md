# 混合特征提取快速参考

## 📋 基本信息

**配置**:
- 文本: MiniLM-L6-v2 (384d)
- 音频: HuBERT-base (768d) → PCA (384d)
- 视频: google/vit-small-patch16-224 (384d)

**效果**:
- 文件大小: 减少 50% (60GB → 30GB)
- 准确率损失: <1%
- 所有模态: 统一 384 维

## 🚀 快速开始

### MOSEI 数据集

```bash
python unimodal_features/batch_extract_hybrid.py \
  --dataset mosei \
  --config unimodal_features/config_hybrid_pca.json \
  --base_dir /path/to/MOSEI \
  --label_file /path/to/MOSEI/label/label.csv \
  --output_dir ./output/mosei_features_hybrid \
  --train_pca \
  --pca_training_samples 1000
```

**说明**:
- `--dataset mosei`: 指定 MOSEI 数据集
- `--label_file`: MOSEI 需要提供标签文件路径
- `--train_pca`: 自动训练音频 PCA 模型
- `--pca_training_samples 1000`: 使用 1000 个样本训练 PCA

### MELD 数据集

```bash
python unimodal_features/batch_extract_hybrid.py \
  --dataset meld \
  --config unimodal_features/config_hybrid_pca.json \
  --base_dir /path/to/MELD \
  --output_dir ./output/meld_features_hybrid \
  --split all \
  --train_pca \
  --pca_training_samples 1000
```

**说明**:
- `--dataset meld`: 指定 MELD 数据集
- `--split all`: 处理所有划分 (train/dev/test)
- 不需要 `--label_file`（MELD 每个划分有独立 label.csv）

**划分选择**:
- `--split all`: 全部划分（推荐）
- `--split train`: 仅训练集
- `--split dev`: 仅验证集
- `--split test`: 仅测试集

## 📂 数据集结构

### MOSEI 结构

```
MOSEI/
├── label/
│   └── label.csv              # 全局标签文件
├── audio/
│   └── {video_id}/
│       └── {clip_id}.wav
└── video/
    └── {video_id}/
        └── {clip_id}.mp4
```

### MELD 结构

```
MELD/
├── train/
│   ├── label.csv              # 训练集标签
│   ├── audio/
│   │   └── {file_id}.wav
│   └── video/
│       └── {file_id}.mp4
├── dev/
│   ├── label.csv              # 验证集标签
│   ├── audio/
│   └── video/
└── test/
    ├── label.csv              # 测试集标签
    ├── audio/
    └── video/
```

## 📤 输出结果

### MOSEI 输出

```
output/mosei_features_hybrid/
├── audio_pca_model.pkl           # PCA 模型
├── MOSEIhappylabel0.pkl
├── MOSEIsadlabel1.pkl
├── MOSEIangerlabel2.pkl
├── MOSEIdisgustlabel3.pkl
├── MOSEIsurpriselabel4.pkl
├── MOSEIfearlabel5.pkl
└── MOSEIneutrallabel6.pkl
```

### MELD 输出

```
output/meld_features_hybrid/
├── audio_pca_model.pkl           # PCA 模型
├── MELD_trainhappylabel0.pkl     # 训练集
├── MELD_trainsadlabel1.pkl
├── ...
├── MELD_devhappylabel0.pkl       # 验证集
├── MELD_devsadlabel1.pkl
├── ...
├── MELD_testhappylabel0.pkl      # 测试集
└── MELD_testsadlabel1.pkl
```

## 🔧 配置文件

### 修改模型路径

编辑 `unimodal_features/config_hybrid_pca.json`:

```json
{
  "text": {
    "model_path": "/path/to/MiniLM-L6-v2",
    "output_dim": 384
  },
  "audio": {
    "model_path": "/path/to/hubert-base-ls960",
    "output_dim": 768,
    "pca_reduction": {
      "enabled": true,
      "target_dim": 384
    }
  },
  "video": {
    "model_path": "/path/to/vit-small-patch16-224",
    "output_dim": 384
  },
  "memory_management": {
    "max_frames": 500,
    "video_batch_size": 32,
    "enable_memory_cleanup": true
  }
}
```

将 `TO_BE_SPECIFIED` 或 `/path/to/` 替换为实际路径。

## ⚙️ 参数说明

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--dataset` | ✅ | 数据集类型 | mosei, meld |
| `--config` | ❌ | 配置文件路径 | config_hybrid_pca.json |
| `--base_dir` | ✅ | 数据集根目录 | /path/to/MOSEI |
| `--label_file` | MOSEI 需要 | 标签文件路径 | /path/to/label.csv |
| `--output_dir` | ✅ | 输出目录 | ./output/features |
| `--split` | MELD 可选 | 处理的划分 | all, train, dev, test |
| `--train_pca` | ❌ | 训练 PCA 模型 | (flag) |
| `--pca_training_samples` | ❌ | PCA 训练样本数 | 1000 (默认) |
| `--pca_model_path` | ❌ | 预训练 PCA 路径 | ./audio_pca_model.pkl |

## 📊 性能预估

### MOSEI (22,856 样本)

| 阶段 | 时间 | 说明 |
|------|------|------|
| PCA 训练 | ~5 分钟 | 使用 1000 样本 |
| 特征提取 | ~5-6 小时 | 24GB GPU |
| 文件大小 | 30 GB | 减少 50% |

### MELD (13,708 样本)

| 阶段 | 时间 | 说明 |
|------|------|------|
| PCA 训练 | ~3 分钟 | 使用 1000 样本 |
| 特征提取 | ~3-4 小时 | 24GB GPU |
| 文件大小 | ~18 GB | 减少 50% |

## 🔍 使用预训练 PCA

如果已经训练过 PCA 模型，可以直接使用：

```bash
# 使用 MOSEI 训练的 PCA 模型提取 MELD
python unimodal_features/batch_extract_hybrid.py \
  --dataset meld \
  --config unimodal_features/config_hybrid_pca.json \
  --base_dir /path/to/MELD \
  --output_dir ./output/meld_features_hybrid \
  --split all \
  --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl
```

**注意**: MOSEI 和 MELD 的 PCA 模型可以通用（都是 HuBERT 768d → 384d）

## 💻 加载和使用特征

```python
import pickle

# 加载 MOSEI 特征
with open('output/mosei_features_hybrid/MOSEIhappylabel0.pkl', 'rb') as f:
    mosei_samples = pickle.load(f)

# 加载 MELD 特征
with open('output/meld_features_hybrid/MELD_trainhappylabel0.pkl', 'rb') as f:
    meld_samples = pickle.load(f)

# 查看特征
for sample in mosei_samples[:3]:
    print(f"样本: {sample['sample_id']}")
    print(f"  文本: {sample['text_features'].shape}")    # [T, 384]
    print(f"  音频: {sample['audio_features'].shape}")   # [T, 384]
    print(f"  视频: {sample['video_features'].shape}")   # [T, 384]
    print(f"  标签: {sample['label']}")
    print(f"  帧数: {sample['num_frames']}")
```

## 🐛 常见问题

### Q1: MOSEI 缺少 --label_file

**错误**: `MOSEI 数据集需要提供 --label_file 参数`

**解决**: 添加 `--label_file` 参数
```bash
--label_file /path/to/MOSEI/label/label.csv
```

### Q2: MELD 找不到标签文件

**错误**: `标签文件不存在: /path/to/MELD/train/label.csv`

**解决**: 检查 MELD 目录结构
```bash
ls /path/to/MELD/train/
# 应该有: label.csv, audio/, video/
```

### Q3: PCA 模型未训练

**错误**: `RuntimeError: PCA 模型未训练`

**解决**: 添加 `--train_pca` 参数
```bash
--train_pca --pca_training_samples 1000
```

### Q4: 模型路径错误

**错误**: `OSError: Can't load tokenizer for 'TO_BE_SPECIFIED'`

**解决**: 编辑配置文件，将 `TO_BE_SPECIFIED` 替换为实际模型路径

## 📚 相关文档

- **详细指南**: `HYBRID_EXTRACTION_GUIDE.md`
- **降维方法**: `FEATURE_REDUCTION_GUIDE.md`
- **配置文件**: `unimodal_features/config_hybrid_pca.json`

## 🎯 下一步

1. **下载小模型** (MiniLM-L6, ViT-small)
2. **配置模型路径** (编辑 config_hybrid_pca.json)
3. **运行提取命令** (使用上面的快速开始命令)
4. **训练 Hypergraph 模型** (使用提取的 384 维特征)
