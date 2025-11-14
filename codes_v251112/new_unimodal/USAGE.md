# 自动配置特征提取使用指南

## 概述

`batch_extract_hybrid_auto.py` 提供了自动从 JSON 配置文件读取参数的功能，简化了特征提取流程。

## 配置文件

系统使用三个 JSON 配置文件：

1. **config_hybrid.json** - 模型配置（文本、音频、视频模型路径）
2. **dataset_paths.json** - 数据集路径配置（MOSEI 和 MELD 的路径）
3. **extraction_config.json** - 提取参数配置（PCA 训练样本数、批处理大小等）

## 快速开始

### 1. 配置数据集路径

编辑 `dataset_paths.json`，设置实际的数据集路径：

```json
{
  "mosei": {
    "base_dir": "/path/to/MOSEI",           // 修改为实际路径
    "label_file": "/path/to/MOSEI/label/label.csv",
    "output_dir": "./output/mosei_features_hybrid"
  },
  "meld": {
    "base_dir": "/path/to/MELD",            // 修改为实际路径
    "output_dir": "./output/meld_features_hybrid",
    "split": "all"
  }
}
```

### 2. 配置模型路径

编辑 `config_hybrid.json`，设置模型路径：

```json
{
  "text": {
    "model_path": "/path/to/models/MiniLM-L6-v2"  // 修改
  },
  "audio": {
    "model_path": "/path/to/models/hubert-base-ls960"  // 修改
  },
  "video": {
    "model_path": "/path/to/models/vit-small-patch16-224"  // 修改
  }
}
```

### 3. 运行提取（推荐方式）

配置好 JSON 文件后，只需简单的命令即可运行：

```bash
# MOSEI 数据集
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py --dataset mosei --train_pca

# MELD 数据集（全部划分）
python batch_extract_hybrid_auto.py --dataset meld --train_pca

# MELD 数据集（单个划分）
python batch_extract_hybrid_auto.py --dataset meld --split train --train_pca
```

## 高级用法

### 覆盖 JSON 配置

命令行参数会覆盖 JSON 配置：

```bash
# 使用自定义的 base_dir 和 output_dir
python batch_extract_hybrid_auto.py --dataset mosei \
    --base_dir /custom/path/to/MOSEI \
    --output_dir /custom/output \
    --train_pca

# 使用不同的 PCA 训练样本数
python batch_extract_hybrid_auto.py --dataset mosei \
    --pca_training_samples 2000 \
    --train_pca
```

### 使用预训练的 PCA 模型

如果已经训练好 PCA 模型，可以直接使用：

```bash
python batch_extract_hybrid_auto.py --dataset mosei \
    --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl
```

### 使用自定义配置文件

```bash
python batch_extract_hybrid_auto.py --dataset mosei \
    --dataset_config my_custom_dataset_paths.json \
    --extraction_config my_custom_extraction_config.json \
    --train_pca
```

## 参数优先级

参数按以下优先级使用：

1. **命令行参数** （最高优先级）
2. **JSON 配置文件**
3. **程序默认值** （最低优先级）

例如：
- JSON 中设置 `base_dir: "/path/to/MOSEI"`
- 命令行使用 `--base_dir /custom/path`
- 最终使用：`/custom/path`（命令行覆盖 JSON）

## 命令行参数说明

### 必需参数

- `--dataset`: 数据集名称（mosei 或 meld）

### 配置文件参数（默认值已设置）

- `--config`: 模型配置文件（默认: config_hybrid.json）
- `--dataset_config`: 数据集路径配置（默认: dataset_paths.json）
- `--extraction_config`: 提取参数配置（默认: extraction_config.json）

### 可选参数（覆盖 JSON 配置）

- `--base_dir`: 数据集根目录
- `--label_file`: 标签文件路径（仅 MOSEI）
- `--output_dir`: 输出目录
- `--split`: MELD 数据集划分（train/dev/test/all）
- `--pca_model_path`: 预训练 PCA 模型路径
- `--train_pca`: 是否训练 PCA 模型
- `--pca_training_samples`: PCA 训练样本数量

## 与原版对比

### 原版（batch_extract_hybrid.py）

需要手动指定所有参数：

```bash
python batch_extract_hybrid.py \
    --dataset mosei \
    --config new_unimodal/config_hybrid.json \
    --base_dir /path/to/MOSEI \
    --label_file /path/to/MOSEI/label/label.csv \
    --output_dir ./output/mosei_features_hybrid \
    --train_pca \
    --pca_training_samples 1000
```

### 自动版（batch_extract_hybrid_auto.py）

自动从 JSON 读取配置：

```bash
python batch_extract_hybrid_auto.py --dataset mosei --train_pca
```

## 故障排除

### 错误：必须提供 base_dir

```
错误：必须通过命令行参数或 JSON 配置提供 base_dir
请在 dataset_paths.json 中设置 mosei.base_dir
```

**解决方法**：编辑 `dataset_paths.json`，设置正确的 `base_dir` 路径。

### 错误：MOSEI 数据集必须提供 label_file

```
错误：MOSEI 数据集必须提供 label_file
请在 dataset_paths.json 中设置 mosei.label_file 或使用 --label_file 参数
```

**解决方法**：编辑 `dataset_paths.json`，设置 `label_file` 路径，或在命令行使用 `--label_file` 参数。

### 配置文件未找到

```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset_paths.json'
```

**解决方法**：确保在 `codes_v251112/new_unimodal/` 目录下运行脚本，或使用 `--dataset_config` 指定完整路径。

## 输出说明

提取完成后，特征文件保存在 `output_dir` 中：

### MOSEI 输出

```
output/mosei_features_hybrid/
├── audio_pca_model.pkl          # PCA 模型（如果训练）
├── MOSEIhappylabel0.pkl        # 各情感的特征文件
├── MOSEIsadlabel1.pkl
├── MOSEIangerlabel2.pkl
└── ...
```

### MELD 输出

```
output/meld_features_hybrid/
├── audio_pca_model.pkl              # PCA 模型（如果训练）
├── MELD_trainhappylabel0.pkl       # train 集特征
├── MELD_trainsadlabel1.pkl
├── MELD_devhappylabel0.pkl         # dev 集特征
├── MELD_testhappylabel0.pkl        # test 集特征
└── ...
```

## 性能提示

1. **首次运行**：使用 `--train_pca` 训练 PCA 模型
2. **后续运行**：使用 `--pca_model_path` 加载已训练的模型
3. **调整样本数**：根据 GPU 显存调整 `pca_training_samples`（默认 1000）
4. **内存不足**：修改 `config_hybrid.json` 中的 `max_frames`（默认 500）

## 完整示例

```bash
# 1. 首次提取 MOSEI，训练 PCA
python batch_extract_hybrid_auto.py --dataset mosei --train_pca

# 2. 提取 MELD，复用 MOSEI 的 PCA 模型
python batch_extract_hybrid_auto.py --dataset meld \
    --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl

# 3. 仅提取 MELD test 集
python batch_extract_hybrid_auto.py --dataset meld --split test \
    --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl

# 4. 使用高质量预设（修改 extraction_config.json 中的 pca_samples: 2000）
python batch_extract_hybrid_auto.py --dataset mosei \
    --pca_training_samples 2000 --train_pca
```
