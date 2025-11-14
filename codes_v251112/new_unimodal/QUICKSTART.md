# 快速开始指南

## 3 步开始使用自动配置特征提取

### 第 1 步：配置数据集路径

编辑 `dataset_paths.json`：

```bash
vim dataset_paths.json  # 或使用你喜欢的编辑器
```

修改以下字段：

```json
{
  "mosei": {
    "base_dir": "/YOUR/PATH/TO/MOSEI",                    ← 修改这里
    "label_file": "/YOUR/PATH/TO/MOSEI/label/label.csv",  ← 修改这里
    "output_dir": "./output/mosei_features_hybrid"
  },
  "meld": {
    "base_dir": "/YOUR/PATH/TO/MELD",                     ← 修改这里
    "output_dir": "./output/meld_features_hybrid",
    "split": "all"
  }
}
```

### 第 2 步：配置模型路径

编辑 `config_hybrid.json`：

```bash
vim config_hybrid.json
```

修改模型路径：

```json
{
  "text": {
    "model_path": "/YOUR/PATH/TO/MiniLM-L6-v2"              ← 修改这里
  },
  "audio": {
    "model_path": "/YOUR/PATH/TO/hubert-base-ls960"        ← 修改这里
  },
  "video": {
    "model_path": "/YOUR/PATH/TO/vit-small-patch16-224"    ← 修改这里
  }
}
```

### 第 3 步：运行特征提取

```bash
# MOSEI 数据集
python batch_extract_hybrid_auto.py --dataset mosei --train_pca

# MELD 数据集
python batch_extract_hybrid_auto.py --dataset meld --train_pca
```

就这么简单！🎉

---

## 常用命令

### 提取 MOSEI

```bash
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py --dataset mosei --train_pca
```

### 提取 MELD（全部划分）

```bash
python batch_extract_hybrid_auto.py --dataset meld --train_pca
```

### 提取 MELD（单个划分）

```bash
# 仅训练集
python batch_extract_hybrid_auto.py --dataset meld --split train --train_pca

# 仅测试集
python batch_extract_hybrid_auto.py --dataset meld --split test --train_pca
```

### 使用已训练的 PCA 模型

```bash
python batch_extract_hybrid_auto.py \
  --dataset meld \
  --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl
```

---

## 临时覆盖配置

如果你想临时使用不同的路径（不修改 JSON 文件）：

```bash
python batch_extract_hybrid_auto.py \
  --dataset mosei \
  --base_dir /temporary/path/to/MOSEI \
  --output_dir /temporary/output \
  --train_pca
```

命令行参数会覆盖 JSON 配置！

---

## 验证配置

运行配置检查工具：

```bash
python check_config.py
```

这会检查：
- ✓ JSON 配置文件是否存在
- ✓ 模型路径是否有效
- ✓ 数据集路径是否存在
- ✓ 依赖包是否安装
- ✓ GPU 是否可用

---

## 输出说明

### MOSEI 输出

```
output/mosei_features_hybrid/
├── audio_pca_model.pkl          # PCA 模型
├── MOSEIhappylabel0.pkl        # 各情感特征
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
├── audio_pca_model.pkl              # PCA 模型
├── MELD_trainhappylabel0.pkl       # train 集
├── MELD_trainsadlabel1.pkl
├── MELD_devhappylabel0.pkl         # dev 集
└── MELD_testhappylabel0.pkl        # test 集
```

每个 `.pkl` 文件包含：
- `audio_features`: [T, 384] 音频特征
- `text_features`: [T, 384] 文本特征
- `video_features`: [T, 384] 视频特征
- `label`: 情感标签 ID (0-6)
- `emotion`: 情感名称
- `sample_id`: 样本 ID
- `num_frames`: 帧数

---

## 性能参考

| 数据集 | 样本数 | PCA 训练 | 特征提取 | 文件大小 | GPU 显存 |
|--------|--------|----------|----------|----------|----------|
| MOSEI  | 22,856 | ~5 分钟  | ~5-6 小时| ~30 GB   | 24GB 推荐|
| MELD   | 13,708 | ~3 分钟  | ~3-4 小时| ~18 GB   | 24GB 推荐|

---

## 需要帮助？

- 查看详细文档: `USAGE.md`
- 查看示例脚本: `./example_usage_auto.sh`
- 查看完整说明: `README.md`
- 运行帮助命令: `python batch_extract_hybrid_auto.py --help`

---

## 故障排除

### 错误：base_dir 未配置

```
错误：必须通过命令行参数或 JSON 配置提供 base_dir
```

**解决**：编辑 `dataset_paths.json`，设置正确的 `base_dir`

### 错误：label_file 未配置（MOSEI）

```
错误：MOSEI 数据集必须提供 label_file
```

**解决**：编辑 `dataset_paths.json`，设置 `mosei.label_file`

### CUDA 内存不足

**解决**：编辑 `config_hybrid.json`，减小 `max_frames`：

```json
"memory_management": {
  "max_frames": 300,  // 从 500 改为 300
  ...
}
```

---

## 对比

### 原版（手动配置）

```bash
python batch_extract_hybrid.py \
  --dataset mosei \
  --config config_hybrid.json \
  --base_dir /path/to/MOSEI \
  --label_file /path/to/MOSEI/label/label.csv \
  --output_dir ./output \
  --train_pca \
  --pca_training_samples 1000
```

### 新版（自动配置）

```bash
python batch_extract_hybrid_auto.py --dataset mosei --train_pca
```

**节省 80% 的命令行输入！** ✨
