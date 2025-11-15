# 自动化特征提取系统

完全自动化的多模态特征提取工具，所有配置通过 JSON 文件管理，无需命令行参数。

## 特性

- ✅ **5 fps 采样率**：快速提取（9-12小时完成 MOSEI）
- ✅ **完全自动化**：无需命令行参数，编辑 JSON 即可
- ✅ **统一 384 维特征**：文本 + 音频 + 视频
- ✅ **自动 PCA 训练**：首次运行自动训练并保存
- ✅ **支持 MOSEI 和 MELD**：两个主流数据集

## 快速开始

### 1. 配置数据集路径

编辑 `extraction_settings.json`：

```json
{
  "dataset": {
    "name": "mosei"  // 或 "meld"
  },

  "mosei": {
    "base_dir": "/your/path/to/MOSEI",
    "label_file": "/your/path/to/MOSEI/label/label.csv",
    "output_dir": "./output/mosei_features_5fps",
    "enabled": true
  },

  "models": {
    "text": {
      "model_path": "/your/path/to/MiniLM-L6-v2"
    },
    "audio": {
      "model_path": "/your/path/to/hubert-base-ls960"
    },
    "video": {
      "model_path": "/your/path/to/vit-small-patch16-224"
    }
  }
}
```

### 2. 直接运行

```bash
cd codes_v251112/unimodal_pretrainedModel
python extract_features.py
```

就这么简单！无需任何命令行参数。

## 配置文件说明

### extraction_settings.json

```json
{
  "dataset": {
    "name": "mosei"  // 选择数据集：mosei 或 meld
  },

  "extraction": {
    "sampling_rate_fps": 5,       // 采样率：5 帧/秒
    "max_frames": 300,             // 最大帧数限制
    "train_pca": true,             // 是否训练 PCA
    "pca_training_samples": 1000   // PCA 训练样本数
  }
}
```

## 性能预期

| 数据集 | 样本数 | 采样率 | 预计时间 | 文件大小 |
|--------|--------|--------|----------|----------|
| MOSEI  | 22,856 | 5 fps  | 9-12小时 | ~15 GB   |
| MELD   | 13,708 | 5 fps  | 5-7小时  | ~9 GB    |

**单样本速度**：约 1.5-2 秒/样本

## 输出格式

### MOSEI

```
output/mosei_features_5fps/
├── audio_pca_model.pkl          # PCA 模型
├── MOSEIhappylabel0.pkl        # 各情感特征
├── MOSEIsadlabel1.pkl
├── MOSEIangerlabel2.pkl
└── ...
```

### MELD

```
output/meld_features_5fps/
├── audio_pca_model.pkl              # PCA 模型
├── MELD_trainhappylabel0.pkl       # train 集
├── MELD_devhappylabel0.pkl         # dev 集
├── MELD_testhappylabel0.pkl        # test 集
└── ...
```

每个 `.pkl` 文件包含：

```python
{
    'audio_features': array([384]),      # 音频特征 (utterance-level)
    'text_features': array([384]),       # 文本特征 (utterance-level)
    'video_features': array([384]),      # 视频特征 (utterance-level)
    'label': int,                         # 情感标签 ID (0-6)
    'emotion': str,                       # 情感名称
    'sample_id': str                      # 样本 ID
}
```

**特征说明**：
- 所有模态都是 **utterance-level**（话语级别）特征
- 每个样本都是固定的 384 维向量
- **文本**：使用 BERT [CLS] token 表示
- **音频**：对所有时间步进行 mean pooling
- **视频**：对所有采样帧进行 mean pooling

## 切换数据集

### 提取 MOSEI

```json
{
  "dataset": {
    "name": "mosei"
  },
  "mosei": {
    "enabled": true
  },
  "meld": {
    "enabled": false
  }
}
```

### 提取 MELD

```json
{
  "dataset": {
    "name": "meld"
  },
  "mosei": {
    "enabled": false
  },
  "meld": {
    "enabled": true,
    "split": "all"  // 或 "train", "dev", "test"
  }
}
```

## 高级配置

### 调整采样率

```json
{
  "extraction": {
    "sampling_rate_fps": 3  // 更快但精度可能降低
    // 3 fps: 更快（约6小时/MOSEI），文件更小（约10GB）
    // 5 fps: 推荐（约10小时/MOSEI），平衡速度和精度
    // 10 fps: 更慢（约15小时/MOSEI），更高精度
  }
}
```

### 使用预训练的 PCA 模型

```json
{
  "extraction": {
    "train_pca": false,
    "pca_model_path": "./output/mosei_features_5fps/audio_pca_model.pkl"
  }
}
```

## 技术细节

### 特征聚合策略（Utterance-Level）

```
所有模态都使用 utterance-level 表示：

文本（MiniLM-L6）：
- 输入: "I love this movie" → [CLS] token embedding
- 输出: [384] 维向量（固定）

音频（HuBERT + PCA）：
- 输入: 音频文件 → HuBERT提取 → [time_steps, 768]
- PCA降维: [time_steps, 384]
- Mean Pooling: [384] 维向量（固定）

视频（ViT-small）：
- 输入: 视频文件 → 5fps采样 → [num_frames, 384]
- Mean Pooling: [384] 维向量（固定）
```

### 优化说明

1. **文本特征**：使用 [CLS] token，代表整句话的语义
2. **音频特征**：对所有时间步做平均池化，捕获整体声学特征
3. **视频特征**：5fps采样后平均池化，捕获整体视觉信息
4. **统一维度**：所有模态都是固定的 384 维向量，便于融合
5. **计算高效**：utterance-level 比 frame-level 更快，内存占用更小

## 故障排除

### 错误：配置文件不存在

确保在 `unimodal_pretrainedModel` 目录下运行：

```bash
cd codes_v251112/unimodal_pretrainedModel
python extract_features.py
```

### 错误：数据集未启用

在 `extraction_settings.json` 中设置：

```json
{
  "mosei": {
    "enabled": true  // 设置为 true
  }
}
```

### CUDA 内存不足

降低 batch size：

```json
{
  "extraction": {
    "video_batch_size": 32  // 从 64 改为 32
  }
}
```

## 文件说明

```
unimodal_pretrainedModel/
├── extraction_settings.json      # 配置文件（编辑这个）
├── extract_features.py           # 主程序（运行这个）
├── feature_extractor_hybrid.py   # 核心提取器
└── README.md                      # 本文件
```

## 与旧版本对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 配置方式 | 命令行参数 | JSON 配置文件 |
| 使用难度 | 需要记忆参数 | 编辑 JSON 即可 |
| 采样率 | 手动计算 | 自动处理 |
| 运行命令 | 长命令（7+参数）| `python extract_features.py` |

## 示例工作流

```bash
# 1. 编辑配置
vim extraction_settings.json

# 2. 运行提取
python extract_features.py

# 3. 查看日志
tail -f extraction_*.log

# 4. 完成后查看输出
ls -lh output/mosei_features_5fps/
```

就是这么简单！🎉
