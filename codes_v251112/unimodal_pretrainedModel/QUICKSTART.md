# 快速开始指南

## 2 步完成特征提取

### 步骤 1：编辑配置文件

打开 `extraction_settings.json`，修改以下内容：

```json
{
  "dataset": {
    "name": "mosei"  // 或 "meld"
  },

  "mosei": {
    "base_dir": "/your/path/to/MOSEI",           // ← 修改这里
    "label_file": "/your/path/to/label.csv",     // ← 修改这里
    "enabled": true
  },

  "models": {
    "text": {
      "model_path": "/your/path/to/MiniLM-L6-v2"  // ← 修改这里
    },
    "audio": {
      "model_path": "/your/path/to/hubert"        // ← 修改这里
    },
    "video": {
      "model_path": "/your/path/to/vit-small"     // ← 修改这里
    }
  }
}
```

### 步骤 2：运行提取

```bash
cd codes_v251112/unimodal_pretrainedModel
python extract_features.py
```

就这么简单！🎉

---

## 预期结果

```
提取 MOSEI：
- 速度：约 1.5-2 秒/样本
- 总时间：9-12 小时
- 输出：./output/mosei_features_5fps/
- 文件大小：约 15 GB

提取 MELD：
- 速度：约 1.5-2 秒/样本
- 总时间：5-7 小时
- 输出：./output/meld_features_5fps/
- 文件大小：约 9 GB
```

---

## 切换到 MELD 数据集

修改 `extraction_settings.json`：

```json
{
  "dataset": {
    "name": "meld"  // ← 改为 meld
  },

  "mosei": {
    "enabled": false  // ← 关闭 MOSEI
  },

  "meld": {
    "base_dir": "/your/path/to/MELD",  // ← 设置路径
    "split": "all",                     // all/train/dev/test
    "enabled": true                     // ← 启用 MELD
  }
}
```

然后再次运行：

```bash
python extract_features.py
```

---

## 常见问题

**Q: 如何查看进度？**

```bash
tail -f extraction_*.log
```

**Q: 提取太慢？**

修改采样率（牺牲一些精度）：

```json
{
  "extraction": {
    "sampling_rate_fps": 3  // 从 5 改为 3，更快
  }
}
```

**Q: 内存不足？**

```json
{
  "extraction": {
    "video_batch_size": 32  // 从 64 改为 32
  }
}
```

---

## 完整示例

```bash
# 1. 克隆/拉取代码
cd /home/user/openMM
git checkout claude/mmsa-feature-extraction-demo-01JV9tGzV7NbLuia1H57NX31

# 2. 进入目录
cd codes_v251112/unimodal_pretrainedModel

# 3. 编辑配置（设置路径）
vim extraction_settings.json

# 4. 运行提取
python extract_features.py

# 5. 监控进度
tail -f extraction_*.log

# 6. 完成后查看输出
ls -lh output/mosei_features_5fps/
```

---

## 输出文件说明

每个 `.pkl` 文件包含一个列表，每个元素是一个字典：

```python
{
    'audio_features': numpy.array([T, 384]),   # 音频特征
    'text_features': numpy.array([T, 384]),    # 文本特征
    'video_features': numpy.array([T, 384]),   # 视频特征
    'label': int,                               # 0-6 (情感ID)
    'emotion': str,                             # happy/sad/...
    'sample_id': str,                           # 样本ID
    'num_frames': int                           # 帧数T
}
```

**使用示例**：

```python
import pickle

with open('output/mosei_features_5fps/MOSEIhappylabel0.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Happy 样本数: {len(data)}")
print(f"第一个样本帧数: {data[0]['num_frames']}")
print(f"音频特征形状: {data[0]['audio_features'].shape}")  # (T, 384)
```

---

## 性能对比

| 版本 | 采样率 | 单样本速度 | MOSEI总时间 | 文件大小 |
|------|--------|-----------|------------|----------|
| 旧版 | ~22fps | 8.78秒 | 56小时 | 60GB |
| 优化后 | 10fps | 2.5秒 | 16小时 | 30GB |
| **当前** | **5fps** | **1.8秒** | **11小时** | **15GB** |

提速 **5倍**，文件减小 **75%** ！

---

需要更多帮助？查看 [README.md](README.md)
