# 快速开始指南

## 步骤 1: 配置

编辑 `extraction_settings.json`:

```json
{
  "mosei": {
    "base_dir": "/your/path/to/MOSEI",
    "label_file": "/your/path/to/label.csv",
    "enabled": true
  },
  "models": {
    "text": {"model_path": "/path/to/roberta-base"},
    "audio": {"model_path": "/path/to/hubert-base-ls960"},
    "video": {"model_path": "/path/to/vit-base-patch16-224"}
  }
}
```

## 步骤 2: 运行

```bash
python extract_features.py
```

## 输出

```python
{
    'text_features': array([768]),    # RoBERTa
    'audio_features': array([768]),   # HuBERT  
    'video_features': array([768]),   # ViT-Base
    'label': 0,
    'emotion': 'happy',
    'sample_id': 'xxx'
}
```

## 技术细节

| 模态 | 模型 | 方法 | 维度 |
|------|------|------|------|
| 文本 | RoBERTa-base | [CLS] token | 768 |
| 音频 | HuBERT-base | Mean pooling | 768 |
| 视频 | ViT-Base | Mean pooling (5fps) | 768 |
