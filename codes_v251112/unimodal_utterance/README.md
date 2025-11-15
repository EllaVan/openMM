# Utterance-Level 多模态特征提取

使用 **RoBERTa/HuBERT/ViT-Base** 提取固定维度的 utterance-level 特征。

## ✨ 特性

- ✅ **Utterance-Level**: 每个样本固定 768维×3模态 = 2304维
- ✅ **无需对齐**: 三个模态独立提取，无需时间对齐
- ✅ **统一768维**: RoBERTa/HuBERT/ViT-Base 都输出 768 维
- ✅ **完全自动化**: JSON 配置，无命令行参数
- ✅ **支持 MOSEI/MELD**: 两个主流情感数据集

## 🚀 快速开始

### 1. 配置数据集路径

编辑 `extraction_settings.json`

### 2. 直接运行

```bash
cd codes_v251112/unimodal_utterance
python extract_features.py
```

## 📊 输出格式

每个样本：`{'text': [768], 'audio': [768], 'video': [768]}`

详见 QUICKSTART.md
