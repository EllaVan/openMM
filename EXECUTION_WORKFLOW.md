# 完整执行流程：从特征提取到超图网络训练

本文档提供从原始数据到超图网络训练的完整执行流程，使用本地 backbone 模型路径。

## 📋 目录
1. [环境准备](#环境准备)
2. [数据集组织](#数据集组织)
3. [配置本地模型路径](#配置本地模型路径)
4. [特征提取](#特征提取)
5. [超图网络训练](#超图网络训练)
6. [完整示例](#完整示例)

---

## 1. 环境准备

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio

# 特征提取依赖
pip install transformers librosa soundfile opencv-python mediapipe pandas tqdm

# 可选：加速训练
pip install tensorboard wandb
```

### 检查环境

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## 2. 数据集组织

### MOSEI 数据集结构

```
/path/to/MOSEI/
├── audio/
│   └── video_id/           # 例如: 03bSnISJMiM
│       ├── 0.wav
│       ├── 1.wav
│       └── ...
├── video/
│   └── video_id/
│       ├── 0.mp4
│       ├── 1.mp4
│       └── ...
└── label/
    └── label.csv           # 包含: video_id, clip_id, text, emotion
```

**label.csv 格式**:
```csv
video_id,clip_id,text,emotion
03bSnISJMiM,0,This is a sample text,happy
03bSnISJMiM,1,Another example,sad
...
```

### MELD 数据集结构

```
/path/to/MELD/
├── train/
│   ├── audio/
│   │   └── video_id/
│   │       ├── clip_id.wav
│   │       └── ...
│   ├── video/
│   │   └── video_id/
│   │       ├── clip_id.mp4
│   │       └── ...
│   └── label/
│       └── merged_label_new.csv
├── dev/
│   └── ... (同上)
└── test/
    └── ... (同上)
```

---

## 3. 配置本地模型路径

### 3.1 创建本地路径配置文件

创建 `config_local_models.json`:

```json
{
  "description": "使用本地模型路径的配置",
  "text": {
    "model": "roberta-base",
    "model_path": "/path/to/local/models/roberta-base",
    "enabled": true,
    "max_length": 512
  },
  "audio": {
    "model": "hubert",
    "model_name": "facebook/hubert-base-ls960",
    "model_path": "/path/to/local/models/hubert-base-ls960",
    "processor_path": "/path/to/local/models/hubert-base-ls960",
    "sample_rate": 16000,
    "enabled": true
  },
  "video": {
    "model": "vit",
    "model_name": "google/vit-base-patch16-224",
    "model_path": "/path/to/local/models/vit-base-patch16-224",
    "processor_path": "/path/to/local/models/vit-base-patch16-224",
    "fps": 25,
    "enabled": true,
    "feature_mode": "cls"
  },
  "alignment": {
    "method": "timestamp_matching",
    "enabled": true,
    "reference": "audio"
  }
}
```

### 3.2 修改特征提取器以支持本地路径

创建 `feature_extraction_demo_local.py`:

```python
"""
支持本地模型路径的特征提取器
基于 feature_extraction_demo.py 修改
"""

import os
import json
from feature_extraction_demo import MultimodalFeatureExtractor


class LocalModelFeatureExtractor(MultimodalFeatureExtractor):
    """支持本地模型路径的特征提取器"""

    def _init_text_extractor(self):
        """初始化文本特征提取器（支持本地路径）"""
        try:
            from transformers import AutoTokenizer, AutoModel

            model_config = self.config['text']

            # 优先使用本地路径，否则使用 model_name
            if 'model_path' in model_config and model_config['model_path']:
                model_path = model_config['model_path']
                print(f"✓ 从本地加载文本模型: {model_path}")
            else:
                model_path = model_config['model']
                print(f"✓ 从 HuggingFace 加载文本模型: {model_path}")

            self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.text_model = AutoModel.from_pretrained(model_path).to(self.device)
            self.text_model.eval()

            model_type = self.text_model.config.model_type
            print(f"  模型类型: {model_type}")
        except ImportError:
            print("⚠ transformers 未安装，文本特征提取将被禁用")
            self.config['text']['enabled'] = False
        except Exception as e:
            print(f"⚠ 加载文本模型失败: {str(e)}")
            self.config['text']['enabled'] = False

    def _init_audio_extractor(self):
        """初始化音频特征提取器（支持本地路径）"""
        audio_model = self.config['audio']['model']

        if audio_model in ['wav2vec2', 'hubert']:
            try:
                from transformers import AutoProcessor, AutoModel

                audio_config = self.config['audio']

                # 获取模型路径
                if 'model_path' in audio_config and audio_config['model_path']:
                    model_path = audio_config['model_path']
                    print(f"✓ 从本地加载音频模型: {model_path}")
                else:
                    if 'model_name' in audio_config:
                        model_path = audio_config['model_name']
                    else:
                        model_path = "facebook/wav2vec2-base-960h" if audio_model == 'wav2vec2' else "facebook/hubert-base-ls960"
                    print(f"✓ 从 HuggingFace 加载音频模型: {model_path}")

                # 获取 processor 路径（通常和模型路径相同）
                processor_path = audio_config.get('processor_path', model_path)

                self.audio_processor = AutoProcessor.from_pretrained(processor_path)
                self.audio_model = AutoModel.from_pretrained(model_path).to(self.device)
                self.audio_model.eval()

                model_type = self.audio_model.config.model_type
                print(f"  模型类型: {model_type}")
            except ImportError:
                print("⚠ transformers 未安装，使用 librosa 作为备选")
                self.config['audio']['model'] = 'librosa'
            except Exception as e:
                print(f"⚠ 加载音频模型失败: {str(e)}")
                print("  使用 librosa 作为备选")
                self.config['audio']['model'] = 'librosa'

        if audio_model == 'librosa' or self.config['audio']['model'] == 'librosa':
            print("✓ 音频提取器已加载: Librosa")

    def _init_video_extractor(self):
        """初始化视频特征提取器（支持本地路径）"""
        video_model = self.config['video']['model']

        if video_model == 'openface':
            print("⚠ OpenFace 需要单独安装")

        elif video_model == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                )
                print("✓ 视频提取器已加载: MediaPipe")
            except ImportError:
                print("⚠ mediapipe 未安装，视频特征提取将被禁用")
                self.config['video']['enabled'] = False

        elif video_model == 'vit':
            try:
                from transformers import AutoImageProcessor, AutoModel

                video_config = self.config['video']

                # 获取模型路径
                if 'model_path' in video_config and video_config['model_path']:
                    model_path = video_config['model_path']
                    print(f"✓ 从本地加载视频模型: {model_path}")
                else:
                    if 'model_name' in video_config:
                        model_path = video_config['model_name']
                    else:
                        model_path = "google/vit-base-patch16-224"
                    print(f"✓ 从 HuggingFace 加载视频模型: {model_path}")

                # 获取 processor 路径
                processor_path = video_config.get('processor_path', model_path)

                self.video_processor = AutoImageProcessor.from_pretrained(processor_path)
                self.video_model = AutoModel.from_pretrained(model_path).to(self.device)
                self.video_model.eval()

                self.vit_feature_mode = video_config.get('feature_mode', 'cls')
                print(f"  特征模式: {self.vit_feature_mode}")
            except ImportError:
                print("⚠ transformers 未安装，视频特征提取将被禁用")
                self.config['video']['enabled'] = False
            except Exception as e:
                print(f"⚠ 加载视频模型失败: {str(e)}")
                self.config['video']['enabled'] = False


def load_config_with_local_paths(config_file: str) -> dict:
    """加载配置文件"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    print("\n" + "="*60)
    print("配置文件加载完成")
    print("="*60)
    print(f"文本模型: {config['text'].get('model_path', config['text']['model'])}")
    print(f"音频模型: {config['audio'].get('model_path', config['audio'].get('model_name', config['audio']['model']))}")
    print(f"视频模型: {config['video'].get('model_path', config['video'].get('model_name', config['video']['model']))}")
    print("="*60 + "\n")

    return config


if __name__ == "__main__":
    # 示例：使用本地路径配置
    config = load_config_with_local_paths('config_local_models.json')

    extractor = LocalModelFeatureExtractor(config=config)

    # 提取特征
    features = extractor.extract_from_files(
        text_file="example_data/sample.txt",
        audio_file="example_data/sample.wav",
        video_file="example_data/sample.mp4",
        output_file="output/sample_features.pkl"
    )
```

保存为 `feature_extraction_demo_local.py`

### 3.3 修改数据集提取器以支持本地路径

创建 `extract_dataset_features_local.py`:

```python
"""
支持本地模型路径的数据集特征提取器
基于 extract_dataset_features.py 修改
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_dataset_features import (
    MOSEIFeatureExtractor,
    MELDFeatureExtractor,
    DatasetFeatureExtractor
)
from feature_extraction_demo_local import LocalModelFeatureExtractor


class LocalModelDatasetExtractor(DatasetFeatureExtractor):
    """支持本地模型的数据集提取器"""

    def __init__(self, dataset_name, base_dir, output_dir, config=None):
        # 不调用父类 __init__，手动初始化
        self.dataset_name = dataset_name.upper()
        self.base_dir = base_dir
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # 使用支持本地路径的提取器
        self.extractor = LocalModelFeatureExtractor(config=config)

        self.emotion_mapping = {
            'happy': 0, 'happiness': 0,
            'sad': 1, 'sadness': 1,
            'anger': 2,
            'disgust': 3,
            'surprise': 4,
            'fear': 5,
            'neutral': 6
        }


class LocalMOSEIFeatureExtractor(LocalModelDatasetExtractor, MOSEIFeatureExtractor):
    """支持本地模型的 MOSEI 提取器"""
    pass


class LocalMELDFeatureExtractor(LocalModelDatasetExtractor, MELDFeatureExtractor):
    """支持本地模型的 MELD 提取器"""
    pass
```

保存为 `extract_dataset_features_local.py`

---

## 4. 特征提取

### 4.1 准备配置文件

**步骤 1**: 修改 `config_local_models.json`，填入你的本地模型路径

```json
{
  "text": {
    "model": "roberta-base",
    "model_path": "/media/sda/pingjm/MTCA/pretraining_model/RoBERTa/roberta-base",
    "enabled": true
  },
  "audio": {
    "model": "hubert",
    "model_path": "/media/sda/pingjm/MTCA/pretraining_model/HuBERT/hubert-base-ls960",
    "processor_path": "/media/sda/pingjm/MTCA/pretraining_model/HuBERT/hubert-base-ls960",
    "sample_rate": 16000,
    "enabled": true
  },
  "video": {
    "model": "vit",
    "model_path": "/media/sda/pingjm/MTCA/pretraining_model/ViT/vit-base-patch16-224-in21k",
    "processor_path": "/media/sda/pingjm/MTCA/pretraining_model/ViT/vit-base-patch16-224-in21k",
    "feature_mode": "cls",
    "enabled": true
  },
  "alignment": {
    "enabled": true,
    "reference": "audio"
  }
}
```

### 4.2 创建提取配置

创建 `extraction_config_local.json`:

```json
{
  "mosei": {
    "base_dir": "/path/to/MOSEI",
    "label_file": "/path/to/MOSEI/label/label.csv",
    "output_dir": "./output/mosei_features",
    "feature_config": "config_local_models.json"
  },
  "meld": {
    "base_dir": "/path/to/MELD",
    "output_dir": "./output/meld_features",
    "split": "all",
    "feature_config": "config_local_models.json"
  }
}
```

### 4.3 运行特征提取

**方式 1: 交互式提取**

```python
python feature_extraction_demo_local.py
```

**方式 2: 批量提取（推荐）**

创建批量提取脚本 `batch_extract_local.py`:

```python
#!/usr/bin/env python
"""
使用本地模型批量提取数据集特征
"""

import json
import logging
from extract_dataset_features_local import (
    LocalMOSEIFeatureExtractor,
    LocalMELDFeatureExtractor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_mosei():
    """提取 MOSEI 数据集"""
    # 加载配置
    with open('extraction_config_local.json', 'r') as f:
        config = json.load(f)

    mosei_config = config['mosei']

    # 加载特征提取配置
    with open(mosei_config['feature_config'], 'r') as f:
        feature_config = json.load(f)

    logger.info("开始提取 MOSEI 数据集特征")

    # 创建提取器
    extractor = LocalMOSEIFeatureExtractor(
        base_dir=mosei_config['base_dir'],
        output_dir=mosei_config['output_dir'],
        label_file=mosei_config['label_file'],
        config=feature_config
    )

    # 提取特征
    results = extractor.process_dataset()

    logger.info("MOSEI 特征提取完成")
    return results


def extract_meld():
    """提取 MELD 数据集"""
    # 加载配置
    with open('extraction_config_local.json', 'r') as f:
        config = json.load(f)

    meld_config = config['meld']

    # 加载特征提取配置
    with open(meld_config['feature_config'], 'r') as f:
        feature_config = json.load(f)

    logger.info("开始提取 MELD 数据集特征")

    # 创建提取器
    extractor = LocalMELDFeatureExtractor(
        base_dir=meld_config['base_dir'],
        output_dir=meld_config['output_dir'],
        config=feature_config
    )

    # 提取特征
    split = meld_config.get('split', 'all')
    results = extractor.process_dataset(split=split)

    logger.info("MELD 特征提取完成")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['mosei', 'meld', 'all'])
    args = parser.parse_args()

    if args.dataset in ['mosei', 'all']:
        extract_mosei()

    if args.dataset in ['meld', 'all']:
        extract_meld()
```

运行:

```bash
# 提取 MOSEI
python batch_extract_local.py --dataset mosei

# 提取 MELD
python batch_extract_local.py --dataset meld

# 提取所有
python batch_extract_local.py --dataset all
```

### 4.4 特征提取输出

提取完成后，输出目录结构：

```
output/
├── mosei_features/
│   ├── MOSEIhappylabel0.pkl
│   ├── MOSEIsadlabel1.pkl
│   ├── MOSEIangerlabel2.pkl
│   ├── MOSEIdisgustlabel3.pkl
│   ├── MOSEIsurpriselabel4.pkl
│   └── MOSEIfearlabel5.pkl
└── meld_features/
    ├── MELD_trainhappylabel0.pkl
    ├── MELD_devhappylabel0.pkl
    ├── MELD_testhappylabel0.pkl
    └── ...
```

每个 `.pkl` 文件包含:

```python
[
    {
        'audio_features': torch.Tensor,  # [num_frames, audio_dim]
        'text_features': torch.Tensor,   # [num_frames, text_dim]
        'video_features': torch.Tensor,  # [num_frames, video_dim]
        'label': int,
        'emotion': str,
        'sample_id': str,
        'num_frames': int
    },
    # ... 更多样本
]
```

---

## 5. 超图网络训练

### 5.1 验证提取的特征

```python
import pickle
import torch

# 加载特征
with open('./output/mosei_features/MOSEIhappylabel0.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"样本数量: {len(data)}")
print(f"第一个样本的键: {data[0].keys()}")
print(f"音频特征: {data[0]['audio_features'].shape}")
print(f"文本特征: {data[0]['text_features'].shape}")
print(f"视频特征: {data[0]['video_features'].shape}")
print(f"标签: {data[0]['label']}")
print(f"帧数: {data[0]['num_frames']}")
```

### 5.2 训练超图网络

**方式 1: 使用训练脚本**

```bash
python train_hypergraph.py \
  --data_dir ./output/mosei_features \
  --dataset MOSEI \
  --emotion happy \
  --label_id 0 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --num_hyperedges 64 \
  --num_conv_layers 2 \
  --use_contrastive \
  --contrastive_weight 0.1 \
  --use_bottleneck \
  --save_dir ./checkpoints \
  --device cuda
```

**方式 2: 自定义训练脚本**

创建 `train_custom.py`:

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from emotion_dataloader import create_dataloaders
from hypergraph_network import HypergraphEmotionClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 加载数据
logger.info("加载数据...")
dataloaders = create_dataloaders(
    data_dir='./output/mosei_features',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32,
    num_workers=4,
    train_ratio=0.7
)

train_loader = dataloaders['train']
test_loader = dataloaders['test']

logger.info(f"训练集: {len(train_loader.dataset)} 样本")
logger.info(f"测试集: {len(test_loader.dataset)} 样本")

# 2. 创建模型
logger.info("创建模型...")
batch = next(iter(train_loader))
feature_dims = {
    'text': batch['text_features'].shape[-1],
    'audio': batch['audio_features'].shape[-1],
    'video': batch['video_features'].shape[-1]
}

logger.info(f"特征维度: {feature_dims}")

config = {
    'encoder_hidden_dim': 256,
    'encoder_output_dim': 256,
    'hypergraph_hidden_dim': 256,
    'num_hyperedges': 64,
    'num_conv_layers': 2,
    'bottleneck_dim': 128,
    'dropout': 0.1,
    'hyperedge_drop_rate': 0.2,
    'use_contrastive': True,
    'contrastive_weight': 0.1,
    'use_bottleneck': True
}

model = HypergraphEmotionClassifier(
    feature_dims=feature_dims,
    num_classes=2,
    config=config
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
logger.info(f"使用设备: {device}")

# 3. 优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# 4. 训练
logger.info("开始训练...")
best_acc = 0

for epoch in range(1, 51):
    # 训练
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        output = model(batch)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = torch.argmax(output['logits'], dim=1)
        train_correct += (predictions == batch['label']).sum().item()
        train_total += batch['label'].size(0)

    train_acc = train_correct / train_total

    # 评估
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            output = model(batch)
            predictions = torch.argmax(output['logits'], dim=1)
            test_correct += (predictions == batch['label']).sum().item()
            test_total += batch['label'].size(0)

    test_acc = test_correct / test_total

    logger.info(
        f"Epoch {epoch:2d} - "
        f"Train Loss: {train_loss/len(train_loader):.4f}, "
        f"Train Acc: {train_acc:.4f}, "
        f"Test Acc: {test_acc:.4f}"
    )

    # 学习率调度
    scheduler.step(test_acc)

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_acc,
            'config': config,
            'feature_dims': feature_dims
        }, './checkpoints/best_model.pth')
        logger.info(f"  ✓ 保存最佳模型 (Acc: {best_acc:.4f})")

logger.info(f"训练完成！最佳准确率: {best_acc:.4f}")
```

运行:

```bash
python train_custom.py
```

---

## 6. 完整示例

### 端到端执行流程

```bash
# ========================================
# 步骤 1: 准备环境和配置
# ========================================

# 1.1 创建目录
mkdir -p output/mosei_features
mkdir -p output/meld_features
mkdir -p checkpoints
mkdir -p logs

# 1.2 修改配置文件中的本地路径
# 编辑 config_local_models.json
# 编辑 extraction_config_local.json

# ========================================
# 步骤 2: 特征提取
# ========================================

# 2.1 提取 MOSEI 特征
echo "开始提取 MOSEI 特征..."
python batch_extract_local.py --dataset mosei 2>&1 | tee logs/extract_mosei.log

# 2.2 提取 MELD 特征
echo "开始提取 MELD 特征..."
python batch_extract_local.py --dataset meld 2>&1 | tee logs/extract_meld.log

# 2.3 验证特征
python -c "
import pickle
with open('./output/mosei_features/MOSEIhappylabel0.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'MOSEI happy: {len(data)} samples')
print(f'Feature dims: audio={data[0][\"audio_features\"].shape}, text={data[0][\"text_features\"].shape}, video={data[0][\"video_features\"].shape}')
"

# ========================================
# 步骤 3: 训练超图网络
# ========================================

# 3.1 训练单个情感分类器
echo "开始训练 happy 分类器..."
python train_hypergraph.py \
  --data_dir ./output/mosei_features \
  --dataset MOSEI \
  --emotion happy \
  --label_id 0 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --num_hyperedges 64 \
  --num_conv_layers 2 \
  --use_contrastive \
  --use_bottleneck \
  --save_dir ./checkpoints \
  2>&1 | tee logs/train_happy.log

# 3.2 训练多个情感分类器
for emotion in happy sad anger disgust surprise fear; do
    echo "训练 ${emotion} 分类器..."
    python train_hypergraph.py \
      --data_dir ./output/mosei_features \
      --dataset MOSEI \
      --emotion ${emotion} \
      --label_id $(case ${emotion} in happy) echo 0;; sad) echo 1;; anger) echo 2;; disgust) echo 3;; surprise) echo 4;; fear) echo 5;; esac) \
      --batch_size 32 \
      --epochs 50 \
      --lr 1e-4 \
      --num_hyperedges 64 \
      --num_conv_layers 2 \
      --use_contrastive \
      --use_bottleneck \
      --save_dir ./checkpoints \
      2>&1 | tee logs/train_${emotion}.log
done

# ========================================
# 步骤 4: 评估模型
# ========================================

# 4.1 加载并测试模型
python -c "
import torch
from hypergraph_network import HypergraphEmotionClassifier
from emotion_dataloader import create_dataloaders

# 加载模型
checkpoint = torch.load('./checkpoints/best_model_MOSEI_happy.pth')
model = HypergraphEmotionClassifier(
    feature_dims=checkpoint['feature_dims'],
    num_classes=2,
    config=checkpoint['config']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载测试数据
dataloaders = create_dataloaders(
    data_dir='./output/mosei_features',
    dataset_name='MOSEI',
    emotion='happy',
    label_id=0,
    batch_size=32
)

# 评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for batch in dataloaders['test']:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        predictions = model.predict(batch)
        correct += (predictions == batch['label']).sum().item()
        total += batch['label'].size(0)

accuracy = correct / total
print(f'测试准确率: {accuracy:.4f}')
"
```

---

## 7. 文件清单

### 需要创建的文件

```
openMM/
├── config_local_models.json              # 本地模型路径配置
├── extraction_config_local.json          # 数据集提取配置
├── feature_extraction_demo_local.py      # 支持本地路径的特征提取器
├── extract_dataset_features_local.py     # 支持本地路径的数据集提取器
├── batch_extract_local.py                # 批量提取脚本
└── train_custom.py                       # 自定义训练脚本（可选）
```

### 已有的文件

```
openMM/
├── feature_extraction_demo.py            # 原始特征提取器
├── extract_dataset_features.py           # 原始数据集提取器
├── emotion_dataloader.py                 # DataLoader
├── hypergraph_modules.py                 # 超图模块
├── hypergraph_network.py                 # 超图网络
├── train_hypergraph.py                   # 训练脚本
└── examples/
    └── hypergraph_example.py             # 使用示例
```

---

## 8. 执行顺序总结

### 标准流程

```
1. 准备数据集
   └─> 组织 MOSEI/MELD 目录结构

2. 配置本地模型
   └─> 修改 config_local_models.json
   └─> 填入本地模型路径

3. 特征提取
   └─> python batch_extract_local.py --dataset mosei
   └─> 输出: output/mosei_features/*.pkl

4. 验证特征
   └─> 检查 pkl 文件是否正确

5. 训练超图网络
   └─> python train_hypergraph.py [参数]
   └─> 输出: checkpoints/best_model_*.pth

6. 评估和使用
   └─> 加载模型进行预测
```

### 快速开始（最少步骤）

```bash
# 1. 修改配置（填入你的路径）
vim config_local_models.json
vim extraction_config_local.json

# 2. 提取特征
python batch_extract_local.py --dataset mosei

# 3. 训练模型
python train_hypergraph.py \
  --data_dir ./output/mosei_features \
  --dataset MOSEI \
  --emotion happy \
  --label_id 0 \
  --batch_size 32 \
  --epochs 50

# 完成！
```

---

## 9. 常见问题

### Q1: 模型路径找不到？

**A**: 检查路径是否正确：
```bash
ls /path/to/local/models/roberta-base/config.json
ls /path/to/local/models/roberta-base/pytorch_model.bin
```

### Q2: 内存不足？

**A**: 减少 batch_size 或使用 CPU：
```bash
python train_hypergraph.py ... --batch_size 16 --device cpu
```

### Q3: 特征提取很慢？

**A**: 确保使用 GPU：
```python
import torch
print(torch.cuda.is_available())  # 应该为 True
```

### Q4: 特征维度不匹配？

**A**: 检查模型输出维度：
- RoBERTa-base: 768
- HuBERT-base: 768
- ViT-base: 768

---

## 10. 性能优化

### 特征提取优化

```python
# 使用混合精度
with torch.cuda.amp.autocast():
    features = extractor.extract_text_features(text)
```

### 训练优化

```python
# 1. 梯度累积
for i, batch in enumerate(train_loader):
    loss = model(batch)['loss'] / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(batch)
    loss = output['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

希望这个完整的执行流程指南对你有帮助！🚀
