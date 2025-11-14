#!/bin/bash
#
# 特征提取使用示例
#

set -e

echo "=========================================="
echo "特征提取使用示例"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# 示例 1: MOSEI 数据集提取
# ==========================================
echo -e "${GREEN}示例 1: MOSEI 数据集提取${NC}"
echo ""

MOSEI_BASE_DIR="/path/to/MOSEI"
MOSEI_LABEL="/path/to/MOSEI/label/label.csv"
MOSEI_OUTPUT="./output/mosei_features_hybrid"

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
python new_unimodal/batch_extract_hybrid.py \
  --dataset mosei \
  --config new_unimodal/config_hybrid.json \
  --base_dir /path/to/MOSEI \
  --label_file /path/to/MOSEI/label/label.csv \
  --output_dir ./output/mosei_features_hybrid \
  --train_pca \
  --pca_training_samples 1000
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 2: MELD 数据集提取（所有划分）
# ==========================================
echo -e "${GREEN}示例 2: MELD 数据集提取（所有划分）${NC}"
echo ""

MELD_BASE_DIR="/path/to/MELD"
MELD_OUTPUT="./output/meld_features_hybrid"

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
python new_unimodal/batch_extract_hybrid.py \
  --dataset meld \
  --config new_unimodal/config_hybrid.json \
  --base_dir /path/to/MELD \
  --output_dir ./output/meld_features_hybrid \
  --split all \
  --train_pca \
  --pca_training_samples 1000
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 3: MELD 数据集提取（仅训练集）
# ==========================================
echo -e "${GREEN}示例 3: MELD 数据集提取（仅训练集）${NC}"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
python new_unimodal/batch_extract_hybrid.py \
  --dataset meld \
  --config new_unimodal/config_hybrid.json \
  --base_dir /path/to/MELD \
  --output_dir ./output/meld_features_hybrid \
  --split train \
  --train_pca \
  --pca_training_samples 1000
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 4: 使用预训练 PCA 模型
# ==========================================
echo -e "${GREEN}示例 4: 使用预训练 PCA 模型${NC}"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
python new_unimodal/batch_extract_hybrid.py \
  --dataset meld \
  --config new_unimodal/config_hybrid.json \
  --base_dir /path/to/MELD \
  --output_dir ./output/meld_features_hybrid \
  --split all \
  --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 5: 调整 PCA 训练样本数
# ==========================================
echo -e "${GREEN}示例 5: 调整 PCA 训练样本数${NC}"
echo ""

echo "快速模式（500 样本，约 3 分钟）:"
echo -e "${YELLOW}--pca_training_samples 500${NC}"
echo ""

echo "标准模式（1000 样本，约 5 分钟）[推荐]:"
echo -e "${YELLOW}--pca_training_samples 1000${NC}"
echo ""

echo "高质量模式（2000 样本，约 10 分钟）:"
echo -e "${YELLOW}--pca_training_samples 2000${NC}"
echo ""

# ==========================================
# 注意事项
# ==========================================
echo "=========================================="
echo "注意事项"
echo "=========================================="
echo ""
echo "1. 首次运行前，请配置 config_hybrid.json 中的模型路径"
echo "2. 确保数据集结构正确"
echo "3. MOSEI 需要提供 --label_file 参数"
echo "4. MELD 不需要 --label_file，每个划分有独立 label.csv"
echo "5. PCA 模型在 MOSEI 和 MELD 之间可复用"
echo ""

echo "=========================================="
echo "预计时间和资源需求"
echo "=========================================="
echo ""
echo "MOSEI (22,856 样本):"
echo "  - PCA 训练: 约 5 分钟"
echo "  - 特征提取: 约 5-6 小时"
echo "  - 文件大小: 约 30 GB"
echo "  - GPU 显存: 推荐 24GB"
echo ""
echo "MELD (13,708 样本):"
echo "  - PCA 训练: 约 3 分钟"
echo "  - 特征提取: 约 3-4 小时"
echo "  - 文件大小: 约 18 GB"
echo "  - GPU 显存: 推荐 24GB"
echo ""

echo "=========================================="
echo "更多信息请查看 README.md"
echo "=========================================="
