#!/bin/bash
#
# 自动配置特征提取使用示例
# 使用 batch_extract_hybrid_auto.py（从 JSON 自动读取配置）
#

set -e

echo "=========================================="
echo "自动配置特征提取使用示例"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}提示：使用自动配置版本前，请先编辑以下 JSON 文件：${NC}"
echo "  1. dataset_paths.json - 设置数据集路径"
echo "  2. config_hybrid.json - 设置模型路径"
echo "  3. extraction_config.json - 调整提取参数（可选）"
echo ""

# ==========================================
# 示例 1: 最简单的用法（推荐）
# ==========================================
echo -e "${GREEN}示例 1: 最简单的用法（推荐）${NC}"
echo "前提：已在 dataset_paths.json 中配置好路径"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py --dataset mosei --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 2: MELD 数据集
# ==========================================
echo -e "${GREEN}示例 2: MELD 数据集（全部划分）${NC}"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py --dataset meld --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 3: 覆盖 JSON 配置
# ==========================================
echo -e "${GREEN}示例 3: 覆盖 JSON 配置中的参数${NC}"
echo "使用命令行参数覆盖 JSON 中的配置"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py \
  --dataset mosei \
  --base_dir /custom/path/to/MOSEI \
  --output_dir /custom/output \
  --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 4: MELD 单个划分
# ==========================================
echo -e "${GREEN}示例 4: MELD 数据集（仅训练集）${NC}"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py --dataset meld --split train --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 5: 使用预训练 PCA 模型
# ==========================================
echo -e "${GREEN}示例 5: 使用预训练 PCA 模型${NC}"
echo "复用已训练的 PCA 模型，跳过 PCA 训练步骤"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py \
  --dataset meld \
  --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 6: 调整 PCA 训练样本数
# ==========================================
echo -e "${GREEN}示例 6: 调整 PCA 训练样本数${NC}"
echo ""

echo "快速模式（500 样本）:"
echo -e "${YELLOW}"
cat << 'EOF'
python batch_extract_hybrid_auto.py --dataset mosei --pca_training_samples 500 --train_pca
EOF
echo -e "${NC}"
echo ""

echo "高质量模式（2000 样本）:"
echo -e "${YELLOW}"
cat << 'EOF'
python batch_extract_hybrid_auto.py --dataset mosei --pca_training_samples 2000 --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 示例 7: 使用自定义配置文件
# ==========================================
echo -e "${GREEN}示例 7: 使用自定义配置文件${NC}"
echo ""

echo "命令:"
echo -e "${YELLOW}"
cat << 'EOF'
cd codes_v251112/new_unimodal
python batch_extract_hybrid_auto.py \
  --dataset mosei \
  --dataset_config my_custom_paths.json \
  --extraction_config my_custom_extraction.json \
  --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 完整工作流程
# ==========================================
echo "=========================================="
echo "完整工作流程示例"
echo "=========================================="
echo ""

echo -e "${BLUE}步骤 1: 配置 JSON 文件${NC}"
echo "  编辑 dataset_paths.json，设置实际路径"
echo ""

echo -e "${BLUE}步骤 2: 首次提取 MOSEI（训练 PCA）${NC}"
echo -e "${YELLOW}"
cat << 'EOF'
  python batch_extract_hybrid_auto.py --dataset mosei --train_pca
EOF
echo -e "${NC}"
echo ""

echo -e "${BLUE}步骤 3: 提取 MELD（复用 PCA 模型）${NC}"
echo -e "${YELLOW}"
cat << 'EOF'
  python batch_extract_hybrid_auto.py \
    --dataset meld \
    --pca_model_path ./output/mosei_features_hybrid/audio_pca_model.pkl
EOF
echo -e "${NC}"
echo ""

# ==========================================
# JSON 配置示例
# ==========================================
echo "=========================================="
echo "JSON 配置示例"
echo "=========================================="
echo ""

echo -e "${BLUE}dataset_paths.json 配置:${NC}"
echo -e "${YELLOW}"
cat << 'EOF'
{
  "mosei": {
    "base_dir": "/media/sda/datasets/MOSEI",
    "label_file": "/media/sda/datasets/MOSEI/label/label.csv",
    "output_dir": "./output/mosei_features_hybrid"
  },
  "meld": {
    "base_dir": "/media/sda/datasets/MELD",
    "output_dir": "./output/meld_features_hybrid",
    "split": "all"
  }
}
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 对比：手动 vs 自动配置
# ==========================================
echo "=========================================="
echo "对比：手动配置 vs 自动配置"
echo "=========================================="
echo ""

echo -e "${RED}手动配置（原版）:${NC}"
echo -e "${YELLOW}"
cat << 'EOF'
python batch_extract_hybrid.py \
  --dataset mosei \
  --config config_hybrid.json \
  --base_dir /path/to/MOSEI \
  --label_file /path/to/MOSEI/label/label.csv \
  --output_dir ./output \
  --train_pca \
  --pca_training_samples 1000
EOF
echo -e "${NC}"
echo ""

echo -e "${GREEN}自动配置（新版）:${NC}"
echo -e "${YELLOW}"
cat << 'EOF'
python batch_extract_hybrid_auto.py --dataset mosei --train_pca
EOF
echo -e "${NC}"
echo ""

# ==========================================
# 注意事项
# ==========================================
echo "=========================================="
echo "注意事项"
echo "=========================================="
echo ""
echo "1. ${GREEN}首次使用${NC}：编辑 dataset_paths.json 和 config_hybrid.json"
echo "2. ${GREEN}路径优先级${NC}：命令行参数 > JSON 配置 > 默认值"
echo "3. ${GREEN}PCA 模型${NC}：MOSEI 和 MELD 可共享同一个 PCA 模型"
echo "4. ${GREEN}调试${NC}：遇到问题时使用命令行参数覆盖 JSON 配置"
echo "5. ${GREEN}性能${NC}：首次训练 PCA 约 5 分钟，后续可复用"
echo ""

# ==========================================
# 故障排除
# ==========================================
echo "=========================================="
echo "常见问题"
echo "=========================================="
echo ""

echo -e "${BLUE}Q: 提示 'base_dir' 未配置？${NC}"
echo "A: 编辑 dataset_paths.json，设置正确的 base_dir 路径"
echo ""

echo -e "${BLUE}Q: 想临时使用不同的路径？${NC}"
echo "A: 使用命令行参数覆盖："
echo -e "${YELLOW}   --base_dir /custom/path${NC}"
echo ""

echo -e "${BLUE}Q: MOSEI 提示缺少 label_file？${NC}"
echo "A: 在 dataset_paths.json 中设置 mosei.label_file"
echo ""

echo -e "${BLUE}Q: 如何查看帮助？${NC}"
echo -e "${YELLOW}   python batch_extract_hybrid_auto.py --help${NC}"
echo ""

# ==========================================
# 预计时间和资源
# ==========================================
echo "=========================================="
echo "预计时间和资源需求"
echo "=========================================="
echo ""
echo "MOSEI (22,856 样本):"
echo "  - PCA 训练: 约 5 分钟（首次）"
echo "  - 特征提取: 约 5-6 小时"
echo "  - 文件大小: 约 30 GB"
echo "  - GPU 显存: 推荐 24GB"
echo ""
echo "MELD (13,708 样本):"
echo "  - PCA 训练: 约 3 分钟（首次）"
echo "  - 特征提取: 约 3-4 小时"
echo "  - 文件大小: 约 18 GB"
echo "  - GPU 显存: 推荐 24GB"
echo ""

echo "=========================================="
echo "更多信息请查看 USAGE.md 和 README.md"
echo "=========================================="
