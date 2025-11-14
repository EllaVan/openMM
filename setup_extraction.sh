#!/bin/bash
#
# 特征提取配置向导
# 用于快速配置并运行 MOSEI/MELD 特征提取
#

set -e

echo "=========================================="
echo "特征提取配置向导"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 选择数据集
echo "请选择数据集:"
echo "  1) MOSEI"
echo "  2) MELD"
read -p "输入选择 [1-2]: " dataset_choice

case $dataset_choice in
    1)
        DATASET="mosei"
        ;;
    2)
        DATASET="meld"
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}已选择: $DATASET${NC}"
echo ""

# 2. 选择提取方式
echo "请选择提取方式:"
echo "  1) 标准提取 (RoBERTa + HuBERT + ViT-base, 768维, ~60GB)"
echo "  2) 混合提取 (MiniLM + HuBERT+PCA + ViT-small, 384维, ~30GB) [推荐]"
read -p "输入选择 [1-2]: " method_choice

case $method_choice in
    1)
        METHOD="standard"
        SCRIPT="batch_extract_efficient.py"
        CONFIG="config_efficient.json"
        OUTPUT_SUFFIX=""
        ;;
    2)
        METHOD="hybrid"
        SCRIPT="batch_extract_hybrid.py"
        CONFIG="config_hybrid_pca.json"
        OUTPUT_SUFFIX="_hybrid"
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}已选择: $METHOD 提取${NC}"
echo ""

# 3. 输入数据集路径
echo "请输入数据集路径信息:"
echo ""

if [ "$DATASET" = "mosei" ]; then
    read -p "MOSEI 根目录 (例: /media/sda/datasets/MOSEI): " BASE_DIR
    read -p "标签文件路径 (例: /media/sda/datasets/MOSEI/label/label.csv): " LABEL_FILE

    # 验证路径
    if [ ! -d "$BASE_DIR" ]; then
        echo -e "${RED}错误: 目录不存在: $BASE_DIR${NC}"
        exit 1
    fi

    if [ ! -f "$LABEL_FILE" ]; then
        echo -e "${RED}错误: 标签文件不存在: $LABEL_FILE${NC}"
        exit 1
    fi

    OUTPUT_DIR="./output/mosei_features${OUTPUT_SUFFIX}"

else  # MELD
    read -p "MELD 根目录 (例: /media/sda/datasets/MELD): " BASE_DIR

    # 验证路径
    if [ ! -d "$BASE_DIR" ]; then
        echo -e "${RED}错误: 目录不存在: $BASE_DIR${NC}"
        exit 1
    fi

    # 检查 train/dev/test 目录
    if [ ! -d "$BASE_DIR/train" ] || [ ! -d "$BASE_DIR/dev" ] || [ ! -d "$BASE_DIR/test" ]; then
        echo -e "${YELLOW}警告: 未找到 train/dev/test 目录，请确认目录结构${NC}"
    fi

    echo ""
    echo "选择处理的划分:"
    echo "  1) all (全部: train + dev + test) [推荐]"
    echo "  2) train (仅训练集)"
    echo "  3) dev (仅验证集)"
    echo "  4) test (仅测试集)"
    read -p "输入选择 [1-4]: " split_choice

    case $split_choice in
        1) SPLIT="all" ;;
        2) SPLIT="train" ;;
        3) SPLIT="dev" ;;
        4) SPLIT="test" ;;
        *)
            echo -e "${RED}无效选择，使用默认值: all${NC}"
            SPLIT="all"
            ;;
    esac

    OUTPUT_DIR="./output/meld_features${OUTPUT_SUFFIX}"
fi

# 4. 创建输出目录
echo ""
echo "输出目录: $OUTPUT_DIR"
read -p "确认使用此输出目录? [Y/n]: " confirm_output
if [[ $confirm_output =~ ^[Nn]$ ]]; then
    read -p "请输入自定义输出目录: " OUTPUT_DIR
fi

mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ 输出目录已创建: $OUTPUT_DIR${NC}"

# 5. 混合提取的额外配置
EXTRA_ARGS=""
if [ "$METHOD" = "hybrid" ]; then
    echo ""
    echo "混合提取配置:"
    read -p "是否训练音频 PCA 模型? [Y/n]: " train_pca
    if [[ ! $train_pca =~ ^[Nn]$ ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --train_pca"
        read -p "PCA 训练样本数 (默认 1000): " pca_samples
        pca_samples=${pca_samples:-1000}
        EXTRA_ARGS="$EXTRA_ARGS --pca_training_samples $pca_samples"
    fi
fi

# 6. 生成命令
echo ""
echo "=========================================="
echo "配置完成，生成提取命令"
echo "=========================================="
echo ""

CMD="python unimodal_features/$SCRIPT"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --config unimodal_features/$CONFIG"
CMD="$CMD --base_dir \"$BASE_DIR\""

if [ "$DATASET" = "mosei" ]; then
    CMD="$CMD --label_file \"$LABEL_FILE\""
else
    CMD="$CMD --split $SPLIT"
fi

CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD $EXTRA_ARGS"

echo "将执行以下命令:"
echo ""
echo -e "${YELLOW}$CMD${NC}"
echo ""

# 7. 确认执行
read -p "是否立即执行? [Y/n]: " confirm_run

if [[ $confirm_run =~ ^[Nn]$ ]]; then
    echo ""
    echo "命令已保存到: run_extraction.sh"
    echo "$CMD" > run_extraction.sh
    chmod +x run_extraction.sh
    echo ""
    echo "稍后可运行: ./run_extraction.sh"
    exit 0
fi

# 8. 执行提取
echo ""
echo "=========================================="
echo "开始特征提取"
echo "=========================================="
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "特征提取完成!"
echo "=========================================="
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
