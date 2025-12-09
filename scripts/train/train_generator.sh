#!/bin/bash
# ============================================================================
# 训练脚本生成器 - 总控脚本
# 
# 使用方法：
# 1. 修改下面的配置参数
# 2. 运行: bash train_generator.sh (注意：必须用bash，不能用sh)
# 3. 自动生成带超参标识的训练脚本并执行
#
# 多数据集使用：
# DATASET_PATHS="/path1.jsonl /path2.jsonl /path3.jsonl"
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 【核心配置】- 修改这里即可
# ============================================================================

# 模型配置
MODEL_SIZE="7B"  # 模型大小: 7B, 32B
TASK_NAME="painting_detection"  # 任务名称: painting_detection, painting_classification

# 数据配置（支持多个数据集，用空格分隔）
# 注意：多个数据集路径用空格分隔，如果路径包含空格请用引号括起来
DATASET_PATHS="/data/dlf/code/Visual-RFT/ch_painting/datasets/tasks/detection/train.jsonl"
# 多个数据集示例：
# DATASET_PATHS="/path/to/dataset1.jsonl /path/to/dataset2.jsonl /path/to/dataset3.jsonl"


# 训练超参
GLOBAL_BATCH_SIZE=16      # 全局batch size
PER_DEVICE_BS=2           # 每张卡的batch size
GRAD_ACCUM_STEPS=4        # 梯度累积步数 (global_bs = per_device_bs * num_gpus * grad_accum)
LEARNING_RATE="1e-6"      # 学习率
NUM_EPOCHS=3              # 训练轮数
NUM_GENERATIONS=4         # 每个prompt生成的样本数

# GPU配置
CUDA_DEVICES="0,1,2,3"  # 使用的GPU
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# 训练配置
TRAIN_TYPE="full"         # full 或 lora
FREEZE_VIT="true"         # 是否冻结视觉编码器
MAX_PIXELS=1003520        # 最大像素数

# Reward配置
REWARD_FUNCS="det_cls_format det_cls_acc"  # 使用的reward插件
REWARD_WEIGHTS="1 1"      # reward权重
BETA=0.01                 # GRPO的beta参数

# 其他配置
DEEPSPEED="zero2"         # deepspeed配置: zero2, zero3
MAX_GRAD_NORM=1.0         # 梯度裁剪
WARMUP_RATIO=0.05         # 预热比例

# 版本标识（可选，用于区分不同实验）
VERSION_TAG=""  # 例如: "v1", "exp1", 留空则只用超参

# ============================================================================
# 【自动计算和生成】- 不需要修改
# ============================================================================

# 计算实际的梯度累积步数（确保global_bs正确）
CALCULATED_GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BS * NUM_GPUS)))
if [ $CALCULATED_GRAD_ACCUM -ne $GRAD_ACCUM_STEPS ]; then
    echo "警告: 梯度累积步数不匹配"
    echo "  期望: $GRAD_ACCUM_STEPS"
    echo "  计算: $CALCULATED_GRAD_ACCUM (global_bs=$GLOBAL_BATCH_SIZE / (per_device_bs=$PER_DEVICE_BS * num_gpus=$NUM_GPUS))"
    echo "  使用计算值: $CALCULATED_GRAD_ACCUM"
    GRAD_ACCUM_STEPS=$CALCULATED_GRAD_ACCUM
fi

# 生成超参标识
LR_TAG=$(echo $LEARNING_RATE | sed 's/\.//g' | sed 's/e-/e/g')  # 1e-6 -> 1e6
HYPERPARAM_ID="bs${GLOBAL_BATCH_SIZE}_lr${LR_TAG}_ep${NUM_EPOCHS}_gen${NUM_GENERATIONS}"

# 添加版本标识
if [ -n "$VERSION_TAG" ]; then
    HYPERPARAM_ID="${VERSION_TAG}_${HYPERPARAM_ID}"
fi

# 生成脚本名称和输出路径
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SCRIPT_NAME="train_${TASK_NAME}_${MODEL_SIZE}_${HYPERPARAM_ID}_${TIMESTAMP}.sh"
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/${TASK_NAME}_${MODEL_SIZE}/${HYPERPARAM_ID}"

# 根据IP后缀设置DATA_PREFIX
IP_SUFFIX=$(hostname -I | awk '{print $1}' | awk -F. '{print $NF}')
if [ "$IP_SUFFIX" = "226" ]; then
    DATA_PREFIX="/data/share"
elif [ "$IP_SUFFIX" = "227" ]; then
    DATA_PREFIX="/home/gpuadmin/share"
else
    DATA_PREFIX="/data/share"
fi

# 模型路径
if [ "$MODEL_SIZE" = "7B" ]; then
    MODEL_PATH="${DATA_PREFIX}/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
elif [ "$MODEL_SIZE" = "32B" ]; then
    MODEL_PATH="${DATA_PREFIX}/hub/models/Qwen/Qwen2___5-VL-32B-Instruct"
else
    echo "错误: 不支持的模型大小 $MODEL_SIZE"
    exit 1
fi

# 插件路径
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/detection_classification.py"

# 飞书Webhook
FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/f120b7d5-8205-4f01-bf1a-86a9e50984a7"

# ============================================================================
# 【生成训练脚本】
# ============================================================================

echo "=" | tr '=' '-' | head -c 70 && echo
echo "训练配置总览"
echo "=" | tr '=' '-' | head -c 70 && echo
echo "任务: ${TASK_NAME}"
echo "模型: Qwen2.5-VL-${MODEL_SIZE}"
echo "超参ID: ${HYPERPARAM_ID}"
echo ""
echo "数据集:"
# 兼容bash/sh: 将空格分隔的字符串转为列表输出
set -- $DATASET_PATHS
idx=1
for ds in "$@"; do
    echo "  [$idx] $ds"
    idx=$((idx+1))
done

echo ""
echo "训练超参:"
echo "  Global Batch Size: ${GLOBAL_BATCH_SIZE}"
echo "  Per-device Batch Size: ${PER_DEVICE_BS}"
echo "  梯度累积步数: ${GRAD_ACCUM_STEPS}"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${NUM_EPOCHS}"
echo "  生成数: ${NUM_GENERATIONS}"
echo ""
echo "GPU配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo "  GPU数量: ${NUM_GPUS}"
echo ""
echo "Reward配置:"
echo "  Reward函数: ${REWARD_FUNCS}"
echo "  Reward权重: ${REWARD_WEIGHTS}"
echo "  Beta: ${BETA}"
echo ""
echo "输出:"
echo "  脚本: ${SCRIPT_NAME}"
echo "  模型目录: ${OUTPUT_DIR}"
echo "=" | tr '=' '-' | head -c 70 && echo

# 创建脚本内容
cat > "${SCRIPT_NAME}" << 'SCRIPT_TEMPLATE'
#!/bin/bash
# ============================================================================
# 自动生成的训练脚本
# 生成时间: __TIMESTAMP__
# 任务: __TASK_NAME__
# 模型: Qwen2.5-VL-__MODEL_SIZE__
# 超参ID: __HYPERPARAM_ID__
# ============================================================================

set -e  # 遇到错误立即退出

# 环境配置
export PYTHONPATH=/data/dlf/code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="__CUDA_DEVICES__"
export NPROC_PER_NODE=__NUM_GPUS__
export MASTER_PORT=12346

# 根据IP后缀设置DATA_PREFIX
IP_SUFFIX=$(hostname -I | awk '{print $1}' | awk -F. '{print $NF}')
if [ "$IP_SUFFIX" = "226" ]; then
    export DATA_PREFIX="/data/share"
    echo "检测到IP后缀226，设置 DATA_PREFIX=/data/share"
elif [ "$IP_SUFFIX" = "227" ]; then
    export DATA_PREFIX="/home/gpuadmin/share"
    echo "检测到IP后缀227，设置 DATA_PREFIX=/home/gpuadmin/share"
else
    export DATA_PREFIX="/data/share"
    echo "使用默认 DATA_PREFIX=/data/share"
fi

# 创建软链接
TARGET="${DATA_PREFIX}/dlf_data/Field-Fidelity/data"
LINK="/data/dlf/code/Field-Fidelity/data"

if [ -L "$LINK" ]; then
    CUR_TARGET=$(readlink -f "$LINK")
    if [ "$CUR_TARGET" != "$TARGET" ]; then
        echo "软链接 $LINK 指向 $CUR_TARGET，重新创建指向 $TARGET"
        rm "$LINK"
        ln -s "$TARGET" "$LINK"
    else
        echo "软链接 $LINK 已存在且指向正确"
    fi
elif [ -e "$LINK" ]; then
    echo "警告: $LINK 已存在但不是软链接，建议手动处理"
else
    echo "创建软链接: $LINK -> $TARGET"
    ln -s "$TARGET" "$LINK"
fi

# 飞书通知函数
FEISHU_WEBHOOK="__FEISHU_WEBHOOK__"
send_feishu_msg() {
    local message="$1"
    curl -X POST "$FEISHU_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{
            \"msg_type\": \"text\",
            \"content\": {
                \"text\": \"$message\"
            }
        }" > /dev/null 2>&1 || true
}

# 训练参数
MODEL="__MODEL_PATH__"
DATASET="__DATASET_PATHS__"
OUTPUT_DIR="__OUTPUT_DIR__"
PLUGIN_FILE="__PLUGIN_FILE__"
export OUTPUT_DIR=$OUTPUT_DIR

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录训练开始
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
send_feishu_msg "🚀 训练开始
时间: $START_TIME
节点: $(hostname)
任务: __TASK_NAME__
模型: Qwen2.5-VL-__MODEL_SIZE__
超参: __HYPERPARAM_ID__
输出: $OUTPUT_DIR"

echo "=" | tr '=' '-' | head -c 70 && echo
echo "训练开始"
echo "=" | tr '=' '-' | head -c 70 && echo
echo "时间: $START_TIME"
echo "输出目录: $OUTPUT_DIR"
echo "=" | tr '=' '-' | head -c 70 && echo

# 启动训练
MAX_PIXELS=__MAX_PIXELS__ \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --dataset $DATASET \
    --external_plugins $PLUGIN_FILE \
    --reward_model_plugin  __REWARD_FUNCS__ \
    --reward_weights __REWARD_WEIGHTS__ \
    --beta __BETA__ \
    --train_type __TRAIN_TYPE__ \
    --freeze_vit __FREEZE_VIT__ \
    --max_grad_norm __MAX_GRAD_NORM__ \
    --torch_dtype bfloat16 \
    --max_completion_length 2048 \
    --num_train_epochs __NUM_EPOCHS__ \
    --per_device_train_batch_size __PER_DEVICE_BS__ \
    --per_device_eval_batch_size __PER_DEVICE_BS__ \
    --learning_rate __LEARNING_RATE__ \
    --gradient_accumulation_steps __GRAD_ACCUM_STEPS__ \
    --eval_strategy epoch \
    --save_strategy epoch  \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio __WARMUP_RATIO__ \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --num_generations __NUM_GENERATIONS__ \
    --temperature 1.0 \
    --top_p 1.0 \
    --deepspeed __DEEPSPEED__ \
    --log_completions true \
    --split_dataset_ratio 0.05

# 记录训练结束
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
TRAIN_STATUS=$?

if [ $TRAIN_STATUS -eq 0 ]; then
    send_feishu_msg "✅ 训练完成
任务: __TASK_NAME__
超参: __HYPERPARAM_ID__
开始: $START_TIME
结束: $END_TIME
输出: $OUTPUT_DIR"
    echo ""
    echo "=" | tr '=' '-' | head -c 70 && echo
    echo "✅ 训练完成"
    echo "=" | tr '=' '-' | head -c 70 && echo
else
    send_feishu_msg "❌ 训练失败
任务: __TASK_NAME__
超参: __HYPERPARAM_ID__
开始: $START_TIME
失败: $END_TIME
退出码: $TRAIN_STATUS
输出: $OUTPUT_DIR"
    echo ""
    echo "=" | tr '=' '-' | head -c 70 && echo
    echo "❌ 训练失败 (退出码: $TRAIN_STATUS)"
    echo "=" | tr '=' '-' | head -c 70 && echo
    exit $TRAIN_STATUS
fi
SCRIPT_TEMPLATE

# 替换占位符
sed -i "s|__TIMESTAMP__|${TIMESTAMP}|g" "${SCRIPT_NAME}"
sed -i "s|__TASK_NAME__|${TASK_NAME}|g" "${SCRIPT_NAME}"
sed -i "s|__MODEL_SIZE__|${MODEL_SIZE}|g" "${SCRIPT_NAME}"
sed -i "s|__HYPERPARAM_ID__|${HYPERPARAM_ID}|g" "${SCRIPT_NAME}"
sed -i "s|__CUDA_DEVICES__|${CUDA_DEVICES}|g" "${SCRIPT_NAME}"
sed -i "s|__NUM_GPUS__|${NUM_GPUS}|g" "${SCRIPT_NAME}"
sed -i "s|__FEISHU_WEBHOOK__|${FEISHU_WEBHOOK}|g" "${SCRIPT_NAME}"
sed -i "s|__MODEL_PATH__|${MODEL_PATH}|g" "${SCRIPT_NAME}"
# DATASET_PATHS已经是字符串格式，直接替换
sed -i "s|__DATASET_PATHS__|${DATASET_PATHS}|g" "${SCRIPT_NAME}"
sed -i "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" "${SCRIPT_NAME}"
sed -i "s|__PLUGIN_FILE__|${PLUGIN_FILE}|g" "${SCRIPT_NAME}"
sed -i "s|__MAX_PIXELS__|${MAX_PIXELS}|g" "${SCRIPT_NAME}"
sed -i "s|__REWARD_FUNCS__|${REWARD_FUNCS}|g" "${SCRIPT_NAME}"
sed -i "s|__REWARD_WEIGHTS__|${REWARD_WEIGHTS}|g" "${SCRIPT_NAME}"
sed -i "s|__BETA__|${BETA}|g" "${SCRIPT_NAME}"
sed -i "s|__TRAIN_TYPE__|${TRAIN_TYPE}|g" "${SCRIPT_NAME}"
sed -i "s|__FREEZE_VIT__|${FREEZE_VIT}|g" "${SCRIPT_NAME}"
sed -i "s|__MAX_GRAD_NORM__|${MAX_GRAD_NORM}|g" "${SCRIPT_NAME}"
sed -i "s|__NUM_EPOCHS__|${NUM_EPOCHS}|g" "${SCRIPT_NAME}"
sed -i "s|__PER_DEVICE_BS__|${PER_DEVICE_BS}|g" "${SCRIPT_NAME}"
sed -i "s|__LEARNING_RATE__|${LEARNING_RATE}|g" "${SCRIPT_NAME}"
sed -i "s|__GRAD_ACCUM_STEPS__|${GRAD_ACCUM_STEPS}|g" "${SCRIPT_NAME}"
sed -i "s|__WARMUP_RATIO__|${WARMUP_RATIO}|g" "${SCRIPT_NAME}"
sed -i "s|__NUM_GENERATIONS__|${NUM_GENERATIONS}|g" "${SCRIPT_NAME}"
sed -i "s|__DEEPSPEED__|${DEEPSPEED}|g" "${SCRIPT_NAME}"

# 赋予执行权限
chmod +x "${SCRIPT_NAME}"

echo ""
echo "✓ 训练脚本已生成: ${SCRIPT_NAME}"
echo ""

# ============================================================================
# 【执行选项】
# ============================================================================

read -p "是否立即执行训练? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始执行训练..."
    echo ""
    bash "${SCRIPT_NAME}"
else
    echo "训练脚本已生成但未执行"
    echo "执行命令: bash ${SCRIPT_NAME}"
fi

