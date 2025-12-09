#!/bin/bash
# ============================================================================
# 自动生成的训练脚本
# 生成时间: 20251208_095134
# 任务: painting_detection
# 模型: Qwen2.5-VL-7B
# 超参ID: bs16_lr1e6_ep3_gen4
# ============================================================================

set -e  # 遇到错误立即退出

# 环境配置
export PYTHONPATH=/data/dlf/code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NPROC_PER_NODE=4
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
FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/f120b7d5-8205-4f01-bf1a-86a9e50984a7"
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
MODEL="/data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
DATASET="/data/dlf/code/Visual-RFT/ch_painting/datasets/tasks/detection/train.jsonl"
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/painting_detection_7B/bs16_lr1e6_ep3_gen4"
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/detection_classification.py"
export OUTPUT_DIR=$OUTPUT_DIR

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录训练开始
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
send_feishu_msg "🚀 训练开始
时间: $START_TIME
节点: $(hostname)
任务: painting_detection
模型: Qwen2.5-VL-7B
超参: bs16_lr1e6_ep3_gen4
输出: $OUTPUT_DIR"

echo "=" | tr '=' '-' | head -c 70 && echo
echo "训练开始"
echo "=" | tr '=' '-' | head -c 70 && echo
echo "时间: $START_TIME"
echo "输出目录: $OUTPUT_DIR"
echo "=" | tr '=' '-' | head -c 70 && echo

# 启动训练
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --dataset $DATASET \
    --external_plugins $PLUGIN_FILE \
    --reward_funcs det_cls_format det_cls_acc\
    --reward_weights 1 1 \
    --beta 0.01 \
    --train_type full \
    --freeze_vit true \
    --max_grad_norm 1.0 \
    --torch_dtype bfloat16 \
    --max_completion_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_strategy epoch \
    --save_strategy epoch  \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --split_dataset_ratio 0.05

# 记录训练结束
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
TRAIN_STATUS=$?

if [ $TRAIN_STATUS -eq 0 ]; then
    send_feishu_msg "✅ 训练完成
任务: painting_detection
超参: bs16_lr1e6_ep3_gen4
开始: $START_TIME
结束: $END_TIME
输出: $OUTPUT_DIR"
    echo ""
    echo "=" | tr '=' '-' | head -c 70 && echo
    echo "✅ 训练完成"
    echo "=" | tr '=' '-' | head -c 70 && echo
else
    send_feishu_msg "❌ 训练失败
任务: painting_detection
超参: bs16_lr1e6_ep3_gen4
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
