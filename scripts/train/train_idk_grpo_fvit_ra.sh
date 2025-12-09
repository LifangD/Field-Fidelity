#!/bin/bash

export PYTHONPATH=/data/dlf/code:$PYTHONPATH
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export NPROC_PER_NODE=6
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
    echo "请确认IP!"
fi

# 创建软链接：将jsonl中的图片路径 /data/dlf 映射到实际位置 ${DATA_PREFIX}/dlf_data
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

# 飞书 Webhook 配置
FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/f120b7d5-8205-4f01-bf1a-86a9e50984a7"

# 发送飞书消息函数
send_feishu_msg() {
    local message="$1"
    curl -X POST "$FEISHU_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{
            \"msg_type\": \"text\",
            \"content\": {
                \"text\": \"$message\"
            }
        }"
}

# 训练参数配置
MODEL="${DATA_PREFIX}/hub/models/Qwen/Qwen2___5-VL-7B-Instruct " 
DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl \
/data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl "  

#DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl /data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl /data/dlf/code/Field-Fidelity/data/if_multi_constraints/formatted/if_multi_constraints_formatted.jsonl"  
#VAL_DATASET="/data/dlf/code/Field-Fidelity/data/vqav2/formatted/vqav2_val_formatted.jsonl"
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/grpo_fvit/reward_anything"
##PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/grpo_skyrm_rewrite.py"
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/reward_anything.py"
export OUTPUT_DIR=$OUTPUT_DIR

# 记录训练开始时间
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
send_feishu_msg "🚀 训练开始\n时间: $START_TIME\n节点: $(hostname)\n模型: Qwen2.5-VL-7B\n输出目录: $OUTPUT_DIR"
# 启动GRPO训练
# 启动GRPO训练
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --dataset $DATASET \
    --external_plugins $PLUGIN_FILE \
    --reward_funcs format \
    --reward_model /data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct  \
    --reward_model_plugin idk_genrm \
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
    --gradient_accumulation_steps 4 \
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
    #--system /data/dlf/code/Field-Fidelity/src/train/prompt/system.txt 
    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.5 \
    # --vllm_max_model_len 16384
    # --lora_rank 16 \
    # --lora_alpha 32 \
    # --target_modules all-linear \

    
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NPROC_PER_NODE=8 \
# swift rlhf \
#     --rlhf_type grpo \
#     --model Qwen/Qwen2.5-7B \
#     --dataset AI-MO/NuminaMath-TIR#5000 \
#     --load_from_cache_file true \
#     --use_vllm true \
#     --vllm_mode colocate \
#     --vllm_gpu_memory_utilization 0.5 \
#     --external_plugins examples/train/grpo/plugin/plugin.py \
#     --reward_funcs format \
#     --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
#     --reward_model_plugin genrm my_rmplugin \
#     --reward_weights 0.1 1 1 \
#     --sleep_level 1 \
#     --offload_model true \
#     --offload_optimizer true \
#     --log_completions true \
#     --deepspeed zero2

    
 
