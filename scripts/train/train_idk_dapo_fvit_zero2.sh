#!/bin/bash

export PYTHONPATH=/data/dlf/code:$PYTHONPATH
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE=8
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
if [ ! -L "/data/dlf/code/Field-Fidelity/data" ] && [ ! -e "/data/dlf/code/Field-Fidelity/data" ]; then
    echo "创建软链接: /data/dlf/code/Field-Fidelity/data -> ${DATA_PREFIX}/dlf_data/Field-Fidelity/data"
    sudo ln -s ${DATA_PREFIX}/dlf_data/Field-Fidelity/data /data/dlf/code/Field-Fidelity/data
elif [ -L "/data/dlf/code/Field-Fidelity/data" ]; then
    echo "软链接 /data/dlf/code/Field-Fidelity/data 已存在"
else
    echo "警告: /data/dlf/code/Field-Fidelity/data 已存在但不是软链接"
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
/data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl \
/data/dlf/code/Field-Fidelity/data/if_multi_constraints/formatted/if_multi_constraints_formatted_5k.jsonl"  

#DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl /data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl /data/dlf/code/Field-Fidelity/data/if_multi_constraints/formatted/if_multi_constraints_formatted.jsonl"  
#VAL_DATASET="/data/dlf/code/Field-Fidelity/data/vqav2/formatted/vqav2_val_formatted.jsonl"
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/dapo_fvit/rewrite_neg_idk_if"
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/grpo_skyrm_rewrite.py"
export OUTPUT_DIR=$OUTPUT_DIR

# 记录训练开始时间
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
send_feishu_msg "🚀 训练开始\n时间: $START_TIME\n节点: $(hostname)\n模型: Qwen2.5-VL-7B\n输出目录: $OUTPUT_DIR"
# 启动GRPO训练
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --dataset $DATASET \
    --external_plugins $PLUGIN_FILE \
    --reward_funcs format soft_overlong  \
    --reward_model ${DATA_PREFIX}/hub/models/Qwen/Qwen2___5-VL-7B-Instruct  \
    --reward_model_plugin idk_genrm \
    --reward_weights 1 1 1 \
    --train_type full \
    --freeze_vit true \
    --max_grad_norm 1.0 \
    --torch_dtype bfloat16 \
    --max_completion_length 16384 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 692 \
    --save_steps 692 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 8192 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --split_dataset_ratio 0.05 \
    --loss_type	bnpo \
    --epsilon_high	0.28 \
    --dynamic_sample  true \
    --max_resample_times 3 \
    --overlong_filter	true \
    --beta 0.0 \
    --system /data/dlf/code/Field-Fidelity/src/train/prompt/system.txt \
    --soft_cache_length	2048 #\
    
    #--resume_from_checkpoint /data/dlf/code/Field-Fidelity/outputs/experiments/grpo_sky/dapo_fvit/v1-20251105-094133/checkpoint-692
    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.5 \
    # --vllm_max_model_len 16384
    # --lora_rank 16 \
    # --lora_alpha 32 \
    # --target_modules all-linear \

# 捕获训练退出状态
TRAIN_EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    send_feishu_msg "✅ 训练成功完成\n开始时间: $START_TIME\n结束时间: $END_TIME\n节点: $(hostname)\n输出目录: $OUTPUT_DIR"
else
    send_feishu_msg "❌ 训练失败\n开始时间: $START_TIME\n结束时间: $END_TIME\n节点: $(hostname)\n退出码: $TRAIN_EXIT_CODE\n输出目录: $OUTPUT_DIR"
fi

exit $TRAIN_EXIT_CODE
    

    
 
