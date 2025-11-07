#!/bin/bash

export PYTHONPATH=/data/dlf/code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export NPROC_PER_NODE=4
export MASTER_PORT=12345

# 训练参数配置


MODEL="/data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct " 
DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl /data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl"  
#VAL_DATASET="/data/dlf/code/Field-Fidelity/data/vqav2/formatted/vqav2_val_formatted.jsonl"
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/grpo/epoch-5_ng-64"
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/grpo_skyrm.py"

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
    --reward_weights 0.3 0.7 \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --max_completion_length 16384 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 4 \
    --eval_steps 300 \
    --save_steps 600 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 32768 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --num_generations 64 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k 50 \
    --deepspeed zero2 \
    --log_completions true \
    --split_dataset_ratio 0.05 \
    --system /data/dlf/code/Field-Fidelity/src/train/prompt/system.txt 
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

    
 
