#!/bin/bash

export PYTHONPATH=/data/dlf/code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPROC_PER_NODE=8
export MASTER_PORT=12346

# 训练参数配置
MODEL="/data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct " 
DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl /data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl"  
#VAL_DATASET="/data/dlf/code/Field-Fidelity/data/vqav2/formatted/vqav2_val_formatted.jsonl"
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/dapo_atomic/fvit"
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/grpo_reward.py"
#Where is the dog and its bed
# 启动GRPO训练
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --dataset $DATASET \
    --external_plugins $PLUGIN_FILE \
    --reward_funcs format soft_overlong  \
    --reward_model /data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct  \
    --reward_model_plugin idk_genrm \
    --reward_weights 0 1 1 \
    --beta 0.0 \
    --train_type full \
    --freeze_vit true \
    --max_grad_norm 1.0 \
    --torch_dtype bfloat16 \
    --max_completion_length 8192 \
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
    --dataset_num_proc 4 \
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
    --soft_cache_length	1024 #\
    #--resume_from_checkpoint /data/dlf/code/Field-Fidelity/outputs/experiments/grpo_sky/dapo_fvit/v1-20251105-094133/checkpoint-692
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

    
 
