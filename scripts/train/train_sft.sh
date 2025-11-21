#!/bin/bash

export PYTHONPATH=/data/dlf/code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
export NPROC_PER_NODE=6
export MASTER_PORT=12346

# 训练参数配置
MODEL="/data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct " 
DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted_sft.jsonl /data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k_sft.jsonl"  
OUTPUT_DIR="/data/dlf/code/Field-Fidelity/outputs/experiments/sft"
PLUGIN_FILE="/data/dlf/code/Field-Fidelity/src/train/plugins/grpo_reward.py"


 MAX_PIXELS=1003520 swift sft \
  --model_type qwen2_5_vl \
  --model  $MODEL \
  --train_type full \
  --dataset $DATASET \
  --torch_dtype bfloat16 \
  --freeze_vit false \
  --freeze_llm false \
  --freeze_aligner false \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-6 \
  --gradient_accumulation_steps 4 \
  --eval_steps 300 \
  --save_steps 300 \
  --save_total_limit 5  \
  --logging_steps 50 \
  --max_length 32768 \
  --output_dir $OUTPUT_DIR \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --report_to tensorboard \
  --attn_impl flash_attn \
  --deepspeed zero2 \
  --run_name raw_sft_v001 \
  --split_ratio 0.95
    
 
