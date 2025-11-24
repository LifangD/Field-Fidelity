
## ppo_trainer不支持reward_funcs
nproc_per_node=8
DATASET="/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl /data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl"  

MAX_PIXELS=1003520 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type ppo \
    --model /data/share/hub/models/Qwen/Qwen2___5-VL-7B-Instruct \
    --reward_funcs format  \
    --external_plugins /data/dlf/code/Field-Fidelity/src/train/plugins/grpo_skyrm.py \
    --reward_model_plugin idk_genrm \
    --reward_weights 0.3 0.7 \
    --train_type full \
    --dataset $DATASET \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 300 \
    --save_steps 300 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir /data/dlf/code/Field-Fidelity/outputs/experiments/grpo_sky/ppo \
     --system /data/dlf/code/Field-Fidelity/src/train/prompt/system.txt \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --response_length 1024 \
    --temperature 0.7 \
    --dataset_num_proc 4 \
    --save_only_model true


    ### sft  pos