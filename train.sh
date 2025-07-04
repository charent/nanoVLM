#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1

export TORCH_CUDA_ARCH_LIST=8.9

# awq lora must be fp16
USE_LORA=False
Q_LORA=False

accelerate launch --main_process_port 29508 \
    --config_file accelerate_config/accelerate_config_multi_gpu.yaml \
    train.py \
    --model_name_or_path  output_model/vlm_pretrain/checkpoint-145 \
    --train_file_dir dataset_dir \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 true \
    --fp16 false \
    --data_seed 2333 \
    --output_dir output_model/vlm_pretrain \
    --num_train_epochs 2 \
    --auto_find_batch_size False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --save_only_model true \
    --load_best_model_at_end false \
    --learning_rate 5e-5 \
    --optim "adamw_torch" \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --log_level "info" \
    --report_to "swanlab" \
    --warmup_steps 10 \
    --gradient_checkpointing false \
    --logging_first_step true \
    --save_on_each_node false \
    --ddp_find_unused_parameters false 

    # --neftune_noise_alpha 10 \
        
    # --save_strategy "steps" \
    # --save_steps 1000 \
    # --ddp_find_unused_parameters false \
    
    # --group_by_length true \
    # --length_column_name "inputs_length" 
    # --deepspeed ${DS_CONFIG_PATH}