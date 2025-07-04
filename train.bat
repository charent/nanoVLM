@echo off
set CUDA_VISIBLE_DEVICES=0,1,2,3
set CUDA_DEVICE_MAX_CONNECTIONS=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set NCCL_P2P_DISABLE=1

set TORCH_CUDA_ARCH_LIST=8.9
set USE_LIBUV=0
REM awq lora must be fp16
set USE_LORA=False
set Q_LORA=False

accelerate launch --main_process_port 29508 ^
    --config_file accelerate_config\accelerate_config.yaml ^
    train.py ^
    --model_name_or_path model_save\my_vlm_model ^
    --train_file_dir dataset_dir ^
    --use_lora %USE_LORA% ^
    --q_lora %Q_LORA% ^
    --lora_r 16 ^
    --lora_alpha 32 ^
    --bf16 true ^
    --fp16 false ^
    --data_seed 2333 ^
    --output_dir output_model\vlm_pretrain ^
    --num_train_epochs 2 ^
    --auto_find_batch_size False ^
    --per_device_train_batch_size 4 ^
    --per_device_eval_batch_size 4 ^
    --gradient_accumulation_steps 16 ^
    --eval_strategy "no" ^
    --save_strategy "epoch" ^
    --save_total_limit 10 ^
    --save_only_model true ^
    --load_best_model_at_end false ^
    --learning_rate 5e-5 ^
    --optim "adamw_torch" ^
    --warmup_ratio 0.01 ^
    --lr_scheduler_type "cosine" ^
    --logging_steps 1 ^
    --log_level "info" ^
    --report_to "swanlab" ^
    --warmup_steps 100 ^
    --gradient_checkpointing false ^
    --logging_first_step true ^
    --save_on_each_node false ^
    --ddp_find_unused_parameters false

REM     --neftune_noise_alpha 10 ^
        
REM     --save_strategy "steps" ^
REM     --save_steps 1000 ^
REM     --ddp_find_unused_parameters false ^
    
REM     --group_by_length true ^
REM     --length_column_name "inputs_length" 
REM     --deepspeed %DS_CONFIG_PATH%