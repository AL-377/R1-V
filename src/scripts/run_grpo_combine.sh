cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.

# follow open-r1-multimodal to install the packages.

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
# for segments

# Add timestamp to log filename
LOG_FILE="/map-vepfs/ljt/R1-V/logs/training_log_$(date +%Y%m%d_%H%M%S).log"

# Add CUDA debugging flags
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir /map-vepfs/ljt/R1-V/models \
    --model_name_or_path /map-vepfs/huggingface/models/Qwen2-VL-2B-Instruct \
    # https://huggingface.co/datasets/Open-MMO1/virgo_qvqbo16_acc_0_3
    --dataset_name /map-vepfs/ljt/R1-V/data/data_0_3/virgo_refined_rl_sample_sharegpt_all_acc.json \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --num_train_epochs 4 \
    --run_name Qwen2-VL-2B-GRPO-Virgo-0_3-0_7-gen8-gradacc2-fixed \
    --save_steps 100 \
    --save_only_model true \
    --logging_first_step true 2>&1 | tee "$LOG_FILE"
