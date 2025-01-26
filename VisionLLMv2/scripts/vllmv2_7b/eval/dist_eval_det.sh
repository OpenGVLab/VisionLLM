#!/bin/bash

OUTPUT_DIR=$1
CONFIG=$2
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
    visionllmv2/eval/eval_mem.py \
    --eval_only True \
    --version v1 \
    --dataset_config ${CONFIG} \
    --model_name_or_path ${OUTPUT_DIR} \
    --vis_encoder_path ${OUTPUT_DIR} \
    --vl_bridge_type mlp2x_gelu \
    --vis_output_layer -2 \
    --use_gdino True \
    --gdino_path checkpoints/grounding-dino-tiny \
    --freeze_vis_encoder True \
    --freeze_llm False \
    --use_llm_lora False \
    --lr_llm_multiplier 0.1 \
    --use_im_start_end False \
    --image_size 336 \
    --image_max_tile 4 \
    --use_pixelshuffle False \
    --image_aspect_ratio anyres \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 12 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --deepspeed scripts/zero2.json \
    --report_to none 

# e.g.
# bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/visionllmv2-7b visionllmv2/datasets/configs/det/coco_val.py