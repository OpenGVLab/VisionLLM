#!/bin/bash

OUTPUT_DIR=$1
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
    visionllmv2/eval/eval_region_caption_vcr.py \
    --model_name ${OUTPUT_DIR} \
    --ann_file data/vcr/vcrvqa_val.jsonl \
    --img_prefix data/vcr/vcr1images/ \
    --conv_mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 \
    --vis_encoder_path checkpoints/clip-vit-large-patch14-336


# e.g.
# bash scripts/vllmv2_7b/eval/dist_eval_region_vcr.sh work_dirs/visionllmv2-7b