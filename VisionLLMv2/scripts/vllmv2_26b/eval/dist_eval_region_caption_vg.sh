#!/bin/bash

OUTPUT_DIR=$1
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
    visionllmv2/eval/eval_region_caption_vg.py \
    --model_name ${OUTPUT_DIR} \
    --ann_file data/vg/annotations/vg_test_caption_coco_format.json \
    --img_prefix data/vg/VG_100K/ \
    --conv_mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True \
    --vis_encoder_path checkpoints/InternViT-6B-448px-V1-5


# e.g.
# bash scripts/vllmv2_26b/eval/dist_eval_region_caption_vg.sh work_dirs/visionllmv2-26b