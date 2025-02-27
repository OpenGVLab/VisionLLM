#!/bin/bash

OUTPUT_DIR=$1
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
    visionllmv2/eval/eval_region_caption_refcoco.py \
    --model_name ${OUTPUT_DIR} \
    --ann_file data/mdetr_annotations/refcocog_val_coco_format.json \
    --img_prefix data/coco2014/train2014/ \
    --conv_mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True \
    --vis_encoder_path ${OUTPUT_DIR} \
    --test_format bbox


# e.g.
# bash scripts/vllmv2_26b/eval/dist_eval_region_caption_refcocog.sh work_dirs/visionllmv2-26b