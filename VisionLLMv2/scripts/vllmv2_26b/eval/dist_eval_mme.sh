#!/bin/bash

OUTPUT_DIR=$1
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
    python3 visionllmv2/eval/mme/eval.py \
    --model_name ${OUTPUT_DIR} \
    --conv_mode internlm2_chat \
    --image_aspect_ratio anyres \
    --image_size 448 \
    --image_max_tile 6 \
    --use_pixelshuffle True \
    --vis_encoder_path ${OUTPUT_DIR}

DIRNAME=`basename ${OUTPUT_DIR}`
python3 visionllmv2/eval/mme/calculation.py --results_dir ${DIRNAME}


# e.g.
# bash scripts/vllmv2_26b/eval/dist_eval_mme.sh work_dirs/visionllmv2-26b