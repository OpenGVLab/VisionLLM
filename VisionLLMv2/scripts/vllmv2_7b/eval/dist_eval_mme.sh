#!/bin/bash

OUTPUT_DIR=$1
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
    python3 visionllmv2/eval/mme/eval.py \
    --model_name ${OUTPUT_DIR} \
    --conv_mode vicuna_v1 \
    --image_aspect_ratio anyres \
    --image_size 336 \
    --image_max_tile 4 \
    --vis_encoder_path ${OUTPUT_DIR}

DIRNAME=`basename ${OUTPUT_DIR}`
python3 visionllmv2/eval/mme/calculation.py --results_dir ${DIRNAME}


# e.g.
# bash scripts/vllmv2_7b/eval/dist_eval_mme.sh work_dirs/visionllmv2-7b