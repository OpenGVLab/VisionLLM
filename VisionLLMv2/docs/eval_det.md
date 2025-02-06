# Evaluation (Object Detection)


Before evaluation, please follow the instructions in [model preparion](./model.md) and [evaluation data prepration (object detection)](./data_det.md) to prepare the model and data.

## Object Detection & Instance Segmentation

### COCO

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/det/coco_val.py
```

### CrowdHuman

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/det/crowdhuman_val.py

# evaluation
python3 visionllmv2/datasets/evaluation/crowdhuman_eval.py
```


### OdinW13

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/det/odinw13_val.py
```


## Referring Expression Comprehension & Segmentation

### RefCOCO/+/g

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/grd/refcoco_val.py
```

### ReasonSeg

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/grd/reasonseg_val.py
```


### Interactive Segmentation

### COCO

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_det.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/visual_prompt/coco_val.py
```