# Evaluation (Region-Level VQA)


Before evaluation, please follow the instructions in [model preparion](./model.md) and [evaluation data prepration (region-level VQA)](./data_region-vqa.md) to prepare the model and data.


## Region Captioning

### RefCOCOg

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_region_caption_refcocog.sh work_dirs/VisionLLMv2
```

### Visual Genome

Specify the subset or full set (`data/vg/annotations/vg_test_caption_coco_format.json`, `data/vg/annotations/vg_test_coco_format.json`) you would like to evaluate in [scripts/vllmv2_7b/eval/dist_eval_region_caption_vg.sh](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/scripts/vllmv2_7b/eval/dist_eval_region_caption_vg.sh#L13).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_region_caption_vg.sh work_dirs/VisionLLMv2
```

Note: The evaluation of full set is very time-consuming.


## Region Recognition / Classification

### COCO

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_region_recognition.sh work_dirs/VisionLLMv2
```

### LVIS & PACO

We follow the evaluation in [Osprey](https://github.com/CircleRadon/Osprey) for these two datasets. Before evaluation, you need to download the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.

```
mkdir checkpoints && cd checkpoints

# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False sentence-transformers/all-MiniLM-L6-v2  --local-dir all-MiniLM-L6-v2

cd ..
```

Specify the datasets (`lvis`, `paco`) you would like to evaluate in [visionllmv2/eval/eval_region_classification.py](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/visionllmv2/eval/eval_region_classification.py#L381).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_region_classification.sh work_dirs/VisionLLMv2
```


## Visual Commensense Reasoning

### VCR

```
GPUS=8  bash scripts/vllmv2_7b/eval/dist_eval_region_vcr.sh work_dirs/VisionLLMv2
```