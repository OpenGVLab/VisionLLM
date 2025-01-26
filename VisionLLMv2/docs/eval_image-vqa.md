# Evaluation (Image-Level VQA)

Before evaluation, please follow the instructions in [model preparion](./model.md) and [evaluation data prepration (image-level VQA)](./data_image-vqa.md) to prepare the model and data.


## Image Captioning

### COCO karpathy test & Flickr30K karpathy test & NoCaps val

Specify the datasets (`flickr30k`, `coco`, `nocaps`) you would like to evaluate in [visionllmv2/eval/eval_image_caption.py](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/visionllmv2/eval/eval_image_caption.py#L333).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_image_caption.sh work_dirs/VisionLLMv2
```


## General VQA

### VQAv2 val & test-dev

Specify the datasets (`vqav2_val`, `vqav2_testdev`) you would like to evaluate in [visionllmv2/eval/vqa/evaluate_vqa.py](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/visionllmv2/eval/vqa/evaluate_vqa.py#L574).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_vqa.sh work_dirs/VisionLLMv2
```

For the test-dev set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

Note: The evaluation of VQAv2 is very time-consuming.


### GQA test-dev

Specify the datasets (`gqa_testdev_llava`) you would like to evaluate in [visionllmv2/eval/vqa/evaluate_vqa.py](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/visionllmv2/eval/vqa/evaluate_vqa.py#L574).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_vqa.sh work_dirs/VisionLLMv2
```

### VizWiz val & test

Specify the datasets (`vizwiz_val`, `vizwiz_test`) you would like to evaluate in [visionllmv2/eval/vqa/evaluate_vqa.py](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/visionllmv2/eval/vqa/evaluate_vqa.py#L574).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_vqa.sh work_dirs/VisionLLMv2
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/my-submission).


### TextVQA val

Specify the datasets (`textvqa_val_ocr`) you would like to evaluate in [visionllmv2/eval/vqa/evaluate_vqa.py](https://github.com/OpenGVLab/VisionLLM/blob/7befe44a38f874fba6835445dbd0177f0b6b46d9/VisionLLMv2/visionllmv2/eval/vqa/evaluate_vqa.py#L574).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_vqa.sh work_dirs/VisionLLMv2
```

### ScienceQA test

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_scienceqa.sh work_dirs/VisionLLMv2
```


## Multimodal Benchmarks

### POPE

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_pope.sh work_dirs/VisionLLMv2
```
 
 The reported number is the average score of F1 scores.


### MME

```
# single GPU testing
CUDA_VISIBLE_DEVICES=0 bash scripts/vllmv2_7b/eval/dist_eval_mme.sh work_dirs/VisionLLMv2
```

### MMBench

The evaluation is performed on EN/CN dev set.

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_mmbench.sh work_dirs/VisionLLMv2
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).


### SEED

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_seed.sh work_dirs/VisionLLMv2
```