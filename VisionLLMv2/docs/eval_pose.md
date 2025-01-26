# Evaluation (Pose Estimation)

Before evaluation, please follow the instructions in [model preparion](./model.md) and [evaluation data prepration (pose estimation)](./data_pose.md) to prepare the model and data.

### COCO & CrowdPose & UniKPT

Specify the datasets you would like to evaluate in [visionllmv2/datasets/configs/pose/unikpt_val.py](https://github.com/OpenGVLab/VisionLLM/blob/release/VisionLLMv2/visionllmv2/datasets/configs/pose/unikpt_val.py).

```
GPUS=8 bash scripts/vllmv2_7b/eval/dist_eval_pose.sh work_dirs/VisionLLMv2 visionllmv2/datasets/configs/pose/unikpt_val.py
```