# VisionLLM [[Paper]()] 


<!-- ## Description -->

Large language models (LLMs) have notably accelerated progress towards artificial general intelligence (AGI), with their impressive zero-shot capacity for user-tailored tasks, endowing them with immense potential across a range of applications. However, in the field of computer vision, despite the availability of numerous powerful vision foundation models (VFMs), they are still restricted to tasks in a pre-defined form, struggling to match the open-ended task capabilities of LLMs. In this work, we present an LLM-based framework for vision-centric tasks, termed **VisionLLM**. This framework provides a unified perspective for vision and language tasks by treating images as a foreign language and aligning vision-centric tasks with language tasks that can be flexibly defined and managed using language instructions. An LLM-based decoder can then make appropriate predictions based on these instructions for open-ended tasks. Extensive experiments show that the proposed VisionLLM can achieve different levels of task customization through language instructions, from fine-grained object-level to coarse-grained task-level customization, all with good results. It's noteworthy that, with a generalist LLM-based framework, our model can achieve over **60\% mAP on COCO**, on par with detection-specific models. We hope this model can set a new baseline for generalist vision and language models. 
 
## 🤖💬 Online Demo

[**NOTE**] VisionLLM will be integrated into [InternGPT](https://github.com/OpenGVLab/InternGPT).

**InternGPT** is online (see [https://igpt.opengvlab.com](https://igpt.opengvlab.com/)). Let's try it!


## 🗓️ Schedule
- [ ] Integrate into [InternGPT](https://github.com/OpenGVLab/InternGPT)
- [ ] Release code and models

## 🏠 Overview
<img width="800" alt="image" src="https://github.com/OpenGVLab/VisionLLM/assets/23737120/8fb174ed-4df7-490d-85be-4ebb524fc4c6">

## 🎁 Major Features 
<img width="800" alt="image" src="https://github.com/OpenGVLab/VisionLLM/assets/23737120/94a23c43-3919-40f1-88ea-a270e4796979">

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE). 

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2023visionllm,
    title={VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks},
    author={Wenhai Wang and Zhe Chen and Xiaokang Chen and Jiannan Wu and Xizhou Zhu and Gang Zeng and Ping Luo and Tong Lu and Jie Zhou and Yu Qiao and Jifeng Dai},
    howpublished = {\url{todo}},
    year={2023}
}
```
