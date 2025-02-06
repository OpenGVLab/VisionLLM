# Model Preparation


|   Model Name    |                                       Vision Part                                       |                                 Language Part                                  |                           HF Link                           |  size |
| :-------------: | :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :---------------------------------------------------------: | :----: |
| VisionLLMv2  | [clip-vit-large](https://huggingface.co/openai/clip-vit-large-patch14-336) |   [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)   | [🤗 link](https://huggingface.co/OpenGVLab/VisionLLMv2)  |  7B |


Download the above model weight and place it in the `work_dirs/` folder.
```
mkdir work_dirs && cd work_dirs

# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/VisionLLMv2 --local-dir VisionLLMv2

cd ..
```

We also need other pretrained models to successfully load the VisionLLMv2 weight. These models are placed in the `checkpoints/` folder.

```
mkdir checkpoints && cd checkpoints

# stable-diffusion-v1.5
huggingface-cli download --resume-download --local-dir-use-symlinks False stable-diffusion-v1-5/stable-diffusion-v1-5  --local-dir stable-diffusion-v1-5

# instruct-pix2pix
huggingface-cli download --resume-download --local-dir-use-symlinks False timbrooks/instruct-pix2pix  --local-dir instruct-pix2pix

cd ..
```