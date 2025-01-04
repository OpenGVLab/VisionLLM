# Installation Guide


## Environment

- CUDA11.8
- GCC7.3.0
- python3.9 / 3.10
- pytorch2.0.1
- transformers4.34


## Installation

- Clone the repository:

```
https://github.com/OpenGVLab/VisionLLM.git
```

- Enter the main folder:

```
cd VisionLLM/VisionLLMv2
```

- Create a conda virtual environment and activate it:

```
conda activate -n vllmv2 python==3.9
conda activate vllmv2
```

- Install the dependencies:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Then, please refer to the `install.sh` to install the necessary packages step by step.

- Additional:

`pycocoevalcap` is used to evaluate the metrics for image/region caption. You can install it by yourself. 
For the convenience, you can directly download it from [here](https://drive.google.com/file/d/1_haRVgvnhwMxjGIgwy3xdxgI9G8nWnaF/view?usp=drive_link). Unzip the file and put it under the main folder `VisionLLM/VisionLLMv2`.
