# Evaluation Data Preparation (Image-Level VQA)

All our datasets are placed under the folder `data/`.


## Image Captioning

### COCO

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

# Step 3: Download and place the annotation files
mkdir -p annotations && cd annotations/
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test.json
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test_gt.json

cd ../../..
```

After prepartion is complete, the directory structure is:

```

data/coco
├── annotations
│   ├── coco_karpathy_test.json
│   └── coco_karpathy_test_gt.json
├── train2014/
├── val2014/
└── test2015/
```

### Flickr30K

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/flickr30k && cd data/flickr30k

# Step 2: Download and unzip image files
# Download images from https://bryanplummer.com/Flickr30kEntities/

# Step 3: Download and place the annotation files
# (Optional) Karpathy split annotations can be downloaded from the following link:
wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt
# This file is provided by the clip-benchmark repository.
# We convert this txt file to json format, download the converted file:
wget https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json

cd ../..
```

After preparation is complete, the directory structure is:

```
data/flickr30k
├── Images
├── flickr30k_test_karpathy.txt
└── flickr30k_test_karpathy.json
```


### NoCaps

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/nocaps && cd data/nocaps

# Step 2: Download and unzip image files
# Download images from https://nocaps.org/download

# Step 3: Download and place the annotation files
# Original annotations can be downloaded from https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json

cd ../..
```

After preparation is complete, the directory structure is:

```
data/nocaps
├── images
└── nocaps_val_4500_captions.json
```


## General VQA

[COCO](https://cocodataset.org/#home) images are used in VQAv2, POPE, and so on. Make sure you have already downloaded COCO images before evaluating on these benchmarks.

```
data/coco
├── train2014
├── val2014
└── test2015
```


### VQAv2

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/vqav2 && cd data/vqav2

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# Step 3: Download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/vqav2
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_train2014_complementary_pairs.json
├── v2_mscoco_val2014_annotations.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_OpenEnded_mscoco_test-dev2015_questions.json
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── vqav2_testdev.jsonl
├── vqav2_train.jsonl
└── vqav2_val.jsonl
```


### GQA

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/gqa && cd data/gqa

# Step 2: Download the official evaluation script
wget https://nlp.stanford.edu/data/gqa/eval.zip
unzip eval.zip

# Step 3: Download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/testdev_balanced.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/train_balanced.jsonl
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_gqa_testdev_balanced_qwen_format.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/gqa
├── challenge_all_questions.json
├── challenge_balanced_questions.json
├── eval.py
├── images/
├── llava_gqa_testdev_balanced_qwen_format.jsonl
├── readme.txt
├── submission_all_questions.json
├── test_all_questions.json
├── test_balanced.jsonl
├── test_balanced_questions.json
├── testdev_all_questions.json
├── testdev_balanced_all_questions.json
├── testdev_balanced_predictions.json
├── testdev_balanced_questions.json
├── train_all_questions
├── train_balanced.jsonl
├── train_balanced_questions.json
├── val_all_questions.json
└── val_balanced_questions.json
```

### VizWiz

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/vizwiz && cd data/vizwiz

# Step 2: Download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip && unzip train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip && unzip val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip && unzip test.zip

# Step 3: Download annotations
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip && unzip Annotations.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/vizwiz
├── annotations/
├── test/
├── train/
├── val/
├── vizwiz_test.jsonl
├── vizwiz_train_annotations.json
├── vizwiz_train.jsonl
├── vizwiz_train_questions.json
├── vizwiz_val_annotations.json
├── vizwiz_val.jsonl
└── vizwiz_val_questions.json
```

### ScienceQA

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# Step 2: Download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ..

# Step 3: Download original questions
wget https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/scienceqa
├── images/
├── problems.json
└── scienceqa_test_img.jsonl
```


### TextVQA

Follow the instructions below to prepare the data：


```
# Step 1: Create the data directory
mkdir -p data/textvqa && cd data/textvqa

# Step 2: Download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# Step 3: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val_llava.jsonl

cd ../..
```


After preparation is complete, the directory structure is:

```
data/textvqa
├── TextVQA_Rosetta_OCR_v0.2_test.json
├── TextVQA_Rosetta_OCR_v0.2_train.json
├── TextVQA_Rosetta_OCR_v0.2_val.json
├── textvqa_train_annotations.json
├── textvqa_train.jsonl
├── textvqa_train_questions.json
├── textvqa_val_annotations.json
├── textvqa_val.jsonl
├── textvqa_val_llava.jsonl
├── textvqa_val_questions.json
└── train_images/
```


## Multimodal Benchmarks

### POPE

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/pope && cd data/pope

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# Step 3: Download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

After preparation is complete, the directory structure is:

```
data/pope
├── coco
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── llava_pope_test.jsonl
└── val2014/
```


### MME

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/mme && cd data/mme

# Step 2: Download MME_Benchmark_release_version.zip
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip
mv MME_Benchmark_release_version images

cd ../..
```

After preparation is complete, the directory structure is:

```
data/mme
└── images/
```
 
 
### MMBench

Follow the instructions below to prepare the data：

```
# Step 1: Create the data directory
mkdir -p data/mmbench && cd data/mmbench

# Step 2: Download csv files
wget http://opencompass.openxlab.space/utils/MMBench/CCBench_legacy.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv

cd ../..
```

After preparation is complete, the directory structure is:

```
data/mmbench
├── CCBench_legacy.tsv
├── mmbench_dev_20230712.tsv
├── mmbench_dev_cn_20231003.tsv
├── mmbench_dev_en_20231003.tsv
├── mmbench_test_cn_20231003.tsv
└── mmbench_test_en_20231003.tsv
```


### SEED

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/SEED && cd data/SEED

# Step 2: Download the dataset
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/SEED-Bench-image.zip
unzip SEED-Bench-image.zip
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/SEED-Bench-old-video-image-1.zip
unzip SEED-Bench-old-video-image-1.zip
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/seed.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/SEED
├── SEED-Bench-image/
├── SEED-Bench-old-video-image-1/
└── seed.jsonl
```
