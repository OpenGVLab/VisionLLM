# Evaluation Data Preparation (Region-Level VQA)

All our datasets are placed under the folder `data/`.


## Region Captioning

### RefCOCOg

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/coco2014 && cd data/coco2014

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
cd ..

# Step 3: Download and place the annotation files
# The RefCOCO/+/g caption annotation files are converted from https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1 into COCO caption format.
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/mdetr_annotations.zip
unzip -qq mdetr_annotations.zip && rm -rf mdetr_annotations.zip

cd ..
```


After prepartion is complete, the directory structure is:

```
data/
├── coco2014/
│   └── train2014/
├── mdetr_annotations/
└── └── refcocog_test_coco_format.json
```



### Visual Genome


Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/vg && cd data/vg

# Step 2: Download and unzip image files
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip && unzip -qq images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip && unzip -qq images2.zip

# Step 3: Download and place the annotation files
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/vg_annotations.zip
unzip -qq vg_annotations.zip && rm -rf vg_annotations.zip

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/vg
├── annotations
│   ├── vg_test_coco_format.json
│   └── vg_test_caption_coco_format.json
├── VG_100K/
└── VG_100K_2/
```

Note: [test.json](https://datarelease.blob.core.windows.net/grit/VG_preprocessed_annotations/test.json) is the full set of test set, provided by [GRiT](https://github.com/JialianW/GRiT). We convert this file into the COCO caption format as `vg_test_coco_format.json`.         
[test_caption.json](https://drive.google.com/file/d/1zF3UGHU1rvgTujinqJ-hZtrCBVsfsuel/view) is a subset of the test set, provided by [GLaMM](https://github.com/mbzuai-oryx/groundingLMM). We convert this file into the COCO caption format as `vg_test_caption_coco_format.json`.



## Region Recognition / Classification

### COCO

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip

# Step 3: Download and place the annotation files
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/coco
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

### LVIS & PACO

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/osprey_val && cd data/osprey_val

# Step 2: Make sure you have downloaded COCO images
# and place them in data/coco

# Step 3: Download and place the annotation files
wget https://huggingface.co/datasets/sunshine-lwt/Osprey-ValData/resolve/main/lvis_val_1k_category.json?download=true
wget https://huggingface.co/datasets/sunshine-lwt/Osprey-ValData/resolve/main/paco_val_1k_category.json?download=true

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/
├── coco/
│   └── val2017/
├── osprey_val/
│   ├── lvis_val_1k_category.json
└── └── paco_val_1k_category.json
```

## Visual Commensense Reasoning

### VCR


Follow the instructions below to prepare the data:

```
Step 1: Create the data directory
mkdir -p data/vcr && cd data/vcr

Step 2: Download and unzip image files
wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip && unzip vcr1images.zip

Step 3: Download and place the annotation files
# We convert the original annotations files into the llava chat format.
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/vcr_annotations.zip && unzip vcr_annotations.zip
mv vcr_annotations/*.jsonl .
rm -rf vcr_annotations

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/vcr
├── vcr1images/
└── vcrvqa_val.jsonl
```