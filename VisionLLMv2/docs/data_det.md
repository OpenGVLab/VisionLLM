# Evaluation Data Preparation (Object Detection)

## Object Detection & Instance Segmentation

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

### CrowdHuman

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/crowdhuman && cd data/crowdhuman

# Step 2: Make sure you have downloaded the images from official website 
# https://www.crowdhuman.org/download.html

# Step 3: Download and place the annotation files
# The annotation files are converted into COCO format.
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/crowdhuman_annotations.zip
unzip -qq crowdhuman_annotations.zip && rm -rf crowdhuman_annotations.zip

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/crowdhuman
├── annotations
│   ├── train.json
│   └── val.json
└── Images/
```


### OdinW13



Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/odinw && cd data/odinw

# Step 2: Please refer to https://github.com/microsoft/GLIP/blob/main/odinw/download.py to download the dataset.
# Note that OdinW35 include all the data for OdinW13.

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/odinw
├── AerialMaritimeDrone/
├── AmericanSignLanguageLetters/
...
```

## Referring Expression Comprehension & Segmentation

### RefCOCO/+/g

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/coco2014 && cd data/coco2014

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip

# Step 3: Download and place the annotation files
# The RefCOCO/+/g annotation files are converted from https://github.com/seanzhuh/SeqTR?tab=readme-ov-file into COCO format.
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/coco2014_annotations.zip
unzip -qq coco2014_annotations.zip && rm -rf coco2014_annotations.zip

cd ../..
```


After prepartion is complete, the directory structure is:

```
data/coco2014
├── annotations
│   ├── refcocog-umd/
│   ├── refcocoplus-unc/
│   ├── refcoco-unc/
│   └── ...
└── train2014/
```

### ReasonSeg

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/reasonseg && cd data/reasonseg

# Step 2: Download the images from official website https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy
unzip train.zip && unzip val.zip

# Step 3: Download and place the annotation files
# The annotation files are converted into COCO format.
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/reasonseg_annotations.zip
unzip -qq reasonseg_annotations.zip && rm -rf reasonseg_annotations.zip

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/reasonseg
├── annotations
│   ├── train.json
│   └── val.json
├── train/
└── val/
```


## Interactive Segmentation

### COCO

Follow the instructions below to prepare the data (We follow the evaluation from [PSALM](https://github.com/zamling/PSALM)):

```
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip

# Step 3: Download and place the annotation files from PSALM
# Download the annotation files from official website https://drive.google.com/file/d/1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3/view

cd ../..
```

After prepartion is complete, the directory structure is:

```
data/coco
├── coco_interactive_train_psalm.json
├── coco_interactive_val_psalm.json
├── train/
└── val/
```