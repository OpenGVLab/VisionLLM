# Evaluation Data Preparation (Pose Estimation)


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
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

### CrowdPose

Follow the instructions below to prepare the data:

```
# Step 1: Create the data directory
mkdir -p data/crowdpose && cd data/crowdpose

# Step 2: Download the images and annotation files
# from official website https://github.com/jeffffffli/CrowdPose
```

After prepartion is complete, the directory structure is:

```
data/crowdpose
├── annotations
│   ├── crowdpose_train.json
│   ├── crowdpose_val.json
│   └── crowdpose_test.json
└── images/
```



### UniKPT

UniKPT is collected by [X-Pose](https://github.com/IDEA-Research/X-Pose) (originally named as [UniPose](https://github.com/IDEA-Research/X-Pose/issues/7)), which include 13 pose datasets. As X-pose did not provide evaluation annotatoin file and the released [training annotation file](https://drive.usercontent.google.com/download?id=1ukLPbTpTfrCQvRY2jY52CgRi-xqvyIsP&export=download&authuser=0) did not specify the data source for each dataset, we collect some of the datasets by ourselves and transform the annotation files into COCO format.

Follow the instructions below to prepare the data:

```
# Step 1: Enter the data directory
cd data

# Step 2: Download the annotation files
wget https://huggingface.co/OpenGVLab/VisionLLMv2/resolve/main/data/pose_annotations.zip
zip -qq pose_annotations.zip && rm -rf pose_annotations.zip

# Step 3: Please download and place the images for each dataset by yourselves
```

After prepartion is complete, the directory structure is (We list the datasets for evaluation here):

```
data/pose_datasets
├── HumanArt
│   ├── annotations
│   │   ├── training_humanart.json
│   │   └── validation_humanart.json
│   └── images/
├── AP10K
│   ├── train.json
│   ├── test.json
│   └── ap-10k/
├── macaque
│   ├── annotations
│   │   ├── macaque_train.json
│   │   └── macaque_test.json
│   └── v1/
├── 300w
│   ├── train.json
│   ├── face_landmarks_300w_train.json
│   ├── face_landmarks_300w_valid.json
│   ├── face_landmarks_300w_test.json
│   └── dataset/
├── animalkingdom
│   ├── train.json
│   ├── train_sambox.json
│   ├── test.json
│   └── dataset/
├── VinegarFly
│   ├── annotations
│   │   ├── fly_train.json
│   │   └── fly_test.json
│   └── images/
├── DesertLocust/
│   ├── annotations
│   │   ├── locust_train.json
│   │   └── locust_test.json
└── └── images/

```