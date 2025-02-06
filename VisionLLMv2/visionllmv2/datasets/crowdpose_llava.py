# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
import copy
import os
import numpy as np
from collections import defaultdict
from mmdet.datasets import CocoDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from PIL import Image

from .llava_data import preprocess, preprocess_multimodal
from ..mm_utils import expand2square, dynamic_preprocess
import visionllmv2.datasets.transforms.transform_crowdpose as T

import cv2
from crowdposetools.coco import COCO
from pathlib import Path

# 30 questions
QUESTIONS = [
    # interrogative
    "Can you analyze the image and identify the <class> present?",
    "In this image, could you detect all instances of <class>?",
    "Are you capable of identifying <class> within this image?",
    "Could you please detect the objects you find that belong to the <class> category in the image?",
    "Can you perform object detection on the image and tell me the <class> you find?",
    "I'm trying to detect <class> in the image. Can you help me?",
    "Can you carry out object detection on this image and identify the <class> it contains?",
    "In the context of the image, I'd like to know which objects fall under the category of <class>. Is that something you can do?",
    "I have an image that needs examination for objects related to <class>. Can you perform that?",
    "Can you determine if there are any <class> present in the image using object detection?",
    "Could you please carry out object detection on this image and list any <class> that you discover?",
    "Could you help me identify the objects corresponding to <class> in the provided image?",
    "Are you capable of detecting and labeling <class> objects within the image?",
    "I'm curious about the objects in the image that correspond to the <class> category. Could you assist in finding them?",
    "Can you detect <class> within the image and provide information about its presence?",
    # declarative
    "Please examine the image and let me know which objects fall under the <class> category.",
    "Please perform object detection on this image for identifying <class>.",
    "I need your expertise to locate <class> in this image.",
    "Please let me know the objects falling into the <class> category in the image.",
    "Please help me identify objects falling under the <class> category in this image.",
    "Please assist me in identifying the <class> objects within the image.",
    "Please provide a breakdown of all the <class> objects visible in the image.",
    "Please analyze the image and let me know if you can find any objects categorized as <class>.",
    "I'm seeking your help in identifying <class> within the contents of the image.",
    "Please conduct object detection on the image to locate any <class> that may be present.",
    "Please execute object detection on this image and provide details about any <class> you detect.",
    "Please identify and list any <class> in the given image using object detection.",
    "Please analyze the image and let me know if there are any recognizable <class> objects.",
    "Detect any <class> in the given image, if possible.",
    "I need assistance in recognizing the <class> shown in the image."
]

POSE_QUESTIONS = [
    "Can you examine the image and pinpoint the keypoint locations of the <class>?",
    "Could you analyze the picture and determine the keypoint placement of the <class>?",
    "Please inspect the image and locate the keypoints for <class>.",
    "Can you evaluate the photo and identify where the keypoints of <class> are situated?",
    "Look at the image and detect the keypoint positions of the <class>.",
    "Please analyze this image and find the keypoints of <class>.",
    "Can you check the image and show me where the keypoints of <class> are located?",
    "Please find the exact keypoint position of the <class>.",
    "Please observe the photo and identify the keypoint locations of the <class>.",
    "Can you review the image and point out the keypoints of <class>?"
]

# 10 yes
YES = [
    "Yes, here are the results for <class> in the image.",
    "Certainly, the image shows the results for <class>.",
    "Absolutely, you can see the results for <class> in the image.",
    "Yes, the detection results for <class> are presented.",
    "Certainly, the image does show the results of <class>.",
    "Certainly, you can spot the results of <class> in the image.",
    "Yes, there is a clear depiction for the results of <class>.",
    "Of course, the image provides a comprehensive results of <class>.",
    "Absolutely, the image showcases the results of <class>.",
    "Sure, the image contains the detection results for <class>."
]

POSE_ANS = [
    "Utilizing keypoints detection, the image analysis reveals the location of <class>.",
    "By focusing on keypoints in the image, you can accurately detect the position of <class>.",
    "The keypoints in the image indicate the precise location of <class>.",
    "Through detailed keypoints analysis, the exact position of <class> in the photo can be identified.",
    "KeyPoints detection techniques allow for the pinpointing of <class> in the image.",
    "In this image, the keypoints clearly show where the <class> is located.",
    "The image, when scanned for keypoints, reveals the specific location of <class>.",
    "By examining the keypoints, the <class> position in the image becomes evident.",
    "The location of <class> can be determined by analyzing the keypoints in this picture.",
    "KeyPoints detection in the image helps to accurately spot the <class>."
]


class CrowdposeLlavaDataset(torch.utils.data.Dataset):
    # incontinuous, in coco-format
    CLASSES = {
        1: 'person',
    }
    # continuous
    POSE_CLASSES = [
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
        "head",
        "neck"
    ]

    def __init__(self, 
                 ann_file, 
                 img_prefix,
                 # conversation
                 tokenizer, 
                 data_args, 
                 # keypoint
                 test_mode = False, 
                 return_masks=False
                 ):
        super(CrowdposeLlavaDataset, self).__init__()
        image_set = 'train' if not test_mode else 'val'
        self.image_set = image_set

        self.task = 'pose'
        self.dataset_name = 'crowdpose'

        # conversation
        self.tokenizer = tokenizer
        self.image_folder = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.num_embs = data_args.num_embs
        self.test_mode = test_mode
        
        # augmentation
        self._transforms = make_coco_transforms(image_set)
        self.prepare = ConvertCocoPolysToMask(return_masks)

        # get img ids
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        if image_set == 'train':
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = [] 
            # filter those imgs without person ann
            for image_id in imgIds:
                if self.coco.getAnnIds(imgIds=image_id) == []:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue
                self.all_imgIds.append(image_id)
        else:  # val
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = [] 
            for image_id in imgIds:
                    self.all_imgIds.append(image_id)

    
    def normalize_box_coordinates(self, bbox, img_shape):
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        img_h, img_w = img_shape[:2]
        bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
        return torch.cat(bbox_new, dim=-1)

    def get_conversations(self, target=None):

        ### conversataions
        class_list = list(self.CLASSES.values()) # coco person keypoint
        pose_class_list = list(self.POSE_CLASSES)
        conversations = []
        if self.image_set == 'train':
            random.shuffle(class_list)
            question_template = random.choice(QUESTIONS)
            answer_template = random.choice(YES)
            
        else:
            question_template = QUESTIONS[0]
            answer_template = YES[0]            
            
        # question
        class_list_str = ', '.join(class_list)
        question = question_template.replace('<class>', class_list_str)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        
        # answer 
        if self.num_embs == 1:
            class_list_str = "[DET][EMB], ".join(class_list)
            class_list_str += "[DET][EMB]"  # the last one
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
            str_temp = "[DET]" + str_temp   # e.g. "[DET][EMB][EMB2][EMB3][EMB4]"
            str_temp_aug = str_temp + ", "
            class_list_str = str_temp_aug.join(class_list)
            class_list_str += str_temp
        answer = answer_template.replace('<class>', class_list_str)
        message2 = {
            'from': 'gpt',
            'value': answer
        }

        #DO POSE
        if self.image_set == 'train':
            question_template = random.choice(POSE_QUESTIONS)
            answer_template = random.choice(POSE_ANS)

            if torch.randn(1) > 0:  # all pose classes
                random.shuffle(pose_class_list)
            else:  # random pose classes, at least 1
                random.shuffle(pose_class_list)
                num_rand = random.randint(1, len(pose_class_list))
                pose_class_list = pose_class_list[:num_rand]

        else:
            question_template = POSE_QUESTIONS[0]
            answer_template = POSE_ANS[0]

        # create pose class idx mapping, for training, change targets['keypoints']
        POSE_CLASS_DICT = {pose_class: i for i, pose_class in enumerate(list(self.POSE_CLASSES))}
        pose_class_list_ids = [POSE_CLASS_DICT[x] for x in pose_class_list]
        pose_class_list_ids = torch.as_tensor(pose_class_list_ids).long()
        # targets keypoints, keep the original kpt order, 
        # set the selected kpt classes as original values, others are 0
        if target['keypoints'] is not None:
            keypoints = target['keypoints']  # [num_gt, 68*3], xy in [0,1], xyxyzz
            num_pose_class = len(pose_class_list_ids)
            Z = keypoints[:, :68*2]
            X = Z[:, 0::2]  # [num_gt, 68]
            Y = Z[:, 1::2]  # [num_gt, 68]
            V = keypoints[:, 68*2:]  # [num_gt, 68]
            new_X = torch.zeros_like(X)
            new_Y = torch.zeros_like(Y)
            new_V = torch.zeros_like(V)
            new_X[:, pose_class_list_ids] = X[:, pose_class_list_ids]
            new_Y[:, pose_class_list_ids] = Y[:, pose_class_list_ids]
            new_Z = torch.stack([new_X, new_Y], dim=-1)   # [num_gt, 68, 2]
            new_Z = new_Z.reshape(keypoints.shape[0], 68*2) # [num_gt, 68*2]
            new_V[:, pose_class_list_ids] = V[:, pose_class_list_ids]  # [num_gt, 68]
            new_keypoints = torch.cat([new_Z, new_V], dim=-1)  # [num_gt, 68*3]
            target['keypoints'] = new_keypoints

        class_list_str = ', '.join(pose_class_list)
        question = question_template.replace('<class>', class_list_str)     

        if self.num_embs == 1:
            class_list_str = "[POSE][EMB], ".join(class_list)
            class_list_str += "[POSE][EMB]"  # the last one
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
            str_temp = "[POSE]" + str_temp  # e.g. "[POSE][EMB][EMB2][EMB3][EMB4]"
            str_temp_aug = str_temp + ", "
            class_list_str = str_temp_aug.join(pose_class_list)
            class_list_str += str_temp
        answer = answer_template.replace('<class>', class_list_str)
        # add space
        question = " " + question
        answer = " " + answer
        message1['value'] += question
        message2['value'] += answer
        #print(message1, message2)

        conversations.append(message1)
        conversations.append(message2)

        sources = preprocess_multimodal(copy.deepcopy([conversations]))
        return sources, class_list, pose_class_list, conversations, target

    def __getitem__(self, idx):
        # ------------------------------
        # keypoint
        if not self.test_mode:  # train
            image_id = self.all_imgIds[idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            target = self.coco.loadAnns(ann_ids)
            target = {'image_id': image_id, 'annotations': target}
        else:
            image_id = self.all_imgIds[idx]
            target = {'image_id': image_id, 'annotations': []}

        img = Image.open(os.path.join(self.image_folder, self.coco.loadImgs(image_id)[0]['file_name']))
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            transformed_img, target = self._transforms(img, target)
        
        image_aug = transformed_img
        image_fname = self.coco.loadImgs(image_id)[0]['file_name']

        # -------------------------------
        # conversation
        sources, class_list, pose_class_list, conversations, target = \
            self.get_conversations(target=target)  # class_list, pose_class_list may be shuffled during training

        # load image and clip preprocess
        processor = self.img_processor
        image = Image.open(os.path.join(self.image_folder, image_fname)).convert('RGB')
        if self.data_args.image_aspect_ratio == 'anyres':
            image = dynamic_preprocess(image, image_size=self.data_args.image_size, max_num=self.data_args.image_max_tile) # list[pil_img]
            image = [processor.preprocess(x, return_tensors='pt')['pixel_values'][0] for x in image]
            image = torch.stack(image)  # [1 + n_tile, 3, h, w]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
            image_token_len = image_token_len * len(image)
        elif self.data_args.image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
        else:  # resize
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
        
        data_dict = preprocess(
            sources=sources,  # list[list[dict]], first list length=1, second list length=num_rounds
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            has_image=True,
            image_token_len=image_token_len,
        ) # keys: "input_ids", "labels", size of [1, L]
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )
        data_dict['image'] = image

        # ------------------------------------
        # update data_dict
        # create continue label id to random index mapping
        name2index = {name: idx for idx, name in enumerate(class_list)}  
        id2index = {idx: name2index[name] for idx, name in self.CLASSES.items() if name in class_list}  # e.g. {1: 0}
        # create self.POSE_CLASSES id (start from 0) to pose_class_list index (start from 0) mapping
        kpt_name2index = {name: idx for idx, name in enumerate(pose_class_list)}
        kpt_id2index = {idx: kpt_name2index[name] for idx, name in enumerate(self.POSE_CLASSES) if name in pose_class_list} # e.g. {0: 2, 1: 0, 2:1}
        # change to same format as det/seg
        img_metas = {}
        img_metas['image_id'] = target['image_id']
        img_metas['ori_shape'] = target['orig_size']
        img_metas['img_shape'] = target['size']
        img_metas['id2index'] = id2index
        img_metas['kpt_id2index'] = kpt_id2index
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations
        if not self.test_mode:  # train
            gt_labels = target['labels']        # [num_gt], category_id
            gt_bboxes = target['boxes']         # [num_gt, 4], cxcywh in [0, 1]
            gt_keypoints = target['keypoints']  # [num_gt, 17*3], xy in [0, 1], xyxyzz
            gt_areas = target['area']           # [num_gt,], for keypoint oks loss
            data_dict_pose = {
                'image_aug': image_aug,
                'class_labels': gt_labels,
                'boxes': gt_bboxes,
                'keypoints': gt_keypoints,
                'area': gt_areas,
                'img_metas': img_metas,
            }
        else:
            data_dict_pose = {
                'image_aug': image_aug,
                'img_metas': img_metas,
            }
        data_dict.update(data_dict_pose)
        return data_dict
    
    def __len__(self):
        return len(self.all_imgIds)


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 14, 3)  
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        if self.return_masks:
            masks = masks[keep]
        target = {}
        target["boxes"] = boxes       # [num_gt, 4], xyxy in image_size
        target["labels"] = classes    # [num_gt,]
        if self.return_masks:  # false
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints  # [num_gt, 17, 3], xyz in image size
        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])  # ori image size
        target["size"] = torch.as_tensor([int(h), int(w)])       # after aug, before padding
        return image, target

def make_coco_transforms(image_set, args=None):
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # data augmentation
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
    elif image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])
    else:
        raise ValueError(f'unknown {image_set}')
    
