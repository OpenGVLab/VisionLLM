# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
import copy
import json
import os
import numpy as np
from collections import defaultdict
from mmdet.datasets import CocoDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from PIL import Image

from .llava_data import preprocess, preprocess_multimodal
from ..mm_utils import expand2square, dynamic_preprocess

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


# General class for handling det dataset, e.g. objects365, openimages
class DetLlavaDataset(CocoDataset):
    """
    Return both chat data and coco detection data.
    """
    def __init__(self,
                 ann_file,
                 img_prefix,
                 # conversation
                 tokenizer,
                 data_args,
                 # detection
                 test_mode=False,
                 max_gt_per_img=100,
                 with_mask=False,
                 dataset_name='objects365'
                 ):
        # read classe from anno file
        coco_gt = json.load(open(ann_file,'r'))
        CocoDataset.CLASSES = tuple(i['name'] for i in coco_gt['categories'])
        self.class_mapping = {cat: idx for idx, cat in enumerate(self.CLASSES)} # class name -> category id (continuous, start from 0)
        self.num_classes = len(self.CLASSES)

        self.task = 'det'
        self.dataset_name = dataset_name

        # conversation
        self.tokenizer = tokenizer
        self.image_folder = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.num_embs = data_args.num_embs

        # detection
        self.with_mask = with_mask
        self.max_gt_per_img = max_gt_per_img

        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True)

        # train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
        # from the default setting in mmdet.
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=self.with_mask),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[
                    [
                        dict(
                            type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True)
                    ],
                    [
                        dict(
                            type='Resize',
                            # The radio of all image in train dataset < 7
                            # follow the original impl
                            img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                            multiscale_mode='value',
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            override=True,
                            keep_ratio=True)
                    ]
                ]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),  
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                    else ['img', 'gt_bboxes', 'gt_labels'])
        ]
        # test_pipeline, NOTE the Pad's size_divisor is different from the default
        # setting (size_divisor=32). While there is little effect on the performance
        # whether we use the default setting or use size_divisor=1.
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=test_mode,
            pipeline=pipeline)
        
        super().__init__(**dataset_cfg)
    
    def normalize_box_coordinates(self, bbox, img_shape):
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        img_h, img_w = img_shape[:2]
        bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
        return torch.cat(bbox_new, dim=-1)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx) # after mmdet pipeline

        ########## llava ############
        # conversation version: v3, output every object
        file_name = data_item['img_metas'].data['ori_filename'] if not self.test_mode \
                        else data_item['img_metas'][0].data['ori_filename']
        class_list = list(self.CLASSES)
        conversations = []
        if not self.test_mode:  # train
            # random.shuffle(class_list)
            question_template = random.choice(QUESTIONS)
            answer_template = random.choice(YES)

            # objects365, openimages have too many categories, we restrict maximum of 80 classes during training
            # keep all positives, random number negatives
            num_gt = len(data_item['gt_labels'].data)
            ann_cat_labels = sorted(torch.unique(data_item['gt_labels'].data).tolist())    # continuous ids
            ann_cat_names = [self.CLASSES[label] for label in ann_cat_labels]   # e.g. ['person', 'car', ...]
            # find exist/unexist catgories
            pos_cat_names = [cat for cat in class_list if cat in ann_cat_names]      # category names exist in the image
            neg_cat_names = [cat for cat in class_list if cat not in ann_cat_names]  # category names not exist in the image
            num_pos = len(pos_cat_names)
            min_num_neg = 1 if num_gt == 0 else 0  # in case of no gt
            max_num_neg = 100 - num_pos
            assert max_num_neg >= 0
            num_neg = random.randint(min_num_neg, max_num_neg)
            random.shuffle(neg_cat_names)
            neg_cat_names = neg_cat_names[:num_neg]
            class_list = pos_cat_names + neg_cat_names  # new class list
            random.shuffle(class_list)
        else:  # TODO: inference need split all categories into groups
            question_template = QUESTIONS[0]
            answer_template = YES[0]            
        # clean class names
        class_list = [x.strip().lower() for x in class_list] 

        # question
        class_list_str = ', '.join(class_list)
        question = question_template.replace('<class>', class_list_str)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        conversations.append(message1)
        # answer 
        if self.num_embs == 1:
            class_list_str = "[DET][EMB], ".join(class_list)
            class_list_str += "[DET][EMB]"  # the last one
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
            str_temp = "[DET]" + str_temp # e.g. "[DET][EMB][EMB2][EMB3][EMB4]"
            str_temp_aug = str_temp + ", "
            class_list_str = str_temp_aug.join(class_list)
            class_list_str += str_temp
        answer = answer_template.replace('<class>', class_list_str)
        message2 = {
            'from': 'gpt',
            'value': answer
        }
        conversations.append(message2)

        # load image and clip preprocess
        processor = self.img_processor
        image = Image.open(os.path.join(self.image_folder, file_name)).convert('RGB')
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

        sources = preprocess_multimodal(copy.deepcopy([conversations]))
        data_dict = preprocess(
            sources=sources,  # list[list[dict]], first list length=1, second list length=num_rounds
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            has_image=True,
            image_token_len=image_token_len
        ) # keys: "input_ids", "labels", size of [1, L]
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )
        data_dict['image'] = image

        ######### detection ##########
        # create continue label id to random index mapping
        CLASSES_NAMES = [x.strip().lower() for x in self.CLASSES]  # clean class names
        name2index = {name: idx for idx, name in enumerate(class_list)}
        id2index = {idx: name2index[name] for idx, name in enumerate(CLASSES_NAMES) if name in class_list}
        if not self.test_mode:
            gt_bboxes = data_item['gt_bboxes'].data
            img_shape = data_item['img_metas'].data['img_shape']
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            gt_bboxes = self.normalize_box_coordinates(gt_bboxes, img_shape) # cxcywh, [0, 1] 
            if self.with_mask:
                gt_masks = data_item['gt_masks'].data
            img_metas = data_item['img_metas'].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            if self.with_mask:
                data_dict_det = {
                    'image_aug': data_item['img'].data,
                    'class_labels': data_item['gt_labels'].data,  # continuous label ids
                    'boxes': gt_bboxes,
                    'mask_labels': gt_masks,
                    'img_metas': data_item['img_metas'].data,     # dict
                }
            else:
                data_dict_det = {
                    'image_aug': data_item['img'].data,
                    'class_labels': data_item['gt_labels'].data,  # continuous label ids
                    'boxes': gt_bboxes,
                    'img_metas': data_item['img_metas'].data,     # dict
                }
        else:
            img_metas = data_item['img_metas'][0].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            data_dict_det = {
                'image_aug': data_item['img'][0].data,
                'img_metas': data_item['img_metas'][0].data   # dict
            }
        data_dict.update(data_dict_det)
        return data_dict