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

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal

# 30 questions
QUESTIONS = [
    # interrogative
    "Can you identify and segment all occurrences of <class> in the given picture?",
    "Is it possible for you to perform instance segmentation specifically for <class> in this image?",
    "Is there a way to automatically segment and identify instances of <class> in this image?",
    "Are you able to help me with the process of segmenting instances belonging to <class>?",
    "Can you walk me through the steps to achieve instance segmentation for <class> objects?",
    "Is there a way to automatically identify and segment instances of <class> in this image?",
    "Could you assist me in the process of obtaining segmented masks for <class> in this image?",
    "Is it possible to automatically generate distinct masks for each <class> object in the image?",
    "In the image, I want to identify and segment all occurrences of <class>. Can you assist?",
    "Could you assist me in the process of obtaining segmented masks for instances of <class>?",
    "Could you segmente instances of <class> with precision and clarity in this picture?",
    "Could you help me identify and segment all objects categorized as <class> with clear and separate masks?",
    "Are you able to generate distinct masks for each occurrence of <class> using instance segmentation?",
    "Would you be able to apply segmentation techniques to highlight individual instances of <class> in the image?",
    "Is it possible to conduct segmentation on the image and generate distinct masks for objects of the category <class>?",
    # declarative
    "I need assistance in segmenting instances of <class> in this image.",
    "I'd like your assistance in segmenting objects, particularly those of the <class> type",
    "Seeking help to segment instances of <class> with clear and separate masks.",
    "Let's focus on segmenting all objects falling under the <class> category in this image.",
    "Examine the image and obtain segmented masks for each <class> object.",
    "I'm trying to perform instance segmentation for <class> in the image. Can you help me?",
    "The task at hand is to achieve instance segmentation specifically for <class>.",
    "I need assistance with instance segmentation, particularly for objects of the <class> class.",
    "Let's work on generating masks for each occurrence of <class> in the image.",
    "Seeking your expertise in identifying and segmenting instances of <class> in this picture.",
    "Please assist me in creating distinct masks for each object categorized as <class>.",
    "Let's focus on achieving instance segmentation for all objects falling under <class>.",
    "Please assist me in segmenting instances of the <class> category in this particular image.",
    "Please generate distinct masks for each instance of <class> in this picture.",
    "Generate masks for each occurrence of <class> in the provided image."
]

# 10 yes
YES = [
    "Yes, here are the results for <class> in the image.",
    "Certainly, the image shows the results for <class>.",
    "Absolutely, you can see the results for <class> in the image.",
    "Yes, the segmentation results for <class> are present.",
    "Certainly, the image does show the results of <class>.",
    "Certainly, you can spot <class> in the image.",
    "Yes, there is a clear depiction for the results of <class>.",
    "Of course, the image provides a comprehensive results of <class>.",
    "Absolutely, the image showcases the results of <class>.",
    "Sure, the image contains the segmentation results for <class>."
]


class CODLlavaDataset(CocoDataset):
    """
    Return both chat data and coco detection data.
    """
    CLASSES = ('Camouflage',)
    def __init__(self,
                 ann_file,
                 img_prefix,
                 # conversation
                 tokenizer,
                 data_args,
                 # detection
                 test_mode=False,
                 max_gt_per_img=100,
                 with_mask=True,
                 ):
        self.task = 'det'
        self.dataset_name = 'cod'

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

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
    
    def __getitem__(self, idx):
        data_item = super().__getitem__(idx) # after mmdet pipeline

        ########## llava ############
        # conversation version: v3, output every object
        file_name = data_item['img_metas'].data['ori_filename'] if not self.test_mode \
                        else data_item['img_metas'][0].data['ori_filename']
        # class_list = list(self.CLASSES)
        class_list = ['camouflage object']
        conversations = []
        if not self.test_mode:  # train
            random.shuffle(class_list)  # only one class
            question_template = random.choice(QUESTIONS)
            answer_template = random.choice(YES)
        else:
            question_template = QUESTIONS[0]
            answer_template = YES[0]            
            
        # question
        # class_list_with_tokens = [DEFAULT_TOKENS['sod'] + cat + DEFAULT_TOKENS['eod'] for cat in class_list]
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
            image_token_len=image_token_len,
        ) # keys: "input_ids", "labels", size of [1, L]
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )
        data_dict['image'] = image

        ######### detection ##########
        # create continue label id to random index mapping
        # only one class
        id2index = {0: 0}
        if not self.test_mode:  # train
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
        else:  # test
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