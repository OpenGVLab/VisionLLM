# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
import numpy as np
import sys
import copy
import torch
import random
from PIL import Image
from mmseg.datasets import ADE20KDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from .llava_data import preprocess_multimodal, preprocess
from ..mm_utils import expand2square, dynamic_preprocess

from transformers import (
    Mask2FormerImageProcessor
)

# 30 questions
QUESTIONS = [
    # interrogative
    "Could you aid me in generating unique masks for every category present in <class> in this image?",
    "Can you help me generate distinct masks for each category that belongs to <class> in this image?",
    "Is it possible for you to help me create distinct masks for the different <class> categories in this image?",
    "Could you assist me in generating masks that correspond to each individual <class> category in this image?",
    "Would you mind helping me generate separate masks for each <class> category detected in this image?",
    "Can you guide me in generating unique masks for all the categories falling under <class> in this image?",
    "Can you provide me with the necessary support to generate masks specific to each <class> category in this image?",
    "Could you please guide me in creating separate masks for each <class> category detected in this image?",
    "Can you support me in generating masks for all the categories encompassed by <class> in this image?",
    "Examine the image and generate masks that correspond to each individual <class> category present.",
    "Is it possible for you to help me generate separate masks for each detected category falling under <class> in this image?",
    "Can you assist me in generating masks that isolate each category belonging to <class> in this image?",
    "Can you provide me with assistance in generating individual masks for every <class> category identified in this image?",
    "Can you help with the process of generating masks that are specific to each <class> category detected in this image?",
    "Generate masks that accurately depict each category belonging to <class> in this image.",
    "I require assistance in producing separate masks for all the <class> categories in this image.",
    "I need your support to generate masks that are specific to each <class> category in this image.",
    "Your task is to produce masks that differentiate each category falling under the <class> category in this image.",
    "Please create masks that are distinct for each category belonging to <class> in this image.",
    "I'm seeking your help to generate masks that isolate every category within the <class> category in this image.",
    "Please segment the different categories falling under <class> in this image and generating masks for each.",
    "Please accurately segment and generate masks for all the categories falling under <class> in this image.",
    "I need your support to create masks that are specific to each <class> category identified in this image.",
    "I'm requesting your aid in generating masks that distinguish each category belonging to <class> in this image.",
    "Please lend me your expertise in creating masks that are unique for each detected <class> category in this image.",
    "Your help is required to generate distinct masks for each category of <class> in this image.",
    "It would be appreciated if you could assist in creating separate masks for each <class> category in this image.",
    "Let's collaborate on segmenting all categories falling under the <class> category in this image and generating masks.",
    "Assisting me in generating distinct masks for each class categorized as <class> would be greatly appreciated.",
    "Providing assistance in generating masks that accurately identify the categories falling under <class> in this image would be greatly helpful."
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


class ADE20KLlavaDataset(ADE20KDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 # conversation
                 tokenizer,
                 data_args,
                 # detection
                 test_mode=False,
                 max_gt_per_img=100,
                 ):
        self.task = 'seg'
        self.dataset_name = 'ade20k'

        # conversation
        self.tokenizer = tokenizer
        self.image_folder = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.num_embs = data_args.num_embs

        self.mask2former_processor = Mask2FormerImageProcessor(
            _max_size=2560,
            ignore_index=255
        )
        self.max_gt_per_img = max_gt_per_img

        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        # 512 x 512
        crop_size = (512, 512)
        # crop_size = (640, 640)
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ]
        # 640 x 640
        # train_pipeline = [
        #     dict(type='LoadImageFromFile'),
        #     dict(type='LoadAnnotations', reduce_zero_label=True),
        #     dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0), keep_ratio=True),
        #     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        #     dict(type='RandomFlip', prob=0.5),
        #     dict(type='PhotoMetricDistortion'),
        #     dict(type='Normalize', **img_norm_cfg),
        #     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        #     dict(type='DefaultFormatBundle'),
        #     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        # ]
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_dir=ann_file,
            img_dir=img_prefix,
            reduce_zero_label=True,
            test_mode=test_mode,
            pipeline=pipeline
        )
        super().__init__(**dataset_cfg)
    

    def process_mask(mask):
        unique_classes = torch.unique(mask)
        num_classes = len(unique_classes)
        masks = {}

        for i in range(num_classes):
            sub_mask = (mask == unique_classes[i]).to(torch.uint8)
            masks[unique_classes[i]] = sub_mask
        return masks

    def get_bounding_boxes(self, masks):
        num_masks = masks.shape[0]
        boxes = torch.zeros(num_masks, 4, dtype=torch.float32)
        x_any = torch.any(masks, dim=1)
        y_any = torch.any(masks, dim=2)
        for idx in range(masks.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
        return boxes

    def normalize_box_coordinates(self, bbox, img_shape):
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        img_h, img_w = img_shape[:2]
        bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
        return torch.cat(bbox_new, dim=-1)

    def __getitem__(self, idx):
        # quick check for some corrupted anno images for training
        if not self.test_mode:  # train
            flag = False
            while not flag:
                data_item = super().__getitem__(idx)  # after mmseg pipeline
                file_name = data_item['img_metas'].data['filename'] if not self.test_mode \
                    else data_item['img_metas'][0].data['filename']
                gt_semantic_seg = data_item['gt_semantic_seg'].data  # [1, h, w]
                unique_classes = torch.unique(gt_semantic_seg) 
                if len(unique_classes) == 1 and unique_classes == 255:
                    print(f"{file_name} annotation image corrupted.")
                    idx = random.randint(0, self.__len__() - 1)
                else:
                    flag = True
        else:  # inference
            data_item = super().__getitem__(idx)  # after mmseg pipeline
            file_name = data_item['img_metas'].data['filename'] if not self.test_mode \
                else data_item['img_metas'][0].data['filename']


        class_list = list(self.CLASSES)
        conversations = []
        if not self.test_mode:  # train
            # random.shuffle(class_list)
            question_template = random.choice(QUESTIONS)
            answer_template = random.choice(YES)

            # use random category number
            if torch.randn(1) > 0:  # all classes
                random.shuffle(class_list)
            else:  # keep all positives, random number negatives
                gt_semantic_seg = data_item['gt_semantic_seg'].data.squeeze(0).cpu().numpy()  # [1, h, w] -> [h, w]
                pre_process_data = self.mask2former_processor(
                    data_item['img'].data,
                    segmentation_maps=gt_semantic_seg,
                    # mmseg pipeline has preprocessed
                    do_rescale=False,
                    do_resize=False,
                    do_normalize=False,
                    return_tensors="pt"
                )
                gt_labels = pre_process_data['class_labels'][0]
                num_gt = len(gt_labels)
                ann_cat_labels = sorted(torch.unique(gt_labels).tolist())    # continuous ids
                ann_cat_names = [self.CLASSES[label] for label in ann_cat_labels]   # e.g. ['person', 'car', ...]
                # find exist/unexist catgories
                pos_cat_names = [cat for cat in class_list if cat in ann_cat_names]      # category names exist in the image
                neg_cat_names = [cat for cat in class_list if cat not in ann_cat_names]  # category names not exist in the image
                min_num_neg = 1 if num_gt == 0 else 0  # in case of no gt
                num_neg = random.randint(min_num_neg, len(class_list))
                random.shuffle(neg_cat_names)
                neg_cat_names = neg_cat_names[:num_neg]
                class_list = pos_cat_names + neg_cat_names  # new class list
                random.shuffle(class_list)
        else:
            question_template = QUESTIONS[0]
            answer_template = YES[0]  
        
        # question
        class_list_str = ', '.join(class_list)
        class_list_str = class_list_str.lower()  # lower
        question = question_template.replace('<class>', class_list_str)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        conversations.append(message1)
        # answer 
        if self.num_embs == 1:
            class_list_str = '[SEG][EMB], '.join(class_list)
            class_list_str += '[SEG][EMB]'  # the last one
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)])
            str_temp = "[SEG]" + str_temp  # e.g. "[SEG][EMB][EMB2][EMB3][EMB4]"
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
        image = Image.open(file_name).convert('RGB')
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
        )
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )
        data_dict['image'] = image

        ######### Segmentation ##########
        # create continue label id to random index mapping
        name2index = {name: idx for idx, name in enumerate(class_list)}
        id2index = {idx: name2index[name] for idx, name in enumerate(self.CLASSES) if name in class_list}
        if not self.test_mode:
            gt_semantic_seg = data_item['gt_semantic_seg'].data.squeeze(0).cpu().numpy()  # [1, h, w] -> [h, w]
            pre_process_data = self.mask2former_processor(
                data_item['img'].data,
                segmentation_maps=gt_semantic_seg,
                # mmseg pipeline has preprocessed
                do_rescale=False,
                do_resize=False,
                do_normalize=False,
                return_tensors="pt"
            )

            img_metas = data_item['img_metas'].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            boxes = self.get_bounding_boxes(pre_process_data['mask_labels'][0])
            boxes = bbox_xyxy_to_cxcywh(boxes)
            img_shape = data_item['img_metas'].data['img_shape']
            boxes = self.normalize_box_coordinates(boxes, img_shape)
            data_dict_seg = {
                'image_aug': pre_process_data['pixel_values'][0],
                'mask_labels': pre_process_data['mask_labels'][0],
                'class_labels': pre_process_data['class_labels'][0],
                # 'pixel_mask': pre_process_data['pixel_mask'],  # not used in data collate
                'img_metas': data_item['img_metas'].data,
                'boxes': boxes
            }
        else:
            img_metas = data_item['img_metas'][0].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            data_dict_seg = {
                'image_aug': data_item['img'][0].data,
                'img_metas': data_item['img_metas'][0].data   # dict
            }
        
        data_dict.update(data_dict_seg)
        return data_dict

 