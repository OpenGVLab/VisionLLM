import copy
import random
import os
import numpy as np
import torch
import re

from PIL import Image
from mmdet.datasets import LVISV1Dataset
from mmdet.datasets.api_wrappers import COCO

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal
from .utils import boxes_to_masks

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
    from .llava_data import TCSLoader
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False


LVIS_QUESTIONS = [
    "Whis is the object category of <regions>? Answer with the category name from LVIS-1203, and use single word or phrase.",
    "Could you tell me what is the object in <regions>? Answer with the category name from LVIS-1203, and use single word or phrase.",
    "What category best describes the area represented by <regions>? Answer with the category name from LVIS-1203, and use single word or phrase.",
    "Can you specify the type of object inside the region labeld by <regions>? Answer with the category name from LVIS-1203, and use single word or phrase.",
    "How would you label the area indicated by <regions> in the image? Answer with the category name from LVIS-1203, and use single word or phrase.",
    "Give a category label to the region outlined by <regions>. Answer with the category name from LVIS-1203, and use single word or phrase.",
    "Please identify the category of the object inside the <regions>. Answer with the category name from LVIS-1203, and use single word or phrase.",
    "Examine and determine the primary subject located within <regions>. Answer with the category name from LVIS-1203, and use single word or phrase.",
    "I need your help to assign an object category to the <regions>, please. Answer with the category name from LVIS-1203, and use single word or phrase.",
    "Evaluate the content of the region shown as <regions> and provide its category. Answer with the category name from LVIS-1203, and use single word or phrase.",
]


def clean_string(expression):
    expression = re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ').replace('_', ' ')
    return expression

class LVISRecognition(LVISV1Dataset):

    def __init__(
            self,
            ann_file,
            img_prefix,
            tokenizer,
            data_args,
            max_gt_per_img=15,
            with_mask=True,
            test_mode=False,
            use_tcs_loader=False,
    ):
        self.task = 'region_recognition'
        self.dataset_name = 'lvis'

        # conversation
        self.tokenizer = tokenizer
        self.img_prefix = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.image_size = data_args.image_size

        self.with_mask = with_mask
        self.max_gt_per_img = max_gt_per_img
        self.test_mode = test_mode

        # tcs loader
        if use_tcs_loader:
            assert has_tcs_loader
            self.tcs_loader = TCSLoader('~/petreloss.conf') 
        else:
            self.tcs_loader = None

        image_mean = self.img_processor.image_mean
        image_mean = [x * 255 for x in image_mean]
        image_std = self.img_processor.image_std
        image_std = [x * 255 for x in image_std]

        img_norm_cfg = dict(
            mean=image_mean,
            std=image_std,
            to_rgb=True)

        # file_client_args
        file_client_args = dict(backend='petrel') if use_tcs_loader else dict(backend='disk')
        
        train_pipeline = [
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
            dict(type='Resize', img_scale=(self.image_size, self.image_size), keep_ratio=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=self.image_size),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                    else ['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
            dict(type='Resize', img_scale=(self.image_size, self.image_size), keep_ratio=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=self.image_size),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                    else ['img', 'gt_bboxes', 'gt_labels']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,  # also need gt for object recognition
            pipeline=pipeline)

        super(LVISV1Dataset, self).__init__(**dataset_cfg)

    def preprocess_data(self, data_item):
        # lvis has mask annotations
        # image = data_item['img'].data
        labels = data_item['gt_labels'].data
        bboxes = data_item['gt_bboxes'].data
        masks = data_item['gt_masks'].data if self.with_mask else None
        img_shape = data_item['img_metas'].data['img_shape']

        # get image
        file_name = data_item['img_metas'].data['ori_filename'] 
        image_path = os.path.join(self.img_prefix, file_name)
        if self.tcs_loader is not None:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        if self.data_args.image_aspect_ratio == 'anyres':
            image = dynamic_preprocess(image, image_size=self.data_args.image_size, max_num=self.data_args.image_max_tile)  # list[pil_img]
            image = [self.img_processor.preprocess(x, return_tensors='pt')['pixel_values'][0] for x in image]
            image = torch.stack(image)  # [1 + n_tile, 3, h, w]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
            image_token_len = image_token_len * len(image)
        else:
            image = self.img_processor.preprocess(image, do_center_crop=False, return_tensors='pt')['pixel_values'][0]  # resize
            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(self.image_size, self.image_size),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4

        # train: randomly select max_gt_per_img gts
        shuffle_ids = torch.randperm(len(labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_labels = labels[shuffle_ids]
        select_bboxes = bboxes[shuffle_ids]  
        select_masks = masks[shuffle_ids] if self.with_mask else None

        # get regions
        if self.with_mask:
            if torch.randn(1) > 0:
                regions = boxes_to_masks(select_bboxes, img_shape)
            else:
                regions = select_masks
        else:
            regions = boxes_to_masks(select_bboxes, img_shape)
        valid_regions = regions.sum(-1).sum(-1) > 0
        regions = regions[valid_regions]
        select_labels = select_labels[valid_regions]
        select_bboxes = select_bboxes[valid_regions]
        select_masks = select_masks[valid_regions] if self.with_mask else None

        # ---------------------------------------------
        # conversation
        conversations = []
        for i in range(len(select_labels)):
            # question
            question_template = LVIS_QUESTIONS[0] if self.test_mode else random.choice(LVIS_QUESTIONS)
            region_str = DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']  # '<reg>region1<region></reg>'
            question = question_template.replace('<regions>', region_str)
            if i == 0:
                question = "<image>\n" + question
            message1 = {
                'from': 'human',
                'value': question
            }
            conversations.append(message1)
            # answer
            answer = self.CLASSES[select_labels[i]]  # str
            answer = clean_string(answer)
            answer = answer.strip().lower().capitalize()
            if not answer.endswith('.'):
                answer = answer + '.'
            message2 = {
                'from': 'gpt',
                'value': answer
            }
            conversations.append(message2)

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

        # -------------------------------------
        # update image and regions
        data_dict['image'] = image
        img_metas = data_item['img_metas'].data
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations
        data_dict['img_metas'] = img_metas
        # update regions
        data_dict['regions'] = regions  # [n, h, w]
        return data_dict

    def __getitem__(self, idx):
        flag = False  # may be no gt in some cases
        while not flag:
            try:
                data_item = super().__getitem__(idx)  # after mmdet pipeline
                data_dict = self.preprocess_data(data_item)
                flag = True
            except Exception as e:
                print("lvis recognition:", e)
                idx = random.randint(0, self.__len__() - 1)
        return data_dict