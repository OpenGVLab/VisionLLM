# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
import os
import json
import numpy as np
from collections import defaultdict
from mmdet.datasets import CocoDataset
from mmdet.datasets.pipelines import Compose
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from PIL import Image
from pycocotools import mask as maskUtils

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal
from .utils import masks_to_boxes, boxes_to_masks

# visual sampler
from .visual_sampler.sampler import ShapeSampler

# 30 questions
QUESTIONS = [
    "Can you examine the image and segment the corresponding objects denoted as <regions>?",
    "Where are the objects marked by <regions> in the image? Could you help me segment these objects?",
    "Could you please segment all the corresponding objects according to the visual prompt as <regions>?",
    "Can you help me draw the instance segmentation masks of <regions> in the picture?",
    "Please help me find all the objects shown as <regions> and segment them.",
    "I'd like to know the objects outlined by <regions>. Please help me draw their masks.",
    "Given the <regions>, I need your help to segment the corresponding object masks.",
    "Examine the image and identify all the objects that belong to the provided <regions>.",
    "I'm interested in the objects labeled as <regions>. Could you please draw their instance masks?",
    "There are some regions represented by <regions>. I need your assistance to find their corresponding objects.",
]

# 10 yes
YES = [
    "Sure, these objects are <regions>.",
    "Yes, the objects masks are <regions>.",
    "Certainly, <regions> are shown in the image.",
    "Absolutely, they are <regions>.",
    "Of course, the objects are <regions>.",
]


# For interactive dataset training
class CocoInteractiveDataset(CocoDataset):
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
                 with_mask=True,
                 mode=None        # for inference, specify one mode
                 ):
        self.task = 'interactive'
        self.dataset_name = 'coco'

        # conversation
        self.tokenizer = tokenizer
        self.image_folder = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.image_size = data_args.image_size
        self.num_embs = data_args.num_embs

        # detection
        assert with_mask, "The dataset must provide mask annotations."
        self.with_mask = with_mask
        self.max_gt_per_img = max_gt_per_img

        # visual sampler
        is_train = ~test_mode  # TODO: pass is_train, need check sampler.py
        self.visual_sampler = ShapeSampler(is_train=True, mode=mode) 

        # CLIP and ImageNet norm, RGB 
        self.clip_mean = (torch.tensor(self.img_processor.image_mean) * 255).view(3, 1, 1)
        self.clip_std = (torch.tensor(self.img_processor.image_std) * 255).view(3, 1, 1)
        self.inet_mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(3, 1, 1)
        self.inet_std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(3, 1, 1)

        # norm
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
            dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),  # filter too small regions, since low-resolution image for clip
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
                    dict(type='DefaultFormatBundleFlickr'),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                            else ['img', 'gt_bboxes', 'gt_labels'])
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
        flag = False
        while not flag:
            data_item = super().__getitem__(idx) # after mmdet pipeline

            ########## llava ############
            # Using ShapeSampler to get visual prompt
            gt_mask = data_item['gt_masks'].data if not self.test_mode else data_item['gt_masks'][0].data  # [n, h, w]
            if gt_mask.shape[0] == 0:  # no valid mask
                idx = random.randint(0, self.__len__() - 1)
                continue
            # resize to clip image size, image also need normalization with clip mean std
            image = data_item['img'].data if not self.test_mode else data_item['img'][0].data  # [3, h, w]
            image = F.interpolate(image[None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
            clip_image = image * self.inet_std + self.inet_mean
            clip_image = (clip_image - self.clip_mean) / self.clip_std
            # the downsample may cause some masks as 0
            clip_gt_mask = F.interpolate(gt_mask[None], size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0].bool()
            valid_mask = clip_gt_mask.sum(1).sum(1) > 0
            clip_gt_mask = clip_gt_mask[valid_mask]  # only keey valid mask
            if clip_gt_mask.shape[0] == 0:
                idx = random.randint(0, self.__len__() - 1)
                continue
            clip_gt_bbox = masks_to_boxes(clip_gt_mask)    
            visual_prompt_dict = self.visual_sampler(clip_gt_bbox, clip_gt_mask)
            visual_prompt = visual_prompt_dict['rand_shape']       # [n, h, w], randomly sample visual prompt
            visual_prompt_indices = visual_prompt_dict['indices']  # list
            n_visual_prompt = len(visual_prompt)
            if n_visual_prompt > 0:  # check visual prompt has valid mask
                flag = True
            else:
                idx = random.randint(0, self.__len__() - 1)
                continue

        # ----------------------- chat ------------------------
        conversations = []
        if not self.test_mode:  # train
            question_template = random.choice(QUESTIONS)
            answer_template = random.choice(YES)
        else:
            question_template = QUESTIONS[0]
            answer_template = YES[0]            
        # question
        region_str = ''
        for i in range(len(visual_prompt)):
            if i != len(visual_prompt) - 1:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor'] + ', '
            else:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
        question = question_template.replace('<regions>', region_str)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        conversations.append(message1)
        # answer 
        region_list = [f'region{i+1}' for i in range(len(visual_prompt))]
        if self.num_embs == 1:
            region_list_str = "[DET][EMB], ".join(region_list)
            region_list_str += "[DET][EMB]"  # the last one
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
            str_temp = "[DET]" + str_temp # e.g. "[DET][EMB][EMB2][EMB3][EMB4]"
            str_temp_aug = str_temp + ", "
            region_list_str = str_temp_aug.join(region_list)
            region_list_str += str_temp
        answer = answer_template.replace('<regions>', region_list_str)
        message2 = {
            'from': 'gpt',
            'value': answer
        }
        conversations.append(message2)

        image_token_len = int((self.image_size // 14) ** 2)
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
        # image for clip
        data_dict['image'] = clip_image       # [3, h, w]
        data_dict['regions'] = visual_prompt  # [n, h, w]

        ######### detection ##########
        # create continue object id to random index mapping
        id2index = {i: i for i in range(n_visual_prompt)}   # {0: 0, 1: 1, ...}
        if not self.test_mode:
            gt_bboxes = data_item['gt_bboxes'].data
            img_shape = data_item['img_metas'].data['img_shape']
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            gt_bboxes = self.normalize_box_coordinates(gt_bboxes, img_shape) # cxcywh, [0, 1] 
            if self.with_mask:
                gt_masks = data_item['gt_masks'].data
            # shuffle gt_boxes/masks according to visual_prompt_indices
            gt_labels = data_item['gt_labels'].data
            gt_labels = gt_labels[visual_prompt_indices]  # [n_visual_prompt]
            for i in range(len(gt_labels)):  # each region is a class
                gt_labels[i] = i
            gt_bboxes = gt_bboxes[visual_prompt_indices]  # [n_visual_prompt, 4]
            gt_masks = gt_masks[visual_prompt_indices]    # [n_visual_prompt, h, w]
            img_metas = data_item['img_metas'].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            if self.with_mask:
                data_dict_det = {
                    'image_aug': data_item['img'].data,
                    'class_labels': gt_labels,  
                    'boxes': gt_bboxes,
                    'mask_labels': gt_masks,
                    'img_metas': data_item['img_metas'].data,     # dict
                }
            else:
                data_dict_det = {
                    'image_aug': data_item['img'].data,
                    'class_labels': gt_labels,
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
    

class CocoInteractiveTest(Dataset):
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
                 test_mode=True,
                 max_gt_per_img=100,
                 with_mask=True,
                 mode='Box'     # for inference, specify one mode
                 ):
        self.task = 'interactive'
        self.dataset_name = 'coco'

        assert mode in ['Box', 'Mask', 'Point', 'Scribble']
        self.mode = mode

        self.data_infos = self.load_annotations(ann_file)

        # conversation
        self.tokenizer = tokenizer
        self.image_folder = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.image_size = data_args.image_size
        self.num_embs = data_args.num_embs

        # detection
        assert with_mask, "The dataset must provide mask annotations."
        self.with_mask = with_mask
        self.max_gt_per_img = max_gt_per_img

        # CLIP and ImageNet norm, RGB 
        self.clip_mean = (torch.tensor(self.img_processor.image_mean) * 255).view(3, 1, 1)
        self.clip_std = (torch.tensor(self.img_processor.image_std) * 255).view(3, 1, 1)
        self.inet_mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(3, 1, 1)
        self.inet_std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(3, 1, 1)

        # norm
        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True)

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

        self.pipeline = Compose(test_pipeline)

    def load_annotations(self, ann_file):
        data = json.load(open(ann_file, 'r'))
        data_infos = []
        for d in data:
            info = d['image_info']
            info['filename'] = info['file_name']
            info['id'] = d['img_id']  # new image id after preprocessing
            info['anns'] = d['anns']  # list[dict], we need the visual prompt as inputs
            data_infos.append(info)
        return data_infos
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.image_folder
        results['seg_prefix'] = None
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):  # polygon
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle) # np.array
        return mask

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        img_info = self.data_infos[idx]   # dict
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        data_item = self.pipeline(results)

        # getting images and regions
        filename, height, width = img_info['filename'], img_info['height'], img_info['width']
        image = Image.open(os.path.join(self.image_folder, filename)).convert('RGB')
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


        # regions
        anns = img_info['anns']
        if self.mode == 'Box':
            visual_prompt = [ann['box_visual_prompt_mask'] for ann in anns]
        elif self.mode == 'Mask':
            visual_prompt = [ann['mask_visual_prompt_mask'] for ann in anns]
        elif self.mode == 'Point':
            visual_prompt = [ann['point_visual_prompt_mask'] for ann in anns]
        elif self.mode == 'Scribble':
            visual_prompt = [ann['scribble_visual_prompt_mask'] for ann in anns]
        visual_prompt = [self.annToMask(mask, height, width) for mask in visual_prompt]
        visual_prompt = torch.from_numpy(np.array(visual_prompt)).float()  # [n, ori_h, ori_w]
        visual_prompt = torch.nn.functional.interpolate(visual_prompt.unsqueeze(0), 
                                                    size=(self.image_size, self.image_size),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0).bool() # [n, h, w], h, w is the clip img size
        
        # gt masks, for evaluation metrics
        gt_masks = [ann['segmentation'] for ann in anns]
        gt_masks = [self.annToMask(mask, height, width) for mask in gt_masks]
        gt_masks = torch.from_numpy(np.array(gt_masks)).float().bool()  # [n, ori_h, ori_w]

        # check valid mask，some visual prompt masks may be 0 due to downsample
        valid = visual_prompt.sum(-1).sum(-1) > 0
        # if (valid==0).any():
        #     print(f'\n{filename} has invalid visual prompt: {valid}')
        visual_prompt = visual_prompt[valid]
        gt_masks = gt_masks[valid]
        assert all(visual_prompt.sum(-1).sum(-1) > 0)  # must have valid visual prompts


        # ----------------------- chat ------------------------
        # test mode
        conversations = []
        question_template = QUESTIONS[0]
        answer_template = YES[0]            
        # question
        region_str = ''
        for i in range(len(visual_prompt)):
            if i != len(visual_prompt) - 1:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor'] + ', '
            else:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
        question = question_template.replace('<regions>', region_str)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        conversations.append(message1)
        # TODO: delete answer for inference
        # answer 
        region_list = [f'region{i+1}' for i in range(len(visual_prompt))]
        if self.num_embs == 1:
            region_list_str = "[DET][EMB], ".join(region_list)
            region_list_str += "[DET][EMB]"  # the last one
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
            str_temp = "[DET]" + str_temp # e.g. "[DET][EMB][EMB2][EMB3][EMB4]"
            str_temp_aug = str_temp + ", "
            region_list_str = str_temp_aug.join(region_list)
            region_list_str += str_temp
        answer = answer_template.replace('<regions>', region_list_str)
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
        # image for clip
        data_dict['image'] = image            # [3, h, w]
        data_dict['regions'] = visual_prompt  # [n, h, w]

        ######### detection ##########
        # create continue object id to random index mapping
        id2index = {i: i for i in range(len(visual_prompt))}   # {0: 0, 1: 1, ...}
        img_metas = data_item['img_metas'][0].data
        img_metas['id2index'] = id2index
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations
        img_metas['gt_masks'] = gt_masks  # for evaluation metrics
        data_dict_det = {
            'image_aug': data_item['img'][0].data,
            'img_metas': data_item['img_metas'][0].data   # dict
        }
        data_dict.update(data_dict_det)
        return data_dict