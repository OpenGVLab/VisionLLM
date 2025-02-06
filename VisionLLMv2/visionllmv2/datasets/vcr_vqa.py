import random
import torch
import copy
import os
import json
import re
import secrets
import string
import numpy as np
from PIL import Image
from collections import defaultdict
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from pycocoevalcap.eval import COCOEvalCap
from torch.utils.data import Dataset

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal
from .utils import boxes_to_masks

class VCRVQA(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        self.task = 'region_vqa'
        self.dataset_name = 'vcr'

        with open(ann_file, 'r') as f:
            data = [json.loads(line) for line in f]
        self.questions = data
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.image_size = data_args.image_size

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        line = self.questions[idx]
        image_file = line["image"]
        boxes = line["boxes"]
        conversations = line['conversations']
        correct_option = line['correct_option']
        category = line['category']

        # get image
        image = Image.open(os.path.join(self.img_prefix, image_file)).convert('RGB')
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

        # get regions
        boxes = torch.as_tensor(boxes)
        scale_fct = torch.as_tensor([self.image_size, self.image_size, self.image_size, self.image_size])[None, :]
        boxes = boxes * scale_fct  # xyxy in image size
        img_shape = [self.image_size, self.image_size]
        regions = boxes_to_masks(boxes, img_shape)

        # preprocess <regions>
        source = copy.deepcopy(conversations)
        region_str = ""
        for i in range(len(boxes)):
            if i != len(boxes) - 1:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor'] + ', '
            else:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
        source[0]["value"] = source[0]["value"].replace("<regions>", region_str)

        # get conversation
        sources = preprocess_multimodal(copy.deepcopy([source]))
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

        
        # -----------------------------------------------------
        # update image and img_metas
        data_dict['image'] = image      # [3, h, w]
        img_metas = dict()
        img_metas['task'] = "refer"
        img_metas['dataset_name'] = "vcr"
        img_metas['conversations'] = source
        img_metas['correct_option'] = correct_option
        img_metas['category'] = category
        data_dict['img_metas'] = img_metas
        # update regions
        data_dict['regions'] = regions  # [n, h, w]
        return data_dict