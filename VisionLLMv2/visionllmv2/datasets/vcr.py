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


# for question
def replace_numbers_with_tokens(s):
    pattern = r'\b(\d+)\b'
    reg, sor, eor = DEFAULT_TOKENS['reg'], DEFAULT_TOKENS['sor'], DEFAULT_TOKENS['eor']
    try:
        result = re.sub(pattern, lambda match: f'{sor}region{match.group(1)}{reg}{eor}', s)
    except:
        # contain number not for instance
        return None
    return result

# for answer and why
def replace_numbers_with_tags(s, class_names):
    pattern = r'\b(\d+)\b'
    try:
        # -1 because region index starts from 1
        result = re.sub(pattern, lambda match: f'{class_names[int(match.group(1))-1]} at region{match.group(1)}', s)
    except:
        # contain number not for instance
        return None
    return result

class VCRDataset(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        self.task = 'region_refer'
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
        flag = False
        while not flag:
            line = self.questions[idx]
            image_file = line["image"]
            boxes = line["boxes"]
            objects = line["objects"]
            conversations = line['conversations']
            assert len(boxes) == len(objects)

            # preprocess <regions>
            rounds = len(conversations) // 2  # 1 or 2
            source = copy.deepcopy(conversations)
            source[0]['value'] = '<image>\n' + source[0]['value']
            source[0]['value'] = replace_numbers_with_tokens(source[0]['value'])  # q
            source[1]['value'] = replace_numbers_with_tags(source[1]['value'], class_names=objects)  # a
            source[1]['value'] = source[1]['value'].lower().capitalize()
            if rounds == 2:
                source[3]['value'] = replace_numbers_with_tags(source[3]['value'], class_names=objects)  # why
                source[3]['value'] = source[3]['value'].lower().capitalize()
            
            # check 
            if source[0]['value'].count(DEFAULT_TOKENS['reg']) != len(boxes):
                print("vcr: region token number != boxes number")
                idx = random.randint(0, self.__len__() - 1)
                continue
            else:
                flag = True

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
        data_dict['img_metas'] = img_metas
        # update regions
        data_dict['regions'] = regions  # [n, h, w]
        return data_dict