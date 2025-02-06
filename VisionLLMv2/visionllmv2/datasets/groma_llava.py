import copy
import random
import os
import re
import json
import numpy as np
import torch
from torchvision.ops import box_convert
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from pycocoevalcap.eval import COCOEvalCap

from PIL import Image
import mmcv
import os.path as osp
import tempfile
from collections import OrderedDict

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal
from .utils import boxes_to_masks

def match_roi_ground_box(text):
    pattern = r'<roi>(.*?)</roi>'
    matches = re.findall(pattern, text)
    return matches

def replace_roi_ground_box(text, replace_text):
    pattern = r'<roi>(.*?)</roi>'
    replaced_text = re.sub(pattern, replace_text, text)
    return replaced_text


class GromaLlavaDataset(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        self.task = 'det_cap'
        self.dataset_name = 'groma'

        with open(ann_file, 'r') as f:
            data = json.load(f)
        self.data = data  # list[dict]
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.img_processor = data_args.img_processor

        self.use_im_start_end = data_args.use_im_start_end
        self.image_size = data_args.image_size
        self.num_embs = data_args.num_embs

        # transform for images_aug
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(size=800, max_size=1333, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.data)
    
    def normalize_box_coordinates(self, bbox, img_shape):
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        img_h, img_w = img_shape[:2]
        bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
        return torch.cat(bbox_new, dim=-1)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        file_name = data_item['file_name']
        height, width = data_item['height'], data_item['width']
        conversations = data_item['conversation']  # list[dict], from human, from gpt...
        boxes = data_item['boxes']  # list[box], box xywh in image size
        boxes = torch.tensor(boxes) # [n_all, 4]
        boxes = box_convert(boxes, 'xywh', 'cxcywh')
        boxes = self.normalize_box_coordinates(boxes, (height, width))  # cxcywh in normalized[0, 1]

        # -------------------------------------
        # load clip and preprocess image
        processor = self.img_processor
        image = Image.open(os.path.join(self.img_prefix, file_name)).convert('RGB')
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

        # -------------------------------------
        # conversations
        gt_label_id = 0
        gt_labels, gt_bboxes = [], []
        new_conversations = []
        for i, conversation in enumerate(conversations):
            chat = conversation['value']
            if i % 2 == 0:  # from human
                if i == 0:
                    chat = "<image>\n" + chat
                chat += " Answer the question and localize each object."
                message1 = {
                    'from': 'human',
                    'value': chat
                }
                new_conversations.append(message1)
            else:  # from gpt
                box_inds = conversation['box_inds']
                matches = match_roi_ground_box(chat)
                chat = chat.replace('<p>', DEFAULT_TOKENS['sod']).replace('</p>', DEFAULT_TOKENS['eod'])
                if self.num_embs == 1:
                    str_temp = "[DET][EMB]"
                else:
                    str_temp = "[DET][EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
                chat = replace_roi_ground_box(chat, replace_text=str_temp)
                if chat.startswith(DEFAULT_TOKENS['sod']):  # in case of internlm2 tokenizer bug for '\n' + special token.
                    chat = 'Sure. ' + chat
                message2 = {
                    'from': 'gpt',
                    'value': chat
                }
                new_conversations.append(message2)
                # append gt_labels and gt_bboxes
                for match in matches:
                    count = match.count('<ground_box>')
                    cur_gt_labels = [gt_label_id for _ in range(count)]
                    gt_labels.extend(cur_gt_labels)
                    gt_label_id += 1
                cur_gt_bboxes = boxes[box_inds]
                gt_bboxes.extend(cur_gt_bboxes)
        conversations = new_conversations
        gt_labels = torch.tensor(gt_labels).long()
        gt_bboxes = torch.stack(gt_bboxes, dim=0)  # cxcywh in normalized[0, 1]
        assert len(gt_labels) == len(gt_bboxes)

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

        # -------------------------------------
        # detection
        # tranform keep the image ratio, so no need to process gt_boxes (normalized)
        image_aug = self.transform(Image.open(os.path.join(self.img_prefix, file_name)).convert('RGB'))
        img_h, img_w = image_aug.shape[-2:]
        # create continue label id to random index mapping
        id2index = {idx: idx for idx in range(len(gt_labels))}
        img_metas = dict()
        img_metas['id2index'] = id2index
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations
        img_metas['img_shape'] = (img_h, img_w)
        img_metas['ori_shape'] = (height, width)
        data_dict_det = {
            'image_aug': image_aug,
            'class_labels': gt_labels,
            'boxes': gt_bboxes,
            'img_metas': img_metas,     # dict
        }
        data_dict.update(data_dict_det)
        return data_dict
