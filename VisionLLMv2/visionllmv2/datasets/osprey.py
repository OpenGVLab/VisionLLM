import copy
import os
import re
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset

import transformers
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal


DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe <region> in the image in detail.',
    'Can you offer a thorough analysis of <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the <region> closely and share its details.'
]


# COCO format
class CustomDataset(Dataset):
    def __init__(self, ann_file: str, img_prefix: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 data_args, max_gt_per_img=20
        ):
        super().__init__()
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.image_size = data_args.image_size
        self.max_gt_per_img = max_gt_per_img

        self.data_infos = self.load_annotations(ann_file)

    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue

            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
    
    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info
    
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
    
    def read_process_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
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
        return image, image_token_len
    
    def get_data_item(self, idx):
        data_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        img_path = os.path.join(self.img_prefix, data_info['filename'])
        image, image_token_len = self.read_process_image(img_path)

        gt_masks = []
        gt_labels = []
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], data_info['height'], data_info['width'])
            gt_masks.append(mask)  # list[np.array]

            cat = self.coco.loadCats(ann['category_id'])
            gt_labels.append(cat[0]['name'])

        data_item = dict(
            img = image,
            gt_masks = gt_masks,
            gt_labels = gt_labels,
            image_token_len = image_token_len
        )
        return data_item
    
    def process_data(self, data_item):
        image = data_item['img']
        ori_labels = data_item['gt_labels']
        ori_masks = np.array(data_item['gt_masks'])
        ori_masks = torch.from_numpy(ori_masks).float() # uint -> float

        # training
        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        # after shuffle
        ori_masks = ori_masks[shuffle_ids]  # [n, h, w]
        ori_labels = [ori_labels[i] for i in shuffle_ids]
        ori_masks = torch.nn.functional.interpolate(ori_masks.unsqueeze(0),
                                                size=(self.image_size, self.image_size),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0).bool()

        # conversation
        conversations = []
        for i in range(len(ori_labels)):
            # question
            question = '<region>'
            region_str = DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
            question = question.replace('<region>', region_str)
            if i == 0:
                question = '<image>\n' + question
            message1 = {
                'from': 'human',
                'value': question
            }
            conversations.append(message1)
            # answer
            answer = ori_labels[i]
            # answer = data_item['caption']  # ??
            message2 = {
                'from': 'gpt',
                'value': answer
            }
            conversations.append(message2)
        
        image_token_len = data_item['image_token_len']
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

        # update images and regions
        data_dict['image'] = image
        data_dict['regions'] = ori_masks
        img_metas = dict()
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations 
        data_dict['img_metas'] = img_metas
        return data_dict
    
    def __getitem__(self, idx):
        data_item = self.get_data_item(idx)
        data_dict = self.process_data(data_item=data_item)
        return data_dict
    
# Osprey Datasets
class OspreyConversationDataset(CustomDataset):
    def __init__(self, ann_file: str, img_prefix: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 data_args, max_gt_per_img=20
        ):
        super().__init__(ann_file, img_prefix, tokenizer, data_args, max_gt_per_img)
        print("Loading Osprey dataset...")
        self.task = 'region_refer'
        self.dataset_name = 'osprey'

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file, 'r'))

        for ann in ann_list:  # for each sample
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]  # coco train2017 
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            region_str = ""
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']  # polygon
                masks.append(mask)  # list[polygon]
                if i != region_num - 1:
                    region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor'] + ', '
                else:
                    region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']

            for i in range(len(ann['conversations'])//2):
                # first round
                if i==0:
                    if region_num==1:
                        mid_str = "Ther are 1 part region in the picture: " + region_str + '. '
                    else:
                        mid_str = "Ther are {} part regions in the picture: ".format(str(region_num)) + region_str + '. '

                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    question = "<image>\n" + mid_str + question
                    qa_s.append({'from': 'human', 'value': question + self.limit})    # self.limit is the answer format prompt.    
                else:
                    question = ann['conversations'][i*2]['value']
                    question = question.replace('<','').replace('>','')
                    qa_s.append({'from': 'human', 'value': question + self.limit})         

                # answer
                answer = ann['conversations'][i*2+1]['value']
                answer = answer.replace('<','').replace('>','')
                qa_s.append({'from': 'gpt', 'value': answer.strip()})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s  # list[dict]
            ))
        return data_infos
    
    def __getitem__(self, i):
        flag = False
        while not flag:
            data_info = self.data_infos[i]
            img_path = data_info['img_path']
            height = data_info['height']
            width = data_info['width']
            masks_raw = data_info['masks']
            masks = []
            for mask_r in masks_raw:
                mask = self.annToMask(mask_r, height, width)
                masks.append(mask)
                
            masks = np.array(masks)
            masks = torch.from_numpy(masks).float()  # uint -> float
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0),
                                                    size=(self.image_size, self.image_size),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0).bool()
            valid_mask = masks.sum(-1).sum(-1) > 0  # in case of no mask after resize
            if not (valid_mask > 0).all():
                i = random.randint(0, self.__len__() - 1)
                continue
            image, image_token_len = self.read_process_image(img_path)
            flag = True
        
        # conversation
        qas = data_info['qas']
        sources = preprocess_multimodal(copy.deepcopy([qas]))
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

        # update images and regions
        data_dict['image'] = image
        data_dict['regions'] = masks
        img_metas = dict()
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = qas
        data_dict['img_metas'] = img_metas
        return data_dict


class OspreyConversations(OspreyConversationDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 tokenizer,
                 data_args,
                 ):
        self.limit = ""
        super().__init__(ann_file, img_prefix, tokenizer, data_args)


class OspreyDetailedDescription(OspreyConversationDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 tokenizer,
                 data_args,
                 ):
        super().__init__(ann_file, img_prefix, tokenizer, data_args)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file, 'r'))

        for ann in ann_list:
            masks = []
            qa_s = []
            filename = ann['file_name'].split('_')[-1]
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']
            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)

                question = random.choice(DETAILED_QUESTIONS)
                region_str = DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
                question = question.replace('<region>', region_str)
                if i==0:
                    qa_s.append({'from': 'human', 'value': "<image>\n" + question})         
                else:
                    qa_s.append({'from': 'human', 'value': question})     
            
                answer = re.findall(r"<.*>:\ (.*)", ann['description'][i])[0]  # <region1>: xxxxx
           
                qa_s.append({'from': 'gpt', 'value': answer.strip()})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
        return data_infos
    

class OspreyLVISPosNeg(OspreyConversationDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 tokenizer,
                 data_args,
                 ):
        
        super().__init__(ann_file, img_prefix, tokenizer, data_args)

    def load_annotations(self, ann_file):
        data_infos = []
        ann_list = json.load(open(ann_file, 'r'))

        for ann in ann_list:
            if len(ann['conversations'])//2 ==0:
                continue
            masks = []
            qa_s = []
            filename = ann['file_name']
            img_path = os.path.join(self.img_prefix, filename)
            region_num = len(ann['annotation'])
            h, w = ann['height'], ann['width']

            for i in range(region_num):
                mask = ann['annotation'][i]['segmentation']
                masks.append(mask)
        
            for i in range(len(ann['conversations'])//2):
                    
                question = ann['conversations'][i*2]['value']
                region_str = DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
                question = re.sub(r'<region\d+>', region_str, question)
                if i==0:
                    question = "<image>\n" + question
                qa_s.append({'from': 'human', 'value': question})         
             
                answer = ann['conversations'][i*2+1]['value']
                qa_s.append({'from': 'gpt', 'value': answer.strip()})

            data_infos.append(dict(
                img_path = img_path,
                masks = masks,
                height = h,
                width = w,
                qas = qa_s
            ))
            # print(qa_s)

        return data_infos
    

class OspreyPartLevel(OspreyConversationDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 tokenizer,
                 data_args,
                 ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(ann_file, img_prefix, tokenizer, data_args)


class OspreyShortForm(OspreyConversationDataset):
    def __init__(self,
                ann_file,
                img_prefix,
                tokenizer,
                data_args,
                ):
        self.limit = ' Answer the question using a single word or phrase.'
        super().__init__(ann_file, img_prefix, tokenizer, data_args)