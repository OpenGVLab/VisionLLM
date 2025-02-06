import io
import json
import os
import copy
import random
import re
import traceback
from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..constant import IGNORE_INDEX, DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal


PREFIX_PROMPTS = [
    "Generate image with caption:",
    "Can you give me the image with caption:",
    "Help me to generate this image:",
    "Generate image with according to caption:",
    "According to caption, generate image:",
    "An image with caption:",
    "Can you visualize this caption:",
    "Create an image based on this caption:",
    "Generate a visual representation for this caption:",
    "Provide me with an image corresponding to this caption:",
    "Craft an image with the following caption:",
    "Generate an image accompanied by this caption:",
    "Turn this caption into an image:",
    "Generate an image reflecting this caption:",
    "Translate this caption into a visual representation:",
    "Produce an image that matches this caption:",
    "Create an image in line with this caption:",
    "Generate an image to illustrate this caption:",
    "Construct an image based on the given caption:",
    "Give me an image associated with this caption:",
]

ANSWER_PROMPTS = [
    "Here it is",
    "There you are",
    "Of course, here is the generated image",
    "No problem, here it is",
    "Certainly, here you go",
    "Presenting the generated image",
    "Absolutely, here it is",
    "Sure, here you go",
    "Here's what you requested",
    "Behold, the generated image",
    "Here's the outcome you were looking for",
    "Delivering the generated image",
    "Certainly, presenting the result",
    "Voila! Here it is",
    "Here's the requested content",
]

def preprocess_caption(caption):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")
    return caption


class Text2ImageDataset(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        super().__init__()
        self.task = 't2i'

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_embs_gen = data_args.num_embs_gen

        self.img_processor = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                # transforms.RandomHorizontalFlip(),  # in case of wrong 'left', 'right' 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, index):
        while True:
            try:
                output_image, caption = self.get_image_caption_pair(index)
                caption = preprocess_caption(caption)

                # create conversations
                conversations = []
                instruction = random.choice(PREFIX_PROMPTS) + " " + caption + "."
                # Here it is [GEN][EMB][EMB]...
                answer = random.choice(ANSWER_PROMPTS)
                answer += f" [GEN]{'[EMB]' * self.num_embs_gen}."

                conversations.append({"from": "human", "value": instruction})
                conversations.append({"from": "gpt", "value": answer})

                sources = preprocess_multimodal(copy.deepcopy([conversations]))
                data_dict = preprocess(
                    sources=sources,  # list[list[dict]], first list length=1, second list length=num_rounds
                    tokenizer=self.tokenizer,
                    data_args=self.data_args,
                    has_image=False,
                ) # keys: "input_ids", "labels", size of [1, L]
                data_dict = dict(
                    input_ids=data_dict["input_ids"][0],
                    labels=data_dict["labels"][0]
                )

                # skip too long input_ids samples
                if data_dict['input_ids'].shape[-1] > 2048:
                    print(f"Warning: Skip t2i sample with too long caption: {caption}.")
                    index = random.randint(0, self.__len__() - 1)
                    continue

                output_image = expand2square(output_image, (255, 255, 255))
                output_image = self.img_processor(output_image)  # [3, h, w]
                data_dict['output_image'] = output_image
                data_dict['caption'] = caption
                # dummy images
                crop_size = self.data_args.img_processor.crop_size
                if self.data_args.image_aspect_ratio == 'anyres':
                    data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
                else:
                    data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                # img_metas
                img_metas = {'task': self.task, 'dataset_name': self.dataset_name}
                data_dict['img_metas'] = img_metas
                return data_dict
            except Exception as e:
                # print("Text2ImgDataset:", e)
                index = random.randint(0, self.__len__() - 1)


class CC3MDataset(Text2ImageDataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        super().__init__(ann_file, img_prefix, tokenizer, data_args)
        self.dataset_name = 'cc3m'
        self.list_data_dict = json.load(open(self.ann_file, "r"))

    def __len__(self):
        return len(self.list_data_dict)

    def get_image_caption_pair(self, index):
        source = self.list_data_dict[index]
        image_file = source["image"]
        image_folder = self.img_prefix
        output_image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        caption = source["conversations"][1]["value"]
        return output_image, caption
    

class LaionDataset(Text2ImageDataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        super().__init__(ann_file, img_prefix, tokenizer, data_args)
        self.dataset_name = 'laion'
        from petrel_client.client import Client
        conf_path = os.environ.get('CEPH_CONFIG_PATH', '~/petreloss.conf')
        self.client = Client(conf_path)
        self.list_data_dict = open(self.ann_file, "r").readlines()

    def __len__(self):
        return len(self.list_data_dict)

    def imread_ceph(self, image_path):
        img_value_str = self.client.get(image_path)
        assert img_value_str is not None, f'{image_path}'
        buff = io.BytesIO(img_value_str)
        return Image.open(buff).convert('RGB')

    def get_image_caption_pair(self, index):
        source = json.loads(self.list_data_dict[index])
        image_file = source["image"]
        image_path = os.path.join(self.img_prefix, image_file)
        output_image = self.imread_ceph(image_path)
        caption = source["caption"]
        return output_image, caption


class MJDataset(Text2ImageDataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        super().__init__(ann_file, img_prefix, tokenizer, data_args)
        self.dataset_name = 'midjourney'
        from petrel_client.client import Client
        conf_path = os.environ.get('CEPH_CONFIG_PATH', '~/petreloss.conf')
        self.client = Client(conf_path)
        self.list_data_dict = json.load(open(self.ann_file, 'r'))

    def __len__(self):
        return len(self.list_data_dict)

    def imread_ceph(self, image_path):
        img_value_str = self.client.get(image_path)
        assert img_value_str is not None, f'{image_path}'
        buff = io.BytesIO(img_value_str)
        image = Image.open(buff).convert('RGB')
        image = np.array(image)[:, :, ::-1]
        image = Image.fromarray(image)
        return image

    def get_image_caption_pair(self, index):
        source = self.list_data_dict[index]
        image_file = source['image_paths'][0]
        image_file = f"{sha256(image_file.encode('utf-8')).hexdigest()}"
        img_path = os.path.join(self.img_prefix, image_file)
        output_image = self.imread_ceph(img_path)
        caption = source["prompt"]
        return output_image, caption
    

class JourneyDBDataset(Text2ImageDataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
        super().__init__(ann_file, img_prefix, tokenizer, data_args)
        from petrel_client.client import Client
        conf_path = os.environ.get('CEPH_CONFIG_PATH', '~/petreloss.conf')
        self.client = Client(conf_path)
        self.mode = 'valid'
        if 'train' in self.ann_file:
            self.mode = 'train'
        self.list_data_dict = open(self.ann_file, 'r').readlines()
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def imread_ceph(self, image_path):
        img_value_str = self.client.get(image_path)
        assert img_value_str is not None, f'{image_path}'
        buff = io.BytesIO(img_value_str)
        return Image.open(buff).convert('RGB')
    
    def get_image_caption_pair(self, index):
        source = json.loads(self.list_data_dict[index])
        image_file = source["img_path"]
        image_file = os.path.join(self.img_prefix, image_file)
        image_file = image_file.replace('/./','/').replace('.png','.jpg')
        output_image = self.imread_ceph(image_file)
        caption = source["prompt"]
        return output_image, caption