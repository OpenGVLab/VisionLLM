import random
from dataclasses import dataclass
from typing import Dict, Sequence
import os
import json
from PIL import Image

import copy
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from ..constant import IGNORE_INDEX, DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal


PREFIX_PROMPTS = [
    "Edit the image according to the caption:",
    "Modify the image as per the caption provided:",
    "Transform the image according to the given description:",
    "Please alter the image based on the following caption:",
    "Adjust the image in line with the caption details:",
    "Can you rework the image following the provided description:",
    "Apply the caption's details to edit the image:",
    "Make changes to the image as described in the caption:",
    "Update the image according to the given text:",
    "Please modify the image as instructed by the caption:",
    "Create an edited image based on the given caption:",
    "Implement the changes described in the caption to the image:",
    "Edit the image to reflect the description in the caption:",
    "Translate the image based on the provided description:",
    "Adjust the image to match the details in the caption:",
    "Can you edit the image according to the description provided:",
    "Transform the image to align with the caption's instructions:",
    "Modify the image based on the provided details:",
    "Recreate the image as per the caption:",
    "Please follow the caption to alter the image:",
    "Edit the image in accordance with the caption's description:",
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

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

class BaseDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        super().__init__()
        self.task = 'edit'

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_embs_gen = data_args.num_embs_gen

        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end

        self.output_img_processor = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, index):
        original_image, output_image, caption = self.get_data(index)

        # load image and clip preprocess
        processor = self.img_processor
        if self.data_args.image_aspect_ratio == 'anyres':
            image = dynamic_preprocess(original_image, image_size=self.data_args.image_size, max_num=self.data_args.image_max_tile) # list[pil_img]
            image = [processor.preprocess(x, return_tensors='pt')['pixel_values'][0] for x in image]
            image = torch.stack(image)  # [1 + n_tile, 3, h, w]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
            image_token_len = image_token_len * len(image)
        elif self.data_args.image_aspect_ratio == 'pad':
            image = expand2square(original_image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
        else:  # resize
            image = processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4

        # conversations
        conversations = []
        if torch.randn(1) > 0:
            instruction = random.choice(PREFIX_PROMPTS) + " " + caption.lower().strip().strip('.') + "."
        else:
            instruction = caption.strip().capitalize().strip('.') + "."
        instruction = '<image>\n' + instruction
        answer = random.choice(ANSWER_PROMPTS)
        answer += f" [EDIT]{'[EMB]' * self.num_embs_gen}."

        conversations.append({"from": "human", "value": instruction})
        conversations.append({"from": "gpt", "value": answer})

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
        data_dict['caption'] = caption

        # input, output images for ip2p
        input_image = self.output_img_processor(original_image)
        data_dict['input_image'] = input_image
        output_image = self.output_img_processor(output_image)
        data_dict['output_image'] = output_image

        # img_metas
        img_metas = {'task': self.task, 'dataset_name': self.dataset_name}
        data_dict['img_metas'] = img_metas
        return data_dict
    
class IP2PDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args) -> None:
        super().__init__(data_path, tokenizer, data_args)
        self.dataset_name = 'ip2p'

        dataset = load_dataset(self.data_path)
        self.train_dataset = dataset["train"]

    def get_data(self, index):
        example = self.train_dataset[index]
        caption = example['edit_prompt']
        output_image = example['edited_image']
        input_image = example['original_image']
        return input_image, output_image, caption

    def __len__(self):
        return len(self.train_dataset)


class SeedXDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args) -> None:
        super().__init__(data_path, tokenizer, data_args)
        self.dataset_name = 'seedx'

        train_dataset = []
        jsonl_files = sorted(list(os.listdir(os.path.join(data_path, 'annotations'))))
        for jsonl_file in jsonl_files:
            jsonl_path = os.path.join(data_path, 'annotations', jsonl_file)
            with open(jsonl_path, 'r') as f:
                data = [json.loads(line) for line in f]
                train_dataset.extend(data)
        self.train_dataset = train_dataset

    def get_data(self, index):
        example = self.train_dataset[index]
        caption = example['instruction']
        input_image_path = os.path.join(self.data_path, 'images', example['source_image'])
        input_image = Image.open(input_image_path).convert('RGB')
        output_image_path = os.path.join(self.data_path, 'images', example['target_image'])
        output_image = Image.open(output_image_path).convert('RGB')
        return input_image, output_image, caption

    def __len__(self):
        return len(self.train_dataset)