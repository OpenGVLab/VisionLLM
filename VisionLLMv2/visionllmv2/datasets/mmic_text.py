from calendar import c
from concurrent.futures import process
from dataclasses import replace
import os
import io
import copy
import json
import numpy as np

import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
import transformers

from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.datasets.pipelines import Compose

from ..constant import IGNORE_INDEX, DEFAULT_TOKENS, IMAGE_TOKEN_INDEX
from ..mm_utils import expand2square, dynamic_preprocess
from ..conversation import get_conv_template
from .. import conversation as conversation_lib

from .llava_data import preprocess_multimodal, preprocess

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')

class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key="sensecore"):
        print(f"[TCSLoader] config_path: {conf_path}")
        print("--> before Client(conf_path)")
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print("--> after Client(conf_path)")

    def __call__(self, fn):
        img_value_str = self.client.get(fn)
        img = pil_loader(img_value_str)
        return img


try:
    TCS_LOADER = TCSLoader("~/petreloss.conf")
except Exception as e:
    TCS_LOADER = None


class InContextTextDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ann_file: str,
        img_prefix: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        mask_prefix: Optional[str] = None,
        test_mode=False,
        use_tcs_loader=False,
    ):
        super(InContextTextDataset, self).__init__()
        print("Formatting inputs...Skip in lazy mode")
        if ann_file.endswith(".json"):
            self.list_data_dict = json.load(open(ann_file, "r"))
        elif ann_file.endswith(".jsonl"):
            with open(ann_file, "r") as f:
                data = [json.loads(line) for line in f]
            self.list_data_dict = data
        else:
            raise NotImplementedError("Annotation file format not supported.")
        self.task = "ic_text"
        self.dataset_name = "ic_text"

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.num_embs = data_args.num_embs
        self.test_mode = test_mode
        # tcs loader
        if use_tcs_loader:
            assert has_tcs_loader and TCS_LOADER is not None, "tcs_loader is not available."
            self.tcs_loader = TCS_LOADER
        else:
            self.tcs_loader = None


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def read_images_in_batch(self, img_prefix, image_file, mode="RGB"):
        if image_file is None:
            return None

        if isinstance(image_file, str):
            image_file = [image_file]

        images = []
        for img in image_file:
            image_path = os.path.join(img_prefix, img)

            # =====================================
            if self.tcs_loader is not None:
                image = self.tcs_loader(image_path)
            else:
                image = Image.open(image_path)
            image = image.convert(mode)
            images.append(image)
        return images # list[PIL]

    @classmethod
    def expand2square(cls, pil_img, background_color): # for multiple images
        if not isinstance(pil_img, list):
            pil_img = [pil_img]
        results = []
        for img in pil_img:
            results.append(expand2square(img, background_color))
        return results


    def _load_data(self, sources, i):
        image_file = self.list_data_dict[i]["image"]
        mask_file = self.list_data_dict[i].get("mask", None)
        img_prefix = self.img_prefix
        mask_prefix = self.mask_prefix
        processor = self.img_processor

        images = self.read_images_in_batch(img_prefix, image_file)           # list[PIL]
        masks = self.read_images_in_batch(mask_prefix, mask_file, mode="L")  # list[PIL]

        # a list of PIL images, process each image onece a time
        if self.data_args.image_aspect_ratio == "anyres":
            new_images = []       # list[tensor], 
            num_splits = []       # list[int], n_split for each image
            image_token_lens = [] # list[int], image_token_len for each image
            for image in images:
                image = dynamic_preprocess(
                    image,
                    image_size=self.data_args.image_size,
                    max_num=self.data_args.image_max_tile,
                )  # list[pil_img]
                image = [
                    processor.preprocess(x, return_tensors="pt")["pixel_values"][0]
                    for x in image
                ]
                image = torch.stack(image)  # [1 + n_tile, 3, h, w]
                image_token_len = int((self.data_args.image_size // 14) ** 2)
                if self.data_args.use_pixelshuffle:
                    image_token_len = image_token_len // 4
                image_token_len = image_token_len * len(image)  # len(image) is the num_splits for an image
                # append
                new_images.append(image)
                num_splits.append(len(image))
                image_token_lens.append(image_token_len)
        elif self.data_args.image_aspect_ratio == "pad":
            new_images = []       # list[tensor], 
            num_splits = []       # list[int], n_split for each image
            image_token_lens = [] # list[int], image_token_len for each image
            for image in images:
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values']  # [1, 3, h, w]
                image_token_len = int((self.data_args.image_size // 14) ** 2)
                if self.data_args.use_pixelshuffle:
                    image_token_len = image_token_len // 4
                # append
                new_images.append(image)
                num_splits.append(1)
                image_token_lens.append(image_token_len)
        else:  # resize
            new_images = []       # list[tensor], 
            num_splits = []       # list[int], n_split for each image
            image_token_lens = [] # list[int], image_token_len for each image
            for image in images:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"] # [1, 3, h, w]
                image_token_len = int((self.data_args.image_size // 14) ** 2)
                if self.data_args.use_pixelshuffle:
                    image_token_len = image_token_len // 4
                # append
                new_images.append(image)
                num_splits.append(1)
                image_token_lens.append(image_token_len)
        # concat images
        # image_token_lens: list[int], image_token_len for each image
        images = torch.cat(new_images, dim=0)  # [n_images_and_splits, 3, h, w]
        del new_images

        # masks
        if masks is not None:
            new_masks = self.expand2square(masks, 0)  # list[PIL]
            new_masks = [torch.from_numpy(np.array(m.resize(image.shape[-2:][::-1], 0), dtype=np.float32)) for m in new_masks]
            new_masks = torch.stack(new_masks) # [n_regions, h, w]
            masks = new_masks
            masks /= 255
            del new_masks

        conversations = copy.deepcopy(sources[0]["conversations"])
        region_str = (
            DEFAULT_TOKENS["sor"]
            + 'region'
            + DEFAULT_TOKENS["reg"]
            + DEFAULT_TOKENS["eor"]
        )  # '<reg>region<region></reg>'
        conversations[0]["value"] = conversations[0]["value"].replace("<mask>", region_str)
        
        sources = preprocess_multimodal(copy.deepcopy([conversations]))
        data_dict = preprocess(
            sources=sources,  # list[list[dict]], first list length=1, second list length=num_rounds
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            has_image="image" in self.list_data_dict[i],
            image_token_len=image_token_lens,  # list[int], image_token_len for each image
        )  # keys: "input_ids", "labels", size of [1, L]

        
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )
        # for identifying num_images in a sample
        data_dict["num_splits"] = num_splits  # list[int], n_splits for each image

        # add regions
        if "mask" in self.list_data_dict[i] and masks is not None:
            data_dict["regions"] = masks  # [n_regions, h, w] 

        # img metas
        img_metas = dict()
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations
        data_dict['img_metas'] = img_metas
            
        # image exists in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = images  # [n_all_images_and_splits, 3, h, w]
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.img_processor.crop_size
            if self.data_args.image_aspect_ratio == "anyres":
                data_dict["image"] = torch.zeros(
                    1, 3, crop_size["height"], crop_size["width"]
                )
            else:
                data_dict["image"] = torch.zeros(
                    3, crop_size["height"], crop_size["width"]
                )
                
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        flag = False
        while not flag:
            try:
                sources = self.list_data_dict[i]  # dict
                if isinstance(i, int):
                    sources = [sources]
                assert (
                    len(sources) == 1
                ), "Don't know why it is wrapped to a list"  # FIXME
                data_dict = self._load_data(sources, i)
                flag = True
            except Exception as e:
                import traceback
                traceback.print_exc()
                # print(e)
                i = random.randint(0, len(self.list_data_dict) - 1)
        return data_dict