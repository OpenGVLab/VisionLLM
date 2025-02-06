from calendar import c
from concurrent.futures import process
from dataclasses import replace
import os
import io
import copy
import json

import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List, Union
import transformers

from ..constant import IGNORE_INDEX, DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from ..conversation import get_conv_template
from .. import conversation as conversation_lib

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

IMAGE_TOKEN_INDEX = -200



def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')

class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn):
        img_value_str = self.client.get(fn)
        img = pil_loader(img_value_str)
        return img


try:
    TCS_LOADER = TCSLoader("~/petreloss.conf")
except Exception as e:
    TCS_LOADER = None



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, ann_file: str, img_prefix: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 data_args, use_tcs_loader=False
        ):
        super(LazySupervisedDataset, self).__init__()
        print("Formatting inputs...Skip in lazy mode")
        if ann_file.endswith('.json'):
            self.list_data_dict = json.load(open(ann_file, "r"))
        elif ann_file.endswith('.jsonl'):
            with open(ann_file, 'r') as f:
                data = [json.loads(line) for line in f]
            self.list_data_dict = data
        else:
            raise NotImplementedError("Annotation file format not supported.")
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.img_processor = data_args.img_processor

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
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        flag = False
        while not flag:
            try:
                sources = self.list_data_dict[i]  # dict
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
                if 'image' in sources[0]:
                    image_file = self.list_data_dict[i]['image']
                    img_prefix = self.img_prefix
                    processor = self.img_processor
                    image_path = os.path.join(img_prefix, image_file)

                    # =====================================
                    if self.tcs_loader is not None:
                        image = self.tcs_loader(image_path)
                    else:
                        image = Image.open(image_path).convert('RGB')
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
                    sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
                else:  # nlp
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                    image_token_len = int((self.data_args.image_size // 14) ** 2)
                    if self.data_args.use_pixelshuffle:
                        image_token_len = image_token_len // 4
                
                data_dict = preprocess(
                    sources=sources,  # list[list[dict]], first list length=1, second list length=num_rounds
                    tokenizer=self.tokenizer,
                    data_args=self.data_args,
                    has_image='image' in self.list_data_dict[i],
                    image_token_len=image_token_len 
                ) # keys: "input_ids", "labels", size of [1, L]
                if isinstance(i, int):
                    data_dict = dict(
                        input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0]
                    )

                # image exists in the data
                if 'image' in self.list_data_dict[i]:
                    data_dict['image'] = image # [3, h, w] or [1 + n_tile, 3, h, w]
                else:
                    # image does not exist in the data, but the model is multimodal
                    crop_size = self.data_args.img_processor.crop_size
                    if self.data_args.image_aspect_ratio == 'anyres':
                        data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
                    else:
                        data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                flag = True
            except Exception as e:
                print(e)
                i = random.randint(0, len(self.list_data_dict) - 1)
        return data_dict


def preprocess_multimodal(
    sources: Sequence[str],
) -> Dict:
    # pass
    # TODO: temp solution, when num(<images>) != num(images)
    # only the first round conversation could have <image>
    for source in sources:
        for i, sentence in enumerate(source):
            if i != 0 and DEFAULT_TOKENS['img'] in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_TOKENS['img'], '').strip()
    # comment the below codes to handle interleaved data.
    # Forcefully place <image> at the beginning for each conversation.
    # for source in sources:
    #     for sentence in source:
    #         if DEFAULT_TOKENS['img'] in sentence['value']:
    #             # move <image> token to begining
    #             sentence['value'] = sentence['value'].replace(DEFAULT_TOKENS['img'], '').strip()
    #             sentence['value'] = DEFAULT_TOKENS['img'] + '\n' + sentence['value']  
    #             sentence['value'] = sentence['value'].strip()
    return sources


def preprocess(
    sources: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args,
    has_image: bool = False,
    image_token_len: Union[int, List[int]] = 576,  # (336 // 14) ** 2
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if data_args.version == 'plain':
        return preprocess_plain(sources, tokenizer, data_args, image_token_len=image_token_len)
    elif data_args.version == 'v1' or data_args.version == 'vicuna_v1':
        return preprocess_v1(sources, tokenizer, data_args, has_image=has_image, image_token_len=image_token_len)
    elif data_args.version == 'internlm2_chat':
        return preprocess_internlm(sources, tokenizer, data_args, has_image=has_image, image_token_len=image_token_len)
    else:
        raise NotImplementedError(f"conv template {data_args.version} is not supported.")


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    image_token_len,
) -> Dict:
    conv = get_conv_template(data_args.version)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2  # only one round
        assert DEFAULT_TOKENS['img'] in source[0]['value']
        # source[0]['value'] = DEFAULT_TOKENS['img']   # llava implementation. only use <image> as question.
        conversation = source[0]['value'] + source[1]['value'] + conv.sep
        conversations.append(conversation)

    # replace tokens
    new_conversations = []
    # image_token_len = data_args.image_token_len
    use_im_start_end = data_args.use_im_start_end
    replace_token = DEFAULT_TOKENS['imp'] * image_token_len
    if use_im_start_end:
        replace_token = DEFAULT_TOKENS['boi'] + replace_token + DEFAULT_TOKENS['eoi']
    for conversation in conversations:
        conversation = conversation.replace(DEFAULT_TOKENS['img'], replace_token)
        new_conversations.append(conversation)
    conversations = new_conversations

    # tokenizer conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids  # [1, L]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer(replace_token).input_ids)  # check here
        target[:tokenized_len] = IGNORE_INDEX

    return dict(
        input_ids=input_ids, 
        labels=targets
    )

def preprocess_v1(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    has_image: bool,
    image_token_len: Union[int, List[int]],  
) -> Dict:
    # conv = conversation_lib.default_conversation.copy()
    conv = get_conv_template(data_args.version)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids  # [1, L]
    # FIXME: quick fix for too long input ids
    if input_ids.shape[-1] > tokenizer.model_max_length:
        input_ids = input_ids[:, :tokenizer.model_max_length]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " # i.e. " ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)  # split by "</s>"
        cur_len = 1   # start token
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:  # compatible with transformers==4.32.0
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

            if i != 0 and not tokenizer.legacy:  # compatible with transformers==4.32.0
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # replace IMAGE_TOKEN (-200) with replace token/target ids
    if has_image:  
        new_input_ids, new_targets = [], []
        # handle interleaved data
        for input_id, target in zip(input_ids, targets): # bs=1
            indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
            if isinstance(image_token_len, list):
                assert len(indices) == len(image_token_len)

            new_input_id = []
            new_target = []
            prev = 0
            for i in range(indices.shape[-1]):  # for each IMAGE_TOKEN_INDEX
                # replace id
                cur_image_token_len = image_token_len[i] if isinstance(image_token_len, list) else image_token_len
                use_im_start_end = data_args.use_im_start_end
                replace_token = DEFAULT_TOKENS['imp'] * cur_image_token_len
                if use_im_start_end:
                    replace_token = DEFAULT_TOKENS['boi'] + replace_token + DEFAULT_TOKENS['eoi']
                replace_token_ids = tokenizer([replace_token], return_tensors="pt").input_ids[0][1:] # [L,], remove start token
                replace_target_ids = torch.ones_like(replace_token_ids) * IGNORE_INDEX

                # start replaceing
                idx = indices[i]
                new_input_id.append(input_id[prev:idx])
                new_target.append(target[prev:idx])
                prev = idx + 1
                new_input_id.append(replace_token_ids)
                new_target.append(replace_target_ids)
                if i == indices.shape[-1] - 1:
                    new_input_id.append(input_id[idx + 1 :])
                    new_target.append(target[idx + 1 :])
            new_input_id = torch.cat(new_input_id, dim=0)
            new_target = torch.cat(new_target, dim=0)
            new_input_ids.append(new_input_id)
            new_targets.append(new_target)
        input_ids, targets = torch.stack(new_input_ids, dim=0), torch.stack(new_targets, dim=0)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_internlm(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    has_image: bool,
    image_token_len: Union[int, List[int]],
) -> Dict:
    conv = get_conv_template(data_args.version)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids  # [1, L]
    # FIXME: quick fix for too long input ids
    if input_ids.shape[-1] > tokenizer.model_max_length:
        input_ids = input_ids[:, :tokenizer.model_max_length]
    targets = input_ids.clone()


    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        if has_image:
            temp_len = len(tokenizer_image_token(info, tokenizer)) - 1
        else:
            temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_INDEX
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            if has_image:
                temp_len = len(tokenizer_image_token(part, tokenizer)) - 1
            else:
                temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_INDEX
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        if has_image:
            temp_len = len(tokenizer_image_token(last_info, tokenizer)) - 1
        else:
            temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # replace IMAGE_TOKEN (-200) with replace token/target ids
    if has_image:  
        new_input_ids, new_targets = [], []
        # handle interleaved data
        for input_id, target in zip(input_ids, targets): # bs=1
            indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
            if isinstance(image_token_len, list):
                assert len(indices) == len(image_token_len)

            new_input_id = []
            new_target = []
            prev = 0
            for i in range(indices.shape[-1]):  # for each IMAGE_TOKEN_INDEX
                # replace id
                cur_image_token_len = image_token_len[i] if isinstance(image_token_len, list) else image_token_len
                use_im_start_end = data_args.use_im_start_end
                replace_token = DEFAULT_TOKENS['imp'] * cur_image_token_len
                if use_im_start_end:
                    replace_token = DEFAULT_TOKENS['boi'] + replace_token + DEFAULT_TOKENS['eoi']
                replace_token_ids = tokenizer([replace_token], return_tensors="pt").input_ids[0][1:] # [L,], remove start token
                replace_target_ids = torch.ones_like(replace_token_ids) * IGNORE_INDEX

                # start replaceing
                idx = indices[i]
                new_input_id.append(input_id[prev:idx])
                new_target.append(target[prev:idx])
                prev = idx + 1
                new_input_id.append(replace_token_ids)
                new_target.append(replace_target_ids)
                if i == indices.shape[-1] - 1:
                    new_input_id.append(input_id[idx + 1 :])
                    new_target.append(target[idx + 1 :])
            new_input_id = torch.cat(new_input_id, dim=0)
            new_target = torch.cat(new_target, dim=0)
            new_input_ids.append(new_input_id)
            new_targets.append(new_target)
        input_ids, targets = torch.stack(new_input_ids, dim=0), torch.stack(new_targets, dim=0)

    return dict(
        input_ids=input_ids,
        labels=targets
    )


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    # compatible for transformers==4.32
    prompt_chunks = []  # compatible with transformers==4.32.0
    for chunk in prompt.split('<image>'):
        if len(chunk) > 0:
            prompt_chunks.append(tokenizer(chunk).input_ids)
        else:
            prompt_chunks.append([tokenizer.bos_token_id])

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids