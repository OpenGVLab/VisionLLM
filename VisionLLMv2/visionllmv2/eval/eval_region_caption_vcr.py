import argparse
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import time
import json
import copy

import requests
from PIL import Image
from io import BytesIO

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import visionllmv2.util.misc as utils
from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.utils import disable_torch_init, init_distributed_mode
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.utils import boxes_to_masks
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

IMAGE_TOKEN_INDEX = -200

class VCRVQATest(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, data_args):
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
        conversations = line["conversations"]
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

        # get conversation
        conv_mode = args.conv_mode
        conv = conv_templates[conv_mode].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        source = copy.deepcopy(conversations)  # list[dict]
        # preprocess <regions>
        region_str = ""
        for i in range(len(boxes)):
            if i != len(boxes) - 1:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor'] + ', '
            else:
                region_str += DEFAULT_TOKENS['sor'] + f'region{i+1}' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']
        source[0]["value"] = source[0]["value"].replace("<regions>", region_str)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{sentence}"
            conv.append_message(role, sentence["value"])
        assert len(source) % 2 == 1, len(source)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # tokenizer conversations
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) # [1, L]
        # replace with 'imp' tokens
        replace_token = DEFAULT_TOKENS['imp'] * image_token_len
        if self.use_im_start_end:
            replace_token = DEFAULT_TOKENS['boi'] + replace_token + DEFAULT_TOKENS['eoi']
        replace_token_ids = self.tokenizer([replace_token], return_tensors="pt").input_ids[0][1:] # [L,], remove start token
        index = input_ids[0].argmin()  # find the index of IMAGE_TOKEN_INDEX
        new_input_ids = torch.cat([input_ids[0, :index], replace_token_ids, input_ids[0, index+1:]], dim=0).unsqueeze(0)
        input_ids = new_input_ids  # [1, L]

        data_dict = dict(
            input_ids=input_ids, # [1, L]
        )

        # -----------------------------------------------------
        # update image and img_metas
        data_dict['image'] = image      # [3, h, w]
        img_metas = dict()
        img_metas['task'] = "region_vqa"
        img_metas['dataset_name'] = "vcr"
        img_metas['conversations'] = conversations
        img_metas['correct_option'] = correct_option
        img_metas['category'] = category
        data_dict['img_metas'] = img_metas
        # update regions
        data_dict['regions'] = regions  # [n, h, w]
        return data_dict
    

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

    
def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']            # [1, L]
    images = batch[0]['image']                   # [3, h, w] or [n_split, 3, h, w]
    regions = [batch[0]['regions']]              # list[tensor], 1 x [n, h, w]
    img_metas = [batch[0]['img_metas']]          # list[dict]
    return input_ids, images, regions, img_metas


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = VisionLLMv2Model.from_pretrained(model_name, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16).cuda()
    model.get_llm().config.use_cache = True
    # init special token ids
    model.init_special_token_ids(tokenizer)

    # create dataset and dataloader
    data_args = {
        'image_aspect_ratio': args.image_aspect_ratio,
        'use_im_start_end': args.use_im_start_end,
        'image_size': args.image_size,
        'image_max_tile': args.image_max_tile,
        'use_pixelshuffle': args.use_pixelshuffle,
        'img_processor': CLIPImageProcessor.from_pretrained(args.vis_encoder_path),
    }
    data_args = Config(data_args)
    dataset = VCRVQATest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    sampler = InferenceSampler(len(dataset))
    dataloader = DataLoader(dataset=dataset, sampler=sampler, collate_fn=custom_collate_fn,
                        batch_size=args.batch_size_per_gpu, num_workers=8, pin_memory=True, drop_last=False)

    # stop criterion, this is needed for internlm2
    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    # begin inference
    model.eval()
    q_a_results, qa_r_results = [], []
    q_a_gt, qa_r_gt = [], []
    rank, world_size = get_dist_info()
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    if rank == 0:
        progress_bar = mmcv.ProgressBar(len(dataset))
    for input_ids, images, regions, img_metas in dataloader:
        input_ids = input_ids.cuda()
        if args.image_aspect_ratio == 'anyres':
            images = [images.cuda().to(torch.bfloat16)]
        else:
            images = images.unsqueeze(0).cuda().to(torch.bfloat16)
        regions = [region.cuda() for region in regions]

        # stop criterion
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=images,
                regions=regions,
                img_metas=img_metas,
                # generation_config=model.get_llm().generation_config,
                do_sample=False,   # greedy search
                temperature=0.,
                max_new_tokens=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        outputs = outputs.strip()
        # print(outputs)

        category = img_metas[0]['category']
        correct_option = img_metas[0]['correct_option']
        if category == 'qa':
            q_a_results.extend([outputs])
            q_a_gt.extend([correct_option])
        elif category == 'qar':
            qa_r_results.extend([outputs])
            qa_r_gt.extend([correct_option])
        else:
            raise ValueError(f"Question category {category} not supported.")

        if rank == 0:
            batch_size = 1
            for _ in range(batch_size * world_size):
                progress_bar.update()

    # collect results from gpus
    torch.distributed.barrier()
    assert len(dataset) % 2 == 0  # q->a and qa->r
    gathered_q_a_results = utils.all_gather(q_a_results)
    gathered_q_a_gt = utils.all_gather(q_a_gt)
    gathered_qa_r_results = utils.all_gather(qa_r_results)
    gathered_qa_r_gt = utils.all_gather(qa_r_gt)
    q_a_results = [p for p_list in gathered_q_a_results for p in p_list]
    q_a_gt = [p for p_list in gathered_q_a_gt for p in p_list]
    qa_r_results = [p for p_list in gathered_qa_r_results for p in p_list]
    qa_r_gt = [p for p_list in gathered_qa_r_gt for p in p_list]

    # evaluation
    if rank == 0:
        # q->a
        q_a_check = []
        for answer, gt in zip(q_a_results, q_a_gt):
            q_a_check.append(answer == gt)
        # qa->r
        qa_r_check = []
        for answer, gt in zip(qa_r_results, qa_r_gt):
            qa_r_check.append(answer == gt)
        # q-ar
        q_ar_check = []
        for q_a, qa_r in zip(q_a_check, qa_r_check):
            q_ar_check.append(q_a and qa_r)   # correct when both q->a, qa->r are correct.
        # print
        print("\n")
        print(f"Category: q->a, Samples: {len(q_a_check)}, Acc: {sum(q_a_check) / len(q_a_check) * 100:.2f}")
        print(f"Category: qa->r, Samples: {len(qa_r_check)}, Acc: {sum(qa_r_check) / len(qa_r_check) * 100:.2f}")
        print(f"Category: q->ar, Samples: {len(q_ar_check)}, Acc: {sum(q_ar_check) / len(q_ar_check) * 100:.2f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--ann_file", type=str, default="data/vcr/vcrvqa_val.jsonl")
    parser.add_argument("--img_prefix", type=str, default="data/vcr/vcr1images/")
    parser.add_argument('--conv_mode', type=str, default='vicuna_v1')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument("--use_im_start_end", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--image_max_tile", type=int, default=6)
    parser.add_argument("--use_pixelshuffle", type=bool, default=False)
    parser.add_argument("--vis_encoder_path", type=str, default="checkpoints/clip-vit-large-patch14-336")
    # dist
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    init_distributed_mode(args)
    

    eval_model(args)

