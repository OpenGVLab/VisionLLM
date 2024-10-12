import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu

from transformers import AutoTokenizer
from transformers import CLIPImageProcessor

from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.utils import disable_torch_init, init_distributed_mode
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

IMAGE_TOKEN_INDEX = -200

ds_collections = {
    'pope': {
        'root': 'data/pope/val2014',
        'question': 'data/pope/llava_pope_test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
        'min_new_tokens': 1,
    },
}

    
class VQADataset(torch.utils.data.Dataset):

    def __init__(self, name, root, data, prompt, tokenizer, data_args):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt

        self.name = name
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)

        # get image
        processor = self.img_processor
        image_path = os.path.join(self.root, image)
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
        
        # get conversation
        question = question + ' ' + self.prompt
        question = '<image>\n' + question
        conv_mode = args.conv_mode
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
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
        img_metas['task'] = "image_vqa"
        img_metas['dataset_name'] = self.name
        img_metas['question'] = question
        img_metas['question_id'] = question_id
        img_metas['annotation'] = annotation
        data_dict['img_metas'] = img_metas
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
    img_metas = [batch[0]['img_metas']]          # list[dict]
    return input_ids, images, img_metas

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = VisionLLMv2Model.from_pretrained(model_name, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16).cuda()
    model.get_llm().config.use_cache = True
    # init special token ids
    model.init_special_token_ids(tokenizer)

    os.makedirs(args.out_dir, exist_ok=True)

    for ds_name in args.datasets:
        data_args = {
            'image_aspect_ratio': args.image_aspect_ratio,
            'use_im_start_end': args.use_im_start_end,
            'image_size': args.image_size,
            'image_max_tile': args.image_max_tile,
            'use_pixelshuffle': args.use_pixelshuffle,
            'img_processor': CLIPImageProcessor.from_pretrained(args.vis_encoder_path),
        }
        data_args = Config(data_args)
        dataset = VQADataset(
            name=ds_name,
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt='',
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
        outputs = []
        rank, world_size = get_dist_info()
        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        if rank == 0:
            progress_bar = mmcv.ProgressBar(len(dataset))
        for input_ids, images, img_metas in dataloader:
            input_ids = input_ids.cuda()
            if args.image_aspect_ratio == 'anyres':
                images = [images.cuda().to(torch.bfloat16)]
            else:
                images = images.unsqueeze(0).cuda().to(torch.bfloat16)

            # stop criterion
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=images,
                    img_metas=img_metas,
                    do_sample=False,   # greedy search
                    temperature=0.,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                    length_penalty=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            pred = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            pred = pred.strip()

            question_id = img_metas[0]['question_id']
            outputs.append({
                'question_id': question_id,
                'text': pred,
                'model_id': args.model_name,
                'metadata': {},
            })

            if rank == 0:
                batch_size = 1
                for _ in range(batch_size * world_size):
                    progress_bar.update()
        
        torch.distributed.barrier()
        # collect results from all gpus
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))
        
        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        
        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))
            cmd = 'python3 visionllmv2/eval/pope/eval_pope.py ' \
                  '--annotation-dir ./data/pope/coco ' \
                  '--question-file ./data/pope/llava_pope_test.jsonl ' \
                  '--result-file ' + results_file
            print(cmd)
            os.system(cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--datasets', type=str, default=['pope'], nargs='+')
    parser.add_argument('--conv_mode', type=str, default='vicuna_v1')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument("--use_im_start_end", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--image_max_tile", type=int, default=6)
    parser.add_argument("--use_pixelshuffle", type=bool, default=False)
    parser.add_argument("--vis_encoder_path", type=str, default="checkpoints/clip-vit-large-patch14-336")
    parser.add_argument('--out-dir', type=str, default='results')
    # dist
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    init_distributed_mode(args)
    
    eval_model(args)