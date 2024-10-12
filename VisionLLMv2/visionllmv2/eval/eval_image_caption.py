import argparse
import itertools
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import time
import json
import copy

import requests
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.utils import disable_torch_init, init_distributed_mode
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.refcoco import RefCOCO, REFG_QUESTIONS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

IMAGE_TOKEN_INDEX = -200

ds_collections = {
    'flickr30k': {
        'root': 'data/flickr30k/',
        'annotation': 'data/flickr30k/flickr30k_test_karpathy.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'coco': {
        'root': 'data/coco/',
        'annotation': ['data/coco/annotations/coco_karpathy_test.json',
                       'data/coco/annotations/coco_karpathy_test_gt.json'],
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
    'nocaps': {
        'root': 'data/nocaps/images',
        'annotation': 'data/nocaps/nocaps_val_4500_captions.json',
        'max_new_tokens': 30,
        'min_new_tokens': 8,
    },
}


class CaptionTest(Dataset):
    def __init__(self, name, ann_file, img_prefix, tokenizer, data_args):
        if name == 'coco':
            self.images = json.load(open(ann_file, 'r'))
        else:
            self.images = json.load(open(ann_file, 'r'))['images']
        self.name = name
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end

        self.begin_str = 'Provide a one-sentence caption for the provided image.'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.name == 'coco':
            filename = self.images[idx]['image']
            image_id = int(filename.split('_')[-1].replace('.jpg', ''))
            image_path = os.path.join(self.img_prefix, filename)
        else:
            image_id = self.images[idx]['id']
            if 'file_name' in self.images[idx]:
                image_path = os.path.join(self.img_prefix, self.images[idx]['file_name'])
            else:
                image_path = os.path.join(self.img_prefix, self.images[idx]['image'])
        
        # get image
        processor = self.img_processor
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
        question = '<image>\n' + self.begin_str
        # answer is None
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
        img_metas['task'] = "image_caption"
        img_metas['dataset_name'] = self.name
        img_metas['image_id'] = image_id
        img_metas['conversations'] = prompt
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

    summaries = []  
    # create dataset and dataloader
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
        if type(ds_collections[ds_name]['annotation']) == list:
            ann_file = ds_collections[ds_name]['annotation'][0]
        else:
            ann_file = ds_collections[ds_name]['annotation']
        dataset = CaptionTest(
            name=ds_name, 
            ann_file=ann_file,
            img_prefix=ds_collections[ds_name]['root'],
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
        image_ids, captions = [], []
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
                    # generation_config=model.get_llm().generation_config,
                    do_sample=False,   # greedy search
                    temperature=0.,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
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

            outputs = outputs.lower().capitalize()
            if not outputs.endswith('.'):
                outputs += '.'
            image_id = img_metas[0]['image_id']
            image_ids.extend([image_id])
            captions.extend([outputs])

            if rank == 0:
                batch_size = 1
                for _ in range(batch_size * world_size):
                    progress_bar.update()

        # collect results from gpus
        torch.distributed.barrier()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
        average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
        print(f'Average caption length: {average_length}')

        # evaluation
        if rank == 0:
            print(f'Evaluating {ds_name} ...')

            results = []
            for image_id, caption in zip(merged_ids, merged_captions):
                results.append({
                    'image_id': int(image_id),
                    'caption': caption,
                })
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(results, open(results_file, 'w'))

            annotation = ds_collections[ds_name]['annotation']
            if type(annotation) == list:
                annotation = annotation[-1]
            coco = COCO(annotation)
            coco_result = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()

            summary = coco_eval.eval.items()
            print(summary)
            summaries.append([args.model_name, ds_name, average_length, summary])

        torch.distributed.barrier()

    # write summarize results
    out_path = '_'.join(args.model_name.split('/')[-1:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'w')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument('--datasets', type=str, default=['flickr30k', 'coco', 'nocaps'], nargs='+')
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
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