import argparse
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import re
import time
import json
import copy
import numpy as np

import requests
from PIL import Image
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from lvis import LVIS, LVISEval, LVISResults

import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import visionllmv2.util.misc as utils
from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.utils import disable_torch_init, init_distributed_mode
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.utils import boxes_to_masks
from visionllmv2.datasets.v3det import COCO_QUESTIONS
from visionllmv2.datasets.lvis import LVIS_QUESTIONS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

IMAGE_TOKEN_INDEX = -200

ds_collections = {
    'coco': {
        'img_path' : 'data/coco/val2017',
        'ann_path' : 'data/coco/annotations/instances_val2017.json',
    },
    'lvis': {
        'img_path' : 'data/lvis/val2017',
        'ann_path' : 'data/lvis/annotations/lvis_v1_val.json',
    },
}

def clean_string(expression):
    expression = re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ').replace('_', ' ')
    return expression

class CoCoRecognitionTest(Dataset):
    def __init__(self, name, ann_file, img_prefix, tokenizer, data_args, test_format='bbox'):
        meta_info = json.load(open(ann_file, 'r'))
        self.anns = meta_info['annotations']
        self.images = {}
        for image in meta_info['images']:
            self.images[image['id']] = image  # int -> dict

        self.name = name
        self.img_prefix = img_prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.test_format = test_format  # 'bbox' or 'mask'
        assert test_format in ['bbox', 'mask']

        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.image_size = data_args.image_size

        assert name in ['coco', 'lvis']
        self.question_template = COCO_QUESTIONS[0] if name == 'coco' else LVIS_QUESTIONS[0]
        
        self.idx2str = {}
        for category_info in meta_info['categories']:
            if name == 'coco':
                self.idx2str[category_info['id']] = category_info['name'] # category_id -> category_name
            elif name == 'lvis':
                self.idx2str[category_info['id']] = clean_string(category_info['name']) # category_id -> category_name, clean string for lvis
        self.str2idx = {v: k for k, v in self.idx2str.items()} # category_name -> category_id

        sorted_key = sorted(list(self.idx2str.keys()))  # in-continuous for coco
        labels_name = [self.idx2str[key] for key in sorted_key]
        self.sorted_key = sorted_key
        self.sorted_labels_name = labels_name

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

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        image_id = self.anns[idx]['image_id']
        image_info = self.images[image_id]  # dict

        # get image
        try:
            image = Image.open(os.path.join(
                self.img_prefix,
                self.images[image_id].get('file_name', f'{image_id:012d}.jpg')
            )).convert('RGB')
        except:
            image = Image.open(os.path.join(
                self.img_prefix.replace('val2017', 'train2017'),
                self.images[image_id].get('file_name', f'{image_id:012d}.jpg')
            )).convert('RGB')
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

        # get label name
        category_id = self.anns[idx]['category_id']
        label_name = self.idx2str[category_id]
        label = f'{label_name}'

        # get bbox or mask as regions
        img_h, img_w = image_info['height'], image_info['width']
        gt_bbox = self.anns[idx]['bbox']  # [x, y, w, h]
        if self.test_format == 'bbox':
            x, y, w, h = self.anns[idx]['bbox']  
            bbox = torch.as_tensor([x, y, x+w, y+h])  # xyxy in original image size
            scale_fct = torch.as_tensor([img_w, img_h, img_w, img_h])
            bbox = bbox / scale_fct  # xyxy in normalized [0, 1]
            scale_fct = torch.as_tensor([self.image_size, self.image_size, self.image_size, self.image_size])
            bbox = bbox * scale_fct  # xyxy in input image size
            bbox = bbox.unsqueeze(0)
            img_shape = [self.image_size, self.image_size]
            regions = boxes_to_masks(bbox, img_shape)
        else:  # mask
            mask = self.anns[idx]['segmentation']
            mask = self.annToMask(mask, img_h, img_w)
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()  # [1, h, w], uint -> float
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0),
                                                    size=(self.image_size, self.image_size),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0).bool() # bool
            regions = mask

        # ----------------------------------------
        # get conversation
        region_str = DEFAULT_TOKENS['sor'] + f'region1' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']  # '<reg>region1<region></reg>'
        question = self.question_template.replace('<regions>', region_str)
        question = "<image>\n" + question
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
        data_dict['regions'] = regions  # [n, h, w]
        img_metas = dict()
        img_metas['task'] = "region_recognition"
        img_metas['dataset_name'] = self.name
        img_metas['image_id'] = image_id
        img_metas['label'] = label  # gt label name
        img_metas['bbox'] = gt_bbox # gt bbox
        img_metas['label_names'] = self.sorted_labels_name  # all category names
        img_metas['str2idx'] = self.str2idx  # category name -> category id
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
    images = batch[0]['image']                   # [3, h, w]
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

    os.makedirs(args.out_dir, exist_ok=True)

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
        dataset = CoCoRecognitionTest(
            name=ds_name, 
            ann_file=ds_collections[ds_name]['ann_path'],
            img_prefix=ds_collections[ds_name]['img_path'],
            tokenizer=tokenizer,
            data_args=data_args,
            test_format=args.test_format
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
        labels_check, predictions = [], []
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
                    max_new_tokens=5,
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
            # print(outputs)  # predicted label name

            outputs = outputs.lower()  # coco and lvis category names are lower case
            if outputs.endswith('.'):
                outputs = outputs[:-1] 

            image_id = img_metas[0]['image_id']
            label = img_metas[0]['label']  # gt label name
            bbox = img_metas[0]['bbox']    # gt bbox
            label_names = img_metas[0]['label_names']  # all category names
            str2idx = img_metas[0]['str2idx']  # category name -> category id
            if outputs not in label_names:  # mis-prediction
                labels_check.extend([False])
                continue
            labels_check.extend([outputs == label])
            # prediction
            pred = {
                'image_id': image_id,
                'category_id': str2idx[outputs],  # dataset original category id
                'bbox': bbox,
                'score': 1.0
            }
            predictions.extend([pred])

            if rank == 0:
                batch_size = 1
                for _ in range(batch_size * world_size):
                    progress_bar.update()

        # collect results from gpus
        torch.distributed.barrier()
        gathered_labels_check = utils.all_gather(labels_check)
        gathered_predictions = utils.all_gather(predictions)
        labels_check = [p for p_list in gathered_labels_check for p in p_list]  # list[bool]
        predictions = [p for p_list in gathered_predictions for p in p_list]    # list[dict]

        # evaluation
        if rank == 0:
            print(f'Evaluating {ds_name} ...')

            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(predictions, open(results_file, 'w'))

            # eval using api
            ann_file = ds_collections[ds_name]['ann_path']
            if ds_name == 'coco':
                print('Evaluation COCO results:')
                print('Accuracy: {:.3}'.format(sum(labels_check) / len(labels_check)))
                coco_gt = COCO(ann_file)
                coco_dt = coco_gt.loadRes(results_file)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            elif ds_name == 'lvis':
                print('Evaluation LVIS results:')
                print('Accuracy: {:.3}'.format(sum(labels_check) / len(labels_check)))
                lvis_gt = LVIS(ann_file)
                lvis_dt = LVISResults(lvis_gt, results_file)
                lvis_eval = LVISEval(lvis_gt, lvis_dt, 'bbox')
                lvis_eval.run()
                lvis_metrics = lvis_eval.get_results()
                for k, v in lvis_metrics.items():
                    print('{}: {}'.format(k, v))

        torch.distributed.barrier()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument('--datasets', type=str, default=['coco'], nargs='+')
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument('--conv_mode', type=str, default='vicuna_v1')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument("--use_im_start_end", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--image_max_tile", type=int, default=6)
    parser.add_argument("--use_pixelshuffle", type=bool, default=False)
    parser.add_argument("--test_format", type=str, default="bbox")  # 'bbox' or 'mask'
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

