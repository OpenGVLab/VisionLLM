import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import time

import requests
from PIL import Image
from io import BytesIO

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
from visionllmv2.datasets.utils import boxes_to_masks
from visionllmv2.datasets.vg import VisualGenome, FINAL_QUESTIONS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

IMAGE_TOKEN_INDEX = -200


class VisualGenomeTest(VisualGenome):
    def preprocess_data(self, data_item):
        # image = data_item['img'].data             # [c, h, w], 336x336
        file_name = data_item['img_metas'].data['ori_filename'] 
        image = Image.open(os.path.join(self.img_prefix, file_name)).convert('RGB')
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

        # test do not know the caption
        # label = data_item['gt_labels'][0]       # caption, annotation file has been preprocessed, each image has one caption.
        bboxes = data_item['gt_bboxes'].data[0].unsqueeze(0)  # [n, 4], xyxy in image shape (after aug), refcoco has 1 regions
        img_shape = data_item['img_metas'].data['img_shape']

        # generate regions
        # box -> mask, [n, h, w]
        regions = boxes_to_masks(bboxes, img_shape)
        
        # -----------------------------------------------------
        # chat
        # question
        question_template = FINAL_QUESTIONS[0] 
        region_str = DEFAULT_TOKENS['sor'] + 'region' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']  # '<reg>region<region></reg>'
        question = question_template.replace('<spi_descript>', region_str)
        question = '<image>\n' + question
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
        img_metas = data_item['img_metas'].data
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = prompt  # TODO: delete this, just for debug
        data_dict['img_metas'] = img_metas
        # update regions
        data_dict['regions'] = regions  # [n, h, w]
        return data_dict
    
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
    dataset = VisualGenomeTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        data_args=data_args,
        test_mode=True
    )
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    # FIXME: num_worker > 0 would cause RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
    # To use CUDA with multiprocessing, you must use the 'spawn' start method.
    dataloader = DataLoader(dataset=dataset, sampler=sampler, collate_fn=custom_collate_fn,
                        batch_size=args.batch_size_per_gpu, num_workers=8, pin_memory=True)  
    
    # stop criterion, this is needed for internlm2
    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    # begin inference
    model.eval()
    results = []
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
                max_new_tokens=64,
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

        # change to coco format for evaluation
        outputs = outputs.lower()
        if outputs.endswith('.'):
            outputs = outputs[:-1]  # remove '.'
        results.extend([outputs])

        if rank == 0:
            batch_size = 1
            for _ in range(batch_size * world_size):
                progress_bar.update()

    # collect results from gpus
    results = collect_results_cpu(results, len(dataset), None)
    if rank == 0:
        dataset.evaluate(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model and data
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--ann_file", type=str, default="data/vg/annotations/vg_test_coco_format.json")
    parser.add_argument("--img_prefix", type=str, default="data/vg/VG_100K/")
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
