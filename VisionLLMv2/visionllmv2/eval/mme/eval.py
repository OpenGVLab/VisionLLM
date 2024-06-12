import argparse
import os
import re

import torch

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import CLIPImageProcessor

from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.utils import disable_torch_init
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

IMAGE_TOKEN_INDEX = -200

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = VisionLLMv2Model.from_pretrained(model_name, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16).cuda()
    model.get_llm().config.use_cache = True
    # init special token ids
    model.init_special_token_ids(tokenizer)

    processor = CLIPImageProcessor.from_pretrained(args.vis_encoder_path)

    output = os.path.basename(args.model_name)
    os.makedirs(output, exist_ok=True)

    # stop criterion, this is needed for internlm2
    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    base_prompt = 'Answer the question using a single word or phrase.'
    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(output, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')

            # get image
            img_path = os.path.join('data/mme/images', filename, img)
            assert os.path.exists(img_path), img_path
            image = Image.open(img_path).convert('RGB')
            if args.image_aspect_ratio == 'anyres':
                image = dynamic_preprocess(image, image_size=args.image_size, max_num=args.image_max_tile) # list[pil_img]
                image = [processor.preprocess(x, return_tensors='pt')['pixel_values'][0] for x in image]
                image = torch.stack(image)  # [n_split, 3, h, w]
                image_token_len = int((args.image_size // 14) ** 2)
                if args.use_pixelshuffle:
                    image_token_len = image_token_len // 4
                image_token_len = image_token_len * len(image)
                image_tensor = [image.cuda().to(torch.bfloat16)]  # list[tensor], 1 x [n_split, 3, h, w]
            elif args.image_aspect_ratio == 'pad':
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_token_len = int((args.image_size // 14) ** 2)
                if args.use_pixelshuffle:
                    image_token_len = image_token_len // 4
                image_tensor = image.unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 3, h, w]
            else:  # resize
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_token_len = int((args.image_size // 14) ** 2)
                if args.use_pixelshuffle:
                    image_token_len = image_token_len // 4
                image_tensor = image.unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 3, h, w]

            # get prompt
            question = question + ' ' + base_prompt
            question = '<image>\n' + question
            conv_mode = args.conv_mode
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # tokenizer conversations
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() # [1, L]
            # replace with 'imp' tokens
            replace_token = DEFAULT_TOKENS['imp'] * image_token_len
            if args.use_im_start_end:
                replace_token = DEFAULT_TOKENS['boi'] + replace_token + DEFAULT_TOKENS['eoi']
            replace_token_ids = tokenizer([replace_token], return_tensors="pt").input_ids[0][1:].cuda() # [L,], remove start token
            index = input_ids[0].argmin()  # find the index of IMAGE_TOKEN_INDEX
            new_input_ids = torch.cat([input_ids[0, :index], replace_token_ids, input_ids[0, index+1:]], dim=0).unsqueeze(0)
            input_ids = new_input_ids  # [1, L]

            # stop criterion
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,  # greedy search
                    temperature=0.,
                    max_new_tokens=20,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            response = response.strip()
            response = post_processing(response)
            question = question.replace('<image>\n', '')
            print(img, question, gt, response, sep='\t', file=fout)

        fin.close()
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--conv_mode', type=str, default='vicuna_v1')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument("--use_im_start_end", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--image_max_tile", type=int, default=6)
    parser.add_argument("--use_pixelshuffle", type=bool, default=False)
    parser.add_argument("--vis_encoder_path", type=str, default="checkpoints/clip-vit-large-patch14-336")
    parser.add_argument('--root', type=str, default='data/mme/eval_tool/Your_Results')
    parser.add_argument('--beam-num', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)
