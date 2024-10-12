# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
import copy
import os
import numpy as np
from collections import defaultdict
from mmdet.datasets import CocoDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from PIL import Image

# evaluation
import contextlib
import io
import logging
import warnings
from collections import OrderedDict

import mmcv
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import RefCOCOeval


from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal

# 30 questions
QUESTIONS = [
    # interrogative
    "Where can we locate the <expression> in the image?",
    "Do you know where the <expression> is within the image?",
    "Have you seen the <expression> in this image? Where is it?",
    "Could you tell me where the <expression> is in the image?",
    "Whereabouts in the image can we find the <expression>?",
    "Do you have any idea where the <expression> might be in this image?",
    "Are you aware of the <expression>'s position within the image?",
    "Where in the image should we be looking for the <expression>?",
    "Is it possible to identify the <expression>'s location in this image?",
    "Have you figured out where the <expression> is in this image?",
    "Could you provide guidance on finding the <expression> in the image?",
    "Do you know where I can locate the <expression> in the picture?",
    "Can you tell me the precise location of the <expression> in the image?",
    "Would you be able to point out the <expression> within the image?",
    "Are you able to discern the <expression> in the image?",
    # declarative
    "Please help me locate the <expression> in the image.",
    "Please find the object indicated by the expression <expression> in the image.",
    "Please assist in identifying the <expression> within the image.",
    "Please determine the exact position of the <expression> in the image.",
    "Please ascertain the whereabouts of the <expression> in this image.",
    "Please assist me in locating the <expression> within the image.",
    "Please take a moment to find the object denoted by the expression <expression> in the image.",
    "Please help us identify the precise location of the <expression> in this image.",
    "Please provide your guidance in finding and marking the <expression> within the image.",
    "Please make it a priority to discover and highlight the <expression> within the image.",
    "Let's determine the specific area where the <expression> is situated in the image.",
    "We're aiming to establish the spatial coordinates of the <expression> in this image.",
    "We need to establish the exact whereabouts of the <expression> within the image.",
    "We are actively engaged in the process of locating the <expression> in the image.",
    "Let's find the <expression> within the image."
]

YES = [
    "Yes, it is <expression>.",
    "Certainly, it is <expression>.",
    "Absolutely, it is <expression>.",
    "Yes, it is <expression>.",
    "Affirmative, it is <expression>.",
    "Sure, it is <expression>.",
    "Of course, it is <expression>.",
    "Without question, it is <expression>.",
    "Certainly, it is <expression>.",
    "Absolutely, it is <expression>."
]


class RefCocoLlavaDataset(CocoDataset):
    """
    Return both chat data and refcoco detection data.
    """
    CLASSES = ('object',)

    def __init__(self,
                 ann_file,
                 img_prefix,
                 # conversation
                 tokenizer,
                 data_args,
                 # detection
                 test_mode=False,
                 max_gt_per_img=100,
                 with_mask=False,
                 ):
        self.task = 'grd'
        self.dataset_name = 'refcoco'

        # conversation
        self.tokenizer = tokenizer
        self.image_folder = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.num_embs = data_args.num_embs

        # detection
        self.with_mask = with_mask
        self.max_gt_per_img = max_gt_per_img
        
        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True)

        # train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
        # from the default setting in mmdet.
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=self.with_mask),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[
                    [
                        dict(
                            type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True)
                    ],
                    [
                        dict(
                            type='Resize',
                            # The radio of all image in train dataset < 7
                            # follow the original impl
                            img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                            multiscale_mode='value',
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            override=True,
                            keep_ratio=True)
                    ]
                ]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),  
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                    else ['img', 'gt_bboxes', 'gt_labels'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'img_norm_cfg', 
                                    'expressions')
            )
        ]
        # test_pipeline, NOTE the Pad's size_divisor is different from the default
        # setting (size_divisor=32). While there is little effect on the performance
        # whether we use the default setting or use size_divisor=1.
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'],
                                meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'img_norm_cfg', 
                                    'expressions')   
                    )
                ])
        ]

        # NOTE: pipeline needs to collect 'expressions'
        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=test_mode,
            pipeline=pipeline)
        
        super().__init__(**dataset_cfg)
    
    def normalize_box_coordinates(self, bbox, img_shape):
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        img_h, img_w = img_shape[:2]
        bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
        return torch.cat(bbox_new, dim=-1)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx) # after mmdet pipeline

        ########## llava ############
        # conversation version: v3, output every object
        file_name = data_item['img_metas'].data['ori_filename'] if not self.test_mode \
                        else data_item['img_metas'][0].data['ori_filename']
        conversations = []
        if not self.test_mode:  # train
            question_template = random.choice(QUESTIONS)
            answer_template = random.choice(YES)
        else:
            question_template = QUESTIONS[0]
            answer_template = YES[0]            
            
        # question
        expression = data_item['img_metas'].data['expressions'] if not self.test_mode \
                        else data_item['img_metas'][0].data['expressions']
        # expression = DEFAULT_TOKENS["sog"] + expression + DEFAULT_TOKENS["eog"]  # add special tokens for grounding
        question = question_template.replace('<expression>', expression)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        conversations.append(message1)
        # answer 
        if self.num_embs == 1:
            str_temp = "[GRD][EMB]"
        else:
            str_temp = "[EMB]" + "".join([f"[EMB{i}]" for i in range(2, self.num_embs+1)]) 
            str_temp = "[GRD]" + str_temp  # e.g. "[GRD][EMB][EMB2][EMB3][EMB4]"
        answer = answer_template.replace('<expression>', str_temp)
        message2 = {
            'from': 'gpt',
            'value': answer
        }
        conversations.append(message2)

        sources = preprocess_multimodal(copy.deepcopy([conversations]))

        # load image and clip preprocess
        processor = self.img_processor
        image = Image.open(os.path.join(self.image_folder, file_name)).convert('RGB')
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

        ######### detection ##########
        # create continue label id to random index mapping
        # there is only one [EMB]
        id2index = {0: 0}
        if not self.test_mode:
            gt_bboxes = data_item['gt_bboxes'].data
            img_shape = data_item['img_metas'].data['img_shape']
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            gt_bboxes = self.normalize_box_coordinates(gt_bboxes, img_shape) # cxcywh, [0, 1] 
            if self.with_mask:
                gt_masks = data_item['gt_masks'].data
            img_metas = data_item['img_metas'].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            if self.with_mask:
                data_dict_det = {
                    'image_aug': data_item['img'].data,
                    'class_labels': data_item['gt_labels'].data,  # continuous label ids
                    'boxes': gt_bboxes,
                    'mask_labels': gt_masks,
                    'img_metas': data_item['img_metas'].data,     # dict
                }
            else:
                data_dict_det = {
                    'image_aug': data_item['img'].data,
                    'class_labels': data_item['gt_labels'].data,  # continuous label ids
                    'boxes': gt_bboxes,
                    'img_metas': data_item['img_metas'].data,     # dict
                }
        else:
            img_metas = data_item['img_metas'][0].data
            img_metas['id2index'] = id2index
            img_metas['task'] = self.task
            img_metas['dataset_name'] = self.dataset_name
            img_metas['conversations'] = conversations
            data_dict_det = {
                'image_aug': data_item['img'][0].data,
                'img_metas': data_item['img_metas'][0].data   # dict
            }
        data_dict.update(data_dict_det)
        return data_dict

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break
            
            # RefCOCO evaluation
            cocoEval = RefCOCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            cocoEval.evaluate()

            # refcoco
            metric_items = ["P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9", "oIoU", "mIoU"]

            if cocoEval is None:
                print_log("No predictions from the model!", logger=logger)
                return {metric: float("nan") for metric in metric_items}

            # the standard metrics
            eval_results = {
                metric: float("nan")
                for idx, metric in enumerate(metric_items)
            }
            ious = np.array([v for (k, v) in cocoEval.ious.items()])
            total_intersection_area = cocoEval.total_intersection_area
            total_union_area = cocoEval.total_union_area
            iou_list = cocoEval.iou_list
            # compute metrics
            eval_results["P@0.5"] = np.sum(ious > 0.5) / len(ious) 
            eval_results["P@0.6"] = np.sum(ious > 0.6) / len(ious) 
            eval_results["P@0.7"] = np.sum(ious > 0.7) / len(ious) 
            eval_results["P@0.8"] = np.sum(ious > 0.8) / len(ious) 
            eval_results["P@0.9"] = np.sum(ious > 0.9) / len(ious) 
            eval_results["oIoU"] = total_intersection_area / total_union_area 
            eval_results["mIoU"] = np.mean(ious) 

            # print output evaluation scores
            print(eval_results)

        return eval_results