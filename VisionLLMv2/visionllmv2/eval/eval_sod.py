import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import mmcv
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu
from mmdet.core import bbox2result
from mmdet.core.mask import mask2bbox, encode_mask_results

from typing import Dict, Optional, Sequence, List
from PIL import Image
import numpy as np

# https://github.com/lartpang/PySODMetrics
from py_sod_metrics import Fmeasure, MAE, Smeasure, Emeasure, WeightedFmeasure

from ..util.box_ops import box_cxcywh_to_xyxy
from ..util.misc import dict_to_cuda


def post_process_instseg_gdino(outputs, target_sizes, image_sizes, num_classes=80, topk=100,  mask_stride=4, threshold=0.):
    # [B, nq, K], [B, nq, 4], [B, nq, H/4, W/4]
    out_logits, out_bbox, out_mask = outputs.gdino_outputs.logits, outputs.gdino_outputs.pred_boxes, outputs.gdino_outputs.pred_masks
    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )
        
    out_logits = out_logits[:, :, :num_classes]

    results = []
    for i, (logits_per_image, box_pred_per_image, target_size, image_size) in enumerate(zip(
        out_logits, out_bbox, target_sizes, image_sizes
    )):
        # select topk predictions
        prob = logits_per_image.sigmoid()   # [nq, k]
        prob = prob.view(-1)                # [nqxk,]
        k_value = min(topk, prob.size(0))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=0)
        scores_per_image = topk_values
        topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode="floor")  # // k

        num_queries = logits_per_image.shape[0] 
        labels_per_image = topk_indexes % logits_per_image.shape[-1]  # % k

        # box
        box_pred_per_image = box_pred_per_image[topk_boxes]
        box_pred_per_image = box_cxcywh_to_xyxy(box_pred_per_image)
        # and from relative [0, 1] to absolute [0, height] coordinates
        ori_h, ori_w = target_size[:2]
        scale_fct = torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float32).to(box_pred_per_image.device)
        box_pred_per_image = box_pred_per_image * scale_fct[None, :]

        # mask
        mask_pred_i = out_mask[i][topk_boxes] # [topk, H/4, W/4]
        H, W = mask_pred_i.shape[-2:]
        # mask = F.interpolate(mask_pred_i[:, None], (ori_h, ori_w), mode='bilinear', align_corners=False)[:, 0] # direct resize, [topk, ori_h, ori_w]
        mask = F.interpolate(mask_pred_i[:, None], size=(H*mask_stride, W*mask_stride), mode='bilinear', align_corners=False)
        mask = mask[:, :, :image_size[0], :image_size[1]]  # remove padding
        mask = F.interpolate(mask, size=(ori_h, ori_w), mode='bilinear', align_corners=False)[:, 0] # [topk, ori_h, ori_w]
        mask = mask.sigmoid() > 0.5 # bool

        results.append({"scores": scores_per_image, "labels": labels_per_image, "boxes": box_pred_per_image,
                        "masks": mask})
    return results


def eval_sod(model, eval_dataloader, num_classes, topk=100, mask_stride=4, save_dir='results/DUTS', gt_dir='data/sod/DUTS/DUTS-TE-Mask'):
    # save dir
    os.makedirs(save_dir, exist_ok=True)

    model = model.cuda()
    model.eval()
    results = []
    dataset = eval_dataloader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        progress_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for data in eval_dataloader:
        with torch.no_grad():
            # data.pop('labels')  # TODO: delete LLM labels
            data = dict_to_cuda(data)
            result = model.forward(**data, return_dict=True)
            target_size = [img_meta['ori_shape'] for img_meta in data['img_metas']]
            image_size = [img_meta['img_shape'] for img_meta in data['img_metas']]
            # instance segmentation
            result = post_process_instseg_gdino(result, target_size, image_size, num_classes=num_classes, topk=topk, mask_stride=mask_stride)
            # list[dict], list length is batch size
            # save img, batch_size=1
            assert len(result) == 1
            filename = data['img_metas'][0]['filename']
            ori_filename = os.path.basename(filename).split('.')[0]  # only name
            ori_img = Image.open(filename).convert('RGB')
            mask = result[0]['masks'][0].detach().cpu().numpy().astype(np.float32)   # [ori_h, ori_w]
            mask = Image.fromarray(mask * 255).convert('L')
            ori_img.save(os.path.join(save_dir, f'{ori_filename}_img.jpg'))
            mask.save(os.path.join(save_dir, f'{ori_filename}.png'))  # for testing with gt

        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                progress_bar.update()
    
    # collect results from gpus
    results = collect_results_cpu(results, len(dataset), None)
    # evaluation
    if rank == 0:
        print(f"\nEvaluating {save_dir}...")
        preds = sorted(os.listdir(save_dir))
        gts = sorted(os.listdir(gt_dir))
        preds = [pred for pred in preds if pred in gts]
        gts = [gt for gt in gts if gt in preds]
        assert len(preds) == len(gts)

        SM = Smeasure()
        WFM = WeightedFmeasure()
        EM = Emeasure()
        Mae = MAE()

        for pred, gt in zip(preds, gts):
            pred_mask = np.array(Image.open(os.path.join(save_dir, pred)).convert('L'))
            gt_mask = np.array(Image.open(os.path.join(gt_dir, gt)).convert('L'))

            assert pred_mask.shape == gt_mask.shape, f"pred shape {pred_mask.shape} does not match the size of gt {gt_mask.shape}"
            SM.step(pred=pred_mask, gt=gt_mask)
            WFM.step(pred=pred_mask, gt=gt_mask)
            EM.step(pred=pred_mask, gt=gt_mask)
            Mae.step(pred=pred_mask, gt=gt_mask)

        sm = SM.get_results()["sm"]
        wfm = WFM.get_results()["wfm"]
        em = EM.get_results()["em"]['curve'].mean()
        mae = Mae.get_results()["mae"]

        print("Smeasure:", sm)
        print("WeightedFmeasure:", wfm)
        print("Emeasure:", em)
        print("MAE:", mae)
            


