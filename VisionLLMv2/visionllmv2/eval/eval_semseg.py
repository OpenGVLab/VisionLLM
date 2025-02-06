import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import mmcv

import numpy as np
from PIL import Image
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, List
import evaluate
from ..util.misc import dict_to_cuda

def process_seg_result(mask_cls, mask_pred, image_size, target_size, sem_seg_postprocess_before_inference=True):
    prob = mask_cls.sigmoid()        # [nq, k]
    mask_pred = mask_pred.sigmoid()  # [nq, h/4, w/4]
    H, W = mask_pred.shape[-2:]

    if sem_seg_postprocess_before_inference:
        # remove padding and resize to original size
        mask_pred = F.interpolate(mask_pred[:, None], size=(H*4, W*4), mode="bilinear", align_corners=False)
        mask_pred = mask_pred[:, :, :image_size[0], :image_size[1]]  # remove padding
        mask_pred = F.interpolate(mask_pred, size=target_size[:2], mode="bilinear", align_corners=False)[:, 0] # [nq, ori_h, ori_w]

        # final results
        semseg = torch.einsum('qc,qhw->chw', prob, mask_pred) 
        semantic_map = semseg.argmax(dim=0)  # [K, H, W] -> [H, W]
    else:
        mask_pred = torch.einsum('qc,qhw->chw', prob, mask_pred) 
        mask_pred = F.interpolate(mask_pred[:, None], size=(H*4, W*4), mode="bilinear", align_corners=False)
        mask_pred = mask_pred[:, :, :image_size[0], :image_size[1]].cpu()  # remove padding, to cpu in case of OOM
        mask_pred = F.interpolate(mask_pred, size=target_size[:2], mode="bilinear", align_corners=False)[:, 0] # [nq, ori_h, ori_w]
        semantic_map = mask_pred.argmax(dim=0)  # [K, H, W] -> [H, W]

    return semantic_map.cpu().numpy()

def post_process_sem_seg(outputs, target_sizes, image_sizes, num_classes=150, topk=100, sem_seg_postprocess_before_inference=True):
    # [1, nq, K], [1, nq, 4], [1, nq, h/4, w/4], inference bs=1
    out_logits, out_bbox, out_mask = outputs.gdino_outputs.logits, outputs.gdino_outputs.pred_boxes, outputs.gdino_outputs.pred_masks
    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )
    postprocess_results = []

    for mask_cls_result, mask_pred_result, image_size, target_size in zip(
            out_logits, out_mask, image_sizes, target_sizes):
        # mask_cls_result:  [nq, K]
        # mask_pred_result: [nq, h/4, w/4]
        mask_cls_result = mask_cls_result[..., :num_classes] # [nq, K]
        mask_pred_result = process_seg_result(
            mask_cls_result,
            mask_pred_result,
            image_size, 
            target_size,
            sem_seg_postprocess_before_inference,
        )  # [H, W]
        postprocess_results.append(mask_pred_result)
    return postprocess_results

def eval_semseg(model, eval_dataloader, num_classes=150, topk=100, sem_seg_postprocess_before_inference=True):
    """
    sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
    """
    # metric = evaluate.load("mean_iou")   # using evaluate (python package)
    model = model.cuda()
    model.eval()
    results = []
    dataset = eval_dataloader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        progress_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)   # This line can prevent deadlock problem in some cases.

    for idx,data in enumerate(eval_dataloader):
        with torch.no_grad():
            data = dict_to_cuda(data)
            result = model.forward(**data, return_dict=True)
            target_size = [img_meta['ori_shape'] for img_meta in data['img_metas']]
            image_size = [img_meta['img_shape'] for img_meta in data['img_metas']]
            file_names = [img_meta['filename'] for img_meta in data['img_metas']]
            anno_names = [i.replace('images','annotations').replace('.jpg','.png') for i in file_names]
            result = post_process_sem_seg(result, target_size, image_size,num_classes=num_classes, topk=topk,
                                sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference)  # [H, W], np.array

            # collect using mmseg
            results.extend(result)
            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    progress_bar.update()

    # collect results from gpus
    results = collect_results_cpu(results, len(dataset), None)
    if rank == 0:
        dataset.evaluate(results, metric='mIoU')

        
    