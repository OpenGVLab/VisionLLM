import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import mmcv
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu
from mmdet.core import bbox2result
from mmdet.core.mask import mask2bbox, encode_mask_results

from typing import Dict, Optional, Sequence, List

from ..util.box_ops import box_cxcywh_to_xyxy
from ..util.misc import dict_to_cuda


# gdino
def post_process_det_gdino(outputs, target_sizes, num_classes, threshold=0., topk=100):
    out_logits, out_bbox = outputs.gdino_outputs.logits, outputs.gdino_outputs.pred_boxes
    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    out_logits = out_logits[:, :, :num_classes]

    # the classes have standard orders
    prob = out_logits.sigmoid()  # [B, num_queries, K]
    prob = prob.view(out_logits.shape[0], -1)   # [B, N*K]
    k_value = min(topk, prob.size(1))
    topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
    scores = topk_values
    topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")

    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    if isinstance(target_sizes, List):
        img_h = torch.Tensor([i[0] for i in target_sizes])
        img_w = torch.Tensor([i[1] for i in target_sizes])
    else:
        img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box})

    return results


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


def eval_det(model, eval_dataloader, num_classes, topk=100, mask_stride=4, with_mask=False, jsonfile_prefix=None):
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
            data = dict_to_cuda(data)
            result = model.forward(**data, return_dict=True)
            target_size = [img_meta['ori_shape'] for img_meta in data['img_metas']]
            image_size = [img_meta['img_shape'] for img_meta in data['img_metas']]
            if not with_mask:  # det
                result = post_process_det_gdino(result, target_size, num_classes=num_classes, topk=topk)
                result = [
                    (torch.cat((x['boxes'], x['scores'].unsqueeze(1)), dim=1), x['labels']) for x in result
                ]
                result = [
                    bbox2result(det_bboxes, det_labels, num_classes)
                    for det_bboxes, det_labels in result
                ]
            else: # instseg
                result = post_process_instseg_gdino(result, target_size, image_size, num_classes=num_classes, topk=topk, mask_stride=mask_stride)
                postprocess_result = []
                for i in range(len(result)):
                    labels_per_image, scores, boxes, mask_pred_binary = \
                        result[i]["labels"], result[i]["scores"], result[i]["boxes"], result[i]["masks"]
                    bboxes = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
                    bbox_results = bbox2result(bboxes, labels_per_image, num_classes)
                    mask_results = [[] for _ in range(num_classes)]
                    for j, label in enumerate(labels_per_image): # each prediction
                        mask = mask_pred_binary[j].detach().cpu().numpy()
                        mask_results[label].append(mask)
                    result_i = (bbox_results, encode_mask_results(mask_results))
                    postprocess_result.append(result_i)
                result = postprocess_result
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                progress_bar.update()
    
    # collect results from gpus
    results = collect_results_cpu(results, len(dataset), None)
    if rank == 0:
        if not with_mask: # det
            dataset.evaluate(results, jsonfile_prefix=jsonfile_prefix)
        else: # det/seg
            dataset.evaluate(results, jsonfile_prefix=jsonfile_prefix, metric=['bbox', 'segm'])