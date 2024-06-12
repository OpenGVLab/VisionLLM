import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import mmcv

import numpy as np
from PIL import Image
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu

from typing import Dict, Optional, Sequence, List
import itertools
from terminaltables import AsciiTable

from ..util.box_ops import box_cxcywh_to_xyxy
from ..util.misc import dict_to_cuda

def post_process_pose(outputs, target_sizes, num_classes=1, topk=100, num_body_points=17, id_mapping=None, threshold=0., ):
    # [bs, nq, k], [bs, nq, 4], [bs, nq, 68*3], xyxyzz
    out_logits, out_bbox, out_keypoints = outputs.unipose_outputs.pred_logits, outputs.unipose_outputs.pred_boxes, outputs.unipose_outputs.pred_keypoints
    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )
        
    out_logits = out_logits[:, :, :num_classes]

    prob = out_logits.sigmoid()  # [bs, nq, k]
    prob = prob.view(out_logits.shape[0], -1)   # [bs, nq*k]
    k_value = min(topk, prob.size(1))
    topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
    scores = topk_values  # [bs, topk]
    topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")

    # label
    labels = topk_indexes % out_logits.shape[2]  # [bs, topk,]
    # convert from continuous category id to dataset category id
    assert id_mapping is not None
    new_labels = torch.zeros_like(labels)  # [bs, topk]
    for batch_idx in range(len(labels)):
        for j in range(labels.shape[-1]):
            new_labels[batch_idx, j] = id_mapping[labels[batch_idx, j].item()]
    labels = new_labels

    # box
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))  # [bs, topk, 4]
    # and from relative [0, 1] to absolute [0, height] coordinates
    if isinstance(target_sizes, List):
        img_h = torch.Tensor([i[0] for i in target_sizes])
        img_w = torch.Tensor([i[1] for i in target_sizes])
    else:
        img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    # keypoint
    topk_keypoints = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
    keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, 68*3))  # [bs, topk, 68*3]
    Z_pred = keypoints[:, :, :(num_body_points*2)]            # [bs, topk, 17*2]
    V_pred = torch.ones_like(Z_pred)[:, :, :num_body_points]  # [bs, topk, 17]
    # and from relative [0, 1] to absolute [0, height] coordinates
    if isinstance(target_sizes, List):
        target_sizes = torch.stack(target_sizes)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h], dim=1).repeat(1, num_body_points)[:, None, :].to(Z_pred.device)
    Z_pred = Z_pred * scale_fct  # [bs, topk, 17*2]
    keypoints = torch.cat([Z_pred, V_pred], dim=-1) # [bs, topk, 17*3], xyxyzz
    # xyxyzz -> xyzxyz
    keypoints_res = torch.zeros_like(keypoints)
    keypoints_res[..., 0::3] = Z_pred[..., 0::2]
    keypoints_res[..., 1::3] = Z_pred[..., 1::2]
    keypoints_res[..., 2::3] = V_pred[..., 0::1]

    results = []
    for s, l, b, k in zip(scores, labels, boxes, keypoints_res):  # for each batch
        score = s[s > threshold]     # [n,]
        label = l[s > threshold]     # [n,]
        box = b[s > threshold]       # [n, 4]
        keypoint = k[s > threshold]  # [n, 17*3], xyzxyz
        results.append({"scores": score, "labels": label, "boxes": box, "keypoints": keypoint})
    return results
    


def eval_pose(model, eval_dataloader, num_classes=1, topk=100, num_body_points=17, dataset_name='coco', ann_file=None):
    model = model.cuda()
    model.eval()
    results = []
    dataset = eval_dataloader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        progress_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)   # This line can prevent deadlock problem in some cases.

    # create evaluator
    iou_types = tuple(k for k in ( 'bbox', 'keypoints'))
    assert ann_file is not None
    if dataset_name == 'coco':
        from ..datasets.evaluation.coco_eval import CocoEvaluator
    elif dataset_name == 'crowdpose':
        from ..datasets.evaluation.crowdpose_eval import CocoEvaluator
    elif dataset_name == 'unikpt':
        from ..datasets.evaluation.unikpt_eval import CocoEvaluator_custom as CocoEvaluator
    else:
        raise NotImplementedError
    coco_evaluator = CocoEvaluator(ann_file, iou_types, useCats=True)

    # id mapping, for converting continuous id to dataset category id
    CLASSES = dataset.CLASSES
    id_mapping =  {idx: k for idx, (k, v) in enumerate(CLASSES.items())}

    for idx, data in enumerate(eval_dataloader):
        with torch.no_grad():
            data = dict_to_cuda(data)
            result = model.forward(**data, return_dict=True)
            image_id = [img_meta['image_id'] for img_meta in data['img_metas']]
            target_size = [img_meta['ori_shape'] for img_meta in data['img_metas']]
            result = post_process_pose(result, target_size, num_classes, topk, num_body_points, id_mapping)  # list[dict], list length=1
            res = {img_id.item(): output for img_id, output in zip(image_id, result)} # dict[dict]
            results.append(res)
            if coco_evaluator is not None:
                coco_evaluator.update(res)
            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    progress_bar.update()

    if dataset_name=='unikpt' and dataset.eval_pck:
        results = collect_results_cpu(results, len(dataset), None) # list[dict]
        coco_anno = coco_evaluator.coco_gt.dataset
        if rank == 0:
            print(f"\nEvaluating {dataset_name}...")
            results_dict = {}
            for result in results:
                results_dict.update(result)
            pck = eval_pck(results_dict, coco_anno)
            print('eval pck result:', pck)
        return
    
    # eval metrics
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        if rank == 0:
            print(f"\nEvaluating {dataset_name}...")
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

            if True:  # Compute per-category AP
                coco_anno = coco_evaluator.coco_gt.dataset
                # from https://github.com/facebookresearch/detectron2/
                precisions = coco_evaluator.coco_eval['bbox'].eval['precision']
                cat_ids = [i['id'] for i in coco_anno['categories']]
                # precision: (iou, recall, cls, area range, max dets)
                assert len(cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    # nm = self.coco.loadCats(catId)[0]
                    nm = [i for i in coco_anno['categories'] if i['id'] == catId][0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                
                table = AsciiTable(table_data)
                print('\n' + table.table)



def eval_pck(results, gt):
    pred_coords = [] #[N,K,D]
    gt_coords = [] #[N,K,D]
    masks = [] #[N,K]
    norm_size_bbox = [] #[N,2]

    #gt = json.load(open(ann_file,'r'))
    assert len(gt['images']) == len(gt['annotations'])
    imgid2anno = {i['image_id']:i for i in gt['annotations']}
    #print(gt['images'][3])
    
    for img_id in results.keys():
        result = results[img_id]
        scores = np.array(result['scores'].cpu())
        top_id = np.argmax(scores)

        pred_kpts = np.array(result['keypoints'].cpu()[top_id]).reshape(-1, 3)[:,:2]
        box_size = np.max(np.array(imgid2anno[img_id]['bbox'][2:]))
        box_size = np.array([box_size,box_size])
        mask = np.array(imgid2anno[img_id]['keypoints']).reshape(-1,3)[:,2] != 0
        gt_kpts = np.array(imgid2anno[img_id]['keypoints']).reshape(-1,3)[:,:2]

        assert gt_kpts.shape == pred_kpts.shape

        pred_coords.append(pred_kpts)
        gt_coords.append(gt_kpts)
        masks.append(mask)
        norm_size_bbox.append(np.array(box_size))
        
    pred_coords = np.stack(pred_coords)
    gt_coords = np.stack(gt_coords)
    mask = np.stack(masks)
    norm_size_bbox = np.stack(norm_size_bbox)
    _, pck, _ = PCKUtils().keypoint_pck_accuracy(pred_coords, gt_coords, mask, 0.2, norm_size_bbox)
    return pck

class PCKUtils:
    def _calc_distances(self, preds: np.ndarray, gts: np.ndarray, mask: np.ndarray,
                        norm_factor: np.ndarray) -> np.ndarray:
        """Calculate the normalized distances between preds and target.
    
        Note:
            - instance number: N
            - keypoint number: K
            - keypoint dimension: D (normally, D=2 or D=3)
    
        Args:
            preds (np.ndarray[N, K, D]): Predicted keypoint location.
            gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
            mask (np.ndarray[N, K]): Visibility of the target. False for invisible
                joints, and True for visible. Invisible joints will be ignored for
                accuracy calculation.
            norm_factor (np.ndarray[N, D]): Normalization factor.
                Typical value is heatmap_size.
    
        Returns:
            np.ndarray[K, N]: The normalized distances. \
                If target keypoints are missing, the distance is -1.
        """
        N, K, _ = preds.shape
        # set mask=0 when norm_factor==0
        _mask = mask.copy()
        _mask[np.where((norm_factor == 0).sum(1))[0], :] = False
    
        distances = np.full((N, K), -1, dtype=np.float32)
        # handle invalid values
        norm_factor[np.where(norm_factor <= 0)] = 1e6
        distances[_mask] = np.linalg.norm(
            ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
        return distances.T


    def _distance_acc(self, distances: np.ndarray, thr: float = 0.5) -> float:
        """Return the percentage below the distance threshold, while ignoring
        distances values with -1.
    
        Note:
            - instance number: N
    
        Args:
            distances (np.ndarray[N, ]): The normalized distances.
            thr (float): Threshold of the distances.
    
        Returns:
            float: Percentage of distances below the threshold. \
                If all target keypoints are missing, return -1.
        """
        distance_valid = distances != -1
        num_distance_valid = distance_valid.sum()
        if num_distance_valid > 0:
            return (distances[distance_valid] < thr).sum() / num_distance_valid
        return -1


    def keypoint_pck_accuracy(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                              thr: np.ndarray, norm_factor: np.ndarray) -> tuple:
        """Calculate the pose accuracy of PCK for each individual keypoint and the
        averaged accuracy across all keypoints for coordinates.
    
        Note:
            PCK metric measures accuracy of the localization of the body joints.
            The distances between predicted positions and the ground-truth ones
            are typically normalized by the bounding box size.
            The threshold (thr) of the normalized distance is commonly set
            as 0.05, 0.1 or 0.2 etc.
    
            - instance number: N
            - keypoint number: K
    
        Args:
            pred (np.ndarray[N, K, 2]): Predicted keypoint location.
            gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
            mask (np.ndarray[N, K]): Visibility of the target. False for invisible
                joints, and True for visible. Invisible joints will be ignored for
                accuracy calculation.
            thr (float): Threshold of PCK calculation.
            norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.
    
        Returns:
            tuple: A tuple containing keypoint accuracy.
    
            - acc (np.ndarray[K]): Accuracy of each keypoint.
            - avg_acc (float): Averaged accuracy across all keypoints.
            - cnt (int): Number of valid keypoints.
        """
        distances = self._calc_distances(pred, gt, mask, norm_factor)
        acc = np.array([self._distance_acc(d, thr) for d in distances])
        valid_acc = acc[acc >= 0]
        cnt = len(valid_acc)
        avg_acc = valid_acc.mean() if cnt > 0 else 0.0
        return acc, avg_acc, cnt
