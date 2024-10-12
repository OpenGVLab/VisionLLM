import os
import contextlib
import copy
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from .coco_eval import CocoEvaluator as CocoEvaluator_base

class COCOeval_custom(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', oks_sigma = None):
        super().__init__(cocoGt=cocoGt,cocoDt=cocoDt,iouType=iouType)
        
        self.params.kpt_oks_sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
            1.07, .87, .87, .89, .89, .25, .25, .25, .25, .25, .25, .25, .25,
            .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
            .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
            .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
        ], dtype=np.float32) / 10.0

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        #k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    k = len(xd)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                
                vars_cutoff = vars[:len(dx)]
                e = (dx**2 + dy**2) / vars_cutoff / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious



class CocoEvaluator_custom(CocoEvaluator_base):
    def __init__(self, coco_gt, iou_types, useCats=True):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = COCO(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_custom(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats