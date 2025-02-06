import sys
import random

import torch
import torch.nn as nn

from .point import Point
from .polygon import Polygon
from .scribble import Scribble
from .circle import Circle

from ..utils import boxes_to_masks, masks_to_boxes


class ShapeSampler(nn.Module):
    def __init__(self, is_train=True, mode=None):
        super().__init__()
        self.max_candidate = 20
        candidate_probs = [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]
        candidate_names = ["Point", "Polygon", "Scribble", "Circle", "Box", "Mask"]
        if not is_train:
            candidate_probs = [0.0 for x in range(len(candidate_names))]
            candidate_probs[candidate_names.index(mode)] = 1.0
        self.shape_prob = candidate_probs      
        self.shape_candidate = candidate_names
        self.is_train = is_train

    def forward(self, gt_boxes, gt_masks):
        masks = gt_masks  # [n, h, w]
        boxes = gt_boxes  # [n, 4], xyxy in image size (after aug)

        if len(masks) == 0:
            gt_masks = torch.zeros(masks.shape[-2:]).bool()
            rand_masks = torch.zeros(masks.shape[-2:]).bool()
            return {'gt_masks': gt_masks[None,:], 'rand_shape': torch.stack([rand_masks]), 'types': ['none']}
        indices = [x for x in range(len(masks))]  # [0, 1, 2, ...]
 
        if self.is_train:
            random.shuffle(indices)
            indices = indices[:self.max_candidate]
            candidate_mask = masks[indices]  # [n, h, w] 
            candidate_box = boxes[indices]   # [n, 4]    
        else:
            candidate_mask = masks
            candidate_box = boxes

        h, w = candidate_mask.shape[-2:]
        img_shape = torch.tensor([h, w])
        
        draw_func_names = random.choices(self.shape_candidate, weights=self.shape_prob, k=len(candidate_mask))  
        # use draw func to sample visual prompts for each gt object.
        rand_shapes = []
        for draw_func_name, mask, box in zip(draw_func_names, candidate_mask, candidate_box):
            # mask: [h, w], box: [4,]
            if draw_func_name in ["Point", "Polygon", "Scribble", "Circle"]:
                draw_func = build_draw_func(draw_func_name, self.is_train)
                rand_shape = draw_func.draw(mask=mask, box=box)  # [h, w]
                rand_shapes.append(rand_shape)
            elif draw_func_name == "Box":
                rand_shape = boxes_to_masks(box.unsqueeze(0), img_shape)[0]
                rand_shapes.append(rand_shape)
            elif draw_func_name == "Mask":
                rand_shape = mask
                rand_shapes.append(rand_shape)
            else:
                raise NotImplementedError

        types = [x for x in draw_func_names]
        for i in range(0, len(rand_shapes)):
            if rand_shapes[i].sum() == 0:
                candidate_mask[i] = candidate_mask[i] * 0
                types[i] = 'none'
        # print("Warning: visual sampler does not sample any foreground points, delete this invalid visual prompt.")
        valid_mask = candidate_mask.sum(1).sum(1) > 0  # [n,]
        candidate_mask = candidate_mask[valid_mask]
        rand_shapes = torch.stack(rand_shapes).bool()
        rand_shapes = rand_shapes[valid_mask]
        indices = [x for i, x in enumerate(indices) if valid_mask[i] == 1]
        types = [x for i, x in enumerate(types) if valid_mask[i] == 1]

        
        # candidate_mask: (c,h,w), bool. rand_shape: (c, iter, h, w), bool. types: list(c)
        return {'gt_masks': candidate_mask, 'rand_shape': rand_shapes, 'types': types, 'sampler': self,
                'indices': indices}

def build_draw_func(draw_name, is_train):
    # ["Point", "Polygon", "Scribble", "Circle", "Box", "Mask"]
    if draw_name == 'Point':
        return Point(is_train=is_train)
    elif draw_name == 'Polygon':
        return Polygon(is_train=is_train)
    elif draw_name == 'Scribble':
        return Scribble(is_train=is_train)
    elif draw_name == "Circle":
        return Circle(is_train)
    else: 
        raise NotImplementedError
    