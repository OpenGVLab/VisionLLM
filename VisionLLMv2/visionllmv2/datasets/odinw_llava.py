import sys
import numpy as np
import transformers
import json

from mmdet.datasets import CocoDataset
from .coco_llava import CocoLlavaDataset

class OdinwLlavaDataset(CocoLlavaDataset):
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
        # read classe from anno file
        coco_gt = json.load(open(ann_file,'r'))
        CocoDataset.CLASSES = tuple(i['name'] for i in coco_gt['categories'])
        self.num_classes = len(self.CLASSES)
        
        super().__init__(
            ann_file,
            img_prefix,
            tokenizer,
            data_args,
            test_mode,
            max_gt_per_img,
            with_mask
        )
        self.task = 'det'
        self.dataset_name = 'odinw'

        
