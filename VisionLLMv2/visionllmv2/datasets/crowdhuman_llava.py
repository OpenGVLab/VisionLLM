import sys
import numpy as np
import transformers

from .coco_llava import CocoLlavaDataset

class CrowdhumanLlavaDataset(CocoLlavaDataset):
    CLASSES = (
        'person',
    )
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
        self.dataset_name = 'crowdhuman'
