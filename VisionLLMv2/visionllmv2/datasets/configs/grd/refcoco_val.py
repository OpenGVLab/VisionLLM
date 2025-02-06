datasets = [
    # RefCOCO
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcoco-unc/instances_val.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcoco-unc/instances_testA.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcoco-unc/instances_testB.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    # RefCOCO+
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcocoplus-unc/instances_val.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcocoplus-unc/instances_testA.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcocoplus-unc/instances_testB.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    # RefCOCOg
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcocog-umd/instances_val.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
    {   
        'type': 'refcoco_llava',
        'ann_file': 'data/coco2014/annotations/refcocog-umd/instances_test.json',
        'img_prefix': 'data/coco2014/train2014',
        'with_mask': True,
        'test_mode': True
    },
]