datasets = [
    {   
        'type': 'coco_pose_llava',
        'ann_file': 'data/coco/annotations/person_keypoints_val2017.json',
        'img_prefix': 'data/coco/val2017',
        'test_mode': True
    },
    {   
        'type': 'coco_pose_llava',
        'ann_file': 'data/pose_datasets/HumanArt/annotations/validation_humanart.json',
        'img_prefix': 'data/pose_datasets',
        'test_mode': True
    },
    {   
        'type': 'crowdpose_llava',
        'ann_file': 'data/crowdpose/annotations/crowdpose_test.json',
        'img_prefix': 'data/crowdpose/images',
        'test_mode': True
    },
    # ------------------------------------------------------------------
    {   
        'type': 'unikpt_llava',
        'ann_file': 'data/pose_datasets/AP10K/test.json',
        'img_prefix': 'data/pose_datasets',
        'test_mode': True
    },
    {   
        'type': 'unikpt_llava',
        'ann_file': 'data/pose_datasets/macaque/annotations/macaque_test.json',
        'img_prefix': 'data/pose_datasets/macaque/v1/images',
        'test_mode': True
    },
    {   
        'type': 'unikpt_llava',
        'ann_file': 'data/pose_datasets/300w/face_landmarks_300w_valid.json',
        'img_prefix': 'data/pose_datasets/300w/dataset',
        'eval_pck': True,
        'test_mode': True
    },
    {   
        'type': 'unikpt_llava',
        'ann_file': 'data/pose_datasets/animalkingdom/test.json',
        'img_prefix': 'data/pose_datasets',
        'eval_pck': True,
        'test_mode': True
    },
    {   
        'type': 'unikpt_llava',
        'ann_file': 'data/pose_datasets/VinegarFly/annotations/fly_test.json',
        'img_prefix': 'data/pose_datasets/VinegarFly/images',
        'eval_pck': True,
        'test_mode': True
    },
    {   
        'type': 'unikpt_llava',
        'ann_file': 'data/pose_datasets/DesertLocust/annotations/locust_test.json',
        'img_prefix': 'data/pose_datasets/DesertLocust/images',
        'eval_pck': True,
        'test_mode': True
    },
]