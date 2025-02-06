import copy
import random
import os
import numpy as np
import torch

from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from pycocoevalcap.eval import COCOEvalCap

from PIL import Image
import mmcv
import os.path as osp
import tempfile
from collections import OrderedDict

from ..constant import DEFAULT_TOKENS
from ..mm_utils import expand2square, dynamic_preprocess
from .llava_data import preprocess, preprocess_multimodal
from .utils import boxes_to_masks

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
    from .llava_data import TCSLoader
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

QUESTIONS = [
    '<spi_descript>',
]

REFG_QUESTIONS = [
    'Can you provide me with a brief description of <spi_descript> in the picture?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in short?",
    'What can you tell me about <spi_descript> in the image?',
    "I'd like to know more about the area in the photo labeled <spi_descript>. Can you give me a brief description?",
    'Could you describe <spi_descript> in the picture in short?',
    'What content can you give me about <spi_descript> in the photo?',
    'Please provide me with a short description of <spi_descript> in the image.',
    'Can you give me a brief account of the region labeled as <spi_descript> in the picture?',
    "I'm interested in learning more about <spi_descript> in the photo. Can you describe it in short?",
    'What is the region outlined by <spi_descript> in the picture like? Could you give me a brief description?',
    'Can you provide me with a brief description of <spi_descript> in the picture, please?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in short, please?",
    'What can you tell me about <spi_descript> in the image, exactly?',
    "I'd like to know more about <spi_descript>. Can you give me a brief description?",
    'Could you describe the region shown as <spi_descript> in the picture in short, please?',
    'What content can you give me about <spi_descript> in the photo, please?',
    'Please provide me with a short description of <spi_descript> in the image, please.',
    'Can you give me a brief account of the region labeled as <spi_descript> in the picture, please?',
    "I'm interested in learning more about <spi_descript> in the photo. Can you describe it in short, please?",
    'What is <spi_descript> in the picture like, please? Could you give me a brief description?',
]


class RefCOCO(CocoDataset):
    CLASSES = ('object',)

    def __init__(self,
                 ann_file, 
                 img_prefix,
                 tokenizer,
                 data_args,
                 test_mode=False,
                 max_gt_per_img=15,
                 with_mask=True,
                 test_format='bbox',
                 filter_captions=True,  # filter too short captions
                 use_tcs_loader=False,
                 ):
        self.task = 'region_refer'
        self.dataset_name = 'refcoco'

        # conversation
        self.tokenizer = tokenizer
        self.img_prefix = img_prefix
        self.data_args = data_args
        self.img_processor = data_args.img_processor
        self.use_im_start_end = data_args.use_im_start_end
        self.image_size = data_args.image_size

        self.with_mask = with_mask
        self.test_format = test_format  # 'bbox' or 'mask'
        self.filter_captions = filter_captions
        self.max_gt_per_img = max_gt_per_img
        self.test_mode = test_mode

        # tcs loader
        if use_tcs_loader:
            assert has_tcs_loader
            self.tcs_loader = TCSLoader('~/petreloss.conf') 
        else:
            self.tcs_loader = None

        image_mean = self.img_processor.image_mean
        image_mean = [x * 255 for x in image_mean]
        image_std = self.img_processor.image_std
        image_std = [x * 255 for x in image_std]

        img_norm_cfg = dict(
            mean=image_mean,
            std=image_std,
            to_rgb=True)

        # file_client_args
        file_client_args = dict(backend='petrel') if use_tcs_loader else dict(backend='disk')

        train_pipeline = [
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
            dict(type='Resize', img_scale=(self.image_size, self.image_size), keep_ratio=False),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=self.image_size),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                    else ['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
            dict(type='Resize', img_scale=(self.image_size, self.image_size), keep_ratio=False),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=self.image_size),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'] if self.with_mask
                                    else ['img', 'gt_bboxes', 'gt_labels']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,  # need gt bbox as region
            pipeline=pipeline)
        
        super(CocoDataset, self).__init__(**dataset_cfg)

        # FIXME:  '<image>\n' as begin_str
        self.begin_str = '<image>\n I will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # TODO: obtain images that contain annotation
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        num_remove_images = 0  # filter too short captions
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            if len(ann_ids) == 0:
                continue
            # filter too short sentences during training.
            if self.filter_captions and not self.test_mode:
                ann = self.coco.load_anns(ann_ids)[0]  # train only has one caption, per sample
                if len(ann['caption'].split(' ')) < 3:
                    num_remove_images += 1
                    continue
            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        if self.filter_captions:
            print(f'Filtered {num_remove_images} from  {self.ann_file} ')
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(ann['caption'])  # caption
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


    def preprocess_data(self, data_item):
        # image = data_item['img'].data           # [c, h, w], 336x336
        label = data_item['gt_labels'][0]       # caption, annotation file has been preprocessed, each image has one caption.
        bboxes = data_item['gt_bboxes'].data    # [n, 4], xyxy in image shape (after aug)
        masks = data_item['gt_masks'].data      # [n, h, w]
        img_shape = data_item['img_metas'].data['img_shape']
        # NOTE: here we do not convert boxes to cxcywh in [0, 1]
        # bboxes = bbox_xyxy_to_cxcywh(bboxes)
        # bboxes = self.normalize_box_coordinates(bboxes, img_shape)

        # get image
        file_name = data_item['img_metas'].data['ori_filename'] 
        image_path = os.path.join(self.img_prefix, file_name)
        if self.tcs_loader is not None:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        if self.data_args.image_aspect_ratio == 'anyres':
            image = dynamic_preprocess(image, image_size=self.data_args.image_size, max_num=self.data_args.image_max_tile)  # list[pil_img]
            image = [self.img_processor.preprocess(x, return_tensors='pt')['pixel_values'][0] for x in image]
            image = torch.stack(image)  # [1 + n_tile, 3, h, w]
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4
            image_token_len = image_token_len * len(image)
        else:
            image = self.img_processor.preprocess(image, do_center_crop=False, return_tensors='pt')['pixel_values'][0]  # resize
            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(self.image_size, self.image_size),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)
            image_token_len = int((self.data_args.image_size // 14) ** 2)
            if self.data_args.use_pixelshuffle:
                image_token_len = image_token_len // 4

        # generate regions
        # randomly select bbox or mask as region
        if not self.test_mode and self.with_mask: # train
            if torch.randn(1) > 0:
                regions = boxes_to_masks(bboxes, img_shape)
            else:
                regions = masks
        else: # test
            if self.test_format == 'bbox':
                regions = boxes_to_masks(bboxes, img_shape)
            elif self.test_format == 'mask':
                regions = masks
            else:
                raise NotImplementedError

        
        # -----------------------------------------------------
        # chat
        conversations = []
        # question
        question_template = REFG_QUESTIONS[0] if self.test_mode else random.choice(REFG_QUESTIONS)
        region_str = DEFAULT_TOKENS['sor'] + 'region1' + DEFAULT_TOKENS['reg'] + DEFAULT_TOKENS['eor']  # '<reg>region1<region></reg>'
        question = question_template.replace('<spi_descript>', region_str)
        question = '<image>\n' + question
        message1 = {
            'from': 'human',
            'value': question
        }
        conversations.append(message1)
        # answer
        answer = label.strip().lower().capitalize()
        if not answer.endswith('.'):
            answer = answer + '.'
        message2 = {
            'from': 'gpt',
            'value': answer
        }
        conversations.append(message2)

        sources = preprocess_multimodal(copy.deepcopy([conversations]))
        data_dict = preprocess(
            sources=sources,  # list[list[dict]], first list length=1, second list length=num_rounds
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            has_image=True,
            image_token_len=image_token_len
        ) # keys: "input_ids", "labels", size of [1, L]
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )

        # -----------------------------------------------------
        # update image and img_metas
        data_dict['image'] = image
        img_metas = data_item['img_metas'].data
        img_metas['task'] = self.task
        img_metas['dataset_name'] = self.dataset_name
        img_metas['conversations'] = conversations
        data_dict['img_metas'] = img_metas
        # update regions
        data_dict['regions'] = regions  # [n, h, w]
        return data_dict


    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)  # after mmdet pipeline
        data_dict = self.preprocess_data(data_item)
        return data_dict
    
    # for pycocoevalcap evaluation
    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.
        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.
        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".
        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()   
        assert isinstance(results[0], str)  # [str1], [str2], ... -> [str1, str2, ...]

        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            data = dict()
            data['image_id'] = img_id
            data['caption'] = result
            json_results.append(data)
        result_files['caption'] = f'{outfile_prefix}.caption.json'
        mmcv.dump(json_results, result_files['caption'])
        return result_files


    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir


    def evaluate(self,
                 results,
                 metric='caption',
                 logger=None,
                 jsonfile_prefix=None):
        """Evaluation in COCO protocol.
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['caption']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
            
        # format results
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix) 
        # get results
        predictions = mmcv.load(result_files[metric])

        # evaluate
        coco_gt = self.coco
        coco_result = coco_gt.loadRes(predictions)

        coco_eval = COCOEvalCap(coco_gt, coco_result)
    
        # evaluate on a subset of images by setting
        # coco_eval.params['image_id'] = coco_result.getImgIds()
        # please remove this line when evaluating the full validation set
        # coco_eval.params['image_id'] = coco_result.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.params['image_id'] = coco_result.getImgIds()
        coco_eval.evaluate()
        eval_results = OrderedDict()

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
            eval_results[metric] = score

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results