import torch
import numpy as np
import random
from mmcv import Config
from torch.utils.data import ConcatDataset

# chat
from visionllmv2.datasets.llava_data import LazySupervisedDataset
# det
from visionllmv2.datasets.coco_llava import CocoLlavaDataset
from visionllmv2.datasets.lvis_llava import LvisLlavaDataset
from visionllmv2.datasets.crowdhuman_llava import CrowdhumanLlavaDataset
from visionllmv2.datasets.det_llava import DetLlavaDataset
from visionllmv2.datasets.odinw_llava import OdinwLlavaDataset
from visionllmv2.datasets.cod_llava import CODLlavaDataset  
from visionllmv2.datasets.sod_llava import SODLlavaDataset
from visionllmv2.datasets.dota_llava import DOTALlavaDataset
from visionllmv2.datasets.sar_llava import SARLlavaDataset
from visionllmv2.datasets.deeppcb_llava import DeepPCBLlavaDataset
from visionllmv2.datasets.neu_llava import NEULlavaDataset
# det cap (referential dialogue)
from visionllmv2.datasets.flickr30k_llava import Flickr30kLlavaDataset
from visionllmv2.datasets.groma_llava import GromaLlavaDataset
# grd
from visionllmv2.datasets.refcoco_llava import RefCocoLlavaDataset
from visionllmv2.datasets.reasonseg_llava import ReasonsegLlavaDataset
# seg
from visionllmv2.datasets.ade20k_llava import ADE20KLlavaDataset
from visionllmv2.datasets.cityscape_llava import CityscapesLlavaDataset
from visionllmv2.datasets.mapillary_llava import MapillaryLlavaDataset
from visionllmv2.datasets.loveda_llava import LovedaLlavaDataset
from visionllmv2.datasets.medical_mr_llava import MedicalMrDataset
# pose
from visionllmv2.datasets.coco_pose_llava import CocoPoseLlavaDataset
from visionllmv2.datasets.crowdpose_llava import CrowdposeLlavaDataset
from visionllmv2.datasets.unikpt_llava import UniKPTLlavaDataset
# region caption
from visionllmv2.datasets.vg import VisualGenome
from visionllmv2.datasets.refcoco import RefCOCO
from visionllmv2.datasets.flickr30k import Flickr30k
from visionllmv2.datasets.vcr import VCRDataset
from visionllmv2.datasets.vcr_vqa import VCRVQA
from visionllmv2.datasets.as_caption import ASCaption
from visionllmv2.datasets.as_vqa import ASVQA
from visionllmv2.datasets.osprey import OspreyConversations, OspreyDetailedDescription, \
    OspreyLVISPosNeg, OspreyPartLevel, OspreyShortForm
# region recognition
from visionllmv2.datasets.v3det import V3DetRecognition, CocoRecognition
from visionllmv2.datasets.lvis import LVISRecognition
# visual prompt
from visionllmv2.datasets.coco_interactive import CocoInteractiveDataset, CocoInteractiveTest
from visionllmv2.datasets.sa_interactive import SAInteractiveDataset
# count
from visionllmv2.datasets.count_object_text_llava import CountObjectTextLlavaDataset
from visionllmv2.datasets.count_object_visual_llava import CountObjectVisualLlavaDataset
# image gen/edit
from visionllmv2.datasets.text2img import CC3MDataset, LaionDataset, MJDataset, JourneyDBDataset
from visionllmv2.datasets.ip2p import IP2PDataset, SeedXDataset
# mmic
from visionllmv2.datasets.mmic_text import InContextTextDataset
from visionllmv2.datasets.mmic_mask import InContextMaskDataset


def build_multi_datasets(dataset_cfg_file, tokenizer, data_args, **kwargs):
    dataset_cfgs = Config.fromfile(dataset_cfg_file)
    dataset_cfgs = dataset_cfgs.datasets
    assert isinstance(dataset_cfgs, list)
    datasets = [build_dataset(cfg, tokenizer=tokenizer, data_args=data_args, **kwargs) for cfg in dataset_cfgs]
    return ConcatDataset(datasets)


def build_dataset(dataset_cfg, tokenizer, data_args, **kwargs):
    dataset_type = dataset_cfg.pop('type')
    ratio = dataset_cfg.pop('ratio', 1)
    # chat
    if dataset_type == 'llava_data':
        dataset = LazySupervisedDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args,
        )    
    # det/insetseg
    elif dataset_type == 'coco_llava':
        dataset = CocoLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'lvis_llava':
        dataset = LvisLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'crowdhuman_llava':
        dataset = CrowdhumanLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'det_llava':
        dataset = DetLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'odinw_llava':
        dataset = OdinwLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # det cap
    elif dataset_type == 'flickr30k_llava':
        dataset = Flickr30kLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'groma_llava':
        dataset = GromaLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # grd
    elif dataset_type == 'refcoco_llava':
        dataset = RefCocoLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'reasonseg_llava':
        dataset = ReasonsegLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'cod_llava':
        dataset = CODLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'sod_llava':
        dataset = SODLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'dota_llava':
        dataset = DOTALlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'sar_llava':
        dataset = SARLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == "neu_llava":
        dataset = NEULlavaDataset(
            **dataset_cfg, 
            tokenizer=tokenizer, 
            data_args=data_args
        )
    elif dataset_type == "deeppcb_llava":
        dataset = DeepPCBLlavaDataset(
            **dataset_cfg, 
            tokenizer=tokenizer, 
            data_args=data_args
        )
    # semseg
    elif dataset_type == 'ade20k_llava':
        dataset = ADE20KLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'cityscape_llava':
        dataset = CityscapesLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'mapillary_llava':
        dataset = MapillaryLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'loveda_llava':
        dataset = LovedaLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'medical_mr_llava':
        dataset = MedicalMrDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # pose
    elif dataset_type == 'coco_pose_llava':
        dataset = CocoPoseLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'crowdpose_llava':
        dataset = CrowdposeLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'unikpt_llava':
        dataset = UniKPTLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # region caption
    elif dataset_type == 'vg':
        dataset = VisualGenome(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'refcoco':
        dataset = RefCOCO(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'flickr30k':
        dataset = Flickr30k(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'vcr':
        dataset = VCRDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'vcr_vqa':
        dataset = VCRVQA(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'as_caption':
        dataset = ASCaption(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'as_vqa':
        dataset = ASVQA(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'osprey_conversation':
        dataset = OspreyConversations(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'osprey_detail':
        dataset = OspreyDetailedDescription(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'osprey_lvis':
        dataset = OspreyLVISPosNeg(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'osprey_part':
        dataset = OspreyPartLevel(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'osprey_short':
        dataset = OspreyShortForm(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # region recognition
    elif dataset_type == 'v3det_recognition':
        dataset = V3DetRecognition(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'coco_recognition':
        dataset = CocoRecognition(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'lvis_recognition':
        dataset = LVISRecognition(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # visual prompt
    elif dataset_type == 'coco_interactive':
        dataset = CocoInteractiveDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'coco_interactive_test':
        dataset = CocoInteractiveTest(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        ) 
    elif dataset_type == 'sa_interactive':
        dataset = SAInteractiveDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # count
    elif dataset_type == 'count_object_text':
        dataset = CountObjectTextLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'count_object_visual':
        dataset = CountObjectVisualLlavaDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # image generation
    elif dataset_type == 'cc3m':
        dataset = CC3MDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        ) 
    elif dataset_type == 'laion':
        dataset = LaionDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        ) 
    elif dataset_type == 'midjourney':
        dataset = MJDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        ) 
    elif dataset_type == 'journeydb':
        dataset = JourneyDBDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        ) 
    # image edit
    elif dataset_type == 'ip2p':
        dataset = IP2PDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        ) 
    elif dataset_type == 'seedx':
        dataset = SeedXDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    # mmic
    elif dataset_type == 'ic_text':
        dataset = InContextTextDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    elif dataset_type == 'ic_mask':
        dataset = InContextMaskDataset(
            **dataset_cfg,
            tokenizer=tokenizer,
            data_args=data_args
        )
    else:
        raise NotImplementedError

    if ratio < 1:
        print(f'randomly sample {ratio} of the dataset {dataset_type}: {int(ratio * len(dataset))}' )
        # random_indices = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        dataset_indices = [idx for idx in range(len(dataset))]
        random_indices = random.sample(dataset_indices, int(ratio * len(dataset)))
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.task = getattr(dataset, 'task', 'chat')
        return subsample_dataset
    return dataset
